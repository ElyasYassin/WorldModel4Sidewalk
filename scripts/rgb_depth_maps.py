"""Monocular depth from the front RGB camera -> E4-C-style elevation grids (E5, option C).

A frozen pretrained depth model (default: Depth Anything V2 Metric Outdoor Small) predicts metric depth per frame.
Pixels are back-projected with the camera intrinsics and mount (go2wm.sim.sensors.RGB_SPEC) and the recorded base
pose, then binned like the range sensors (bin_elevation, 5 cm robot-centred grid). Two variants are written:

  rgbda      raw metric depth from the model;
  rgbdacal   the same depth recalibrated per frame with a scale and shift in inverse depth (1/d = a/d_pred + b),
             fitted on the pixels whose rays would meet a flat ground CAL_BAND ahead at the ground height under the
             base. That is what a robot can do without other range sensors: it knows its camera height and attitude
             from leg kinematics and the IMU (simulator truth here, as for the other maps). The fit drops outlier
             pixels, but a step inside the band still biases it; that is part of the method.

The raw metric output is ~16x too far on these 224 px, 100 deg sim frames (the outdoor model expects car-camera
scenes), so only rgbdacal is usable; rgbda is kept to document that.

Writes <data>/derived/<variant>/<episode_id>.npz with sensor_<variant>_elev_m (float16) and sensor_<variant>_count
(uint8), so `terrain_source: "sensor:rgbda@10"` and the accuracy tooling work unchanged. Done episodes are skipped.

  python scripts/rgb_depth_maps.py --data data/go2_e4c_test
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from go2wm.common.config import resolve_path  # noqa: E402
from go2wm.common.schema import STATE_FIELDS  # noqa: E402
from go2wm.data.episodes import list_episodes, load_episode, load_rgb  # noqa: E402
from go2wm.models.terrain_map import episode_height_fn  # noqa: E402
from go2wm.sim.sensors import RGB_SPEC, bin_elevation, pinhole_directions, pitch_matrix  # noqa: E402

MODEL = "depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf"
RANGE = (0.15, 8.0)
CAL_BAND = (0.5, 2.5)  # forward distance [m] of the flat-ground pixels used for the per-frame calibration


def quat_to_R(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def camera_rays() -> np.ndarray:
    """Body-frame-oriented camera rays scaled so that ray * z_depth = point (z-depth along the optical axis)."""
    d = pinhole_directions(RGB_SPEC["hfov_deg"], RGB_SPEC["hfov_deg"] * RGB_SPEC["height"] / RGB_SPEC["width"],
                           RGB_SPEC["width"], RGB_SPEC["height"])
    d = d / d[:, :1]  # unit forward component
    return d @ pitch_matrix(RGB_SPEC["pitch_deg"]).T


def frame_points(depth: np.ndarray, state: np.ndarray, rays_b: np.ndarray, z_offset: float) -> np.ndarray:
    """depth [H*W] z-depth -> world-frame points [N, 3] in the sensor-grid height frame."""
    R = quat_to_R(state[STATE_FIELDS["quat_w"]])
    # only rays below the horizon (known from the IMU): monocular depth for sky and far background is
    # unreliable and, after recalibration, lands as phantom tall points a few metres ahead
    ok = (depth > RANGE[0]) & (depth < RANGE[1]) & ((rays_b @ R.T)[:, 2] < -0.02)
    p_b = np.asarray(RGB_SPEC["mount_pos"]) + rays_b[ok] * depth[ok, None]
    p = p_b @ R.T + state[STATE_FIELDS["pos_w"]]
    p[:, 2] += z_offset
    return p


def calib_inverse_affine(depth: np.ndarray, state: np.ndarray, rays_b: np.ndarray,
                         ground_z_local: float) -> tuple[float, float]:
    """(a, b) with 1/depth_true = a / depth_pred + b, fitted on pixels whose rays meet a flat ground CAL_BAND
    ahead (target: the z-depth of that flat-ground hit). Pixels are chosen from the camera geometry alone
    (known height and attitude), not from the predicted depth. Robust: one refit after dropping residuals
    beyond 2.5 MAD (steps, obstacles). Returns (nan, nan) when too few pixels qualify."""
    R = quat_to_R(state[STATE_FIELDS["quat_w"]])
    mount_w = R @ np.asarray(RGB_SPEC["mount_pos"]) + state[STATE_FIELDS["pos_w"]]
    ray_w = rays_b @ R.T  # world direction per unit z-depth
    down = ray_w[:, 2] < -0.02
    t = np.full(len(depth), np.nan)
    t[down] = (ground_z_local - mount_w[2]) / ray_w[down, 2]  # z-depth of the flat-ground hit
    fwd = t * rays_b[:, 0]  # forward distance in the base frame
    sel = down & (fwd > CAL_BAND[0]) & (fwd < CAL_BAND[1]) & (depth > 1e-3)
    if sel.sum() < 50:
        return float("nan"), float("nan")
    x, y = 1.0 / depth[sel], 1.0 / t[sel]
    keep = np.ones(len(x), bool)
    for _ in range(2):
        a, b = np.polyfit(x[keep], y[keep], 1)
        r = y - (a * x + b)
        mad = np.median(np.abs(r[keep] - np.median(r[keep]))) + 1e-9
        keep = np.abs(r - np.median(r[keep])) < 2.5 * 1.4826 * mad
        if keep.sum() < 30:
            break
    return float(a), float(b)


def apply_inverse_affine(depth: np.ndarray, a: float, b: float) -> np.ndarray:
    inv = a / np.maximum(depth, 1e-6) + b
    return np.where(inv > 1e-3, 1.0 / inv, np.nan)  # non-positive inverse depth = beyond infinity: dropped


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--max-episodes", type=int, default=0)
    a = ap.parse_args()
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    proc = AutoImageProcessor.from_pretrained(a.model)
    net = AutoModelForDepthEstimation.from_pretrained(a.model).to(dev).eval().half()
    root = resolve_path(a.data)
    outs = {v: root / "derived" / v for v in ("rgbda", "rgbdacal")}
    for d in outs.values():
        d.mkdir(parents=True, exist_ok=True)
    rays_b = camera_rays()
    H, W = RGB_SPEC["height"], RGB_SPEC["width"]
    paths = list_episodes(root)[: a.max_episodes or None]
    t0 = time.time()
    scales_all = []
    for n, p in enumerate(paths):
        ep = load_episode(p)
        if all((d / f"{ep.episode_id}.npz").exists() for d in outs.values()):
            continue
        frames = load_rgb(p)
        S = ep["states"]
        hfn = episode_height_fn(ep.meta)
        # sensor grids are in the tile-surface frame; states are in the env frame. The per-episode offset is
        # the median of (surface under the base - recorded ground height) (constant per tile, see E4-C (2)).
        surf = hfn(S[:, 0], S[:, 1]) if hfn is not None else np.zeros(len(S))
        z_off = float(np.nanmedian(surf - ep["gt_ground_z_m"])) if "gt_ground_z_m" in ep.arrays else 0.0
        ground_local = ep["gt_ground_z_m"] if "gt_ground_z_m" in ep.arrays else surf - z_off
        depths = []
        for i in range(0, len(frames), a.batch):
            x = proc(images=list(frames[i : i + a.batch]), return_tensors="pt")
            with torch.no_grad():
                pred = net(pixel_values=x["pixel_values"].to(dev).half()).predicted_depth.float()
            pred = torch.nn.functional.interpolate(pred[:, None], size=(H, W), mode="bilinear", align_corners=False)[:, 0]
            depths.append(pred.cpu().numpy())
        depths = np.concatenate(depths).reshape(len(frames), -1)
        res = {v: ([], []) for v in outs}
        yaw = np.arctan2(2 * (S[:, 3] * S[:, 6] + S[:, 4] * S[:, 5]), 1 - 2 * (S[:, 5] ** 2 + S[:, 6] ** 2))
        for t in range(len(frames)):
            ca, cb = calib_inverse_affine(depths[t], S[t], rays_b, float(ground_local[t]))
            scales_all.append(ca)
            for v, dep in (("rgbda", depths[t]), ("rgbdacal", apply_inverse_affine(depths[t], ca, cb))):
                e, c = bin_elevation(frame_points(dep, S[t], rays_b, z_off), S[t, :2], float(yaw[t]), z_ref=0.0)
                res[v][0].append(e)
                res[v][1].append(c)
        for v, d in outs.items():
            np.savez_compressed(d / f"{ep.episode_id}.npz", **{f"sensor_{v}_elev_m": np.stack(res[v][0]),
                                                               f"sensor_{v}_count": np.stack(res[v][1])})
        if (n + 1) % 8 == 0:
            print(f"  {n + 1}/{len(paths)} ({time.time() - t0:.0f}s), median inverse-depth scale so far {np.nanmedian(scales_all):.3f}",
                  flush=True)
    if scales_all:
        print(f"calibration a (inverse-depth scale): median {np.nanmedian(scales_all):.3f}, p10-p90 {np.nanpercentile(scales_all, 10):.3f}"
              f"-{np.nanpercentile(scales_all, 90):.3f}")
    print(f"done: {len(paths)} episodes -> {root / 'derived'}/rgbda*")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
