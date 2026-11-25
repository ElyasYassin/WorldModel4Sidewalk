"""Exteroceptive sensor models for the Go2 sensor comparison (E4-C).

Range sensors are ray casts against the terrain mesh with mount poses, fields of view, point
budgets, range limits, dropouts and range noise modelled on real devices (approximate, from public
data sheets). Every frame is reduced to a robot-centred, yaw-aligned elevation grid (mean hit height
per 5 cm cell + hit count), which is what is recorded (raw point clouds would be ~100x larger).

  depth_d435   RealSense D435-like depth camera on the head, 30 deg down; 87 x 58 deg, 106 x 60 rays
               (848 x 480 / 8), 0.2-4 m, sigma(z) = 2 mm + 5 mm * z^2, 3 % dropout
  depth_d555   RealSense D555-like (95 mm baseline, D450 optics, as D455): same mount and FOV, 0.52-6 m
               (datasheet ideal range 0.6-6 m; ~0.52 m min-z at full resolution), half the D435 noise
               (sigma(z) = 2 mm + 2.5 mm * z^2, i.e. <~1 % at 4 m; D455 is quoted <2 % at 4 m), 3 % dropout
  depth_d405   RealSense D405-like close-range stereo (18 mm baseline, no projector): on the chin, pitched 60 deg
               down so it looks at the ground ~0.3-0.75 m ahead of the base; 87 x 58 deg, 0.07-0.5 m (datasheet ideal
               range 7-50 cm), sigma(z) = 0.2 mm + 4 mm * z^2 (~0.5 mm at 25 cm, ~1.2 mm at 50 cm), 3 % dropout.
               Legs are not ray-cast: the front feet that swing through its view are not modelled.
  l1           Unitree L1-like (stock Go2 Pro/EDU): 360 x 90 deg dome whose axis points forward, 15 deg
               down, under the head; 21.6 k pts/s -> 2160 per 0.1 s, non-repetitive (fresh random
               directions every frame), 0.05-30 m, sigma 2 cm
  mid360       Livox Mid-360-like on the back, upright: 360 deg x (-7, +52) deg, 200 k pts/s -> 20 k per
               frame, non-repetitive, 0.1-40 m, sigma 2 cm
  mid360_tilt  the same, pitched 30 deg forward (a common Go2 mount to see the ground near the robot)
  height_scan  the rough policy's own scanner (reference; recorded separately as scan_height_m)

RGB (front camera, 224 x 224, ~100 deg) is rendered by Isaac and stored per episode as an MP4.

Frames: sensor frame x forward, y left, z up. Mount = (position in the base frame [m], pitch [deg],
positive = nose down). Pure numpy.
"""

from __future__ import annotations

from typing import Any

import numpy as np

ELEV_GRID = {"x0": -1.0, "x1": 3.5, "y0": -1.25, "y1": 1.25, "res": 0.05}  # robot frame, yaw-aligned
GRID_SHAPE = (int(round((ELEV_GRID["x1"] - ELEV_GRID["x0"]) / ELEV_GRID["res"])),
              int(round((ELEV_GRID["y1"] - ELEV_GRID["y0"]) / ELEV_GRID["res"])))

SENSOR_SPECS: dict[str, dict[str, Any]] = {
    "depth_d435": {"kind": "pinhole", "mount_pos": [0.30, 0.0, 0.08], "pitch_deg": 30.0, "hfov_deg": 87.0,
                   "vfov_deg": 58.0, "width": 106, "height": 60, "range_m": [0.2, 4.0],
                   "noise": {"model": "depth_quadratic", "a": 0.002, "b": 0.005}, "dropout": 0.03,
                   "device": "Intel RealSense D435 (approx.)"},
    "depth_d555": {"kind": "pinhole", "mount_pos": [0.30, 0.0, 0.08], "pitch_deg": 30.0, "hfov_deg": 87.0,
                   "vfov_deg": 58.0, "width": 106, "height": 60, "range_m": [0.52, 6.0],
                   "noise": {"model": "depth_quadratic", "a": 0.002, "b": 0.0025}, "dropout": 0.03,
                   "device": "Intel RealSense D555 (approx.: 95 mm baseline like D455; no official RMS figure)"},
    "depth_d405": {"kind": "pinhole", "mount_pos": [0.33, 0.0, -0.03], "pitch_deg": 60.0, "hfov_deg": 87.0,
                   "vfov_deg": 58.0, "width": 106, "height": 60, "range_m": [0.07, 0.5],
                   "noise": {"model": "depth_quadratic", "a": 0.0002, "b": 0.004}, "dropout": 0.03,
                   "device": "Intel RealSense D405 (approx.: 18 mm baseline, close range)"},
    "l1": {"kind": "dome", "mount_pos": [0.28, 0.0, -0.02], "pitch_deg": 15.0, "n_rays": 2160, "range_m": [0.05, 30.0],
           "noise": {"model": "range_gaussian", "sigma": 0.02}, "dropout": 0.01, "device": "Unitree L1 4D LiDAR (approx.)"},
    "mid360": {"kind": "band", "mount_pos": [-0.05, 0.0, 0.12], "pitch_deg": 0.0, "elev_deg": [-7.0, 52.0],
               "n_rays": 20000, "range_m": [0.1, 40.0], "noise": {"model": "range_gaussian", "sigma": 0.02},
               "dropout": 0.01, "device": "Livox Mid-360 (approx.), upright"},
    "mid360_tilt": {"kind": "band", "mount_pos": [0.05, 0.0, 0.12], "pitch_deg": 30.0, "elev_deg": [-7.0, 52.0],
                    "n_rays": 20000, "range_m": [0.1, 40.0], "noise": {"model": "range_gaussian", "sigma": 0.02},
                    "dropout": 0.01, "device": "Livox Mid-360 (approx.), pitched 30 deg forward"},
}
RGB_SPEC = {"mount_pos": [0.32, 0.0, 0.04], "pitch_deg": 10.0, "hfov_deg": 100.0, "width": 224, "height": 224,
            "device": "Go2 front camera (approx.)"}


def apply_mount_overrides(overrides: dict[str, dict[str, Any]] | str | None) -> None:
    """Patch sensor mounts in place, e.g. {"depth_d405": {"mount_pos": [0.33, 0, 0.08], "pitch_deg": 50}} (mount
    studies). Also read from the GO2WM_SENSOR_MOUNTS environment variable (JSON) at import, so worker processes
    inherit it."""
    import json

    if isinstance(overrides, str):
        overrides = json.loads(overrides)
    for name, upd in (overrides or {}).items():
        if name not in SENSOR_SPECS:
            raise KeyError(f"unknown sensor {name!r}")
        bad = set(upd) - {"mount_pos", "pitch_deg"}
        if bad:
            raise KeyError(f"only mount_pos / pitch_deg can be overridden, got {sorted(bad)}")
        SENSOR_SPECS[name].update(upd)


def _env_mounts() -> None:
    import os

    if os.environ.get("GO2WM_SENSOR_MOUNTS"):
        apply_mount_overrides(os.environ["GO2WM_SENSOR_MOUNTS"])


_env_mounts()


def pitch_matrix(pitch_deg: float) -> np.ndarray:
    """Rotation about +y; positive pitch turns the sensor's +x towards -z (looks down)."""
    t = np.radians(pitch_deg)
    return np.array([[np.cos(t), 0.0, np.sin(t)], [0.0, 1.0, 0.0], [-np.sin(t), 0.0, np.cos(t)]])


def pinhole_directions(hfov_deg: float, vfov_deg: float, width: int, height: int) -> np.ndarray:
    """Unit ray directions [H*W, 3] in the sensor frame (optical axis +x, image right = -y,
    image down = -z), row-major from the top-left pixel."""
    u = (np.arange(width) + 0.5) / width * 2 - 1
    v = (np.arange(height) + 0.5) / height * 2 - 1
    U, V = np.meshgrid(u, v)
    d = np.stack([np.ones_like(U), -U * np.tan(np.radians(hfov_deg) / 2), -V * np.tan(np.radians(vfov_deg) / 2)], -1)
    d = d.reshape(-1, 3)
    return d / np.linalg.norm(d, axis=1, keepdims=True)


def sample_directions(spec: dict[str, Any], rng: np.random.Generator) -> np.ndarray:
    """Sensor-frame unit directions for one frame (fixed for pinhole, fresh random for LiDARs)."""
    if spec["kind"] == "pinhole":
        return pinhole_directions(spec["hfov_deg"], spec["vfov_deg"], spec["width"], spec["height"])
    n = int(spec["n_rays"])
    if spec["kind"] == "dome":  # uniform on the hemisphere around the mounting axis (+x)
        v = rng.normal(size=(n, 3))
        v[:, 0] = np.abs(v[:, 0])
        return v / np.linalg.norm(v, axis=1, keepdims=True)
    if spec["kind"] == "band":  # uniform on the sphere band elev in [lo, hi], all azimuths
        lo, hi = np.radians(spec["elev_deg"])
        s = rng.uniform(np.sin(lo), np.sin(hi), n)
        az = rng.uniform(0, 2 * np.pi, n)
        c = np.sqrt(1 - s**2)
        return np.stack([c * np.cos(az), c * np.sin(az), s], 1)
    raise ValueError(spec["kind"])


def base_frame_rays(spec: dict[str, Any], rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """(starts [N, 3], directions [N, 3]) in the robot base frame."""
    d = sample_directions(spec, rng) @ pitch_matrix(spec["pitch_deg"]).T
    return np.broadcast_to(np.asarray(spec["mount_pos"], dtype=np.float64), d.shape).copy(), d


def n_rays(spec: dict[str, Any]) -> int:
    return spec["width"] * spec["height"] if spec["kind"] == "pinhole" else int(spec["n_rays"])


def corrupt_hits(starts_w: np.ndarray, dirs_w: np.ndarray, hits_w: np.ndarray, spec: dict[str, Any],
                 rng: np.random.Generator, optical_axis_w: np.ndarray | None = None) -> np.ndarray:
    """Apply range limits, dropout and range noise along each ray; misses / invalid -> NaN."""
    hits = np.asarray(hits_w, dtype=np.float64)
    r = np.linalg.norm(hits - starts_w, axis=1)
    ok = np.isfinite(r) & (r >= spec["range_m"][0]) & (r <= spec["range_m"][1])
    ok &= rng.uniform(size=len(r)) >= spec.get("dropout", 0.0)
    r = np.where(np.isfinite(r), r, 0.0)  # misses: masked below, keep the arithmetic finite
    nz = spec.get("noise") or {}
    if nz.get("model") == "range_gaussian":
        sig = np.full(len(r), nz["sigma"])
    elif nz.get("model") == "depth_quadratic":
        z = r * (dirs_w @ optical_axis_w if optical_axis_w is not None else 1.0)  # depth along the optical axis
        sig = nz["a"] + nz["b"] * np.nan_to_num(z) ** 2
    else:
        sig = np.zeros(len(r))
    r_noisy = r + rng.normal(size=len(r)) * sig
    out = starts_w + dirs_w * r_noisy[:, None]
    out[~ok] = np.nan
    return out


def bin_elevation(points_w: np.ndarray, robot_xy: np.ndarray, robot_yaw: float, z_ref: float = 0.0,
                  grid: dict[str, float] = ELEV_GRID) -> tuple[np.ndarray, np.ndarray]:
    """Mean hit height (z - z_ref) per cell of the robot-centred, yaw-aligned grid -> (elev [nx, ny]
    float16, NaN where empty; count [nx, ny] uint8, capped at 255). Cell (i, j) covers
    x in [x0 + i res, x0 + (i+1) res), y likewise (robot frame: x forward, y left)."""
    nx, ny = GRID_SHAPE if grid is ELEV_GRID else (int(round((grid["x1"] - grid["x0"]) / grid["res"])),
                                                    int(round((grid["y1"] - grid["y0"]) / grid["res"])))
    p = np.asarray(points_w, dtype=np.float64)
    p = p[np.all(np.isfinite(p), axis=1)]
    c, s = np.cos(robot_yaw), np.sin(robot_yaw)
    dx, dy = p[:, 0] - robot_xy[0], p[:, 1] - robot_xy[1]
    xr, yr = c * dx + s * dy, -s * dx + c * dy
    i = np.floor((xr - grid["x0"]) / grid["res"]).astype(np.int64)
    j = np.floor((yr - grid["y0"]) / grid["res"]).astype(np.int64)
    m = (i >= 0) & (i < nx) & (j >= 0) & (j < ny)
    flat = i[m] * ny + j[m]
    cnt = np.bincount(flat, minlength=nx * ny)
    tot = np.bincount(flat, weights=p[m, 2] - z_ref, minlength=nx * ny)
    with np.errstate(invalid="ignore", divide="ignore"):
        elev = np.where(cnt > 0, tot / np.maximum(cnt, 1), np.nan)
    return elev.reshape(nx, ny).astype(np.float16), np.minimum(cnt, 255).reshape(nx, ny).astype(np.uint8)


def grid_centres(grid: dict[str, float] = ELEV_GRID) -> tuple[np.ndarray, np.ndarray]:
    nx, ny = GRID_SHAPE
    return grid["x0"] + (np.arange(nx) + 0.5) * grid["res"], grid["y0"] + (np.arange(ny) + 0.5) * grid["res"]
