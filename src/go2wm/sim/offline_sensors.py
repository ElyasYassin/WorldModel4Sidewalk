"""Offline range-sensor simulator with realistic error sources (no Isaac needed).

Re-casts the E4-C sensors' rays (go2wm.sim.sensors.SENSOR_SPECS) against a tile's height field along a recorded
robot path, then applies an error profile. Unlike the Isaac rig (random noise only, perfect calibration and poses),
a profile adds the errors that do NOT average away:

  depth_scale      per-session depth scale error of the stereo camera (z_meas = z (1 + e), e ~ N(0, depth_scale))
  static_frac      part of the depth noise that is a fixed per-pixel pattern (same every frame)
  mixed_pixels     probability that a pixel at a depth discontinuity (> mixed_thresh between neighbours) returns
                   the average of the two surfaces (stereo "flying pixels" at step edges)
  noise_mult, dropout   multiplier on the device range noise, and the dropout probability (outdoor sun)
  calib_pitch_deg, calib_roll_deg, calib_z_m   per-session extrinsic errors: rays leave the TRUE (perturbed)
                   mount, points are reconstructed with the NOMINAL mount the robot believes in
  lidar_bias_m     per-session range offset of the LiDAR
  odom_xy, odom_yaw, odom_z (per metre travelled) and odom_z_jitter (per step)   random-walk drift of the
                   robot's own pose estimate; matters only when frames are fused (registered with the estimate)

Points are binned into the robot-centred 5 cm grid of sensors.ELEV_GRID (heights relative to the base, i.e. what
the robot can compute without a world frame). The ray caster marches each ray in 5 cm steps and bisects the
crossing, against the same quantised height field Isaac meshes (bilinear between grid points).
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from .sensors import ELEV_GRID, GRID_SHAPE, SENSOR_SPECS, pitch_matrix, sample_directions

PROFILES: dict[str, dict[str, float]] = {
    "ideal": {},
    "realistic": {"depth_scale": 0.005, "static_frac": 0.5, "mixed_pixels": 0.5, "mixed_thresh": 0.03,
                  "calib_pitch_deg": 0.5, "calib_roll_deg": 0.3, "calib_z_m": 0.005, "lidar_bias_m": 0.005,
                  "odom_xy": 0.015, "odom_yaw": 0.01, "odom_z": 0.003, "odom_z_jitter": 0.001},
    "pessimistic": {"depth_scale": 0.01, "static_frac": 0.5, "mixed_pixels": 0.8, "mixed_thresh": 0.03,
                    "noise_mult": 2.0, "dropout": 0.10, "calib_pitch_deg": 1.0, "calib_roll_deg": 0.5,
                    "calib_z_m": 0.010, "lidar_bias_m": 0.010, "odom_xy": 0.04, "odom_yaw": 0.025, "odom_z": 0.008,
                    "odom_z_jitter": 0.002},
}


def roll_matrix(roll_deg: float) -> np.ndarray:
    t = np.radians(roll_deg)
    return np.array([[1.0, 0.0, 0.0], [0.0, np.cos(t), -np.sin(t)], [0.0, np.sin(t), np.cos(t)]])


def quat_to_R(q) -> np.ndarray:
    w, x, y, z = np.asarray(q, dtype=np.float64) / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def cast(starts: np.ndarray, dirs: np.ndarray, height: Callable, r_max: float, step: float = 0.05,
         iters: int = 10) -> np.ndarray:
    """Range to the first crossing of z = height(x, y) along each ray (inf = no hit within r_max)."""
    n = len(starts)
    r_hit = np.full(n, np.inf)
    prev = np.zeros(n)
    above = starts[:, 2] > height(starts[:, 0], starts[:, 1])
    active = above.copy()
    r = 0.0
    while r < r_max and active.any():
        r += step
        idx = np.flatnonzero(active)
        p = starts[idx] + dirs[idx] * r
        below = p[:, 2] <= height(p[:, 0], p[:, 1])
        if below.any():
            hit = idx[below]
            lo, hi = prev[hit], np.full(len(hit), r)
            for _ in range(iters):
                mid = 0.5 * (lo + hi)
                q = starts[hit] + dirs[hit] * mid[:, None]
                b = q[:, 2] <= height(q[:, 0], q[:, 1])
                hi = np.where(b, mid, hi)
                lo = np.where(b, lo, mid)
            r_hit[hit] = 0.5 * (lo + hi)
            active[hit] = False
        prev[active] = r
    return r_hit


def odometry_estimate(pose: np.ndarray, z: np.ndarray, prof: dict[str, float], rng) -> tuple[np.ndarray, np.ndarray]:
    """Drifting estimate of (x, y, yaw) and base z along a true path (random walk scaled by distance moved)."""
    d = np.r_[0.0, np.linalg.norm(np.diff(pose[:, :2], axis=0), axis=1)]
    exy = np.cumsum(rng.normal(size=(len(pose), 2)) * (prof.get("odom_xy", 0.0) * d)[:, None], axis=0)
    eyaw = np.cumsum(rng.normal(size=len(pose)) * prof.get("odom_yaw", 0.0) * d)
    ez = np.cumsum(rng.normal(size=len(pose)) * prof.get("odom_z", 0.0) * d) + \
        rng.normal(size=len(pose)) * prof.get("odom_z_jitter", 0.0)
    est = pose.copy()
    est[:, :2] += exy
    est[:, 2] += eyaw
    return est, z + ez


class OfflineRig:
    """One session (one robot, one calibration) of the D555 / L1 sensors under an error profile."""

    def __init__(self, sensors: list[str], profile: str | dict = "ideal", seed: int = 0,
                 defects: list[dict] | None = None):
        self.defects = defects or []  # standing water (puddles) loses returns, as in the Isaac rig
        self.prof = PROFILES[profile] if isinstance(profile, str) else dict(profile)
        self.rng = np.random.default_rng(seed)
        self.sensors = sensors
        self.session = {}
        for s in sensors:
            spec = SENSOR_SPECS[s]
            g = self.rng.normal(size=6)
            ses = {"pitch_err": g[0] * self.prof.get("calib_pitch_deg", 0.0), "roll_err": g[1] * self.prof.get("calib_roll_deg", 0.0),
                   "z_err": g[2] * self.prof.get("calib_z_m", 0.0), "scale": 1.0 + g[3] * self.prof.get("depth_scale", 0.0),
                   "bias": g[4] * self.prof.get("lidar_bias_m", 0.0)}
            if spec["kind"] == "pinhole":
                ses["pattern"] = self.rng.normal(size=spec["width"] * spec["height"])
            self.session[s] = ses

    def frame(self, s: str, base_pos: np.ndarray, base_R: np.ndarray, height: Callable) -> np.ndarray:
        """Measured points of one frame in the robot's base frame [N, 3] (NaN = no return)."""
        spec, ses, prof, rng = SENSOR_SPECS[s], self.session[s], self.prof, self.rng
        d_s = sample_directions(spec, rng)  # sensor frame (fixed for the camera, fresh for the LiDAR)
        d_nom = d_s @ pitch_matrix(spec["pitch_deg"]).T  # base frame, as the robot believes it is mounted
        R_true = pitch_matrix(spec["pitch_deg"] + ses["pitch_err"]) @ roll_matrix(ses["roll_err"])
        d_true = d_s @ R_true.T  # the same rays from the actually (mis)mounted sensor
        mount_nom = np.asarray(spec["mount_pos"], dtype=np.float64)
        mount_true = mount_nom + np.array([0.0, 0.0, ses["z_err"]])
        starts_w = base_pos + base_R @ mount_true
        r = cast(np.broadcast_to(starts_w, d_true.shape).copy(), d_true @ base_R.T, height,
                 r_max=min(spec["range_m"][1], 8.0))
        ok = np.isfinite(r) & (r >= spec["range_m"][0]) & (r <= spec["range_m"][1])
        ok &= rng.uniform(size=len(r)) >= prof.get("dropout", spec.get("dropout", 0.0))
        if self.defects:
            from .defects import range_dropout

            hit = starts_w + (d_true @ base_R.T) * np.where(np.isfinite(r), r, 0.0)[:, None]
            ok &= rng.uniform(size=len(r)) >= range_dropout(hit[:, 0], hit[:, 1], self.defects, spec["kind"])
        r = np.where(np.isfinite(r), r, 0.0)
        nz, mult = spec["noise"], prof.get("noise_mult", 1.0)
        if nz["model"] == "depth_quadratic":
            axis_true = R_true @ np.array([1.0, 0.0, 0.0])
            z = r * (d_true @ axis_true)  # depth along the optical axis
            if prof.get("mixed_pixels", 0.0) > 0:  # flying pixels at depth discontinuities (range image order)
                # A discontinuity is a sudden change of the depth GRADIENT (second difference), not a large first
                # difference: on flat ground seen obliquely neighbouring rays differ by 10-20 cm but smoothly.
                img = np.where(ok, z, np.nan).reshape(spec["height"], spec["width"])
                out = img.copy()
                thr = prof.get("mixed_thresh", 0.03)
                for ax in (0, 1):
                    prv, nxt = np.roll(img, 1, ax), np.roll(img, -1, ax)
                    if ax == 0:  # no wrap-around: the image border has no neighbour beyond it
                        prv[0], nxt[-1] = np.nan, np.nan
                    else:
                        prv[:, 0], nxt[:, -1] = np.nan, np.nan
                    db, df = img - prv, nxt - img
                    edge = np.abs(df - db) > thr + 0.3 * np.minimum(np.abs(db), np.abs(df))
                    nb = np.where(np.abs(df) > np.abs(db), nxt, prv)  # the neighbour across the jump
                    pick = edge & np.isfinite(nb) & (rng.uniform(size=img.shape) < prof["mixed_pixels"])
                    out[pick] = 0.5 * (img[pick] + nb[pick])
                z = np.where(np.isfinite(out.ravel()), out.ravel(), z)
            sig = (nz["a"] + nz["b"] * z ** 2) * mult
            sf = prof.get("static_frac", 0.0)
            z_meas = z * ses["scale"] + sig * (np.sqrt(1 - sf ** 2) * rng.normal(size=len(z)) + sf * ses["pattern"])
            r_meas = z_meas / np.maximum(d_true @ axis_true, 1e-6)
        else:
            r_meas = r + ses["bias"] + rng.normal(size=len(r)) * nz["sigma"] * mult
        pts = mount_nom + d_nom * r_meas[:, None]  # reconstructed with the NOMINAL mount and directions
        pts[~ok] = np.nan
        return pts


def bin_base_points(pts_b: np.ndarray, base_R: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Base-frame points -> robot-centred yaw-aligned 5 cm grid (mean height relative to the base, count).
    Roll and pitch come from the IMU (assumed exact); yaw alignment removes heading."""
    p = pts_b[np.all(np.isfinite(pts_b), axis=1)]
    yaw = np.arctan2(base_R[1, 0], base_R[0, 0])
    c, s = np.cos(yaw), np.sin(yaw)
    Rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    q = p @ (Rz.T @ base_R).T  # level, yaw-aligned frame centred on the base
    nx, ny = GRID_SHAPE
    i = np.floor((q[:, 0] - ELEV_GRID["x0"]) / ELEV_GRID["res"]).astype(int)
    j = np.floor((q[:, 1] - ELEV_GRID["y0"]) / ELEV_GRID["res"]).astype(int)
    m = (i >= 0) & (i < nx) & (j >= 0) & (j < ny)
    flat = i[m] * ny + j[m]
    cnt = np.bincount(flat, minlength=nx * ny)
    tot = np.bincount(flat, weights=q[m, 2], minlength=nx * ny)
    with np.errstate(invalid="ignore", divide="ignore"):
        elev = np.where(cnt > 0, tot / np.maximum(cnt, 1), np.nan)
    return elev.reshape(nx, ny), np.minimum(cnt, 255).reshape(nx, ny)


def simulate_path(states: np.ndarray, height: Callable, sensors: list[str], profile: str | dict = "ideal",
                  seed: int = 0, frames: list[int] | None = None) -> dict[str, Any]:
    """Grids (heights relative to the base) for every requested frame, plus the robot's own pose estimate."""
    from ..data.dataset import state_pose2d

    rig = OfflineRig(sensors, profile, seed)
    pose = state_pose2d(states)
    est_pose, est_z = odometry_estimate(pose, states[:, 2], rig.prof, np.random.default_rng(seed + 1))
    frames = list(range(len(states))) if frames is None else frames
    out = {s: {"elev": {}, "count": {}} for s in sensors}
    for t in frames:
        R = quat_to_R(states[t, 3:7])
        for s in sensors:
            e, c = bin_base_points(rig.frame(s, states[t, :3], R, height), R)
            out[s]["elev"][t], out[s]["count"][t] = e, c
    return {"grids": out, "pose": pose, "est_pose": est_pose, "est_z": est_z, "true_z": states[:, 2].copy(),
            "profile": rig.prof, "session": {s: {k: v for k, v in d.items() if k != "pattern"} for s, d in rig.session.items()}}
