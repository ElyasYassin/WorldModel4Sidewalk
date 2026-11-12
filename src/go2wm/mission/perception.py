"""Phase 1 perception: candidate defects from the D555, measurements from raw points of whichever sensor is close.

Detection (one D555 frame, robot-centred 5 cm grid, heights relative to the base): each cell is compared with the
median of its 35 x 35 cm neighbourhood; cells deviating by more than `thresh` form connected clusters. Negative
compact clusters are depressions (holes, puddle dips), positive ones are raised edges (upheaval faults, steps).
Smooth grade and cross slope are removed by the local median; cells outside the walkway (curb, verge) are ignored.

Measurement (raw base-frame points of one frame; geometry as in scripts/near_far_step_study.py):
  depression  depth = local ground plane (fitted on a ring around the candidate) minus the 5th percentile of the
              points inside; extent = span of the points more than 1 cm below the plane
  edge        step height = difference of two planes fitted either side of the edge line (transverse to the walk)

Every output is a measured observation with its sensor, range and a confidence from the sensor/range class
(D405 at its stand-off > D555 within 1.2 m > D555 detection only). Ground truth is never read here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..sim.sensors import ELEV_GRID, grid_centres

CX, CY = grid_centres()


@dataclass
class Observation:
    t: float
    sensor: str
    kind: str  # "depression" | "edge" | "dropout"
    xy: np.ndarray  # world (robot's estimate)
    range_m: float
    value_m: float | None = None  # depth (depression) or step height (edge); None = detection only
    extent_m: float | None = None
    confidence: float = 0.4
    provenance: str = "measured"
    extra: dict[str, Any] = field(default_factory=dict)


def _median_filter(e: np.ndarray, r: int = 3) -> np.ndarray:
    """NaN-aware (2r+1)^2 median."""
    nx, ny = e.shape
    pad = np.pad(e, r, constant_values=np.nan)
    stack = np.stack([pad[i : i + nx, j : j + ny] for i in range(2 * r + 1) for j in range(2 * r + 1)])
    with np.errstate(all="ignore"):
        return np.nanmedian(stack, axis=0)


def _clusters(mask: np.ndarray) -> list[np.ndarray]:
    """4-connected components of a boolean grid -> list of [k, 2] index arrays."""
    seen = np.zeros_like(mask, bool)
    out = []
    for i0, j0 in zip(*np.nonzero(mask)):
        if seen[i0, j0]:
            continue
        stack, comp = [(i0, j0)], []
        seen[i0, j0] = True
        while stack:
            i, j = stack.pop()
            comp.append((i, j))
            for a, b in ((i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)):
                if 0 <= a < mask.shape[0] and 0 <= b < mask.shape[1] and mask[a, b] and not seen[a, b]:
                    seen[a, b] = True
                    stack.append((a, b))
        out.append(np.array(comp))
    return out


def to_world(pose, x, y):
    c, s = np.cos(pose[2]), np.sin(pose[2])
    return pose[0] + c * x - s * y, pose[1] + s * x + c * y


def to_robot(pose, X, Y):
    c, s = np.cos(pose[2]), np.sin(pose[2])
    dx, dy = X - pose[0], Y - pose[1]
    return c * dx + s * dy, -s * dx + c * dy


def detect_candidates(elev: np.ndarray, count: np.ndarray, pose: np.ndarray, t: float, walk_half: float,
                      thresh: float = 0.012, thresh_per_m: float = 0.008, max_range: float = 2.5, min_cells: int = 3,
                      sensor: str = "depth_d555", edge_min_span: float = 0.6) -> list[Observation]:
    """Clusters of cells deviating from the local median. The threshold grows with range beyond 1 m (noise grows
    with depth). A step produces a raised cluster on its high side and a depressed one on its low side, and a hole
    a raised far rim: touching clusters of opposite sign are merged into the dominant one (peak x cells)."""
    res = elev - _median_filter(elev)
    X, Y = to_world(pose, CX[:, None], CY[None, :])
    xr_grid = np.broadcast_to(CX[:, None], res.shape)
    thr = thresh + thresh_per_m * np.maximum(0.0, xr_grid - 1.0)
    ok = np.isfinite(res) & (np.abs(Y) < walk_half - 0.12) & (xr_grid > 0.3) & (xr_grid < max_range)
    comps = []
    for sign, kind in ((-1, "depression"), (1, "edge")):
        for comp in _clusters(ok & (sign * res > thr)):
            peak = float(np.max(sign * res[comp[:, 0], comp[:, 1]]))
            thr_c = float(np.max(thr[comp[:, 0], comp[:, 1]]))
            # far away a small hole is only a couple of cells: accept 2 cells if the deviation is strong
            if len(comp) >= min_cells or (len(comp) >= 2 and peak > 2.0 * thr_c):
                i, j = comp[:, 0], comp[:, 1]
                comps.append({"kind": kind, "idx": comp, "score": float(np.max(sign * res[i, j])) * len(comp),
                              "peak": float(np.max(sign * res[i, j]))})
    keep = [True] * len(comps)
    for a in range(len(comps)):
        for b in range(len(comps)):
            if a == b or comps[a]["kind"] == comps[b]["kind"] or not keep[a] or not keep[b]:
                continue
            ia, ib = comps[a]["idx"], comps[b]["idx"]
            if np.min(np.abs(ia[:, None, :] - ib[None, :, :]).sum(-1)) <= 2:  # touching (within 2 cells)
                lo = a if comps[a]["score"] < comps[b]["score"] else b
                keep[lo] = False
                hi = b if lo == a else a
                comps[hi]["idx"] = np.vstack([comps[hi]["idx"], comps[lo]["idx"]])
    comps = [c for c, k in zip(comps, keep) if k]
    # A transverse edge (upheaval fault, step) is seen as several pieces at the same distance spread across the
    # walk: group clusters whose centres are within 15 cm along x; a group spanning >= edge_min_span across (and
    # much wider than deep) is one edge. Anything compact is a depression (a raised compact cluster is the far
    # wall of a hole seen obliquely).
    for c in comps:
        c["xc"] = float(CX[c["idx"][:, 0]].mean())
    comps.sort(key=lambda c: c["xc"])
    groups, cur = [], []
    for c in comps:
        if cur and c["xc"] - np.mean([g["xc"] for g in cur]) > 0.15:
            groups.append(cur)
            cur = []
        cur.append(c)
    if cur:
        groups.append(cur)
    merged = []
    for g in groups:
        idx = np.vstack([c["idx"] for c in g])
        x_span, y_span = np.ptp(CX[idx[:, 0]]) + ELEV_GRID["res"], np.ptp(CY[idx[:, 1]]) + ELEV_GRID["res"]
        if y_span >= edge_min_span and y_span >= 2.0 * x_span:
            merged.append({"kind": "edge", "idx": idx, "peak": max(c["peak"] for c in g)})
        else:
            for c in g:
                merged.append({**c, "kind": "depression"} if c["kind"] == "edge" else c)
    out = []
    for c in merged:
        i, j = c["idx"][:, 0], c["idx"][:, 1]
        xr, yr = CX[i].mean(), CY[j].mean()
        wx, wy = to_world(pose, xr, yr)
        out.append(Observation(t, sensor, c["kind"], np.array([wx, wy]), float(np.hypot(xr, yr)),
                               value_m=None, extent_m=float(max(np.ptp(CX[i]), np.ptp(CY[j])) + ELEV_GRID["res"]),
                               confidence=0.4, provenance="detected",
                               extra={"peak_dev_m": c["peak"], "cells": int(len(c["idx"])), "view_xy": [float(pose[0]), float(pose[1])],
                                      "y_span": [float(to_world(pose, xr, CY[j].min())[1]),
                                                 float(to_world(pose, xr, CY[j].max())[1])]}))
    return out


def expected_hits(sensor: str, base_R: np.ndarray, base_height: float) -> np.ndarray:
    """Cells of the robot grid a pinhole sensor's rays would hit on flat ground (from its known geometry)."""
    from ..sim.sensors import GRID_SHAPE, SENSOR_SPECS, pitch_matrix, sample_directions

    spec = SENSOR_SPECS[sensor]
    d = sample_directions(spec, np.random.default_rng(0)) @ pitch_matrix(spec["pitch_deg"]).T @ base_R.T
    o = base_R @ np.asarray(spec["mount_pos"]) + np.array([0.0, 0.0, base_height])
    down = d[:, 2] < -1e-3
    r = -o[2] / d[down, 2]
    r = r[(r >= spec["range_m"][0]) & (r <= spec["range_m"][1])]
    hit = o + d[down][: len(r)] * r[:, None] if len(r) == down.sum() else None
    if hit is None:  # recompute with the range mask applied consistently
        dd = d[down]
        rr = -o[2] / dd[:, 2]
        m = (rr >= spec["range_m"][0]) & (rr <= spec["range_m"][1])
        hit = o + dd[m] * rr[m, None]
    yaw = np.arctan2(base_R[1, 0], base_R[0, 0])
    c, s = np.cos(yaw), np.sin(yaw)
    xr, yr = c * hit[:, 0] + s * hit[:, 1], -s * hit[:, 0] + c * hit[:, 1]
    nx, ny = GRID_SHAPE
    i = np.floor((xr - ELEV_GRID["x0"]) / ELEV_GRID["res"]).astype(int)
    j = np.floor((yr - ELEV_GRID["y0"]) / ELEV_GRID["res"]).astype(int)
    m = (i >= 0) & (i < nx) & (j >= 0) & (j < ny)
    cnt = np.zeros((nx, ny), int)
    np.add.at(cnt, (i[m], j[m]), 1)
    return cnt


def detect_dropouts(count: np.ndarray, expected: np.ndarray, pose: np.ndarray, t: float, walk_half: float,
                    min_cells: int = 6, sensor: str = "depth_d555") -> list[Observation]:
    """Clusters of cells that should have >= 2 returns but have none: possible water / specular surface."""
    X, Y = to_world(pose, CX[:, None], CY[None, :])
    xr = np.broadcast_to(CX[:, None], count.shape)
    # only where ray landing points are stable: beyond ~1.4 m range noise at grazing angles moves hits between cells
    miss = (expected >= 3) & (count == 0) & (np.abs(Y) < walk_half - 0.12) & (xr > 0.7) & (xr < 1.4)
    out = []
    for comp in _clusters(miss):
        if len(comp) < min_cells:
            continue
        i, j = comp[:, 0], comp[:, 1]
        x_, y_ = CX[i].mean(), CY[j].mean()
        wx, wy = to_world(pose, x_, y_)
        out.append(Observation(t, sensor, "dropout", np.array([wx, wy]), float(np.hypot(x_, y_)), value_m=None,
                               extent_m=float(max(np.ptp(CX[i]), np.ptp(CY[j])) + ELEV_GRID["res"]), confidence=0.35,
                               provenance="detected", extra={"cells": int(len(comp)), "note": "no returns: possible water", "view_xy": [float(pose[0]), float(pose[1])]}))
    return out


def _plane(q: np.ndarray):
    A = np.c_[np.ones(len(q)), q[:, 0], q[:, 1]]
    coef, *_ = np.linalg.lstsq(A, q[:, 2], rcond=None)
    return coef


def classify_shape(points_b: np.ndarray, pose: np.ndarray, target_xy, extent: float | None = None,
                   half_window: float = 0.2, min_dev: float = 0.006) -> str | None:
    """Step or pit? From one D555 frame, along the walk (world x) in a lateral strip through the finding: fit ONE
    ground plane (with its grade) to the ground before (0.15-0.6 m) and beyond (past the finding's extent, 0.6 m)
    plus a step offset for the far side. A step has a clear offset (upheaval fault); a pit has none but is low at
    the centre (hole, dip, puddle). None if the frame does not cover both sides or nothing stands out."""
    p = points_b[np.all(np.isfinite(points_b), axis=1)]
    if len(p) < 30:
        return None
    wx_, wy_ = to_world(pose, p[:, 0], p[:, 1])
    tx, ty = float(target_xy[0]), float(target_xy[1])
    W = np.c_[wx_ - tx, wy_ - ty, p[:, 2]]
    W = W[np.abs(W[:, 1]) < half_window]
    r = float(np.clip(0.5 * (extent or 0.0) + 0.05, 0.35, 0.6))  # clear of the largest holes (r ~0.32)
    near = W[(W[:, 0] > -r - 0.45) & (W[:, 0] < -r)]
    far = W[(W[:, 0] > r) & (W[:, 0] < r + 0.6)]
    mid = W[np.abs(W[:, 0]) < 0.1]
    if len(near) < 15 or len(far) < 15 or len(mid) < 5:
        return None
    q = np.vstack([near, far])
    side = np.r_[np.zeros(len(near)), np.ones(len(far))]
    A = np.c_[np.ones(len(q)), q[:, 0], q[:, 1], side]
    (a, bx, by, step), *_ = np.linalg.lstsq(A, q[:, 2], rcond=None)
    rc = float(np.percentile(mid[:, 2] - (a + bx * mid[:, 0] + by * mid[:, 1]), 10))
    if abs(step) >= min_dev and abs(step) >= 0.6 * abs(min(rc, 0.0)):
        return "step"
    if rc <= -min_dev and abs(step) < 0.5 * abs(rc):
        return "pit"
    return None


def measure(points_b: np.ndarray, pose: np.ndarray, target_xy: np.ndarray, kind: str, sensor: str, t: float,
            radius: float = 0.35) -> Observation | None:
    """Measure a candidate from one frame's raw base-frame points (heights relative to the base)."""
    p = points_b[np.all(np.isfinite(points_b), axis=1)]
    if len(p) < 30:
        return None
    xr, yr = to_robot(pose, *target_xy)
    d = np.hypot(p[:, 0] - xr, p[:, 1] - yr)
    rng = float(np.hypot(xr, yr))
    conf = 0.95 if sensor == "depth_d405" else 0.8 if rng <= 1.2 else 0.55
    if kind == "depression":
        ring = p[(d > radius) & (d < radius + 0.15)]
        inner = p[d <= radius]
        if len(ring) < 20 or len(inner) < 20:
            return None
        a, bx, by = _plane(ring)
        dev = inner[:, 2] - (a + bx * inner[:, 0] + by * inner[:, 1])
        deep = inner[dev < -0.01]
        depth = float(-np.percentile(dev, 5))
        ext = float(max(np.ptp(deep[:, 0]), np.ptp(deep[:, 1]))) if len(deep) >= 3 else 0.0
        return Observation(t, sensor, kind, np.asarray(target_xy, float), rng, value_m=depth, extent_m=ext,
                           confidence=conf)
    if kind == "edge":
        # sidewalk edges (joints, faults) run ACROSS the walk: fit along the walk direction (world x), whatever the
        # robot's heading (a robot-frame fit at 25 deg yaw straddles the step obliquely and under-reads it)
        wx_, wy_ = to_world(pose, p[:, 0], p[:, 1])
        W = np.c_[wx_, wy_, p[:, 2]]
        tx, ty = float(target_xy[0]), float(target_xy[1])
        lat = W[np.abs(W[:, 1] - ty) < 0.25]
        best = None
        for xe in tx + np.arange(-0.08, 0.081, 0.01):  # the detected position is only good to a few cm
            z_at = []
            for lo, hi in ((xe - 0.20, xe - 0.02), (xe + 0.02, xe + 0.20)):
                q = lat[(lat[:, 0] >= lo) & (lat[:, 0] <= hi)]
                if len(q) < 12:
                    break
                a, bx, by = _plane(q - [xe, ty, 0])
                z_at.append(a)
            if len(z_at) == 2 and (best is None or abs(z_at[1] - z_at[0]) > abs(best[1])):
                best = (xe, z_at[1] - z_at[0])
        if best is None:
            return None
        return Observation(t, sensor, kind, np.array([best[0], ty]), rng, value_m=float(best[1]), confidence=conf)
    return None


def in_view(sensor: str, pose: np.ndarray, xy: np.ndarray) -> bool:
    """Is a world point well inside the sensor's measurement zone (robot frame)?"""
    xr, yr = to_robot(pose, *xy)
    if sensor == "depth_d405":
        return 0.42 <= xr <= 0.62 and abs(yr) <= 0.2
    return 0.6 <= xr <= 1.2 and abs(yr) <= 0.6  # D555 measuring zone (detection reaches farther)
