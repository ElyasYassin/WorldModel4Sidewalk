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


def curb_mask(elev: np.ndarray, X: np.ndarray, walk: np.ndarray, drop: float = 0.08, min_len: float = 0.8,
              pad: int = 4) -> np.ndarray:
    """Cells at or next to a curb, found in the frame itself (not from the map + pose estimate, which fails once
    the lateral pose error exceeds the walkway margin): cells more than `drop` below the walkway's median height
    that form a strip at least `min_len` long along the walk (world x), dilated by `pad` cells. A hole, even a deep
    one, is compact and is kept."""
    fin = np.isfinite(elev)
    if not (walk & fin).any():
        return np.zeros_like(fin)
    low = fin & (elev < np.nanmedian(elev[walk & fin]) - drop)
    strip = np.zeros_like(fin)
    for comp in _clusters(low):
        if np.ptp(X[comp[:, 0], comp[:, 1]]) >= min_len:
            strip[comp[:, 0], comp[:, 1]] = True
    if not strip.any():
        return strip
    out = strip.copy()
    for _ in range(pad):  # 4-neighbour dilation
        g = out.copy()
        g[1:] |= out[:-1]; g[:-1] |= out[1:]; g[:, 1:] |= out[:, :-1]; g[:, :-1] |= out[:, 1:]
        out = g
    return out


def curb_lateral(elev: np.ndarray, drop: float = 0.08, min_len: float = 0.8, min_rows: int = 8,
                 with_angle: bool = False):
    """Robot-frame lateral position [m] of a curb edge running along the robot's heading, or None. Per grid row
    (forward distance) the curb is the first cell, going outward from the robot, more than `drop` below the
    walkway median; the edge is the median over rows of the midpoint between that cell and the last walkway cell.
    Only rows 0.5-2 m ahead (dense, low noise) are used, and only one side (the side with more rows) is returned.
    with_angle: returns (offset at the robot, angle of the curb line in the robot frame [rad]) from a line fit
    (Theil-Sen: median of pairwise slopes), or None."""
    res, fin = ELEV_GRID["res"], np.isfinite(elev)
    rows = (CX > 0.5) & (CX < 2.0)
    if not fin[rows].any():
        return None
    ref = np.nanmedian(elev[rows][:, np.abs(CY) < 0.4]) if fin[rows][:, np.abs(CY) < 0.4].any() else None
    if ref is None:
        return None
    best = []
    for side in (-1, 1):
        cols = np.nonzero(side * CY > 0.15)[0]
        cols = cols[np.argsort(np.abs(CY[cols]))]  # outward from the robot
        edges = []
        for i in np.nonzero(rows)[0]:
            e = elev[i, cols]
            f = np.nonzero(np.isfinite(e))[0]  # finite cells, outward (shadows and dropouts are skipped)
            if len(f) < 2:
                continue
            ef = e[f]
            low = np.nonzero(ef < ref - drop)[0]
            if len(low) == 0 or low[0] == 0:
                continue
            # last walkway cell before the drop, then the first cell clearly below it (often a mixed pixel)
            w = np.nonzero(ef[: low[0]] > ref - 0.03)[0]
            if len(w) == 0:
                continue
            k = w[-1] + 1
            if f[k] - f[w[-1]] > 3:  # too large a gap to place the edge
                continue
            edges.append((CX[i], 0.5 * (CY[cols[f[w[-1]]]] + CY[cols[f[k]]])))
        if len(edges) >= min_rows and np.ptp(CX[rows]) >= min_len and len(edges) > len(best):
            best = edges
    if not best:
        return None
    if not with_angle:
        return float(np.median([b[1] for b in best]))
    xe, ye = np.array(best).T
    i, j = np.triu_indices(len(xe), 1)
    ok = np.abs(xe[j] - xe[i]) > 0.5
    if ok.sum() < 5:
        return None
    k = float(np.median((ye[j] - ye[i])[ok] / (xe[j] - xe[i])[ok]))
    return float(np.median(ye - k * xe)), float(np.arctan(k))


def detect_candidates(elev: np.ndarray, count: np.ndarray, pose: np.ndarray, t: float, walk_half: float,
                      thresh: float = 0.012, thresh_per_m: float = 0.008, max_range: float = 2.5, min_cells: int = 3,
                      sensor: str = "depth_d555", edge_min_span: float = 0.6,
                      curb_drop: float = 0.0, step_thresh: float = 0.0, step_range: tuple[float, float] = (0.5, 1.8),
                      step_max: float = float("inf"), step_clear: float = 0.3) -> list[Observation]:
    """Clusters of cells deviating from the local median. The threshold grows with range beyond 1 m (noise grows
    with depth). A step produces a raised cluster on its high side and a depressed one on its low side, and a hole
    a raised far rim: touching clusters of opposite sign are merged into the dominant one (peak x cells).
    `curb_drop` > 0 also drops cells at or next to a curb found in the frame itself (`curb_mask`).
    `step_thresh` > 0 adds the transverse step detector (`transverse_steps`) for faults below `thresh`; `step_max`
    drops its steps above that size (large faults are found by the residual path, and a step down seen from its high
    side casts an occlusion shadow that the step detector places at the shadow's far end, i.e. at the wrong x)."""
    res = elev - _median_filter(elev)
    X, Y = to_world(pose, CX[:, None], CY[None, :])
    xr_grid = np.broadcast_to(CX[:, None], res.shape)
    thr = thresh + thresh_per_m * np.maximum(0.0, xr_grid - 1.0)
    ok = np.isfinite(res) & (np.abs(Y) < walk_half - 0.12) & (xr_grid > 0.3) & (xr_grid < max_range)
    if curb_drop > 0:
        ok &= ~curb_mask(elev, X, ok, curb_drop)
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
    if step_thresh > 0:  # small transverse faults the local-median residual cannot reach (it keeps a clean edge)
        # only where the residual path saw nothing in this frame: next to a large fault the occluded low side shows
        # up as residual clusters (edge or depression) and the step detector would place the fault at the wrong x
        known = [float(CX[c["idx"][:, 0]].mean()) for c in merged]
        for st in transverse_steps(elev, ok, step_thresh, step_range, edge_min_span):
            if st["step"] <= step_max and all(abs(st["xr"] - k) > step_clear for k in known):
                merged.append({"kind": "edge", "idx": st["idx"], "peak": st["step"], "via": "step"})
    out = []
    for c in merged:
        i, j = c["idx"][:, 0], c["idx"][:, 1]
        xr, yr = CX[i].mean(), CY[j].mean()
        wx, wy = to_world(pose, xr, yr)
        out.append(Observation(t, sensor, c["kind"], np.array([wx, wy]), float(np.hypot(xr, yr)),
                               value_m=None, extent_m=float(max(np.ptp(CX[i]), np.ptp(CY[j])) + ELEV_GRID["res"]),
                               confidence=0.4, provenance="detected",
                               extra={"peak_dev_m": c["peak"], "cells": int(len(c["idx"])), "via": c.get("via", "residual"), "view_xy": [float(pose[0]), float(pose[1])],
                                      "y_span": [float(to_world(pose, xr, CY[j].min())[1]),
                                                 float(to_world(pose, xr, CY[j].max())[1])]}))
    return out


def transverse_steps(elev: np.ndarray, ok: np.ndarray, thresh: float = 0.006, rng: tuple[float, float] = (0.5, 1.8),
                     min_span: float = 0.6, w: int = 3, max_gap: int = 1) -> list[dict[str, Any]]:
    """Transverse level shifts (upheaval faults) in one frame: per grid column, the median of the `w` rows ahead
    minus the `w` rows behind each row boundary, detrended by the mean of the same response one window further on
    either side (a uniform grade gives the same response everywhere, a step only at its boundary). A step is a run
    of columns (gaps <= `max_gap`) of one sign above `thresh` spanning >= `min_span` across the walk, in rows
    `rng` ahead (dense, low noise). Responses within 0.3 m along x are one step (the strongest run kept).
    Returns dicts: xr (robot-frame x of the boundary), step (median |shift| over the run [m], sign of the shift
    going away from the robot in "sign"), idx ([k, 2] cells of the run on the boundary row)."""
    nx, ny = elev.shape
    e = np.where(ok, elev, np.nan)
    s = np.full((nx, ny), np.nan)
    with np.errstate(all="ignore"):
        for i in range(w, nx - w + 1):
            s[i] = np.nanmedian(e[i:i + w], 0) - np.nanmedian(e[i - w:i], 0)
        d = np.full_like(s, np.nan)
        d[2 * w:nx - 2 * w] = s[2 * w:nx - 2 * w] - 0.5 * (s[w:nx - 3 * w] + s[3 * w:nx - w])
    cand = []
    for i in np.nonzero((CX - 0.5 * ELEV_GRID["res"] > rng[0]) & (CX - 0.5 * ELEV_GRID["res"] < rng[1]))[0]:
        for sign in (-1.0, 1.0):
            hit = np.nan_to_num(sign * d[i], nan=-1.0) > thresh
            j, runs = 0, []
            while j < ny:
                if not hit[j]:
                    j += 1
                    continue
                k, last = j, j
                while k + 1 < ny and (hit[k + 1] or (k + 1 + max_gap < ny and hit[k + 1:k + 2 + max_gap].any())):
                    k += 1
                    if hit[k]:
                        last = k
                runs.append((j, last))
                j = k + 1
            for j0, j1 in runs:
                span = CY[j1] - CY[j0] + ELEV_GRID["res"]
                if span >= min_span:
                    cols = np.arange(j0, j1 + 1)
                    cols = cols[hit[cols]]
                    cand.append({"xr": float(CX[i] - 0.5 * ELEV_GRID["res"]), "span": float(span), "sign": sign,
                                 "step": float(np.median(sign * d[i, cols])),
                                 "idx": np.stack([np.full(len(cols), i), cols], 1)})
    cand.sort(key=lambda c: -c["span"] * c["step"])
    out = []
    for c in cand:
        if all(abs(c["xr"] - o["xr"]) > 0.3 for o in out):
            out.append(c)
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


def in_view(sensor: str, pose: np.ndarray, xy: np.ndarray, d405_zone=None) -> bool:
    """Is a world point well inside the sensor's measurement zone (robot frame)? `d405_zone` = (x_lo, x_hi, half
    width) overrides the D405 zone (inspection posture: the camera looks closer and steeper)."""
    xr, yr = to_robot(pose, *xy)
    if sensor == "depth_d405":
        lo, hi, hw = d405_zone or (0.42, 0.62, 0.2)
        return lo <= xr <= hi and abs(yr) <= hw
    return 0.6 <= xr <= 1.2 and abs(yr) <= 0.6  # D555 measuring zone (detection reaches farther)
