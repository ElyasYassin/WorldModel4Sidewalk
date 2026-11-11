"""Body-frame terrain height maps for terrain-aware dynamics (P2).

At a window's start pose the model receives a height map on a fixed grid in the start body frame
(x forward, y left), heights relative to the ground under the base. During an open-loop rollout
it samples a small local patch around each *predicted* pose from that map (bilinear), so terrain
under future poses enters the prediction. Maps come from the analytic tile surface
(`sidewalk_height`, i.e. a perfect elevation map); on flat ground they are zero.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

MAP_X = np.round(np.arange(-1.0, 3.0001, 0.1), 3)  # 41, forward
MAP_Y = np.round(np.arange(-1.2, 1.2001, 0.1), 3)  # 25, left
PATCH_X = np.array([-0.3, 0.0, 0.3, 0.6, 0.9])
PATCH_Y = np.array([-0.2, 0.0, 0.2])
PATCH_DIM = len(PATCH_X) * len(PATCH_Y) + 1  # relative heights + centre height vs start ground
HEIGHT_SCALE = 0.1  # model sees heights in units of 10 cm


def episode_height_fn(meta: dict[str, Any]):
    """Height function (x, y) -> z in scene coordinates for an episode, or None on flat ground."""
    terrain = ((meta.get("scene") or {}).get("params") or {}).get("terrain")
    if not terrain:
        return None
    from ..sim.terrain import sidewalk_height

    return lambda x, y: sidewalk_height(x, y, terrain)


def start_maps(pose: np.ndarray, height_fn) -> np.ndarray:
    """pose [N, 3] scene-frame (x, y, yaw) -> maps [N, len(MAP_Y), len(MAP_X)] relative heights."""
    n = len(pose)
    if height_fn is None:
        return np.zeros((n, len(MAP_Y), len(MAP_X)), np.float32)
    gx, gy = np.meshgrid(MAP_X, MAP_Y)  # [NY, NX]
    c, s = np.cos(pose[:, 2])[:, None, None], np.sin(pose[:, 2])[:, None, None]
    wx = pose[:, 0, None, None] + c * gx - s * gy
    wy = pose[:, 1, None, None] + s * gx + c * gy
    z0 = height_fn(pose[:, 0], pose[:, 1])[:, None, None]
    return (height_fn(wx, wy) - z0).astype(np.float32)


def scan_maps(scans: np.ndarray, pattern_xy: np.ndarray) -> np.ndarray:
    """Height-scanner scans [N, R] (scene-frame hit heights; rays at yaw-aligned body offsets
    pattern_xy [R, 2], a regular grid) -> maps [N, len(MAP_Y), len(MAP_X)] on the MAP grid, heights
    relative to the scan at the base. Outside the scanned area the edge values are held (bilinear
    with clamping), i.e. the model only sees as far as the sensor. Missed rays take the scan median."""
    pattern_xy = np.asarray(pattern_xy, dtype=np.float64)
    xs = np.unique(np.round(pattern_xy[:, 0], 3))
    ys = np.unique(np.round(pattern_xy[:, 1], 3))
    ix = np.searchsorted(xs, np.round(pattern_xy[:, 0], 3))
    iy = np.searchsorted(ys, np.round(pattern_xy[:, 1], 3))
    scans = np.asarray(scans, dtype=np.float64)
    fin = np.isfinite(scans)
    med = np.nanmedian(np.where(fin, scans, np.nan), axis=1, keepdims=True)
    scans = np.where(fin, scans, np.nan_to_num(med))
    grid = np.zeros((len(scans), len(ys), len(xs)))
    grid[:, iy, ix] = scans

    def bilinear(qx, qy):  # qx, qy broadcastable arrays in metres -> [N, *shape]
        fx = np.clip((qx - xs[0]) / (xs[1] - xs[0]), 0, len(xs) - 1)
        fy = np.clip((qy - ys[0]) / (ys[1] - ys[0]), 0, len(ys) - 1)
        x0, y0 = np.minimum(fx.astype(int), len(xs) - 2), np.minimum(fy.astype(int), len(ys) - 2)
        ax, ay = fx - x0, fy - y0
        g = lambda j, i: grid[:, j, i]  # noqa: E731
        return ((1 - ax) * (1 - ay) * g(y0, x0) + ax * (1 - ay) * g(y0, x0 + 1)
                + (1 - ax) * ay * g(y0 + 1, x0) + ax * ay * g(y0 + 1, x0 + 1))

    gx, gy = np.meshgrid(MAP_X, MAP_Y)
    z0 = bilinear(np.zeros(1), np.zeros(1))[:, 0]
    return (bilinear(gx, gy) - z0[:, None, None]).astype(np.float32)


def sensor_maps(elev: np.ndarray, ground_z: np.ndarray, fill_iters: int = 60) -> np.ndarray:
    """E4-C sensor elevation grids [N, nx, ny] (5 cm cells of go2wm.sim.sensors.ELEV_GRID, terrain-frame
    heights, NaN where empty) -> maps [N, len(MAP_Y), len(MAP_X)] on the MAP grid, heights relative to
    the ground under the base (ground_z [N], same frame as the grids; simulator truth here, standing in
    for the foot-contact height a real robot gets from leg kinematics).

    Each MAP point averages the (up to 2 x 2) observed cells around it. Unobserved points take the mean
    of observed neighbours, grown outward iteratively (so, like the scanner map, values near the edge
    of what the sensor saw are held); points with no observation within reach are 0 (flat)."""
    from ..sim.sensors import ELEV_GRID, grid_centres

    cx, cy = grid_centres(ELEV_GRID)
    e = np.asarray(elev, dtype=np.float32)
    n = len(e)
    acc = np.zeros((n, len(MAP_Y), len(MAP_X)), np.float32)
    cnt = np.zeros_like(acc)
    for dx in (-0.025, 0.025):
        ix = np.round((MAP_X + dx - cx[0]) / ELEV_GRID["res"]).astype(int)
        okx = (ix >= 0) & (ix < len(cx))
        for dy in (-0.025, 0.025):
            iy = np.round((MAP_Y + dy - cy[0]) / ELEV_GRID["res"]).astype(int)
            oky = (iy >= 0) & (iy < len(cy))
            sub = np.full((n, len(MAP_Y), len(MAP_X)), np.nan, np.float32)
            sub[:, np.ix_(oky, okx)[0], np.ix_(oky, okx)[1]] = e[:, ix[okx]][:, :, iy[oky]].transpose(0, 2, 1)
            fin = np.isfinite(sub)
            acc += np.where(fin, sub, 0)
            cnt += fin
    with np.errstate(invalid="ignore", divide="ignore"):
        m = np.where(cnt > 0, acc / np.maximum(cnt, 1), np.nan)
    return _fill_and_reference(m, ground_z, fill_iters)


def _fill_and_reference(m: np.ndarray, ground_z: np.ndarray, fill_iters: int = 60) -> np.ndarray:
    """Fill unobserved MAP points from observed neighbours (grown outward), else 0; subtract ground_z."""
    for _ in range(fill_iters):
        miss = ~np.isfinite(m)
        if not miss.any():
            break
        p = np.pad(m, ((0, 0), (1, 1), (1, 1)), constant_values=np.nan)
        nb = np.stack([p[:, :-2, 1:-1], p[:, 2:, 1:-1], p[:, 1:-1, :-2], p[:, 1:-1, 2:]])
        f = np.isfinite(nb)
        s, c = np.where(f, nb, 0).sum(0), f.sum(0)
        m = np.where(miss & (c > 0), s / np.maximum(c, 1), m)
    m = np.where(np.isfinite(m), m - np.asarray(ground_z, np.float32)[:, None, None], 0.0)
    return m.astype(np.float32)


def parse_sensor_source(source: str) -> tuple[list[str], int]:
    """"sensor:depth_d555" -> (["depth_d555"], 1); "sensor:depth_d555+l1@10" -> (["depth_d555", "l1"], 10)."""
    spec = source.split(":", 1)[1]
    names, _, k = spec.partition("@")
    return names.split("+"), int(k) if k else 1


def fused_sensor_maps(elevs: list[np.ndarray], pose: np.ndarray, starts: np.ndarray, ground_z: np.ndarray,
                      frames: int, fill_iters: int = 60) -> np.ndarray:
    """Elevation map accumulated over the last `frames` frames (0.1 s each) of one or more sensors.

    elevs: per sensor [T, nx, ny] grids (go2wm.sim.sensors.ELEV_GRID, terrain-frame heights, NaN empty);
    pose [T, 3] scene-frame (x, y, yaw) of each frame; starts [N] the window-start frames. Every observed
    cell of frames s-frames+1..s is moved into the frame-s robot frame with the recorded poses (perfect
    registration: odometry error is not modelled) and averaged into the MAP grid point whose 10 cm cell
    holds it; then filled and referenced as in sensor_maps. frames=1 equals sensor_maps for one sensor."""
    from ..sim.sensors import ELEV_GRID, grid_centres

    cx, cy = grid_centres(ELEV_GRID)
    GX, GY = np.meshgrid(cx, cy, indexing="ij")  # [nx, ny], like the grids
    res = MAP_X[1] - MAP_X[0]
    nyx = (len(MAP_Y), len(MAP_X))
    out = np.empty((len(starts), *nyx), np.float32)
    for n, s in enumerate(starts):
        acc = np.zeros(nyx[0] * nyx[1])
        cnt = np.zeros_like(acc)
        x0, y0, th0 = pose[s]
        c0, s0 = np.cos(th0), np.sin(th0)
        for t in range(max(0, s - frames + 1), s + 1):
            xt, yt, tht = pose[t]
            ct, st = np.cos(tht), np.sin(tht)
            for e in elevs:
                z = e[t].astype(np.float64)
                ok = np.isfinite(z)
                if not ok.any():
                    continue
                gx, gy = GX[ok], GY[ok]
                wx, wy = xt + ct * gx - st * gy - x0, yt + st * gx + ct * gy - y0
                bx, by = c0 * wx + s0 * wy, -s0 * wx + c0 * wy
                ix = np.round((bx - MAP_X[0]) / res).astype(int)
                iy = np.round((by - MAP_Y[0]) / res).astype(int)
                m = (ix >= 0) & (ix < nyx[1]) & (iy >= 0) & (iy < nyx[0])
                flat = iy[m] * nyx[1] + ix[m]
                acc += np.bincount(flat, weights=z[ok][m], minlength=acc.size)
                cnt += np.bincount(flat, minlength=acc.size)
        with np.errstate(invalid="ignore", divide="ignore"):
            out[n] = np.where(cnt > 0, acc / np.maximum(cnt, 1), np.nan).reshape(nyx)
    return _fill_and_reference(out, ground_z, fill_iters)


def sample_patch(maps: torch.Tensor, pose: torch.Tensor) -> torch.Tensor:
    """maps [B, NY, NX], pose [B, 3] relative to the start pose -> patch features [B, PATCH_DIM]."""
    B = maps.shape[0]
    px = torch.as_tensor(np.append(np.repeat(PATCH_X, len(PATCH_Y)), 0.0), dtype=maps.dtype, device=maps.device)
    py = torch.as_tensor(np.append(np.tile(PATCH_Y, len(PATCH_X)), 0.0), dtype=maps.dtype, device=maps.device)
    c, s = torch.cos(pose[:, 2:3]), torch.sin(pose[:, 2:3])
    x = pose[:, 0:1] + c * px - s * py  # [B, P]
    y = pose[:, 1:2] + s * px + c * py
    gx = (x - MAP_X[0]) / (MAP_X[-1] - MAP_X[0]) * 2 - 1
    gy = (y - MAP_Y[0]) / (MAP_Y[-1] - MAP_Y[0]) * 2 - 1
    grid = torch.stack([gx, gy], -1).unsqueeze(1)  # [B, 1, P, 2]
    z = F.grid_sample(maps.unsqueeze(1), grid, mode="bilinear", padding_mode="border", align_corners=True)[:, 0, 0]
    centre = z[:, -1:]
    rel = z[:, :-1] - centre  # local relief around the predicted pose
    return torch.cat([rel, centre], -1) / HEIGHT_SCALE
