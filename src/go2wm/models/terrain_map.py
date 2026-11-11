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
