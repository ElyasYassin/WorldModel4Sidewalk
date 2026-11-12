"""Synthetic RGB cameras without Isaac: pinhole rays onto the tile surface, coloured by the sidewalk texture.

For crack studies only: geometry is the tile surface (ray-plane intersection refined by fixed-point iterations on
the height field: fine for smooth ground, no occlusion), colour is go2wm.sim.sidewalk_texture.sidewalk_albedo
sampled with supersampling (anti-aliasing of sub-pixel cracks), then optics blur (Gaussian, in pixels), sensor
noise and 8-bit quantisation. No lighting, shadows, glare or motion blur: an optimistic but honest test of whether
a crack is RESOLVED at a given range.

Cameras (approximate; check datasheets): `d555_rgb` (~1 MP colour of the D555 on its mount, 90 x 64 deg,
1280 x 800) and `d405_rgb` (the D405's colour imagers, 87 x 58 deg, 1280 x 720, on the D405 mount).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .sensors import SENSOR_SPECS, pitch_matrix
from .sidewalk_texture import sidewalk_albedo
from .terrain import params_with_defaults, sidewalk_height

CAMERAS: dict[str, dict[str, Any]] = {
    "d555_rgb": {"mount_pos": SENSOR_SPECS["depth_d555"]["mount_pos"], "pitch_deg": SENSOR_SPECS["depth_d555"]["pitch_deg"],
                 "hfov_deg": 90.0, "vfov_deg": 64.0, "width": 1280, "height": 800, "blur_px": 0.8, "noise": 0.012},
    "d405_rgb": {"mount_pos": SENSOR_SPECS["depth_d405"]["mount_pos"], "pitch_deg": SENSOR_SPECS["depth_d405"]["pitch_deg"],
                 "hfov_deg": 87.0, "vfov_deg": 58.0, "width": 1280, "height": 720, "blur_px": 0.8, "noise": 0.012},
}


def _rays(cam, u, v):
    """Base-frame unit directions for (sub)pixel coordinates u (cols), v (rows) in pixels."""
    tx, ty = np.tan(np.radians(cam["hfov_deg"]) / 2), np.tan(np.radians(cam["vfov_deg"]) / 2)
    xn = (u / cam["width"]) * 2 - 1
    yn = (v / cam["height"]) * 2 - 1
    d = np.stack([np.ones_like(xn), -xn * tx, -yn * ty], -1)
    d /= np.linalg.norm(d, axis=-1, keepdims=True)
    return d @ pitch_matrix(cam["pitch_deg"]).T


def ground_points(cam, u, v, base_pos, base_R, terrain, iters: int = 3):
    d = _rays(cam, u, v) @ base_R.T
    o = base_pos + base_R @ np.asarray(cam["mount_pos"], float)
    z0 = float(sidewalk_height(base_pos[0], base_pos[1], terrain))
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (o[2] - z0) / -d[..., 2]
        for _ in range(iters):
            p = o + d * t[..., None]
            t = (o[2] - sidewalk_height(p[..., 0], p[..., 1], terrain)) / -d[..., 2]
    t = np.where((t > 0) & np.isfinite(t), t, np.nan)
    return o + d * t[..., None], t


def project(cam, xyz, base_pos, base_R):
    """World points -> pixel coordinates (u, v) (NaN if behind the camera)."""
    o = base_pos + base_R @ np.asarray(cam["mount_pos"], float)
    q = (xyz - o) @ base_R @ pitch_matrix(cam["pitch_deg"])  # camera frame: x forward, y left, z up
    tx, ty = np.tan(np.radians(cam["hfov_deg"]) / 2), np.tan(np.radians(cam["vfov_deg"]) / 2)
    with np.errstate(divide="ignore", invalid="ignore"):
        xn, yn = -q[..., 1] / q[..., 0] / tx, -q[..., 2] / q[..., 0] / ty
    u, v = (xn + 1) / 2 * cam["width"], (yn + 1) / 2 * cam["height"]
    bad = q[..., 0] <= 0
    return np.where(bad, np.nan, u), np.where(bad, np.nan, v)


def _blur(img, sigma):
    if sigma <= 0:
        return img
    r = int(np.ceil(3 * sigma))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    out = img
    for ax in (0, 1):
        pad = [(0, 0)] * img.ndim
        pad[ax] = (r, r)
        q = np.pad(out, pad, mode="edge")
        out = sum(k[i] * np.take(q, np.arange(i, i + img.shape[ax]), axis=ax) for i in range(2 * r + 1))
    return out


def render(cam_name: str, base_pos, base_R, terrain, window=None, ss: int = 3, seed: int = 0, texture_seed: int = 0):
    """RGB image (float [0, 1]) of a pixel window (u0, u1, v0, v1) (default: whole frame) + ground points of
    the pixel centres + metres per pixel (ground footprint) per pixel."""
    cam = CAMERAS[cam_name]
    p = params_with_defaults(terrain)
    u0, u1, v0, v1 = window or (0, cam["width"], 0, cam["height"])
    us, vs = np.arange(u0, u1), np.arange(v0, v1)
    off = (np.arange(ss) + 0.5) / ss
    acc = 0.0
    for a in off:
        for b in off:
            U, V = np.meshgrid(us + a, vs + b)
            g, _ = ground_points(cam, U, V, base_pos, base_R, p)
            col = sidewalk_albedo(np.nan_to_num(g[..., 0]), np.nan_to_num(g[..., 1]), p, seed=texture_seed, cracks=False)
            col[~np.isfinite(g[..., 0])] = 0.55  # sky / no ground: flat grey
            acc = acc + col
    img = _blur(acc / ss**2, cam["blur_px"])
    rng = np.random.default_rng(seed)
    img = np.clip(img + rng.normal(0, cam["noise"], img.shape), 0, 1)
    img = np.round(img * 255) / 255
    Uc, Vc = np.meshgrid(us + 0.5, vs + 0.5)
    gc, _ = ground_points(cam, Uc, Vc, base_pos, base_R, p)
    g1, _ = ground_points(cam, Uc + 1, Vc, base_pos, base_R, p)
    g2, _ = ground_points(cam, Uc, Vc + 1, base_pos, base_R, p)
    mpp = np.sqrt(np.linalg.norm(g1 - gc, axis=-1) * np.linalg.norm(g2 - gc, axis=-1))
    return img, gc, mpp


# --- crack measurement from an image (no ground truth used) ---------------------------------------------------

def orthorectify(img, cam_name, base_pos, base_R, terrain, window, x_range, y_range, res):
    """Resample a rendered window onto a ground grid (bilinear in the image)."""
    cam = CAMERAS[cam_name]
    p = params_with_defaults(terrain)
    xs = np.arange(x_range[0], x_range[1], res) + res / 2
    ys = np.arange(y_range[0], y_range[1], res) + res / 2
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    Z = sidewalk_height(X, Y, p)
    u, v = project(cam, np.stack([X, Y, Z], -1), base_pos, base_R)
    u, v = u - window[0] - 0.5, v - window[2] - 0.5
    gray = img.mean(-1)
    ok = (u >= 0) & (u < gray.shape[1] - 1) & (v >= 0) & (v < gray.shape[0] - 1)
    ui, vi = np.floor(np.where(ok, u, 0)).astype(int), np.floor(np.where(ok, v, 0)).astype(int)
    fu, fv = np.where(ok, u, 0) - ui, np.where(ok, v, 0) - vi
    val = (gray[vi, ui] * (1 - fu) * (1 - fv) + gray[vi, ui + 1] * fu * (1 - fv) + gray[vi + 1, ui] * (1 - fu) * fv
           + gray[vi + 1, ui + 1] * fu * fv)
    return np.where(ok, val, np.nan), xs, ys


def _box_median(a, r):
    nx, ny = a.shape
    pad = np.pad(a, r, mode="edge")
    step = max(1, r // 3)
    offs = [(i, j) for i in range(0, 2 * r + 1, step) for j in range(0, 2 * r + 1, step)]
    return np.nanmedian(np.stack([pad[i : i + nx, j : j + ny] for i, j in offs]), axis=0)


def measure_crack(ortho, res, dark_ratio: float = 0.8, min_px: int = 20):
    """Width and length of the largest dark linear structure in an orthorectified grey image (None if none).

    Width from a distance transform: for every crack pixel, the distance to the nearest non-crack pixel; along the
    centre line (local maxima) that distance is half the width (width = 2 d - 1 px, exact for odd pixel widths,
    -1 px for even). Length = area / width. (A per-tile PCA extent counts the width into the length and
    under-reads wide cracks.)"""
    from ..mission.perception import _clusters

    bg = _box_median(ortho, max(3, int(round(0.03 / res))))
    mask = np.isfinite(ortho) & (ortho < dark_ratio * bg)
    comps = [c for c in _clusters(mask) if len(c) >= min_px]
    if not comps:
        return None
    comp = max(comps, key=len)
    lo = comp.min(0) - 1
    sub = np.zeros(tuple(comp.max(0) - lo + 2), bool)
    sub[comp[:, 0] - lo[0], comp[:, 1] - lo[1]] = True
    skel = _skeleton(sub)
    w0 = len(comp) / max(_skel_length(skel), 1.0)  # first width guess (spurs make it low)
    k = int(np.ceil(max(2.0, w0)))  # prune spurs shorter than ~ the width (jagged edges of a thick crack)
    pruned = _prune(skel, k)
    length_px = _skel_length(pruned) + (2 * k if pruned.any() else 0.0)  # pruning also shortened the two real ends
    if not pruned.any():
        length_px = _skel_length(skel)
    width_px = len(comp) / max(length_px, 1.0)
    return {"width_m": width_px * res, "length_m": length_px * res, "pixels": int(len(comp))}


def _nb8(S: np.ndarray) -> np.ndarray:
    P = np.pad(S, 1).astype(np.int8)
    return sum(np.roll(np.roll(P, a, 0), b, 1) for a in (-1, 0, 1) for b in (-1, 0, 1) if (a, b) != (0, 0))[1:-1, 1:-1]


def _prune(skel: np.ndarray, k: int) -> np.ndarray:
    s = skel.copy()
    for _ in range(k):
        ends = s & (_nb8(s) <= 1)
        if not ends.any():
            break
        s &= ~ends
    return s


def _skel_length(skel: np.ndarray, tile: int = 12) -> float:
    """Centre-line length: in tiles of `tile` px, the extent of the skeleton pixels along their principal direction
    (+1 px), summed. Robust to the 2-px staircases thinning leaves on diagonals (link counting over-reads those)."""
    pts = np.argwhere(skel).astype(float)
    if len(pts) < 2:
        return float(len(pts))
    keys = np.floor(pts / tile).astype(int)
    total = 0.0
    for k in np.unique(keys, axis=0):
        q = pts[(keys == k).all(1)]
        if len(q) == 1:
            total += 1.0
            continue
        c = q - q.mean(0)
        _, v = np.linalg.eigh(c.T @ c)
        total += float(np.ptp(c @ v[:, -1])) + 1.0
    return total


def _skeleton(img: np.ndarray) -> np.ndarray:
    """Zhang-Suen thinning of a boolean image to one-pixel-wide centre lines."""
    sk = np.pad(img.astype(bool), 1)
    while True:
        changed = False
        for step in (0, 1):
            P = sk.astype(np.int8)
            n = [np.roll(np.roll(P, a, 0), b, 1) for a, b in ((1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1))]
            # neighbours P2..P9 clockwise from north (row - 1): roll by +1 shifts row r-1 into r
            p2, p3, p4, p5, p6, p7, p8, p9 = n[0], n[7], n[6], n[5], n[4], n[3], n[2], n[1]
            seq = [p2, p3, p4, p5, p6, p7, p8, p9, p2]
            B = sum(seq[:8])
            A = sum(((seq[i] == 0) & (seq[i + 1] == 1)).astype(np.int8) for i in range(8))
            if step == 0:
                cond = (p2 * p4 * p6 == 0) & (p4 * p6 * p8 == 0)
            else:
                cond = (p2 * p4 * p8 == 0) & (p2 * p6 * p8 == 0)
            rm = sk & (B >= 2) & (B <= 6) & (A == 1) & cond
            if rm.any():
                sk = sk & ~rm
                changed = True
        if not changed:
            return sk[1:-1, 1:-1]
