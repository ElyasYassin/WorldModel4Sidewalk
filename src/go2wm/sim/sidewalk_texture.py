"""Surface appearance (albedo) of a parametric sidewalk tile, for rendering only.

`sidewalk_albedo(x, y, p, seed)` paints scene coordinates (same frame as terrain.sidewalk_height)
with sRGB colours: a concrete walkway cut into panels by tooled joints (1.5 m along, one joint
down the middle on wide walks), per-panel tint, wear and a few hairline cracks; heaved slabs in a
slightly different (older) concrete; beyond a curb a concrete curb top, a gutter pan and asphalt with
a painted edge line; otherwise a grass verge. Geometry is not changed: the same image is applied as
a texture to the Isaac terrain mesh (sim/isaac_go2.py, `terrain_texture`) and drawn under the 2D
replays (scripts/render_replays.py), so both views show the same sidewalk.

Pure numpy; deterministic from (params, seed).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .terrain import params_with_defaults

CONCRETE = np.array([0.70, 0.69, 0.66])
OLD_CONCRETE = np.array([0.63, 0.61, 0.57])
CURB_TOP = np.array([0.74, 0.73, 0.70])
GUTTER = np.array([0.60, 0.59, 0.56])
ASPHALT = np.array([0.25, 0.25, 0.26])
GRASS = np.array([0.33, 0.45, 0.20])
DIRT = np.array([0.42, 0.36, 0.26])
PAINT = np.array([0.90, 0.88, 0.80])

PANEL_LEN = 1.5  # m between transverse joints
JOINT_W = 0.012  # m, tooled joint width
CURB_W = 0.15  # m, curb top
GUTTER_W = 0.35  # m, gutter pan in front of the curb


def value_noise(x, y, cell: float, seed: int) -> np.ndarray:
    """Smooth noise in [-1, 1] on a lattice of spacing `cell` [m] (bilinear, smoothstep), defined
    for any coordinates (lattice values hashed from integer cell indices)."""
    gx, gy = np.floor(x / cell), np.floor(y / cell)
    fx, fy = x / cell - gx, y / cell - gy
    fx, fy = fx * fx * (3 - 2 * fx), fy * fy * (3 - 2 * fy)

    def h(i, j):
        k = (i.astype(np.int64) * 73856093) ^ (j.astype(np.int64) * 19349663) ^ (seed * 83492791)
        k = (k ^ (k >> 13)) * 1274126177
        return ((k & 0xFFFF) / 32767.5) - 1.0

    a, b = h(gx, gy), h(gx + 1, gy)
    c, d = h(gx, gy + 1), h(gx + 1, gy + 1)
    return (a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy


def fbm(x, y, cell: float, seed: int, octaves: int = 4) -> np.ndarray:
    out, amp, tot = 0.0, 1.0, 0.0
    for o in range(octaves):
        out = out + amp * value_noise(x, y, cell / 2**o, seed + 101 * o)
        tot += amp
        amp *= 0.5
    return out / tot


def joint_positions(p: dict[str, Any], x_lo: float, x_hi: float) -> np.ndarray:
    """Transverse joints along the walk: every PANEL_LEN, plus both ends of every slab (a heaved
    panel breaks at its joints)."""
    js = list(np.arange(np.floor(x_lo / PANEL_LEN) * PANEL_LEN, x_hi + PANEL_LEN, PANEL_LEN))
    for x0, length, _ in p["slabs"]:
        js += [x0, x0 + length]
    return np.unique(np.round(js, 4))


def crack_segments(p: dict[str, Any], seed: int, x_lo: float = -1.0, x_hi: float = 14.0) -> list[np.ndarray]:
    """A few hairline cracks (polylines [n, 2]) on the walkway; deterministic from seed."""
    rng = np.random.default_rng(seed + 7)
    half = 0.5 * p["width"]
    out = []
    for _ in range(int(rng.integers(2, 6))):
        pt = np.array([rng.uniform(x_lo, x_hi), rng.uniform(-half + 0.2, half - 0.2)])
        ang = rng.uniform(0, np.pi)
        pts = [pt.copy()]
        for _ in range(int(rng.integers(6, 16))):
            ang += rng.normal(0, 0.5)
            pt = pt + 0.08 * np.array([np.cos(ang), np.sin(ang)])
            pt[1] = np.clip(pt[1], -half + 0.02, half - 0.02)
            pts.append(pt.copy())
        out.append(np.array(pts))
    return out


def _dist_to_polyline(x, y, pts: np.ndarray) -> np.ndarray:
    d = np.full(np.shape(x), np.inf)
    for a, b in zip(pts[:-1], pts[1:]):
        ab = b - a
        t = np.clip(((x - a[0]) * ab[0] + (y - a[1]) * ab[1]) / max(ab @ ab, 1e-12), 0, 1)
        d = np.minimum(d, np.hypot(x - a[0] - t * ab[0], y - a[1] - t * ab[1]))
    return d


def sidewalk_albedo(x, y, p: dict[str, Any] | None, seed: int = 0, cracks: bool = True) -> np.ndarray:
    """sRGB albedo [..., 3] in [0, 1] at scene coordinates x, y (broadcast)."""
    p = params_with_defaults(p)
    x, y = np.broadcast_arrays(np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64))
    half = 0.5 * p["width"]
    img = np.empty(x.shape + (3,))
    walk = np.abs(y) <= half
    curb = p["curb_drop"] > 0
    road_side = y < -half
    img[:] = GRASS

    # verge: grass with dirt patches and blade-scale speckle
    g = fbm(x, y, 1.2, seed + 1)
    dirt = np.clip((g - 0.25) * 3, 0, 1)[..., None]
    img = img * (1 - dirt) + DIRT * dirt
    img *= (1 + 0.10 * fbm(x, y, 0.06, seed + 2, 2) + 0.06 * fbm(x, y, 0.5, seed + 3))[..., None]

    if curb:
        top = road_side & (y >= -half - CURB_W)
        gut = road_side & (y < -half - CURB_W) & (y >= -half - CURB_W - GUTTER_W)
        asph = road_side & (y < -half - CURB_W - GUTTER_W)
        img[top] = CURB_TOP
        img[gut] = GUTTER * (1 + 0.04 * fbm(x[gut], y[gut], 0.3, seed + 4))[..., None]
        a = (1 + 0.07 * fbm(x[asph], y[asph], 0.03, seed + 5, 2) + 0.05 * fbm(x[asph], y[asph], 1.5, seed + 6))
        img[asph] = ASPHALT * a[..., None]
        line = asph & (np.abs(y + half + CURB_W + GUTTER_W + 0.30) < 0.05)
        img[line] = PAINT * (1 - 0.15 * np.clip(fbm(x[line], y[line], 0.2, seed + 8), 0, 1))[..., None]
        img[gut & (np.abs(y + half + CURB_W) < 0.006)] *= 0.7  # curb / gutter seam

    # walkway: concrete panels
    xw, yw = x[walk], y[walk]
    js = joint_positions(p, float(np.min(x)) if x.size else 0.0, float(np.max(x)) if x.size else 0.0)
    idx = np.searchsorted(js, xw, side="right")  # panel index along the walk
    long_joint = p["width"] >= 2.0
    side = (yw > 0).astype(np.int64) if long_joint else np.zeros_like(idx)
    rng = np.random.default_rng(seed + 11)
    tint = rng.normal(0, 0.035, (len(js) + 2, 2))
    base = np.broadcast_to(CONCRETE, (len(xw), 3)).copy()
    for x0, length, _ in p["slabs"]:
        on = (xw >= x0) & (xw < x0 + length)
        base[on] = OLD_CONCRETE
    shade = 1 + tint[idx, side] + 0.05 * fbm(xw, yw, 0.4, seed + 12) + 0.035 * fbm(xw, yw, 0.02, seed + 13, 2)
    stain = np.clip((fbm(xw, yw, 0.8, seed + 14) - 0.35) * 2.5, 0, 1) * 0.18  # wear / water stains
    w = base * (shade - stain)[..., None]
    # tooled joints (dark grooves) and a slightly smoother rim along each joint
    dj = np.min(np.abs(xw[:, None] - js[None, :]), axis=1) if len(js) else np.full(len(xw), np.inf)
    w[dj < JOINT_W / 2] *= 0.55
    w[(dj >= JOINT_W / 2) & (dj < 0.05)] *= 1.03
    if long_joint:
        w[np.abs(yw) < JOINT_W / 2] *= 0.55
    w[half - np.abs(yw) < 0.02] *= 0.8  # walkway edge
    if cracks:
        for pts in crack_segments(p, seed):
            w[_dist_to_polyline(xw, yw, pts) < 0.004] *= 0.5
    img[walk] = w
    if p["defects"]:
        from .defects import paint_defects

        img = paint_defects(img, x, y, p["defects"], half, value_noise)
    return np.clip(img, 0.0, 1.0)
