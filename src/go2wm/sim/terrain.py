"""Parametric sidewalk terrain for the Isaac Go2 backend (P1 / E4).

Scene frame (same as SceneSpec): x along the sidewalk, y across it, robot starts at (0, 0),
the walkway is y in [-w/2, w/2]. The terrain tile is centred on the scene origin, because Isaac
Lab puts the env origin at the tile centre.

Features (all optional, all deterministic from the params dict):
  cross_slope   rise per metre across the walkway (+y up); ADA limit is 0.02
  grade         list of [x_start, x_end, rise_per_m] running-slope sections (heights carry over)
  slabs         list of [x_start, length, height_m] raised slabs across the full width
                (heaved / faulted panels; a vertical step at both ends)
  curb_drop     drop [m] from the walkway edge at y = -w/2 down to a road surface (a curb)
  friction      terrain static = dynamic friction coefficient (Isaac multiplies it with the
                robot foot material, 0.8 static / 0.6 dynamic in the Go2 rough task)
  defects       list of sidewalk defects (upheaval, crack, hole, puddle, obstruction): see sim/defects.py.
                Defect tiles use a finer grid (horizontal_scale 0.02, vertical_scale 0.001).

`sidewalk_height(x, y, p)` is the analytic surface [m]; `sidewalk_heightfield` samples it on the
Isaac grid. Both are pure numpy so they can be tested without the simulator.
"""

from __future__ import annotations

from typing import Any

import numpy as np

DEFAULTS: dict[str, Any] = {
    "size": [28.0, 8.0],
    "width": 3.0,
    "cross_slope": 0.0,
    "grade": [],
    "slabs": [],
    "curb_drop": 0.0,
    "friction": 1.0,
    "defects": [],
    "horizontal_scale": 0.05,
    "vertical_scale": 0.005,
}


def params_with_defaults(p: dict[str, Any] | None) -> dict[str, Any]:
    out = dict(DEFAULTS)
    out.update(p or {})
    return out


def sidewalk_height(x, y, p: dict[str, Any], grid: float = 0.0) -> np.ndarray:
    """Surface height [m] at scene coordinates (broadcasts). 0 at the start (0, 0) unless a defect is there.
    `grid` > 0 rasterises sub-grid defects (narrow cracks) for a height field of that spacing."""
    p = params_with_defaults(p)
    x, y = np.broadcast_arrays(np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64))
    half = 0.5 * p["width"]
    z = p["cross_slope"] * np.clip(y, -half, half)
    for x0, x1, rise in p["grade"]:
        z = z + rise * (np.clip(x, x0, x1) - x0)  # height gained on the section carries on after it
    for x0, length, h in p["slabs"]:
        z = z + np.where((x >= x0) & (x < x0 + length), h, 0.0)
    if p["curb_drop"] > 0:
        z = np.where(y < -half, z - p["curb_drop"], z)
    if p["defects"]:
        from .defects import defect_height

        z = z + defect_height(x, y, p["defects"], grid)
    return z


def sidewalk_heightfield(size: tuple[float, float], horizontal_scale: float, vertical_scale: float,
                         p: dict[str, Any]) -> np.ndarray:
    """int16 height field [nx, ny] in units of vertical_scale, index (0, 0) at (-size_x/2, -size_y/2)
    in scene coordinates. Isaac's height_field_to_mesh calls this with the tile size minus a
    one-pixel border on each side and places the result inside that border, so the grid stays
    centred on the tile (= env origin = scene origin)."""
    # round, not floor: Isaac passes size = (n - 2) * scale, which floors to n - 3 in floating point
    nx = int(round(size[0] / horizontal_scale))
    ny = int(round(size[1] / horizontal_scale))
    xs = np.arange(nx) * horizontal_scale - 0.5 * size[0]
    ys = np.arange(ny) * horizontal_scale - 0.5 * size[1]
    z = sidewalk_height(xs[:, None], ys[None, :], p, grid=horizontal_scale)
    return np.round(z / vertical_scale).astype(np.int16)


def random_sidewalk_tile(rng: np.random.Generator, slab_max: float = 0.18, grade_max: float = 0.25,
                         cross_max: float = 0.12, curb_max: float = 0.20) -> dict[str, Any]:
    """One randomized sidewalk tile for data collection (features away from the start, x in [2, 11])."""
    p: dict[str, Any] = {"width": float(rng.uniform(2.5, 3.5)),
                         "cross_slope": float(rng.choice([-1, 1]) * rng.uniform(0, cross_max))}
    slabs, x = [], 2.0
    for _ in range(int(rng.integers(0, 3))):
        x0 = float(rng.uniform(x, x + 2.5))
        L = float(rng.uniform(0.8, 2.0))
        if x0 + L > 11.0:
            break
        slabs.append([round(x0, 2), round(L, 2), round(float(rng.uniform(0.03, slab_max)), 3)])
        x = x0 + L + 0.5
    p["slabs"] = slabs
    if rng.uniform() < 0.6:
        x0 = float(rng.uniform(x if x < 8 else 2.0, 9.0))
        x1 = min(x0 + float(rng.uniform(1.5, 4.0)), 12.0)
        p["grade"] = [[round(x0, 2), round(x1, 2), round(float(rng.choice([-1, 1]) * rng.uniform(0.03, grade_max)), 3)]]
    else:
        p["grade"] = []
    p["curb_drop"] = round(float(rng.uniform(0.08, curb_max)), 3) if rng.uniform() < 0.6 else 0.0
    return p


def random_defect_tile(rng: np.random.Generator, length: float = 24.0, n_defects: int | None = None,
                       types=None) -> dict[str, Any]:
    """A sidewalk tile with mild geometry (grade, cross slope, curb; no flat-raised slabs: upheavals replace
    them) and 4-8 defects on the fine defect grid. The tile is centred on the scene origin, so the walk spans
    x in [-length/2, length/2]; defects go in [-length/2 + 3, length/2 - 1.5] (the first 3 m are left clear for
    a route start at x = -length/2 + 1)."""
    from .defects import DEFECT_TYPES, random_defects

    p = random_sidewalk_tile(rng, slab_max=0.04, grade_max=0.08, cross_max=0.03, curb_max=0.15)  # slabs dropped below
    p["slabs"] = []
    p.update({"size": [length, 7.0], "horizontal_scale": 0.02, "vertical_scale": 0.001})
    p["defects"] = random_defects(rng, p, (-length / 2 + 3.0, length / 2 - 1.5), n_defects, types or DEFECT_TYPES)
    return p


def make_isaac_terrain_generator(p: dict[str, Any] | list[dict[str, Any]]):
    """TerrainGeneratorCfg with one sidewalk tile, or one column per tile for a list of tiles
    (curriculum layout with equal proportions puts tile i in column i). Import only after the
    Isaac app is running."""
    from isaaclab.terrains import TerrainGeneratorCfg
    from isaaclab.terrains.height_field.hf_terrains_cfg import HfTerrainBaseCfg
    from isaaclab.terrains.height_field.utils import height_field_to_mesh
    from isaaclab.utils import configclass

    tiles = [params_with_defaults(t) for t in (p if isinstance(p, list) else [p])]
    size = tuple(tiles[0]["size"])
    hs, vs = tiles[0]["horizontal_scale"], tiles[0]["vertical_scale"]
    if any(tuple(t["size"]) != size or t["horizontal_scale"] != hs or t["vertical_scale"] != vs for t in tiles):
        raise ValueError("all terrain tiles must share size and grid scales")

    @height_field_to_mesh
    def _sidewalk(difficulty: float, cfg) -> np.ndarray:  # difficulty unused: geometry is fixed
        return sidewalk_heightfield(cfg.size, cfg.horizontal_scale, cfg.vertical_scale, cfg.params)

    @configclass
    class SidewalkHfCfg(HfTerrainBaseCfg):
        function = _sidewalk
        params: dict = {}

    subs = {f"sidewalk_{i:03d}": SidewalkHfCfg(proportion=1.0, params=t, horizontal_scale=hs, vertical_scale=vs,
                                              border_width=0.0, slope_threshold=0.75)
            for i, t in enumerate(tiles)}
    return TerrainGeneratorCfg(
        size=size, border_width=5.0, num_rows=1, num_cols=len(tiles),
        horizontal_scale=hs, vertical_scale=vs, slope_threshold=0.75,
        curriculum=len(tiles) > 1, use_cache=False, sub_terrains=subs,
    )
