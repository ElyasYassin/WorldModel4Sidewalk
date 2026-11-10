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
    "horizontal_scale": 0.05,
    "vertical_scale": 0.005,
}


def params_with_defaults(p: dict[str, Any] | None) -> dict[str, Any]:
    out = dict(DEFAULTS)
    out.update(p or {})
    return out


def sidewalk_height(x, y, p: dict[str, Any]) -> np.ndarray:
    """Surface height [m] at scene coordinates (broadcasts). 0 at the start (0, 0)."""
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
    z = sidewalk_height(xs[:, None], ys[None, :], p)
    return np.round(z / vertical_scale).astype(np.int16)


def make_isaac_terrain_generator(p: dict[str, Any]):
    """TerrainGeneratorCfg with one sidewalk tile. Import only after the Isaac app is running."""
    from dataclasses import MISSING  # noqa: F401

    from isaaclab.terrains import TerrainGeneratorCfg
    from isaaclab.terrains.height_field.hf_terrains_cfg import HfTerrainBaseCfg
    from isaaclab.terrains.height_field.utils import height_field_to_mesh
    from isaaclab.utils import configclass

    p = params_with_defaults(p)

    @height_field_to_mesh
    def _sidewalk(difficulty: float, cfg) -> np.ndarray:  # difficulty unused: geometry is fixed
        return sidewalk_heightfield(cfg.size, cfg.horizontal_scale, cfg.vertical_scale, cfg.params)

    @configclass
    class SidewalkHfCfg(HfTerrainBaseCfg):
        function = _sidewalk
        params: dict = {}

    tile = SidewalkHfCfg(proportion=1.0, params=p, horizontal_scale=p["horizontal_scale"],
                         vertical_scale=p["vertical_scale"], border_width=0.0, slope_threshold=0.75)
    return TerrainGeneratorCfg(
        size=tuple(p["size"]), border_width=5.0, num_rows=1, num_cols=1,
        horizontal_scale=p["horizontal_scale"], vertical_scale=p["vertical_scale"], slope_threshold=0.75,
        curriculum=False, use_cache=False, sub_terrains={"sidewalk": tile},
    )
