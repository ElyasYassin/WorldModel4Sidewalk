"""Sidewalk mission scenes on defect tiles (Phase 1, plan step 1).

A tile (terrain.py params with `defects`) is centred on the scene origin, so the walk spans x in [-L/2, L/2]. The
mission scene starts the robot at x = -L/2 + 1 on the centreline, puts the goal at x = L/2 - 1.5, bounds the robot
to the walkway (plus `edge_margin` so it can use the full width) and turns obstructions (and street trees that
reach the walkway) into obstacles. Holes, puddles, cracks and upheavals are NOT obstacles here: what the robot does
about them is the job of perception, memory and the inspection policy. `params["route"]` holds evenly spaced
inspection waypoints along the centreline for the mission runner.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..sim.base import SceneSpec
from ..sim.defects import obstacles_from_defects
from ..sim.terrain import params_with_defaults
from .site import static_obstacles


def sidewalk_mission_scene(terrain: dict[str, Any], scene_id: str = "mission_0000", n_waypoints: int = 5,
                           fixed_radius: float | None = None, edge_margin: float = 0.1) -> SceneSpec:
    p = params_with_defaults(terrain)
    L, half = float(p["size"][0]), 0.5 * float(p["width"])
    start = np.array([-L / 2 + 1.0, 0.0, 0.0])
    goal = np.array([L / 2 - 1.5, 0.0])
    obs = obstacles_from_defects(p["defects"])
    obs = obs[np.abs(obs[:, 1]) - obs[:, 2] < half + edge_margin] if len(obs) else obs  # only those reaching the walk
    if fixed_radius is not None and len(obs):  # the simulator spawns one cylinder size; plan with what is there
        obs[:, 2] = fixed_radius
    site = static_obstacles(p.get("site"))  # work-zone barriers, cones, workers: planned with their true size
    site = site[np.abs(site[:, 1]) - site[:, 2] < half + edge_margin] if len(site) else site
    obs = np.vstack([obs.reshape(-1, 3), site])
    n_waypoints = int(p.get("n_waypoints", n_waypoints))
    route = np.stack([np.linspace(start[0], goal[0], n_waypoints + 1)[1:], np.zeros(n_waypoints)], 1)
    return SceneSpec(scene_id=scene_id, family="sidewalk_mission",
                     bounds=(-L / 2 + 0.5, L / 2 - 0.5, -half - edge_margin, half + edge_margin),
                     obstacles=obs.reshape(-1, 3), start=start, goal=goal,
                     params={"terrain": p, "route": route.round(3).tolist()})
