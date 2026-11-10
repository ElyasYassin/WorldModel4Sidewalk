"""Non-learned reactive local controller: pure pursuit + potential-field obstacle repulsion."""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from ..common.frames import world_to_body_2d
from ..common.schema import Observation
from ..sim.base import SceneSpec
from .safety import SafetyFilter


class ReactiveController:
    name = "conventional"

    def __init__(self, cfg: dict[str, Any], safety: SafetyFilter):
        # slow_floor: minimum fraction of vx kept when slowing near an obstacle.
        # min_speed: deadband-aware floor on forward speed (0 = off). The pretrained flat Go2 policy
        #   ignores planar commands below ~0.2-0.4 m/s and cannot turn in place, so a controller that
        #   slows or turns on the spot freezes (E3a/E3b). With a floor it turns in forward arcs instead.
        # bound_repulsion: also repel from the scene bounds (sidewalk edges), not only obstacles.
        # Defaults reproduce the controller used in E0-E3e.
        c = {"v_max": 0.8, "k_yaw": 1.8, "influence_m": 1.0, "k_rep": 1.2, "slow_m": 0.6,
             "slow_floor": 0.2, "min_speed": 0.0, "bound_repulsion": False}
        c.update(cfg.get("reactive", {}))
        self.c = c
        self.safety = safety
        self.dt = cfg["control_dt"]
        self.robot_radius = cfg["robot_radius"]

    def reset(self, scene: SceneSpec | None, obs: Observation) -> None:
        self.scene = scene
        self.prev = np.zeros(3)

    def act(self, obs: Observation, waypoint: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        t0 = time.perf_counter()
        c = self.c
        pose = obs.pose2d
        tgt_b = world_to_body_2d(waypoint - pose[:2], pose[2])
        dist = float(np.linalg.norm(tgt_b))
        heading_err = float(np.arctan2(tgt_b[1], tgt_b[0]))
        wz = c["k_yaw"] * heading_err
        vx = c["v_max"] * max(0.0, np.cos(heading_err)) * min(1.0, dist / 0.8)
        if self.scene is not None:
            # repulsors: (body-frame vector to nearest point, clearance)
            reps = []
            for ox, oy, r in np.asarray(self.scene.obstacles).reshape(-1, 3):
                rel = world_to_body_2d(np.array([ox, oy]) - pose[:2], pose[2])
                reps.append((rel, np.linalg.norm(rel) - r - self.robot_radius))
            if c["bound_repulsion"]:
                xmin, xmax, ymin, ymax = self.scene.bounds
                x, y = pose[:2]
                for pt in ((xmin, y), (xmax, y), (x, ymin), (x, ymax)):
                    rel = world_to_body_2d(np.array(pt) - pose[:2], pose[2])
                    reps.append((rel, np.linalg.norm(rel) - self.robot_radius))
            for rel, d in reps:
                if d < c["influence_m"] and rel[0] > -0.2:
                    push = c["k_rep"] * (c["influence_m"] - d) / c["influence_m"]
                    side = -np.sign(rel[1]) if abs(rel[1]) > 0.05 else (1.0 if heading_err >= 0 else -1.0)
                    wz += side * push * 1.5
                    if rel[0] > 0:
                        vx *= float(np.clip(d / c["slow_m"], c["slow_floor"], 1.0))
        if c["min_speed"] > 0 and dist > 1e-6:
            vx = max(vx, c["min_speed"])  # the safety filter still stops motion that reduces clearance
        a, reason = self.safety(np.array([vx, 0.0, wz]), self.prev, self.dt, pose, self.scene, True)
        self.prev = a
        return a, {"latency_s": time.perf_counter() - t0, "safety": reason}
