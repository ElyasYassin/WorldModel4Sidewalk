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
        c = {"v_max": 0.8, "k_yaw": 1.8, "influence_m": 1.0, "k_rep": 1.2, "slow_m": 0.6}
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
        min_clear = np.inf
        if self.scene is not None and len(self.scene.obstacles):
            for ox, oy, r in self.scene.obstacles:
                rel = world_to_body_2d(np.array([ox, oy]) - pose[:2], pose[2])
                d = np.linalg.norm(rel) - r - self.robot_radius
                min_clear = min(min_clear, d)
                if d < c["influence_m"] and rel[0] > -0.2:
                    push = c["k_rep"] * (c["influence_m"] - d) / c["influence_m"]
                    side = -np.sign(rel[1]) if abs(rel[1]) > 0.05 else (1.0 if heading_err >= 0 else -1.0)
                    wz += side * push * 1.5
                    if rel[0] > 0:
                        vx *= float(np.clip(d / c["slow_m"], 0.2, 1.0))
        a, reason = self.safety(np.array([vx, 0.0, wz]), self.prev, self.dt, pose, self.scene, True)
        self.prev = a
        return a, {"latency_s": time.perf_counter() - t0, "safety": reason}
