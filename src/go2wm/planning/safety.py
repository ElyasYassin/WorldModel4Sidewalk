"""Hard safety constraints applied after any planner (learned or not)."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..sim.base import SceneSpec


class SafetyFilter:
    def __init__(self, cfg: dict[str, Any]):
        self.lim = cfg["limits"]
        self.max_accel = np.array(cfg.get("max_accel", [2.0, 2.0, 4.0]), dtype=np.float64)  # per second
        self.stop_clearance = float(cfg.get("stop_clearance_m", 0.05))
        self.robot_radius = float(cfg.get("robot_radius", 0.3))
        self.stops = 0

    def clip(self, a) -> np.ndarray:
        a = np.asarray(a, dtype=np.float64)
        return np.array([np.clip(a[i], *self.lim[k]) for i, k in enumerate(("vx", "vy", "yaw_rate"))])

    def __call__(self, action, prev_action, dt: float, pose, scene: SceneSpec | None, feasible: bool = True) -> tuple[np.ndarray, str]:
        a = np.asarray(action, dtype=np.float64)
        if not np.all(np.isfinite(a)):
            self.stops += 1
            return np.zeros(3), "invalid_action"
        if not feasible:
            self.stops += 1
            return np.zeros(3), "no_feasible_candidate"
        a = self.clip(a)
        step = self.max_accel * dt
        a = np.clip(a, prev_action - step, prev_action + step)
        if scene is not None:
            c = float(scene.clearance(np.asarray(pose[:2]), self.robot_radius))
            if c < self.stop_clearance:
                # allow only motion that increases clearance: probe 0.1 s ahead in body frame
                yaw = pose[2]
                ahead = np.asarray(pose[:2]) + 0.1 * np.array(
                    [np.cos(yaw) * a[0] - np.sin(yaw) * a[1], np.sin(yaw) * a[0] + np.cos(yaw) * a[1]]
                )
                if scene.clearance(ahead, self.robot_radius) < c:
                    self.stops += 1
                    return np.array([0.0, 0.0, a[2]]), "clearance_stop"
        return a, ""
