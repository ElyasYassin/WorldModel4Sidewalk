"""Command generators for terrain (E4) data collection."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..common.schema import VelocityCommand
from .recorder import ExplorationCommands


def _wrap(a: float) -> float:
    return float(np.arctan2(np.sin(a), np.cos(a)))


class LaneFollow:
    """Forward speed vx along +x with P-control on lateral offset; y_ref may change with x.
    schedule: list of (x_from, y_ref) applied in order; before the first entry y_ref = the scene goal's y."""

    def __init__(self, vx: float, dt: float, schedule=None, k_y: float = 0.8, k_yaw: float = 1.5):
        self.vx, self.dt, self.schedule, self.k_y, self.k_yaw = vx, dt, schedule, k_y, k_yaw

    def __call__(self, obs, scene) -> VelocityCommand:
        x, y, yaw = obs.pose2d
        y_ref = float(scene.goal[1])
        for x_from, yr in self.schedule or []:
            if x >= x_from:
                y_ref = yr
        yaw_ref = float(np.clip(-self.k_y * (y - y_ref), -0.5, 0.5))
        wz = float(np.clip(self.k_yaw * _wrap(yaw_ref - yaw), -1.0, 1.0))
        return VelocityCommand(self.vx, 0.0, wz, self.dt)


class TerrainMixCommands:
    """Per episode either the standard exploration generator (training command support) or a
    lane-follower with a random speed in [0.3, 0.9] m/s and 0-2 random lateral target changes
    (some cross the walkway edge / curb). Chosen from the seed; `kind` records which."""

    def __init__(self, seed: int, limits: dict[str, list[float]], control_dt: float,
                 exploration_cfg: dict[str, Any] | None = None, p_lane: float = 0.5):
        rng = np.random.default_rng(seed + 7919)
        if rng.uniform() < p_lane:
            self.kind = "lane"
            sched = sorted((float(rng.uniform(1.0, 10.0)), float(rng.uniform(-2.3, 1.2)))
                           for _ in range(int(rng.integers(0, 3))))
            self.speed = float(rng.uniform(0.3, 0.9))
            self.gen = LaneFollow(self.speed, control_dt, schedule=sched)
            self.schedule = sched
        else:
            self.kind = "exploration"
            self.gen = ExplorationCommands(seed, limits, control_dt, exploration_cfg)

    def describe(self) -> dict[str, Any]:
        if self.kind == "lane":
            return {"kind": "lane", "vx": self.speed, "schedule": self.schedule}
        return {"kind": "exploration"}

    def __call__(self, obs, scene) -> VelocityCommand:
        return self.gen(obs, scene)
