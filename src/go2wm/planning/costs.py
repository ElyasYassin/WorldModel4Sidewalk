"""Trajectory costs for the local planner.

Only geometric, map-based terms plus goal progress and smoothness. No learned
risk/collision head is used (none is trained/validated yet; spec section 9.2).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..sim.base import SceneSpec


def trajectory_cost(
    poses_w: np.ndarray,  # [N, H, 3] predicted world poses
    actions: np.ndarray,  # [N, H, 3]
    prev_action: np.ndarray,  # [3]
    waypoint: np.ndarray,  # [2]
    scene: SceneSpec | None,
    w: dict[str, Any],
    robot_radius: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    d = np.linalg.norm(poses_w[..., :2] - waypoint, axis=-1)  # [N, H]
    goal = d[:, -1] + w.get("running_goal", 0.2) * d.mean(1)
    # heading alignment at the end (helps waypoint following when close)
    to_wp = waypoint - poses_w[:, -1, :2]
    hd = np.arctan2(to_wp[:, 1], to_wp[:, 0]) - poses_w[:, -1, 2]
    heading = (1 - np.cos(hd)) * np.minimum(1.0, d[:, -1])
    terms = {"goal": w["goal"] * goal, "heading": w.get("heading", 0.0) * heading}
    feasible = np.ones(len(poses_w), dtype=bool)
    if scene is not None:
        clear = scene.clearance(poses_w[..., :2], robot_radius)  # [N, H]
        margin = w.get("clearance_margin_m", 0.3)
        terms["collision"] = w["collision"] * (np.maximum(0.0, margin - clear) ** 2).sum(1)
        feasible = clear.min(1) > 0.0
    da = np.diff(np.concatenate([prev_action[None, None].repeat(len(actions), 0), actions], 1), axis=1)
    terms["smooth"] = w["smooth"] * (da**2).sum((1, 2))
    terms["effort"] = w.get("effort", 0.0) * (actions[..., 1:] ** 2).sum((1, 2))  # lateral + yaw effort
    total = sum(terms.values())
    total = np.where(feasible, total, total + w.get("infeasible_penalty", 1e3))
    return total, feasible, terms
