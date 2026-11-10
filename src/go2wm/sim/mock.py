"""Deterministic lightweight mock of a velocity-commanded quadruped base.

SOFTWARE SMOKE TESTS ONLY. This is a first-order-lag planar base with slip and
noise, not Go2 physics. Results on this backend say nothing about Go2
locomotion or navigation performance.
"""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np

from ..common.frames import wrap_angle, yaw_to_quat
from ..common.schema import Observation, StepResult, VelocityCommand
from .base import SceneSpec

DEFAULTS: dict[str, Any] = {
    "control_dt": 0.1,
    "physics_dt": 0.01,
    "dt_jitter_s": 0.002,
    "action_latency_s": 0.001,
    "robot_radius": 0.3,
    "base_height": 0.32,
    "vel_limits": {"vx": [-0.6, 1.0], "vy": [-0.5, 0.5], "yaw_rate": [-1.5, 1.5]},
    "tau_lin_s": 0.20,
    "tau_yaw_s": 0.15,
    "process_noise": 0.02,
    "obs_noise": 0.01,
    "families": ["open", "obstacles", "sidewalk", "slippery"],
}

FAMILY_PARAMS = {
    "open": {"gain_lin": 1.0, "gain_yaw": 1.0, "drift": 0.05, "n_obstacles": 0},
    "obstacles": {"gain_lin": 0.95, "gain_yaw": 0.95, "drift": 0.05, "n_obstacles": 7},
    "sidewalk": {"gain_lin": 0.95, "gain_yaw": 0.9, "drift": 0.08, "n_obstacles": 4},
    "slippery": {"gain_lin": 0.75, "gain_yaw": 0.7, "drift": 0.2, "n_obstacles": 3},
}


def _stable_seed(*parts: Any) -> int:
    h = hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()
    return int(h[:8], 16)


def _route_exists(bounds, obstacles, start, goal, inflation: float) -> bool:
    from ..planning.global_waypoints import OccupancyGrid, astar

    sc = SceneSpec("probe", "probe", bounds, np.array(obstacles, dtype=np.float64).reshape(-1, 3), start, goal)
    grid = OccupancyGrid.from_scene(sc, resolution=0.1, inflation=inflation)
    s, g = grid.to_cell(start[:2]), grid.to_cell(goal)
    if grid.occupied[s] or grid.occupied[g]:
        return False
    return astar(grid, start[:2], goal) is not None


def make_scene(scene_id: str, robot_radius: float = 0.3, fixed_radius: float | None = None,
               max_obstacles: int | None = None, route_margin: float = 0.15) -> SceneSpec:
    """Deterministically build a scene from its id '<family>_<index>'.

    Backend-neutral geometry (also used by the Isaac adapter, which needs a fixed
    obstacle radius and a cap on obstacle count because prims are spawned once).
    """
    family, _, idx = scene_id.rpartition("_")
    if family not in FAMILY_PARAMS or not idx.isdigit():
        raise ValueError(f"Bad mock scene_id '{scene_id}'; expected e.g. 'obstacles_0003'")
    rng = np.random.default_rng(_stable_seed("scene", scene_id))
    p = dict(FAMILY_PARAMS[family])
    # per-scene parameter variation
    p["gain_lin"] *= float(rng.uniform(0.95, 1.05))
    p["gain_yaw"] *= float(rng.uniform(0.95, 1.05))
    if family == "sidewalk":
        bounds = (-1.0, 13.0, -1.5, 1.5)
        start = np.array([0.0, 0.0, 0.0])
        goal = np.array([12.0, float(rng.uniform(-0.5, 0.5))])
    else:
        bounds = (-2.0, 12.0, -5.0, 5.0)
        start = np.array([0.0, float(rng.uniform(-2, 2)), float(rng.uniform(-0.5, 0.5))])
        goal = np.array([10.0, float(rng.uniform(-3, 3))])
    obstacles = []
    tries = 0
    n_target = p["n_obstacles"] if max_obstacles is None else min(p["n_obstacles"], max_obstacles)
    while len(obstacles) < n_target and tries < 500:
        tries += 1
        r = float(rng.uniform(0.25, 0.6)) if family != "sidewalk" else float(rng.uniform(0.2, 0.35))
        if fixed_radius is not None:
            r = fixed_radius
        x = float(rng.uniform(2.0, goal[0] - 1.5))
        y = float(rng.uniform(bounds[2] + 0.3, bounds[3] - 0.3))
        c = np.array([x, y])
        if np.linalg.norm(c - start[:2]) < r + 1.2 or np.linalg.norm(c - goal) < r + 1.2:
            continue
        if any(np.linalg.norm(c - np.array(o[:2])) < r + o[2] + 0.9 for o in obstacles):
            continue  # keep a traversable gap between obstacles
        if family == "sidewalk":
            # leave one side passable: gap >= robot diameter + margin
            if min(y - r - bounds[2], bounds[3] - (y + r)) < 0:
                continue
            if max(y - r - bounds[2], bounds[3] - (y + r)) < 2 * robot_radius + 0.4:
                continue
        if not _route_exists(bounds, obstacles + [(x, y, r)], start, goal, robot_radius + route_margin):
            continue  # this obstacle (with the others) would cut off every route to the goal
        obstacles.append((x, y, r))
    return SceneSpec(
        scene_id=scene_id,
        family=family,
        bounds=bounds,
        obstacles=np.array(obstacles, dtype=np.float64).reshape(-1, 3),
        start=start,
        goal=goal,
        params=p,
    )


class MockGo2Sim:
    """Implements SimulatorAdapter. Deterministic given (seed, scene_id)."""

    backend = "mock"

    def __init__(self, cfg: dict[str, Any] | None = None):
        c = dict(DEFAULTS)
        c.update((cfg or {}).get("mock", cfg or {}))
        self.cfg = c
        self.control_dt = float(c["control_dt"])
        self.robot_radius = float(c["robot_radius"])
        self.scene: SceneSpec | None = None
        self._rng = np.random.default_rng(0)
        self._t_ns = 0

    # --- SimulatorAdapter -------------------------------------------------
    def reset(self, seed: int, scene_id: str | None = None) -> Observation:
        if scene_id is None:
            fams = self.cfg["families"]
            scene_id = f"{fams[seed % len(fams)]}_{seed:04d}"
        self.scene = make_scene(scene_id, self.robot_radius)
        self._rng = np.random.default_rng(_stable_seed("episode", seed, scene_id))
        self._pose = self.scene.start.astype(np.float64).copy()  # x, y, yaw
        self._v = np.zeros(3)  # realized body vx, vy, wz
        self._accel = np.zeros(2)
        self._t_ns = 0
        self._steps = 0
        return self._observe()

    def step(self, action: VelocityCommand) -> StepResult:
        assert self.scene is not None, "call reset() first"
        c = self.cfg
        p = self.scene.params
        lim = c["vel_limits"]
        cmd = np.array(
            [
                np.clip(action.vx_m_s, *lim["vx"]),
                np.clip(action.vy_m_s, *lim["vy"]),
                np.clip(action.yaw_rate_rad_s, *lim["yaw_rate"]),
            ]
        )
        action_ts = self._t_ns + int(round(c["action_latency_s"] * 1e9))
        n_periods = max(1, int(round(action.duration_s / self.control_dt)))
        dt_total = sum(
            self.control_dt + self._rng.uniform(-c["dt_jitter_s"], c["dt_jitter_s"]) for _ in range(n_periods)
        )
        n_sub = max(1, int(round(dt_total / c["physics_dt"])))
        h = dt_total / n_sub
        target = np.array([p["gain_lin"] * cmd[0], p["gain_lin"] * cmd[1], p["gain_yaw"] * cmd[2]])
        a_lin = 1.0 - np.exp(-h / c["tau_lin_s"])
        a_yaw = 1.0 - np.exp(-h / c["tau_yaw_s"])
        collision = False
        v_prev = self._v.copy()
        for _ in range(n_sub):
            self._v[:2] += a_lin * (target[:2] - self._v[:2])
            self._v[2] += a_yaw * (target[2] - self._v[2])
            # slip: turning at speed produces outward lateral drift
            self._v[1] += -p["drift"] * self._v[0] * self._v[2] * h / c["tau_lin_s"]
            self._v += self._rng.normal(0.0, c["process_noise"], 3) * np.sqrt(h)
            yaw = self._pose[2]
            cs, sn = np.cos(yaw), np.sin(yaw)
            self._pose[0] += (cs * self._v[0] - sn * self._v[1]) * h
            self._pose[1] += (sn * self._v[0] + cs * self._v[1]) * h
            self._pose[2] = float(wrap_angle(yaw + self._v[2] * h))
            if self.scene.clearance(self._pose[:2], self.robot_radius) < 0.0:
                collision = True
                break
        self._accel = (self._v[:2] - v_prev[:2]) / dt_total
        self._t_ns += int(round(dt_total * 1e9))
        self._steps += 1
        obs = self._observe()
        terminated, reason = False, ""
        if collision:
            xmin, xmax, ymin, ymax = self.scene.bounds
            x, y = self._pose[:2]
            inside = xmin + self.robot_radius < x < xmax - self.robot_radius and ymin + self.robot_radius < y < ymax - self.robot_radius
            terminated, reason = True, ("collision" if inside else "out_of_bounds")
        return StepResult(
            observation=obs,
            action_timestamp_ns=action_ts,
            collision=collision,
            terminated=terminated,
            termination_reason=reason,
            info={"applied_command": cmd.tolist(), "dt_s": dt_total},
        )

    def close(self) -> None:
        pass

    def metadata(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "WARNING": "mock backend: smoke-test dynamics, not Go2 physics",
            "control_dt": self.control_dt,
            "robot_radius": self.robot_radius,
            "config": {k: v for k, v in self.cfg.items() if k != "families"},
        }

    # --- internals --------------------------------------------------------
    def _observe(self) -> Observation:
        c = self.cfg
        n = c["obs_noise"]
        rng = self._rng
        yaw = self._pose[2]
        # small pitch/roll proxy from acceleration (shows up in projected gravity)
        pitch = -0.02 * self._accel[0]
        roll = 0.02 * self._accel[1]
        g_b = np.array([np.sin(pitch), -np.sin(roll) * np.cos(pitch), -np.cos(roll) * np.cos(pitch)])
        g_b = g_b + rng.normal(0, n * 0.1, 3)
        g_b /= np.linalg.norm(g_b)
        pos = np.array([self._pose[0], self._pose[1], c["base_height"] + rng.normal(0, 0.002)])
        quat = yaw_to_quat(yaw)
        lin = np.array([self._v[0], self._v[1], 0.0]) + rng.normal(0, n, 3)
        ang = np.array([0.0, 0.0, self._v[2]]) + rng.normal(0, n, 3)
        assert self.scene is not None
        return Observation(
            timestamp_ns=self._t_ns,
            pos_w=pos,
            quat_w=quat,
            lin_vel_b=lin,
            ang_vel_b=ang,
            gravity_b=g_b,
            ground_truth={
                "pose2d": self._pose.copy(),
                "clearance_m": float(self.scene.clearance(self._pose[:2], self.robot_radius)),
            },
            valid={"rgb": False, "depth": False, "joints": False},
        )
