"""Receding-horizon CEM over bounded body-frame velocity command sequences."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import torch

from ..common.frames import compose_se2
from ..common.schema import Observation
from ..models.features import state_features
from ..models.state_dynamics import DynamicsBase
from ..sim.base import SceneSpec
from .costs import trajectory_cost
from .safety import SafetyFilter

ACT_KEYS = ("vx", "vy", "yaw_rate")


class ModelPredictor:
    """Wraps any DynamicsBase (learned or kinematic) with a rolling observation history."""

    def __init__(self, model: DynamicsBase, device: str | torch.device = "cpu"):
        self.model = model.eval().to(device)
        self.device = torch.device(device)
        self.K = model.history

    def reset(self, obs: Observation, dt_nominal: float) -> None:
        f = state_features(obs.state_vector()).astype(np.float32)
        self.hf = np.repeat(f[None], self.K, 0)
        self.ha = np.zeros((self.K, 3), np.float32)
        self.hd = np.full(self.K, dt_nominal, np.float32)
        self.t_ns = obs.timestamp_ns
        self.dt_nominal = dt_nominal

    def observe(self, obs: Observation, applied_action: np.ndarray) -> None:
        dt = (obs.timestamp_ns - self.t_ns) * 1e-9
        self.t_ns = obs.timestamp_ns
        f = state_features(obs.state_vector()).astype(np.float32)
        self.hf = np.concatenate([self.hf[1:], f[None]])
        self.ha = np.concatenate([self.ha[1:], np.asarray(applied_action, np.float32)[None]])
        self.hd = np.concatenate([self.hd[1:], np.float32([dt])])

    @torch.no_grad()
    def predict(self, actions: np.ndarray) -> np.ndarray:
        """actions [N,H,3] -> poses [N,H,3] relative to the current body frame."""
        N, H, _ = actions.shape
        t = lambda a: torch.as_tensor(a, device=self.device)  # noqa: E731
        out = self.model.rollout(
            t(self.hf[None]).expand(N, -1, -1),
            t(self.ha[None]).expand(N, -1, -1),
            t(self.hd[None]).expand(N, -1),
            t(actions.astype(np.float32)),
            torch.full((N, H), self.dt_nominal, device=self.device),
        )
        return out["poses"].cpu().numpy().astype(np.float64)


def project_deadband(A: np.ndarray, snap: float, vmin: float) -> np.ndarray:
    """Map commands the locomotion policy ignores onto ones it executes.

    Planar speed < snap -> pure stop (planar and yaw zeroed: the pretrained flat Go2 policy
    neither translates nor turns in place there). snap <= speed < vmin -> rescaled to vmin
    along the same direction. Thresholds come from measured tracking (docs/ISAAC_INTEGRATION.md).
    """
    A = A.copy()
    sp = np.linalg.norm(A[..., :2], axis=-1)
    dead = sp < snap
    A[dead] = 0.0
    up = (~dead) & (sp < vmin)
    A[..., :2][up] *= (vmin / sp[up])[:, None]
    return A


class CEMPlanner:
    def __init__(self, cfg: dict[str, Any], limits: dict[str, list[float]], seed: int = 0):
        self.H = cfg["horizon"]
        self.N = cfg["samples"]
        self.n_elite = cfg["elites"]
        self.iters = cfg["iterations"]
        self.init_std = np.array(cfg.get("init_std", [0.4, 0.2, 0.6]), dtype=np.float64)
        self.min_std = cfg.get("min_std", 0.02)
        self.w = cfg["weights"]
        self.lo = np.array([limits[k][0] for k in ACT_KEYS])
        self.hi = np.array([limits[k][1] for k in ACT_KEYS])
        self.rng = np.random.default_rng(seed)
        self.mean = np.zeros((self.H, 3))
        db = cfg.get("deadband") or {}
        self.db = (float(db["planar_snap"]), float(db["planar_min"])) if db.get("enabled") else None

    def reset(self) -> None:
        self.mean = np.zeros((self.H, 3))

    def plan(self, predictor: ModelPredictor, pose_w: np.ndarray, waypoint: np.ndarray, scene: SceneSpec | None,
             prev_action: np.ndarray, robot_radius: float) -> dict[str, Any]:
        mean = np.concatenate([self.mean[1:], self.mean[-1:]])  # warm start (shifted)
        std = np.tile(self.init_std, (self.H, 1))
        best = None
        for _ in range(self.iters):
            A = mean + std * self.rng.standard_normal((self.N, self.H, 3))
            A[0] = mean  # always evaluate the current mean
            A[1] = 0.0  # and stopping
            A = np.clip(A, self.lo, self.hi)
            if self.db is not None:
                A = np.clip(project_deadband(A, *self.db), self.lo, self.hi)
            rel = predictor.predict(A)
            P = compose_se2(pose_w[None, None], rel)
            cost, feas, _ = trajectory_cost(P, A, prev_action, waypoint, scene, self.w, robot_radius)
            if not np.all(np.isfinite(cost)):
                cost = np.where(np.isfinite(cost), cost, np.inf)
            elite = np.argsort(cost)[: self.n_elite]
            mean = A[elite].mean(0)
            std = np.maximum(A[elite].std(0), self.min_std)
            i = elite[0]
            if best is None or cost[i] < best["cost"]:
                best = {"cost": float(cost[i]), "actions": A[i].copy(), "feasible": bool(feas[i]), "poses": P[i]}
        self.mean = best["actions"]
        return best


class MPCController:
    def __init__(self, predictor: ModelPredictor, cfg: dict[str, Any], safety: SafetyFilter, seed: int = 0, name: str = "mpc"):
        self.predictor = predictor
        self.cem = CEMPlanner(cfg["mpc"], cfg["limits"], seed)
        self.safety = safety
        self.robot_radius = cfg["robot_radius"]
        self.dt = cfg["control_dt"]
        self.name = name

    def reset(self, scene: SceneSpec | None, obs: Observation) -> None:
        self.scene = scene
        self.prev = np.zeros(3)
        self.predictor.reset(obs, self.dt)
        self.cem.reset()
        self._first = True

    def act(self, obs: Observation, waypoint: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        if not self._first:
            self.predictor.observe(obs, self.prev)
        self._first = False
        t0 = time.perf_counter()
        pose = obs.pose2d
        best = self.cem.plan(self.predictor, pose, waypoint, self.scene, self.prev, self.robot_radius)
        a, reason = self.safety(best["actions"][0], self.prev, self.dt, pose, self.scene, best["feasible"])
        latency = time.perf_counter() - t0
        self.prev = a
        return a, {"latency_s": latency, "safety": reason, "plan_cost": best["cost"], "pred_poses": best["poses"]}
