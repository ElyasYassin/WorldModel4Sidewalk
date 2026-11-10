"""Simulator-independent adapter protocol and scene description."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np

from ..common.schema import Observation, StepResult, VelocityCommand


@dataclass
class SceneSpec:
    """Known-map description of a scene (used by the global planner and evaluation).

    obstacles: [N, 3] circles (x, y, radius) in world frame [m].
    bounds: (xmin, xmax, ymin, ymax) of the traversable area [m].
    """

    scene_id: str
    family: str
    bounds: tuple[float, float, float, float]
    obstacles: np.ndarray
    start: np.ndarray  # (x, y, yaw)
    goal: np.ndarray  # (x, y)
    params: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["obstacles"] = np.asarray(self.obstacles).tolist()
        d["start"] = np.asarray(self.start).tolist()
        d["goal"] = np.asarray(self.goal).tolist()
        d["bounds"] = list(self.bounds)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SceneSpec":
        return cls(
            scene_id=d["scene_id"],
            family=d["family"],
            bounds=tuple(d["bounds"]),
            obstacles=np.asarray(d["obstacles"], dtype=np.float64).reshape(-1, 3),
            start=np.asarray(d["start"], dtype=np.float64),
            goal=np.asarray(d["goal"], dtype=np.float64),
            params=dict(d.get("params", {})),
        )

    def clearance(self, xy: np.ndarray, robot_radius: float = 0.0) -> np.ndarray:
        """Signed distance from robot footprint to nearest obstacle or boundary. [..., 2] -> [...]."""
        xy = np.asarray(xy, dtype=np.float64)
        xmin, xmax, ymin, ymax = self.bounds
        d = np.minimum.reduce(
            [xy[..., 0] - xmin, xmax - xy[..., 0], xy[..., 1] - ymin, ymax - xy[..., 1]]
        )
        obs = np.asarray(self.obstacles).reshape(-1, 3)
        if len(obs):
            diff = xy[..., None, :] - obs[:, :2]
            dist = np.linalg.norm(diff, axis=-1) - obs[:, 2]
            d = np.minimum(d, dist.min(axis=-1))
        return d - robot_radius


@runtime_checkable
class SimulatorAdapter(Protocol):
    backend: str
    control_dt: float  # nominal high-level control period [s]
    scene: SceneSpec | None

    def reset(self, seed: int, scene_id: str | None = None) -> Observation: ...

    def step(self, action: VelocityCommand) -> StepResult: ...

    def close(self) -> None: ...

    def metadata(self) -> dict[str, Any]: ...


def make_adapter(backend: str, cfg: dict[str, Any]) -> SimulatorAdapter:
    if backend == "mock":
        from .mock import MockGo2Sim

        return MockGo2Sim(cfg)
    if backend == "isaac":
        from .isaac_go2 import IsaacGo2Adapter

        return IsaacGo2Adapter(cfg)
    raise ValueError(f"Unknown backend '{backend}' (expected 'mock' or 'isaac')")
