"""Canonical observation / command / step types and the episode state vector layout."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .frames import quat_to_yaw

SCHEMA_VERSION = "1.0"

# Canonical recorded state vector (one row per observation). Units: m, rad, m/s, rad/s.
# pos_w / quat_w are world-frame and used for evaluation + feature derivation;
# dynamics models only consume frame-invariant features (see models/features.py).
STATE_FIELDS: dict[str, slice] = {
    "pos_w": slice(0, 3),  # base position, world frame [m]
    "quat_w": slice(3, 7),  # base orientation wxyz, body->world
    "lin_vel_b": slice(7, 10),  # realized base linear velocity, body frame [m/s]
    "ang_vel_b": slice(10, 13),  # realized base angular velocity, body frame [rad/s]
    "gravity_b": slice(13, 16),  # projected gravity unit vector, body frame
}
STATE_DIM = 16

# Command vector layout (body frame). duration_s = intended hold time of the command.
COMMAND_FIELDS = ("vx_m_s", "vy_m_s", "yaw_rate_rad_s", "duration_s")
COMMAND_DIM = 4
ACTION_DIM = 3  # (vx, vy, yaw_rate) as used by models/planners


@dataclass(frozen=True)
class VelocityCommand:
    """High-level body-frame (FLU) velocity command for the locomotion controller."""

    vx_m_s: float
    vy_m_s: float
    yaw_rate_rad_s: float
    duration_s: float = 0.1

    def as_array(self) -> np.ndarray:
        return np.array([self.vx_m_s, self.vy_m_s, self.yaw_rate_rad_s, self.duration_s], dtype=np.float64)

    @property
    def action(self) -> np.ndarray:
        return np.array([self.vx_m_s, self.vy_m_s, self.yaw_rate_rad_s], dtype=np.float64)

    @classmethod
    def from_action(cls, a, duration_s: float) -> "VelocityCommand":
        a = np.asarray(a, dtype=np.float64)
        return cls(float(a[0]), float(a[1]), float(a[2]), float(duration_s))


@dataclass
class Observation:
    timestamp_ns: int
    pos_w: np.ndarray  # (3,)
    quat_w: np.ndarray  # (4,) wxyz
    lin_vel_b: np.ndarray  # (3,)
    ang_vel_b: np.ndarray  # (3,)
    gravity_b: np.ndarray  # (3,)
    joint_pos: np.ndarray | None = None
    joint_vel: np.ndarray | None = None
    rgb: np.ndarray | None = None  # HxWx3 uint8
    depth: np.ndarray | None = None
    # Simulator ground truth. Evaluation only; never a model input.
    ground_truth: dict[str, Any] = field(default_factory=dict)
    valid: dict[str, bool] = field(default_factory=dict)

    def state_vector(self) -> np.ndarray:
        s = np.zeros(STATE_DIM, dtype=np.float64)
        s[STATE_FIELDS["pos_w"]] = self.pos_w
        s[STATE_FIELDS["quat_w"]] = self.quat_w
        s[STATE_FIELDS["lin_vel_b"]] = self.lin_vel_b
        s[STATE_FIELDS["ang_vel_b"]] = self.ang_vel_b
        s[STATE_FIELDS["gravity_b"]] = self.gravity_b
        return s

    @property
    def pose2d(self) -> np.ndarray:
        """(x, y, yaw) in world frame."""
        return np.array([self.pos_w[0], self.pos_w[1], float(quat_to_yaw(self.quat_w))])


@dataclass
class StepResult:
    observation: Observation
    action_timestamp_ns: int  # when the command was applied
    collision: bool = False
    terminated: bool = False
    truncated: bool = False
    termination_reason: str = ""  # "", "collision", "fall", "out_of_bounds", "goal", "timeout"
    info: dict[str, Any] = field(default_factory=dict)


TERMINATION_CODES = {"": 0, "collision": 1, "fall": 2, "out_of_bounds": 3, "goal": 4, "timeout": 5, "error": 6}


def state_pose2d(states: np.ndarray) -> np.ndarray:
    """[..., STATE_DIM] -> [..., 3] (x, y, yaw) world frame."""
    states = np.asarray(states)
    xy = states[..., 0:2]
    yaw = quat_to_yaw(states[..., STATE_FIELDS["quat_w"]])
    return np.concatenate([xy, yaw[..., None]], axis=-1)
