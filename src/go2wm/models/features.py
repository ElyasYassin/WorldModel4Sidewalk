"""Frame-invariant model features and delta targets derived from canonical states.

No absolute world position/heading enters the model: inputs are body-frame
velocities + projected gravity; targets are the relative SE(2) motion between
consecutive observations (expressed in the earlier body frame) plus feature deltas.
Heading change comes from quaternions via sin/cos (no wrapped-yaw subtraction).
"""

from __future__ import annotations

import numpy as np

from ..common.frames import angle_diff, quat_to_yaw, world_to_body_2d
from ..common.schema import STATE_FIELDS

FEATURE_NAMES = ["vx_b", "vy_b", "vz_b", "wx_b", "wy_b", "wz_b", "gx_b", "gy_b", "gz_b"]
FEATURE_DIM = len(FEATURE_NAMES)
MOTION_NAMES = ["dx_b", "dy_b", "dz", "dyaw"]
TARGET_NAMES = MOTION_NAMES + [f"d_{n}" for n in FEATURE_NAMES]
TARGET_DIM = len(TARGET_NAMES)


def state_features(states: np.ndarray) -> np.ndarray:
    s = np.asarray(states)
    return np.concatenate(
        [s[..., STATE_FIELDS["lin_vel_b"]], s[..., STATE_FIELDS["ang_vel_b"]], s[..., STATE_FIELDS["gravity_b"]]],
        axis=-1,
    )


def relative_motion(states_t: np.ndarray, states_t1: np.ndarray) -> np.ndarray:
    """(dx_b, dy_b, dz, dyaw) of t1 relative to t, planar part in body frame of t."""
    yaw0 = quat_to_yaw(states_t[..., STATE_FIELDS["quat_w"]])
    yaw1 = quat_to_yaw(states_t1[..., STATE_FIELDS["quat_w"]])
    dxy = world_to_body_2d(states_t1[..., 0:2] - states_t[..., 0:2], yaw0)
    dz = states_t1[..., 2] - states_t[..., 2]
    dyaw = angle_diff(yaw1, yaw0)
    return np.concatenate([dxy, dz[..., None], dyaw[..., None]], axis=-1)


def transition_targets(states_t: np.ndarray, states_t1: np.ndarray) -> np.ndarray:
    return np.concatenate(
        [relative_motion(states_t, states_t1), state_features(states_t1) - state_features(states_t)], axis=-1
    )
