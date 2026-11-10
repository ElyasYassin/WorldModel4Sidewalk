"""Frame conventions and transforms.

Conventions (recorded in every episode's metadata as FRAME_CONVENTION):
  * World frame: right-handed, z up (ENU-like; x/y are simulator world axes).
  * Body frame: FLU (x forward, y left, z up), origin at the robot base.
  * Quaternions are (w, x, y, z), body -> world rotation.
  * Heading (yaw) differences are always computed via atan2 of relative
    rotations / sin-cos, never by subtracting wrapped angles.
"""

from __future__ import annotations

import numpy as np

FRAME_CONVENTION = {
    "world": "right-handed, z-up (ENU-like simulator world axes)",
    "body": "FLU: x forward, y left, z up",
    "quaternion": "wxyz, rotates body vectors into world frame",
    "velocity_command": "body frame (vx forward m/s, vy left m/s, yaw_rate rad/s CCW)",
    "lin_vel_b": "realized base linear velocity expressed in body frame",
    "ang_vel_b": "realized base angular velocity expressed in body frame",
    "gravity_b": "unit gravity direction expressed in body frame ((0,0,-1) when level)",
}


def wrap_angle(a):
    """Wrap angle(s) to [-pi, pi)."""
    a = np.asarray(a, dtype=np.float64)
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def angle_diff(a, b):
    """Shortest signed difference a - b, robust to wraparound (via sin/cos)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return np.arctan2(np.sin(a - b), np.cos(a - b))


def quat_normalize(q):
    q = np.asarray(q, dtype=np.float64)
    n = np.linalg.norm(q, axis=-1, keepdims=True)
    if np.any(n < 1e-9):
        raise ValueError("Zero-norm quaternion")
    return q / n


def quat_to_yaw(q):
    """Yaw (rotation about world z) of a wxyz quaternion, [..., 4] -> [...]."""
    q = quat_normalize(q)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def yaw_to_quat(yaw):
    """Pure-yaw wxyz quaternion(s)."""
    yaw = np.asarray(yaw, dtype=np.float64)
    out = np.zeros(yaw.shape + (4,))
    out[..., 0] = np.cos(yaw / 2.0)
    out[..., 3] = np.sin(yaw / 2.0)
    return out


def quat_rotate_inverse(q, v):
    """Rotate world vector(s) v into the body frame of wxyz quaternion(s) q."""
    q = quat_normalize(q)
    v = np.asarray(v, dtype=np.float64)
    w = q[..., :1]
    u = -q[..., 1:]  # conjugate
    t = 2.0 * np.cross(u, v)
    return v + w * t + np.cross(u, t)


def world_to_body_2d(dxy_world, yaw):
    """Express world-frame planar displacement(s) [..., 2] in a body frame with heading yaw."""
    dxy_world = np.asarray(dxy_world, dtype=np.float64)
    c, s = np.cos(yaw), np.sin(yaw)
    x = c * dxy_world[..., 0] + s * dxy_world[..., 1]
    y = -s * dxy_world[..., 0] + c * dxy_world[..., 1]
    return np.stack([x, y], axis=-1)


def body_to_world_2d(dxy_body, yaw):
    dxy_body = np.asarray(dxy_body, dtype=np.float64)
    c, s = np.cos(yaw), np.sin(yaw)
    x = c * dxy_body[..., 0] - s * dxy_body[..., 1]
    y = s * dxy_body[..., 0] + c * dxy_body[..., 1]
    return np.stack([x, y], axis=-1)


def relative_se2(pose_a, pose_b):
    """Pose of b expressed in frame a. Poses are [..., 3] = (x, y, yaw)."""
    pose_a = np.asarray(pose_a, dtype=np.float64)
    pose_b = np.asarray(pose_b, dtype=np.float64)
    dxy = world_to_body_2d(pose_b[..., :2] - pose_a[..., :2], pose_a[..., 2])
    dyaw = angle_diff(pose_b[..., 2], pose_a[..., 2])
    return np.concatenate([dxy, dyaw[..., None]], axis=-1)


def compose_se2(pose_a, delta_b):
    """Apply body-frame delta (dx, dy, dyaw) to pose a. [..., 3] each."""
    pose_a = np.asarray(pose_a, dtype=np.float64)
    delta_b = np.asarray(delta_b, dtype=np.float64)
    xy = pose_a[..., :2] + body_to_world_2d(delta_b[..., :2], pose_a[..., 2])
    yaw = wrap_angle(pose_a[..., 2] + delta_b[..., 2])
    return np.concatenate([xy, yaw[..., None]], axis=-1)
