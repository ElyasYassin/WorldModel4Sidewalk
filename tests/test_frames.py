import numpy as np
import pytest

from go2wm.common import frames as F


def test_wrap_and_diff_across_pi():
    assert np.isclose(F.angle_diff(np.pi - 0.1, -np.pi + 0.1), -0.2)
    assert np.isclose(F.angle_diff(-np.pi + 0.1, np.pi - 0.1), 0.2)
    assert -np.pi <= F.wrap_angle(3 * np.pi) < np.pi


def test_quat_yaw_roundtrip_and_normalization():
    for yaw in np.linspace(-np.pi + 1e-3, np.pi - 1e-3, 17):
        q = F.yaw_to_quat(yaw) * 3.0  # non-unit input gets normalized
        assert np.isclose(F.angle_diff(F.quat_to_yaw(q), yaw), 0.0, atol=1e-9)
    with pytest.raises(ValueError):
        F.quat_normalize(np.zeros(4))


def test_body_world_inverse():
    rng = np.random.default_rng(0)
    v = rng.normal(size=(10, 2))
    yaw = rng.uniform(-np.pi, np.pi, 10)
    assert np.allclose(F.body_to_world_2d(F.world_to_body_2d(v, yaw), yaw), v)


def test_se2_compose_relative_inverse():
    a = np.array([1.0, -2.0, 3.0])  # near +pi
    b = np.array([0.5, 1.0, -3.0])  # near -pi: wraparound
    rel = F.relative_se2(a, b)
    assert abs(rel[2]) < 0.5  # small turn across the wrap, not ~6 rad
    assert np.allclose(F.compose_se2(a, rel)[:2], b[:2])
    assert np.isclose(F.angle_diff(F.compose_se2(a, rel)[2], b[2]), 0)


def test_quat_rotate_inverse_yaw():
    q = F.yaw_to_quat(np.pi / 2)
    assert np.allclose(F.quat_rotate_inverse(q, np.array([0.0, 1.0, 0.0])), [1.0, 0.0, 0.0])
