"""Curb-aware perception: curb found in the depth frame itself, and the pose correction it drives."""

import numpy as np

from go2wm.mission.perception import CX, CY, curb_lateral, curb_mask
from go2wm.mission.runner import PoseErrors


def _street(curb_y=-0.6, slope=0.0, drop=0.15):
    """Robot-frame elevation grid: flat walkway, road `drop` lower beyond y < curb_y + slope * x."""
    X, Y = np.meshgrid(CX, CY, indexing="ij")
    e = np.zeros_like(X)
    e[Y < curb_y + slope * X] = -drop
    return X, Y, e


def test_curb_lateral_finds_straight_curb():
    _, _, e = _street(-0.6)
    y = curb_lateral(e)
    assert y is not None and abs(y + 0.6) < 0.05
    off, ang = curb_lateral(e, with_angle=True)
    assert abs(off + 0.6) < 0.05 and abs(ang) < 0.02


def test_curb_lateral_measures_angle_of_skewed_curb():
    off, ang = curb_lateral(_street(-0.7, slope=0.08)[2], with_angle=True)
    assert abs(ang - np.arctan(0.08)) < 0.02 and abs(off + 0.7) < 0.06


def test_curb_lateral_none_without_curb_or_with_small_step():
    assert curb_lateral(np.zeros((len(CX), len(CY)))) is None
    assert curb_lateral(_street(-0.6, drop=0.05)[2]) is None  # below the 8 cm drop


def test_curb_mask_drops_curb_strip_but_keeps_a_deep_hole():
    X, Y, e = _street(-0.8)
    hole = (np.abs(X - 1.5) < 0.15) & (np.abs(Y - 0.2) < 0.15)
    e[hole] = -0.11
    walk = Y > -0.8
    m = curb_mask(e, X, walk)
    assert m[(Y < -0.85) & (X > 0) & (X < 3)].all()  # road strip masked
    assert m[(Y > -0.75) & (Y < -0.65)].any()  # dilated onto the walkway edge (mixed pixels)
    assert not m[hole].any()  # compact, so kept


def test_curb_mask_empty_without_curb():
    X, Y, e = _street(-5.0)
    assert not curb_mask(e, X, np.ones_like(e, bool)).any()


def _drifted():
    pe = PoseErrors({"odom_xy": 0.05, "odom_yaw": 0.02}, seed=3)
    pe.origin = np.zeros(2)
    for x in np.linspace(0, 6, 61):
        pe.estimate(np.array([x, 0.0, 0.0]))
    return pe


def test_pose_correct_lateral_shifts_estimate_by_dy():
    pe = _drifted()
    p = pe.last.copy()
    before = pe.estimate_static(p)
    pe.correct(0.07, 0.0)
    after = pe.estimate_static(p)
    assert np.allclose(after - before, [0.0, 0.07, 0.0], atol=1e-9)


def test_pose_correct_heading_keeps_position():
    pe = _drifted()
    p = pe.last.copy()
    before = pe.estimate_static(p)
    pe.correct(0.0, 0.03)
    after = pe.estimate_static(p)
    assert np.allclose(after[:2], before[:2], atol=1e-9) and np.isclose(after[2] - before[2], -0.03)


def test_pose_correct_noop_before_first_estimate():
    pe = PoseErrors({}, seed=0)
    pe.correct(0.1, 0.1)
    assert np.allclose(pe.err, 0)
