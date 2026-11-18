"""Transverse step detector for small upheaval faults (below the local-median threshold)."""

import numpy as np

from go2wm.mission.perception import CX, CY, transverse_steps


def _grid(step_x=1.2, lift=0.011, y0=-1.0, y1=0.0, grade=0.0, noise=0.002, seed=0):
    X, Y = np.meshgrid(CX, CY, indexing="ij")
    e = grade * X + np.random.default_rng(seed).normal(0, noise, X.shape)
    e[(X < step_x) & (Y > y0) & (Y < y1)] += lift  # high side nearer the robot, drop going away
    return X, Y, e, np.abs(Y) < 1.1


def test_finds_11mm_fault_with_sign_and_size():
    X, Y, e, ok = _grid()
    st = transverse_steps(e, ok)
    assert len(st) == 1
    assert abs(st[0]["xr"] - 1.2) < 0.06 and st[0]["sign"] < 0 and abs(st[0]["step"] - 0.011) < 0.003


def test_uniform_grade_and_noise_give_nothing():
    for g in (0.0, 0.05, 0.08):
        X, Y, e, ok = _grid(lift=0.0, grade=g, seed=int(g * 100))
        assert transverse_steps(e, ok) == []


def test_fault_on_a_grade_is_still_found():
    X, Y, e, ok = _grid(grade=0.06)
    assert len(transverse_steps(e, ok)) == 1


def test_narrow_compact_dip_is_not_a_step():
    X, Y, e, ok = _grid(lift=0.0)
    e[(np.abs(X - 1.2) < 0.2) & (np.abs(Y) < 0.2)] -= 0.03  # 40 cm hole: below the 0.6 m span
    assert transverse_steps(e, ok) == []


def test_step_max_drops_large_faults_in_detect_candidates():
    from go2wm.mission.perception import detect_candidates

    X, Y, e, ok = _grid(lift=0.06, noise=0.0005)
    pose = np.zeros(3)
    on = [o for o in detect_candidates(e, np.ones_like(e), pose, 0.0, 1.5, step_thresh=0.006) if o.extra["via"] == "step"]
    capped = [o for o in detect_candidates(e, np.ones_like(e), pose, 0.0, 1.5, step_thresh=0.006, step_max=0.02)
              if o.extra["via"] == "step"]
    assert capped == [] and (on == [] or on[0].extra["peak_dev_m"] > 0.02)
    X, Y, e, ok = _grid(lift=0.011, noise=0.0005)
    small = [o for o in detect_candidates(e, np.ones_like(e), pose, 0.0, 1.5, step_thresh=0.006, step_max=0.02)
             if o.extra["via"] == "step"]
    assert len(small) == 1
