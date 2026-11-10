"""Parametric sidewalk terrain (pure numpy part) and terrain ground-truth recording."""

import numpy as np

from go2wm.common.schema import VelocityCommand
from go2wm.data.episodes import ARRAY_SPEC
from go2wm.data.recorder import EpisodeRecorder
from go2wm.sim.mock import MockGo2Sim
from go2wm.sim.terrain import params_with_defaults, sidewalk_height, sidewalk_heightfield

S1 = {"cross_slope": 0.04, "slabs": [[4.0, 1.5, 0.08]], "grade": [[7.0, 10.0, 0.06]], "curb_drop": 0.15}


def test_flat_by_default():
    assert np.all(sidewalk_height(np.linspace(-5, 13, 50), np.linspace(-3, 3, 50), {}) == 0)


def test_features():
    h = lambda x, y: float(sidewalk_height(x, y, S1))  # noqa: E731
    assert h(0, 0) == 0.0  # start at zero height
    assert np.isclose(h(2, 1.0) - h(2, -1.0), 0.08)  # 4 % across 2 m
    assert np.isclose(h(4.5, 0) - h(3.9, 0), 0.08)  # slab step up
    assert np.isclose(h(5.6, 0), 0.0)  # and back down
    assert np.isclose(h(8.5, 0), 0.06 * 1.5)  # halfway up the grade
    assert np.isclose(h(12, 0), 0.18)  # stays up after the grade
    assert np.isclose(h(2, -1.6) - h(2, -1.5), -0.15)  # curb drop to the road
    assert np.isclose(h(2, 3.0), h(2, 1.5))  # cross-slope clipped outside the walkway


def test_heightfield_grid_alignment():
    p = params_with_defaults(S1)
    hs, vs = 0.05, 0.005
    size = (28.0 - 2 * hs, 8.0 - 2 * hs)  # what Isaac passes (tile minus a one-pixel border)
    hf = sidewalk_heightfield(size, hs, vs, p)
    assert hf.shape == (int(round(size[0] / hs)), int(round(size[1] / hs))) and hf.dtype == np.int16
    # pixel i sits at scene x = i*hs - size/2: check the slab edge at x = 4.0 lands within a pixel
    xs = np.arange(hf.shape[0]) * hs - size[0] / 2
    j0 = hf.shape[1] // 2
    col = hf[:, j0] * vs
    edge = xs[np.argmax(np.diff(col) > 0.05) + 1]
    assert abs(edge - 4.0) <= hs + 1e-9


def test_recorder_stores_declared_terrain_truth_only():
    sim = MockGo2Sim()
    o = sim.reset(seed=1)
    extra = {"height_above_ground_m": 0.3, "ground_z_m": 0.0, "foot_contact": [1, 1, 0, 1], "foot_slip_m_s": [0, 0, 0.1, 0]}
    o.ground_truth.update(extra)
    rec = EpisodeRecorder({"episode_id": "t"}, o)
    for _ in range(5):
        r = sim.step(VelocityCommand(0.4, 0.0, 0.0, 0.1))
        r.observation.ground_truth.update(extra)
        rec.add(VelocityCommand(0.4, 0.0, 0.0, 0.1), r)
    ep = rec.finish()
    assert ep.arrays["gt_foot_contact"].shape == (6, 4)
    assert ep.arrays["gt_height_above_ground_m"].shape == (6,)
    assert "gt_pose2d" not in ep.arrays  # mock's pose2d is not a declared array
    assert all(k in ARRAY_SPEC for k in ep.arrays)
