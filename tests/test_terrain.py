"""Parametric sidewalk terrain (pure numpy part) and terrain ground-truth recording."""

import numpy as np
import pytest

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


def test_sweep_kinematic_error_and_feature_mask():
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location("ats", Path(__file__).parents[1] / "scripts" / "analyze_terrain_sweep.py")
    ats = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ats)
    # robot that follows a constant arc command exactly -> zero kinematic error
    dt, T = 0.1, 30
    cmds = np.tile([0.6, 0.0, 0.3, dt], (T, 1))
    pose = np.zeros((T + 1, 3))
    for t in range(T):
        x, y, th = pose[t]
        pose[t + 1] = [x + 0.6 * np.cos(th) * dt, y + 0.6 * np.sin(th) * dt, th + 0.3 * dt]
    assert np.allclose(ats.kin_errors(pose, cmds, dt), 0.0, atol=1e-9)
    # robot that stands still while commanded 0.5 m/s -> 0.5 m error after 1 s
    still = np.zeros((T + 1, 3))
    assert np.allclose(ats.kin_errors(still, np.tile([0.5, 0, 0, dt], (T, 1)), dt), 0.5)
    # slab feature mask covers the slab +-0.5 m only; curb mask the walkway edge
    xy = np.array([[2.0, 0.0], [3.6, 0.0], [5.0, 0.0], [6.1, 0.0], [3.0, -1.4]])
    assert ats.feature_mask(xy, {"slabs": [[4.0, 1.5, 0.1]]}).tolist() == [False, True, True, False, False]
    assert ats.feature_mask(xy, {"curb_drop": 0.15, "width": 3.0}).tolist() == [False, False, False, False, True]


def test_terrain_map_and_patch():
    import torch

    from go2wm.models.terrain_map import MAP_X, MAP_Y, PATCH_DIM, sample_patch, start_maps
    from go2wm.sim.terrain import sidewalk_height

    p = {"slabs": [[1.0, 1.0, 0.1]]}
    hf = lambda x, y: sidewalk_height(x, y, p)  # noqa: E731
    pose = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, np.pi]])
    m = start_maps(pose, hf)
    assert m.shape == (2, len(MAP_Y), len(MAP_X))
    iy = int(np.argmin(np.abs(MAP_Y)))
    ix = lambda x: int(np.argmin(np.abs(MAP_X - x)))  # noqa: E731
    assert m[0, iy, ix(1.5)] == pytest.approx(0.1) and m[0, iy, ix(0.5)] == pytest.approx(0.0)
    assert m[1, iy, ix(1.5)] == pytest.approx(0.0)  # facing -x: the slab is behind
    assert np.all(start_maps(pose, None) == 0)
    # patch at the start pose (relative pose 0): relief 1 m ahead of a robot standing at x = 0.5 on the start map
    z = sample_patch(torch.as_tensor(m[:1]), torch.tensor([[0.6, 0.0, 0.0]]))
    assert z.shape == (1, PATCH_DIM)
    # local x = +0.9 at robot x = 0.6 -> world 1.5 (on the slab): relative height +0.1 m = +1.0 in 10 cm units
    assert float(z[0, 4 * 3 + 1]) == pytest.approx(1.0, abs=1e-4)


@pytest.mark.parametrize("kind", ["terrain_mlp", "terrain_gru"])
def test_terrain_mlp_rollout_shapes(kind):
    import torch

    from go2wm.models.state_dynamics import MODEL_TYPES
    from go2wm.models.terrain_map import MAP_X, MAP_Y

    m = MODEL_TYPES[kind](history=4, hidden=32, layers=2 if kind == "terrain_mlp" else 1)
    B, H = 3, 5
    out = m.rollout(torch.zeros(B, 4, 9), torch.zeros(B, 4, 3), torch.full((B, 4), 0.1), torch.zeros(B, H, 3),
                    torch.full((B, H), 0.1), terrain_map=torch.zeros(B, len(MAP_Y), len(MAP_X)))
    assert out["poses"].shape == (B, H, 3)
    assert m.rollout(torch.zeros(B, 4, 9), torch.zeros(B, 4, 3), torch.full((B, 4), 0.1), torch.zeros(B, H, 3),
                     torch.full((B, H), 0.1))["poses"].shape == (B, H, 3)


def test_commanded_path_straight_and_turn():
    import numpy as np

    from go2wm.eval.prediction import commanded_path

    p = commanded_path(np.array([1.0, 2.0, 0.0]), np.array([[0.5, 0.0, 0.0]] * 10), np.full(10, 0.1))
    assert p.shape == (11, 2) and np.allclose(p[-1], [1.5, 2.0])
    q = commanded_path(np.array([0.0, 0.0, np.pi / 2]), np.array([[1.0, 0.0, 0.0]]), np.array([1.0]))
    assert np.allclose(q[-1], [0.0, 1.0], atol=1e-9)


def test_scan_maps_match_analytic_inside_scan_and_clamp_outside():
    import numpy as np

    from go2wm.models.terrain_map import MAP_X, MAP_Y, scan_maps, start_maps
    from go2wm.sim.terrain import sidewalk_height

    p = {"width": 3.0, "cross_slope": 0.0, "grade": [], "slabs": [[0.5, 1.0, 0.1]], "curb_drop": 0.0}
    xs, ys = np.round(np.arange(-0.8, 0.8001, 0.1), 3), np.round(np.arange(-0.5, 0.5001, 0.1), 3)
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    pat = np.stack([gx.ravel(), gy.ravel()], 1)
    pose = np.array([[0.0, 0.0, 0.0]])
    scan = sidewalk_height(pat[:, 0], pat[:, 1], p)[None] + 0.37  # arbitrary scene offset
    m = scan_maps(scan, pat)[0]
    ref = start_maps(pose, lambda x, y: sidewalk_height(x, y, p))[0]
    # on scan grid points (0.1 m) inside the scanned area the two maps agree exactly
    inside = (np.abs(MAP_X)[None, :] <= 0.8) & (np.abs(MAP_Y)[:, None] <= 0.5)
    assert np.allclose(m[inside], ref[inside], atol=1e-6)
    # beyond the 0.8 m scan range the map holds the edge value (the slab ends at 1.5 m, unseen)
    edge = int(np.argmin(np.abs(MAP_X - 0.8)))
    assert np.allclose(m[:, MAP_X >= 0.8], m[:, [edge]], atol=1e-6)
    assert not np.allclose(ref[:, MAP_X >= 0.8], ref[:, [edge]])


def test_sidewalk_albedo_regions_and_determinism():
    from go2wm.sim.sidewalk_texture import ASPHALT, CONCRETE, GRASS, sidewalk_albedo

    p = {"width": 3.0, "curb_drop": 0.15, "slabs": [[4.0, 1.0, 0.1]], "grade": []}
    x = np.linspace(0.2, 10.0, 400)
    walk = sidewalk_albedo(x, np.full_like(x, 0.7), p, seed=1, cracks=False)
    grass = sidewalk_albedo(x, np.full_like(x, 3.0), p, seed=1)
    road = sidewalk_albedo(x, np.full_like(x, -3.5), p, seed=1)
    assert walk.shape == (400, 3) and np.all((walk >= 0) & (walk <= 1))
    # region colours are near their base colour on average (noise, tint and joints aside)
    assert np.abs(np.median(walk, 0) - CONCRETE).max() < 0.08
    assert np.abs(np.median(road, 0) - ASPHALT).max() < 0.06
    assert np.median(grass[:, 1]) > np.median(grass[:, 0])  # green dominates
    np.testing.assert_array_equal(walk, sidewalk_albedo(x, np.full_like(x, 0.7), p, seed=1, cracks=False))
    assert not np.array_equal(walk, sidewalk_albedo(x, np.full_like(x, 0.7), p, seed=2, cracks=False))
    # no curb: the road side is grass too
    p2 = {**p, "curb_drop": 0.0}
    side = sidewalk_albedo(x, np.full_like(x, -3.5), p2, seed=1)
    assert np.median(side[:, 1]) > np.median(side[:, 2]) and np.abs(np.median(side, 0) - GRASS).max() < 0.12


def test_bake_albedo_assigns_nearest_tile():
    from go2wm.sim.isaac_texture import bake_albedo

    tiles = [{"width": 3.0, "curb_drop": 0.15}, {"width": 3.0, "curb_drop": 0.0}]
    origins = np.array([[0.0, 0.0], [0.0, 8.0]])
    img = bake_albedo(tiles, origins, (-2.0, 2.0), (-4.0, 12.0), res=0.1)
    assert img.shape == (160, 40, 3) and img.dtype == np.uint8
    row = lambda y: img[int((y + 4.0) / 0.1)]  # noqa: E731
    assert row(-3.5)[:, 2].mean() > row(-3.5)[:, 1].mean() - 5  # tile 0 road side: grey asphalt
    assert row(4.5)[:, 1].mean() > row(4.5)[:, 2].mean() + 30  # tile 1 road side (y = -3.5 local): grass


def test_generator_tile_origins_match_isaac_layout():
    from go2wm.sim.isaac_texture import generator_tile_origins_xy

    # observed in Isaac Lab 2.3 (terrain_origins) for the 6 E4 test tiles of 28 x 8 m
    want = [[0.0, -20.0], [0.0, -12.0], [0.0, -4.0], [0.0, 4.0], [0.0, 12.0], [0.0, 20.0]]
    np.testing.assert_allclose(generator_tile_origins_xy(6, (28.0, 8.0)), want)


def test_sensor_maps_resample_fill_and_reference():
    from go2wm.models.terrain_map import MAP_X, MAP_Y, sensor_maps
    from go2wm.sim.sensors import GRID_SHAPE, grid_centres

    cx, cy = grid_centres()
    X, _ = np.meshgrid(cx, cy, indexing="ij")
    elev = np.where(X > 0.5, 0.10, 0.02).astype(np.float16)[None]  # a 8 cm step at x = 0.5 m, ground at 2 cm
    elev[0, X < 0.3] = np.nan  # unobserved under/behind the robot
    m = sensor_maps(elev, np.array([0.02]))
    assert m.shape == (1, len(MAP_Y), len(MAP_X)) and np.isfinite(m).all()
    ix = lambda x: int(np.argmin(np.abs(MAP_X - x)))  # noqa: E731
    assert abs(m[0, 12, ix(1.0)] - 0.08) < 1e-3 and abs(m[0, 12, ix(0.4)]) < 1e-3
    assert abs(m[0, 12, ix(-0.5)]) < 1e-3  # filled from the nearest observed (flat) ground
    assert np.all(sensor_maps(np.full((1, *GRID_SHAPE), np.nan, np.float16), np.zeros(1)) == 0)
