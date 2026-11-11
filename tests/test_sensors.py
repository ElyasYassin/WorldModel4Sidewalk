import numpy as np

from go2wm.data.episodes import RANGE_SENSORS
from go2wm.sim.sensors import (GRID_SHAPE, SENSOR_SPECS, base_frame_rays, bin_elevation, corrupt_hits, grid_centres,
                               n_rays, pinhole_directions, pitch_matrix, sample_directions)


def test_specs_match_episode_arrays():
    assert set(SENSOR_SPECS) == set(RANGE_SENSORS)


def test_pitch_and_pinhole_geometry():
    assert np.allclose(pitch_matrix(90) @ [1, 0, 0], [0, 0, -1], atol=1e-12)  # positive pitch looks down
    d = pinhole_directions(90.0, 60.0, 4, 2)
    assert d.shape == (8, 3) and np.allclose(np.linalg.norm(d, axis=1), 1)
    assert d[0, 1] > 0 and d[0, 2] > 0  # top-left pixel looks left and up
    assert np.all(d[:, 0] > 0)


def test_lidar_directions_cover_their_fov():
    rng = np.random.default_rng(0)
    band = sample_directions(SENSOR_SPECS["mid360"], rng)
    el = np.degrees(np.arcsin(band[:, 2]))
    assert band.shape == (20000, 3) and el.min() >= -7.001 and el.max() <= 52.001
    dome = sample_directions(SENSOR_SPECS["l1"], rng)
    assert np.all(dome[:, 0] >= 0) and len(dome) == 2160
    assert not np.allclose(dome, sample_directions(SENSOR_SPECS["l1"], rng))  # non-repetitive
    s, d = base_frame_rays(SENSOR_SPECS["depth_d435"], rng)
    assert len(d) == n_rays(SENSOR_SPECS["depth_d435"]) and np.allclose(s[0], SENSOR_SPECS["depth_d435"]["mount_pos"])
    assert np.mean(d[:, 2] < 0) > 0.9  # head depth camera pitched 30 deg down sees mostly ground


def test_corrupt_hits_limits_dropout_noise():
    rng = np.random.default_rng(1)
    n = 5000
    starts = np.zeros((n, 3))
    dirs = np.tile([1.0, 0, 0], (n, 1))
    r = np.linspace(0.0, 6.0, n)
    hits = dirs * r[:, None]
    hits[::10] = np.inf  # misses
    spec = SENSOR_SPECS["depth_d435"]
    out = corrupt_hits(starts, dirs, hits, spec, rng, optical_axis_w=np.array([1.0, 0, 0]))
    ok = np.isfinite(out[:, 0])
    assert not ok[::10].any() and not ok[r < 0.2].any() and not ok[r > 4.0].any()
    err = out[ok, 0] - r[ok]
    far = r[ok] > 3.0
    assert np.std(err[far]) > 3 * np.std(err[r[ok] < 1.0])  # depth noise grows with distance


def test_bin_elevation_robot_frame():
    xs, ys = grid_centres()
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    yaw = 0.7
    c, s = np.cos(yaw), np.sin(yaw)
    pts_r = np.stack([X.ravel(), Y.ravel(), 0.1 * X.ravel()], 1)  # a ramp along robot x
    pts_w = np.stack([5 + c * pts_r[:, 0] - s * pts_r[:, 1], 2 + s * pts_r[:, 0] + c * pts_r[:, 1], pts_r[:, 2] + 1.0], 1)
    pts_w = np.vstack([pts_w, [[np.nan, 0, 0], [100, 100, 0]]])  # invalid and out-of-grid points are ignored
    elev, cnt = bin_elevation(pts_w, np.array([5.0, 2.0]), yaw, z_ref=1.0)
    assert elev.shape == GRID_SHAPE and elev.dtype == np.float16 and cnt.dtype == np.uint8
    assert np.all(cnt == 1)
    assert np.allclose(elev.astype(float), 0.1 * X, atol=2e-3)
