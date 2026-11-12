import numpy as np

from go2wm.sim.offline_sensors import OfflineRig, bin_base_points, cast, odometry_estimate
from go2wm.sim.sensors import grid_centres

FLAT = lambda x, y: np.zeros(np.shape(x))  # noqa: E731
STEP = lambda x, y: np.where(np.asarray(x) > 1.5, 0.10, 0.0)  # noqa: E731


def test_cast_flat_and_step():
    starts = np.array([[0.0, 0.0, 0.5], [0.0, 0.0, 0.5], [0.0, 0.0, 0.5]])
    d = np.array([[1.0, 0.0, -0.5], [0.0, 0.0, -1.0], [1.0, 0.0, 0.2]])
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    r = cast(starts, d, FLAT, r_max=5.0)
    assert abs(r[0] - np.hypot(1.0, 0.5)) < 1e-3 and abs(r[1] - 0.5) < 1e-3 and np.isinf(r[2])
    # a ray that would land beyond x = 1.5 on the floor hits the 10 cm step face/top instead
    d2 = np.array([[1.0, 0.0, -0.25]]) / np.hypot(1.0, 0.25)
    p = starts[:1] + d2 * cast(starts[:1], d2, STEP, 5.0)[:, None]
    assert abs(p[0, 2] - 0.10) < 1e-3 and 1.5 <= p[0, 0] <= 1.6 + 1e-3


def _grid(rig, sensor, height=FLAT):
    R = np.eye(3)
    e, c = bin_base_points(rig.frame(sensor, np.array([0.0, 0.0, 0.33]), R, height), R)
    return e, c


def test_ideal_rig_reproduces_floor_and_calibration_error_tilts_it():
    cx, _ = grid_centres()
    e, c = _grid(OfflineRig(["depth_d555"], "ideal", seed=0), "depth_d555")
    ok = np.isfinite(e)
    assert ok.mean() > 0.1 and abs(np.median(e[ok]) + 0.33) < 0.003  # floor 33 cm below the base
    # a 1 deg pitch error the robot does not know about: the floor appears tilted by ~tan(1 deg) per metre
    rig = OfflineRig(["depth_d555"], {"calib_pitch_deg": 1.0}, seed=0)
    rig.session["depth_d555"]["pitch_err"] = 1.0
    e2, _ = _grid(rig, "depth_d555")
    rows = [i for i in range(len(cx)) if np.isfinite(e2[i]).sum() > 5]
    xs = np.array([cx[i] for i in rows])
    zs = np.array([np.nanmedian(e2[i]) for i in rows])
    slope = np.polyfit(xs, zs, 1)[0]
    assert 0.6 * np.tan(np.radians(1.0)) < abs(slope) < 1.6 * np.tan(np.radians(1.0))


def test_odometry_estimate_ideal_is_exact_and_realistic_drifts():
    pose = np.stack([np.linspace(0, 10, 101), np.zeros(101), np.zeros(101)], 1)
    z = np.full(101, 0.33)
    est, ez = odometry_estimate(pose, z, {}, np.random.default_rng(0))
    assert np.allclose(est, pose) and np.allclose(ez, z)
    est, ez = odometry_estimate(pose, z, {"odom_xy": 0.015, "odom_yaw": 0.01, "odom_z": 0.003}, np.random.default_rng(0))
    assert 0 < np.abs(est[-1, :2] - pose[-1, :2]).max() < 0.5 and np.abs(ez - z).max() < 0.1


def test_flying_pixels_only_at_real_discontinuities():
    # oblique flat ground has large but smooth depth differences between neighbouring rays: no artefacts there
    ideal, _ = _grid(OfflineRig(["depth_d555"], "ideal", seed=0), "depth_d555")
    mixed, _ = _grid(OfflineRig(["depth_d555"], {"mixed_pixels": 1.0, "mixed_thresh": 0.03}, seed=0), "depth_d555")
    e_ideal = np.abs(ideal[np.isfinite(ideal)] + 0.33)  # error vs the true floor (the two rigs draw different noise)
    e_mixed = np.abs(mixed[np.isfinite(mixed)] + 0.33)
    assert np.percentile(e_mixed, 90) < np.percentile(e_ideal, 90) + 0.001 and e_mixed.max() < 0.012
