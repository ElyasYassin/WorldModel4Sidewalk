import numpy as np

from go2wm.common.frames import yaw_to_quat
from go2wm.common.schema import Observation
from go2wm.models.state_dynamics import KinematicBaseline
from go2wm.planning.global_waypoints import GlobalPlanner, WaypointTracker
from go2wm.planning.local_mpc import MPCController, ModelPredictor
from go2wm.planning.safety import SafetyFilter
from go2wm.sim.base import SceneSpec

LIMITS = {"vx": [-0.3, 0.9], "vy": [-0.3, 0.3], "yaw_rate": [-1.2, 1.2]}
CFG = {
    "control_dt": 0.1,
    "robot_radius": 0.3,
    "limits": LIMITS,
    "mpc": {"horizon": 10, "samples": 128, "elites": 16, "iterations": 3,
            "weights": {"goal": 1.0, "collision": 30.0, "smooth": 0.05, "heading": 0.3}},
}


def _obs(x=0.0, y=0.0, yaw=0.0):
    return Observation(0, np.array([x, y, 0.3]), yaw_to_quat(yaw), np.zeros(3), np.zeros(3), np.array([0, 0, -1.0]))


def _scene(obstacles=()):
    return SceneSpec("t_0000", "t", (-5, 15, -5, 5), np.array(obstacles, dtype=float).reshape(-1, 3), np.zeros(3), np.array([10.0, 0.0]))


def _ctrl(seed=0):
    safety = SafetyFilter({"limits": LIMITS, "robot_radius": 0.3, "max_accel": [100, 100, 100]})
    return MPCController(ModelPredictor(KinematicBaseline(1, "command")), CFG, safety, seed)


def test_mpc_makes_progress_toward_waypoint():
    c = _ctrl()
    sc = _scene()
    c.reset(sc, _obs())
    a, info = c.act(_obs(), np.array([5.0, 0.0]))
    assert a[0] > 0.5 and abs(a[2]) < 0.4


def test_mpc_turns_toward_waypoint_behind_left():
    c = _ctrl()
    c.reset(_scene(), _obs())
    a, _ = c.act(_obs(), np.array([0.0, 3.0]))
    assert a[2] > 0.3


def test_mpc_respects_bounds_and_avoids_obstacle():
    c = _ctrl()
    sc = _scene([(1.2, 0.0, 0.4)])  # obstacle straight ahead
    c.reset(sc, _obs())
    acts = []
    o = _obs()
    for _ in range(3):
        a, info = c.act(o, np.array([5.0, 0.0]))
        acts.append(a)
        assert all(LIMITS[k][0] - 1e-9 <= a[i] <= LIMITS[k][1] + 1e-9 for i, k in enumerate(("vx", "vy", "yaw_rate")))
        # predicted plan must not intersect the obstacle
        assert sc.clearance(info["pred_poses"][:, :2], 0.3).min() > 0


def test_safety_filter_rate_limits_and_stops():
    sf = SafetyFilter({"limits": LIMITS, "max_accel": [1.0, 1.0, 2.0], "robot_radius": 0.3})
    a, r = sf(np.array([0.9, 0.0, 0.0]), np.zeros(3), 0.1, np.zeros(3), None)
    assert np.isclose(a[0], 0.1) and r == ""
    a, r = sf(np.array([0.5, 0, 0]), np.zeros(3), 0.1, np.zeros(3), None, feasible=False)
    assert np.allclose(a, 0) and r == "no_feasible_candidate"
    a, r = sf(np.array([np.nan, 0, 0]), np.zeros(3), 0.1, np.zeros(3), None)
    assert np.allclose(a, 0) and r == "invalid_action"
    # touching an obstacle and commanding into it -> stop translation
    sc = _scene([(0.62, 0.0, 0.3)])
    a, r = sf(np.array([0.1, 0, 0]), np.array([0.1, 0, 0]), 0.1, np.zeros(3), sc)
    assert r == "clearance_stop" and a[0] == 0


def test_global_planner_routes_around_wall():
    sc = _scene([(5.0, y, 0.3) for y in np.arange(-5, 3.1, 0.4)])  # wall with gap near y=4
    wps = GlobalPlanner(inflation=0.45).plan(sc, np.zeros(2), sc.goal)
    assert np.allclose(wps[-1], sc.goal, atol=0.1)
    assert sc.clearance(wps, 0.3).min() > 0
    assert wps[:, 1].max() > 3.0  # had to go through the gap
    t = WaypointTracker(wps, 0.6)
    assert np.allclose(t.update(np.zeros(2)), wps[0])
