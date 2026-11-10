"""P0 integrity tooling: held-out command patterns, all-test manifests, split audit, final-test guard,
reactive-baseline tuning knobs."""

import json
import sys
from pathlib import Path

import numpy as np

from go2wm.data.dataset import build_windows, load_split, make_all_test_manifest, make_split_manifest, save_manifest
from go2wm.data.episodes import load_all
from go2wm.data.recorder import HeldOutCommandPatterns, collect_episodes
from go2wm.sim.mock import MockGo2Sim

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))


def _cmds(pattern, n=200, seed=0):
    g = HeldOutCommandPatterns(seed, 0.1, pattern)
    return np.array([g(None, None).as_array()[:3] for _ in range(n)])


def test_heldout_patterns_differ_from_exploration():
    sine = _cmds("sine")
    assert len(np.unique(np.round(sine[:, 0], 6))) > 100  # continuous, never piecewise constant
    fast = _cmds("fast_switch")
    runs = np.flatnonzero(np.any(np.diff(fast, axis=0) != 0, axis=1))
    assert np.max(np.diff(runs)) <= 3  # held at most 0.3 s (training segments >= 0.5 s)
    lat = _cmds("lateral_ood")
    assert np.all(np.abs(lat[:, 1]) >= 0.45) and np.all(np.abs(lat[:, 1]) <= 0.7)  # outside training |vy|<=0.4
    for p in HeldOutCommandPatterns.PATTERNS:
        assert np.all(np.abs(_cmds(p)) <= 1.0)  # inside the policy's trained ranges


def test_pattern_meta_and_all_test_manifest(tmp_path):
    collect_episodes(MockGo2Sim(), tmp_path, num_episodes=4, base_seed=40000, max_steps=60,
                     limits={"vx": [-1, 1], "vy": [-1, 1], "yaw_rate": [-1, 1]}, goal_tolerance_m=-1.0,
                     scene_ids=[f"open_{40000 + i:05d}" for i in range(4)],
                     command_fn_factory=lambda s: HeldOutCommandPatterns(s, 0.1),
                     extra_meta={"data_role": "cmdpattern_test"}, log=lambda s: None)
    eps = load_all(tmp_path)
    assert sorted(e.meta["command_pattern"] for e in eps) == sorted(HeldOutCommandPatterns.PATTERNS)
    assert all(e.meta["data_role"] == "cmdpattern_test" and e.meta["collector"]["type"] == "HeldOutCommandPatterns"
               for e in eps)
    m = make_all_test_manifest(tmp_path)
    save_manifest(tmp_path, m, "all_test")
    assert len(load_split(tmp_path, "test", "all_test")) == 4 and m["episodes"]["train"] == []


def test_windows_never_cross_episodes(mock_dataset):
    eps = load_all(mock_dataset)
    w = build_windows(eps, 4, 10)
    lens = np.array([e.num_transitions for e in eps])
    assert np.all(w.start_index - 4 >= 0)
    assert np.all(w.start_index + 10 <= lens[w.episode_index])


def test_audit_clean_and_detects_leak(mock_dataset, tmp_path, monkeypatch):
    import audit_splits

    save_manifest(mock_dataset, make_split_manifest(mock_dataset, seed=0), "audit_ok")
    m = make_split_manifest(mock_dataset, seed=0)
    m["episodes"]["test"].append(m["episodes"]["train"][0])  # leak one training episode into test
    save_manifest(mock_dataset, m, "audit_leak")
    try:
        out = tmp_path / "a.json"
        rc = audit_splits.main(["--data", str(mock_dataset), "--outputs", str(tmp_path), "--json", str(out)])
        rep = json.loads(out.read_text())
        assert rc == 1
        assert rep["manifests"]["audit_ok"]["problems"] == []
        assert rep["manifests"]["audit_leak"]["problems"]
    finally:
        for n in ("audit_ok", "audit_leak"):
            (Path(mock_dataset) / "splits" / f"{n}.json").unlink()


def test_final_test_guard_ranges():
    import run_navigation
    from go2wm.common.config import load_config
    from go2wm.eval.navigation import scenarios_from_cfg

    c = load_config("planner_isaac.yaml")
    c["navigation"]["scenarios"].update(first_index=0, per_family=17)
    assert run_navigation.final_test_scenes(scenarios_from_cfg(c, "isaac"), "isaac") == []
    c["navigation"]["scenarios"].update(first_index=16, per_family=2)
    ft = run_navigation.final_test_scenes(scenarios_from_cfg(c, "isaac"), "isaac")
    assert ft == ["open_10017", "obstacles_10017", "sidewalk_10017"]
    assert run_navigation.final_test_scenes(scenarios_from_cfg(c, "mock"), "mock") == []


def test_reactive_defaults_unchanged_and_min_speed():
    from go2wm.planning.baseline import ReactiveController
    from go2wm.planning.safety import SafetyFilter
    from go2wm.common.config import load_config

    cfg = load_config("planner_isaac.yaml")
    sim = MockGo2Sim()
    obs = sim.reset(seed=1, scene_id="sidewalk_0003")
    wp = sim.scene.goal + np.array([0.0, 0.0])
    behind = obs.pos_w[:2] - np.array([3.0, 0.0])  # waypoint behind the robot: default wants to turn in place

    def act(extra, target):
        c = {**cfg, "reactive": {**cfg["reactive"], **extra}}
        ctl = ReactiveController(c, SafetyFilter({**c["safety"], "limits": c["limits"], "robot_radius": c["robot_radius"]}))
        ctl.reset(sim.scene, obs)
        ctl.prev = np.array([0.5, 0.0, 0.0])
        return ctl.act(obs, target)[0]

    assert np.allclose(act({}, wp), act({"min_speed": 0.0, "bound_repulsion": False, "slow_floor": 0.2}, wp))
    assert act({}, behind)[0] < 0.4  # slows below the policy's deadband
    assert act({"min_speed": 0.4}, behind)[0] >= 0.4  # turns in a forward arc instead
