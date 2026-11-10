"""Mock backend -> collect -> validate -> split -> train -> evaluate -> navigate. Software smoke test only."""

import numpy as np

from go2wm.data.dataset import get_or_make_manifest, load_split
from go2wm.data.validate import validate_dataset
from go2wm.eval.navigation import aggregate, run_episode
from go2wm.eval.prediction import default_baselines, evaluate_models
from go2wm.models.state_dynamics import load_checkpoint
from go2wm.models.training import train
from go2wm.planning.local_mpc import ModelPredictor, MPCController
from go2wm.planning.safety import SafetyFilter
from go2wm.sim.mock import MockGo2Sim

CFG = {
    "data": {"history": 3, "horizon_train": 5, "stride": 2, "eval_horizons_s": [0.5, 1.0], "split_seed": 0,
             "val_frac": 0.2, "test_frac": 0.2},
    "model": {"type": "mlp", "hidden": 64, "layers": 2},
    "train": {"seed": 0, "device": "cpu", "epochs": 8, "batch_size": 128, "lr": 2e-3, "weight_decay": 0.0,
              "grad_clip": 1.0, "rollout_weight": 0.5, "huber_delta": 1.0},
}
NAV = {
    "control_dt": 0.1, "robot_radius": 0.3,
    "limits": {"vx": [-0.3, 0.9], "vy": [-0.3, 0.3], "yaw_rate": [-1.2, 1.2]},
    "safety": {"max_accel": [2.0, 2.0, 4.0]},
    "navigation": {"grid_resolution": 0.2, "inflation_margin": 0.15, "waypoint_spacing": 1.0, "waypoint_tolerance": 0.6,
                   "goal_tolerance": 0.5, "max_steps": 250, "stall_speed": 0.05, "stall_steps": 20, "max_reroutes": 1},
    "mpc": {"horizon": 10, "samples": 96, "elites": 12, "iterations": 3,
            "weights": {"goal": 1.0, "running_goal": 0.2, "heading": 0.3, "collision": 30.0, "smooth": 0.05}},
}


def test_end_to_end(mock_dataset, tmp_path):
    per_ep, glob = validate_dataset(mock_dataset, manifest_name=None)
    assert not per_ep and not glob
    m = get_or_make_manifest(mock_dataset, "e2e", seed=0, val_frac=0.2, test_frac=0.2)
    CFG["data"]["manifest"] = "e2e"
    res = train(CFG, mock_dataset, tmp_path / "out", log=lambda s: None)
    assert np.isfinite(res["best_val_loss"])
    assert res["history"][-1]["train_loss"] < res["history"][0]["train_loss"]
    model, ck = load_checkpoint(tmp_path / "out" / "best.pt")
    assert set(ck["train_episodes"]) == set(m["episodes"]["train"])

    test_eps = load_split(mock_dataset, "test", "e2e")
    ev = evaluate_models({"mlp": model, **default_baselines(3)}, test_eps, 3, [0.5, 1.0])
    for r in ev["models"].values():
        assert np.isfinite(r["open_loop"]["1.0"]["pos_err_m_mean"])

    sim = MockGo2Sim()
    safety = SafetyFilter({**NAV["safety"], "limits": NAV["limits"], "robot_radius": 0.3})
    ctrl = MPCController(ModelPredictor(model), NAV, safety, seed=0)
    out = [run_episode(sim, ctrl, seed, sid, NAV) for seed, sid in [(10000, "open_10000"), (10001, "obstacles_10001")]]
    agg = aggregate(out)
    assert agg["n_episodes"] == 2 and agg["collision_rate"] == 0.0
    # progress toward goal even if not all reach it in this tiny smoke config
    for r in out:
        tr = np.array(r["trajectory"])
        assert tr[-1, 0] > tr[0, 0] + 3.0
