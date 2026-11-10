import numpy as np
import pytest

from go2wm.common.schema import VelocityCommand
from go2wm.data.episodes import load_all, load_episode, list_episodes
from go2wm.data.recorder import EpisodeRecorder
from go2wm.data.validate import validate_dataset, validate_episode
from go2wm.sim.mock import MockGo2Sim


def test_mock_is_deterministic():
    runs = []
    for _ in range(2):
        sim = MockGo2Sim()
        o = sim.reset(seed=3)
        traj = [o.state_vector()]
        for k in range(30):
            traj.append(sim.step(VelocityCommand(0.5, 0.0, 0.3 * np.sin(k / 5))).observation.state_vector())
        runs.append(np.array(traj))
    assert np.array_equal(runs[0], runs[1])


def test_recorded_alignment(mock_dataset):
    for ep in load_all(mock_dataset):
        a = ep.arrays
        T = ep.num_transitions
        assert a["states"].shape[0] == T + 1
        # action applied between o_t and o_t+1
        assert np.all(a["action_timestamp_ns"] >= a["obs_timestamp_ns"][:-1])
        assert np.all(a["action_timestamp_ns"] < a["obs_timestamp_ns"][1:])
        assert np.all(a["dt_s"] > 0)
        # only the final transition may be terminal
        end = a["terminated"] | a["truncated"]
        assert end[-1] and not end[:-1].any()
        assert validate_episode(ep) == []


def test_unique_episode_ids(mock_dataset):
    ids = [p.stem for p in list_episodes(mock_dataset)]
    assert len(ids) == len(set(ids)) == 12


def test_recorder_rejects_bad_timestamps():
    sim = MockGo2Sim()
    o = sim.reset(seed=0)
    rec = EpisodeRecorder({"episode_id": "x"}, o)
    res = sim.step(VelocityCommand(0.2, 0, 0))
    res.observation.timestamp_ns = o.timestamp_ns  # zero interval
    with pytest.raises(ValueError):
        rec.add(VelocityCommand(0.2, 0, 0), res)


def test_validator_catches_corruption(mock_dataset, tmp_path):
    ep = load_episode(list_episodes(mock_dataset)[0])
    ep.arrays["obs_timestamp_ns"] = ep.arrays["obs_timestamp_ns"].copy()
    ep.arrays["obs_timestamp_ns"][5] = ep.arrays["obs_timestamp_ns"][4]
    assert any("timestamp" in e for e in validate_episode(ep))
    ep2 = load_episode(list_episodes(mock_dataset)[0])
    ep2.arrays["terminated"] = ep2.arrays["terminated"].copy()
    ep2.arrays["terminated"][3] = True
    assert any("boundary" in e for e in validate_episode(ep2))
    per_ep, glob = validate_dataset(mock_dataset, manifest_name=None)
    assert per_ep == {} and glob == []
