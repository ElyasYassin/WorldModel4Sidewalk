import numpy as np

from go2wm.common.frames import relative_se2
from go2wm.common.schema import state_pose2d
from go2wm.data.dataset import build_windows, make_split_manifest
from go2wm.data.episodes import load_all, load_episode, list_episodes


def test_split_is_scene_disjoint_and_deterministic(mock_dataset):
    m1 = make_split_manifest(mock_dataset, seed=1)
    m2 = make_split_manifest(mock_dataset, seed=1)
    assert m1 == m2
    sets = [set(m1["episodes"][k]) for k in ("train", "val", "test")]
    assert all(s for s in sets)
    assert not (sets[0] & sets[1]) and not (sets[0] & sets[2]) and not (sets[1] & sets[2])
    assert set().union(*sets) == {p.stem for p in list_episodes(mock_dataset)}
    scenes = {k: set(v) for k, v in m1["scenes"].items()}
    assert not (scenes["train"] & scenes["test"])


def test_holdout_family_goes_to_test(mock_dataset):
    m = make_split_manifest(mock_dataset, seed=0, holdout_families=["slippery"])
    fam = {p.stem: load_episode(p, load_arrays=False).meta["scene_family"] for p in list_episodes(mock_dataset)}
    assert all(fam[e] != "slippery" for k in ("train", "val") for e in m["episodes"][k])
    assert any(fam[e] == "slippery" for e in m["episodes"]["test"])


def test_windows_do_not_cross_episodes_and_no_leakage(mock_dataset):
    eps = load_all(mock_dataset)
    K, H = 4, 5
    w = build_windows(eps, K, H)
    assert w.hist_feat.shape[1:] == (K, 9) and w.fut_act.shape[1:] == (H, 3)
    for j in np.random.default_rng(0).choice(len(w), 25, replace=False):
        ep = eps[w.episode_index[j]]
        i = w.start_index[j]
        assert K <= i and i + H <= ep.num_transitions
        # future action i is the command applied right after the last history observation
        assert np.allclose(w.fut_act[j, 0], ep["commands"][i, :3])
        # history actions are strictly before i
        assert np.allclose(w.hist_act[j, -1], ep["commands"][i - 1, :3])
        pose = state_pose2d(ep["states"])
        assert np.allclose(w.fut_pose[j], relative_se2(pose[i], pose[i + 1 : i + H + 1]), atol=1e-5)
