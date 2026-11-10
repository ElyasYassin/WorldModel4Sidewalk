"""Scene-disjoint splits and windowed transition datasets."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..common.frames import relative_se2
from ..common.schema import ACTION_DIM, state_pose2d
from ..models.features import state_features, transition_targets
from .episodes import Episode, list_episodes, load_all, load_episode

SPLITS = ("train", "val", "test")


def make_split_manifest(
    root: str | Path,
    seed: int = 0,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    holdout_families: list[str] | None = None,
) -> dict[str, Any]:
    """Assign whole scenes to splits (never adjacent frames).

    holdout_families: scene families placed entirely in test (generalization).
    """
    metas = [load_episode(p, load_arrays=False).meta for p in list_episodes(root)]
    if not metas:
        raise FileNotFoundError(f"no episodes under {root}")
    holdout = set(holdout_families or [])
    by_scene: dict[str, list[str]] = {}
    fam_of: dict[str, str] = {}
    for m in metas:
        sid = m.get("scene_id") or m["episode_id"]
        by_scene.setdefault(sid, []).append(m["episode_id"])
        fam_of[sid] = m.get("scene_family") or "unknown"
    scenes = sorted(s for s in by_scene if fam_of[s] not in holdout)
    rng = np.random.default_rng(seed)
    rng.shuffle(scenes)
    n = len(scenes)
    n_test = int(round(test_frac * n))
    n_val = int(round(val_frac * n))
    if n >= 3:
        n_test, n_val = max(n_test, 1), max(n_val, 1)
    assign = {"test": scenes[:n_test], "val": scenes[n_test : n_test + n_val], "train": scenes[n_test + n_val :]}
    assign["test"] = assign["test"] + sorted(s for s in by_scene if fam_of[s] in holdout)
    manifest = {
        "method": "scene-disjoint",
        "seed": seed,
        "val_frac": val_frac,
        "test_frac": test_frac,
        "holdout_families": sorted(holdout),
        "scenes": {k: sorted(v) for k, v in assign.items()},
        "episodes": {k: sorted(e for s in v for e in by_scene[s]) for k, v in assign.items()},
    }
    return manifest


def make_all_test_manifest(root: str | Path, note: str = "") -> dict[str, Any]:
    """Every episode under root in 'test' (for a dataset collected only for evaluation, e.g. the
    frozen final test or the held-out command-pattern set). Records the episode list so later
    additions to the directory are detectable."""
    ids = sorted(load_episode(p, load_arrays=False).meta["episode_id"] for p in list_episodes(root))
    if not ids:
        raise FileNotFoundError(f"no episodes under {root}")
    return {"method": "all-test", "note": note, "scenes": {"train": [], "val": [], "test": ids},
            "episodes": {"train": [], "val": [], "test": ids}}


def save_manifest(root: str | Path, manifest: dict[str, Any], name: str = "default") -> Path:
    p = Path(root) / "splits" / f"{name}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return p


def load_manifest(root: str | Path, name: str = "default") -> dict[str, Any]:
    with open(Path(root) / "splits" / f"{name}.json", "r", encoding="utf-8") as f:
        return json.load(f)


def get_or_make_manifest(root: str | Path, name: str = "default", **kw) -> dict[str, Any]:
    p = Path(root) / "splits" / f"{name}.json"
    if p.exists():
        return load_manifest(root, name)
    m = make_split_manifest(root, **kw)
    save_manifest(root, m, name)
    return m


@dataclass
class WindowArrays:
    """Stacked training windows. N windows, K history, H horizon."""

    hist_feat: np.ndarray  # [N, K, F]   features of states i-K+1..i
    hist_act: np.ndarray  # [N, K, A]   commands applied before each history state
    hist_dt: np.ndarray  # [N, K]
    fut_act: np.ndarray  # [N, H, A]   commands at i..i+H-1
    fut_dt: np.ndarray  # [N, H]      actual intervals
    targets: np.ndarray  # [N, H, T]   per-step delta targets
    fut_pose: np.ndarray  # [N, H, 3]   poses rel. to pose at i (for eval)
    fut_feat: np.ndarray  # [N, H, F]
    episode_index: np.ndarray  # [N]
    start_index: np.ndarray  # [N]

    def __len__(self) -> int:
        return len(self.hist_feat)


def build_windows(episodes: list[Episode], history: int, horizon: int, stride: int = 1) -> WindowArrays:
    """Windows never cross episode boundaries; history needs K prior transitions."""
    K, H = history, horizon
    out: dict[str, list] = {k: [] for k in WindowArrays.__dataclass_fields__}
    for ei, ep in enumerate(episodes):
        S = ep["states"]
        C = ep["commands"][:, :ACTION_DIM]
        dt = ep["dt_s"]
        T = len(C)
        feat = state_features(S)
        tgt = transition_targets(S[:-1], S[1:])
        pose = state_pose2d(S)
        for i in range(K, T - H + 1, stride):
            out["hist_feat"].append(feat[i - K + 1 : i + 1])
            out["hist_act"].append(C[i - K : i])
            out["hist_dt"].append(dt[i - K : i])
            out["fut_act"].append(C[i : i + H])
            out["fut_dt"].append(dt[i : i + H])
            out["targets"].append(tgt[i : i + H])
            out["fut_pose"].append(relative_se2(pose[i], pose[i + 1 : i + H + 1]))
            out["fut_feat"].append(feat[i + 1 : i + H + 1])
            out["episode_index"].append(ei)
            out["start_index"].append(i)
    if not out["hist_feat"]:
        raise ValueError(f"no windows (history={K}, horizon={H}); episodes too short?")
    return WindowArrays(**{k: np.asarray(v, dtype=np.float32 if k not in ("episode_index", "start_index") else np.int64) for k, v in out.items()})


def load_split(root: str | Path, split: str, manifest_name: str = "default", **manifest_kw) -> list[Episode]:
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}")
    m = get_or_make_manifest(root, manifest_name, **manifest_kw)
    ids = m["episodes"][split]
    if not ids:
        raise ValueError(f"split '{split}' is empty in manifest '{manifest_name}'")
    return load_all(root, ids)
