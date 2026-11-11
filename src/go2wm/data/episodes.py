"""Per-episode storage: <root>/episodes/<episode_id>.npz + <episode_id>.json.

Transition i is (states[i], commands[i], states[i+1]); see docs/DATA_FORMAT.md.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

# name -> (dtype, "obs" for length T+1 / "tr" for length T)
ARRAY_SPEC: dict[str, tuple[str, str]] = {
    "states": ("float64", "obs"),
    "obs_timestamp_ns": ("int64", "obs"),
    "commands": ("float64", "tr"),
    "action_timestamp_ns": ("int64", "tr"),
    "dt_s": ("float64", "tr"),
    "collision": ("bool", "tr"),
    "terminated": ("bool", "tr"),
    "truncated": ("bool", "tr"),
    "termination_code": ("int8", "tr"),
    "progress_to_waypoint_m": ("float64", "tr"),
    "distance_to_goal_m": ("float64", "obs"),
    # ground truth, evaluation only (prefix gt_)
    "gt_clearance_m": ("float64", "obs"),
    # terrain episodes only (Isaac with a `terrain:` config; P1/E4)
    "gt_height_above_ground_m": ("float64", "obs"),
    "gt_ground_z_m": ("float64", "obs"),  # ground under the base, scene frame
    "gt_foot_contact": ("float64", "obs"),  # [T+1, 4] 0/1, FL FR RL RR order as the robot's body list
    "gt_foot_slip_m_s": ("float64", "obs"),  # [T+1, 4] horizontal speed of feet in contact
    # robot sensor (model input allowed): height-scanner ray hit heights, scene frame [T+1, R]; ray
    # offsets in the yaw-aligned body frame are in meta.simulator.height_scan_pattern_xy
    "scan_height_m": ("float64", "obs"),
}
TERRAIN_GT = ("gt_height_above_ground_m", "gt_ground_z_m", "gt_foot_contact", "gt_foot_slip_m_s")
SENSOR_ARRAYS = ("scan_height_m",)
OPTIONAL_ARRAYS = {"gt_clearance_m", "distance_to_goal_m", "progress_to_waypoint_m", *TERRAIN_GT, *SENSOR_ARRAYS}


@dataclass
class Episode:
    meta: dict[str, Any]
    arrays: dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def episode_id(self) -> str:
        return self.meta["episode_id"]

    @property
    def num_transitions(self) -> int:
        return int(len(self.arrays["commands"]))

    def __getitem__(self, k: str) -> np.ndarray:
        return self.arrays[k]


def episodes_dir(root: str | Path) -> Path:
    return Path(root) / "episodes"


def save_episode(root: str | Path, ep: Episode) -> Path:
    d = episodes_dir(root)
    d.mkdir(parents=True, exist_ok=True)
    stem = d / ep.episode_id
    np.savez_compressed(stem.with_suffix(".npz"), **ep.arrays)
    with open(stem.with_suffix(".json"), "w", encoding="utf-8") as f:
        json.dump(ep.meta, f, indent=2, default=_json_default)
    return stem.with_suffix(".npz")


def load_episode(path: str | Path, load_arrays: bool = True) -> Episode:
    p = Path(path)
    with open(p.with_suffix(".json"), "r", encoding="utf-8") as f:
        meta = json.load(f)
    arrays: dict[str, np.ndarray] = {}
    if load_arrays:
        with np.load(p.with_suffix(".npz")) as z:
            arrays = {k: z[k] for k in z.files}
    return Episode(meta=meta, arrays=arrays)


def list_episodes(root: str | Path) -> list[Path]:
    return sorted(episodes_dir(root).glob("*.json"))


def load_all(root: str | Path, ids: list[str] | None = None) -> list[Episode]:
    paths = list_episodes(root)
    if ids is not None:
        wanted = set(ids)
        paths = [p for p in paths if p.stem in wanted]
    return [load_episode(p) for p in paths]


def _json_default(o: Any):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.bool_):
        return bool(o)
    raise TypeError(f"not JSON serializable: {type(o)}")
