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
# E4-C sensor rig (go2wm.sim.sensors): per frame, a robot-centred yaw-aligned elevation grid [T+1, nx, ny]
# (mean hit height, scene frame, NaN = no hit) and hit count; grid and device specs in meta.simulator.sensor_rig
RANGE_SENSORS = ("depth_d435", "depth_d555", "l1", "mid360", "mid360_tilt")
for _s in RANGE_SENSORS:
    ARRAY_SPEC[f"sensor_{_s}_elev_m"] = ("float16", "obs")
    ARRAY_SPEC[f"sensor_{_s}_count"] = ("uint8", "obs")
TERRAIN_GT = ("gt_height_above_ground_m", "gt_ground_z_m", "gt_foot_contact", "gt_foot_slip_m_s")
SENSOR_ARRAYS = ("scan_height_m", *(f"sensor_{s}_{k}" for s in RANGE_SENSORS for k in ("elev_m", "count")))
OPTIONAL_ARRAYS = {"gt_clearance_m", "distance_to_goal_m", "progress_to_waypoint_m", *TERRAIN_GT, *SENSOR_ARRAYS}


@dataclass
class Episode:
    meta: dict[str, Any]
    arrays: dict[str, np.ndarray] = field(default_factory=dict)
    rgb_frames: np.ndarray | None = None  # [T+1, H, W, 3] uint8; saved as <id>_rgb.mp4, not in the npz

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
    if ep.rgb_frames is not None and len(ep.rgb_frames):
        ep.meta["rgb_video"] = save_rgb_video(d / f"{ep.episode_id}_rgb.mp4", ep.rgb_frames)
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


def save_rgb_video(path: Path, frames: np.ndarray, fps: float = 10.0) -> dict[str, Any]:
    """H.264 MP4, one frame per observation (T+1 frames at the control rate)."""
    import imageio_ffmpeg

    frames = np.asarray(frames, dtype=np.uint8)
    h, w = frames.shape[1:3]
    wr = imageio_ffmpeg.write_frames(str(path), (w, h), fps=fps, codec="libx264", quality=8, pix_fmt_out="yuv420p",
                                     macro_block_size=1, ffmpeg_log_level="error")
    wr.send(None)
    for f in frames:
        wr.send(np.ascontiguousarray(f[..., :3]))
    wr.close()
    return {"file": path.name, "frames": int(len(frames)), "width": int(w), "height": int(h), "fps": fps}


def load_rgb(path: str | Path) -> np.ndarray | None:
    """RGB frames [T+1, H, W, 3] of an episode (path = its .json/.npz), or None if it has no video."""
    p = Path(path)
    with open(p.with_suffix(".json"), "r", encoding="utf-8") as f:
        info = json.load(f).get("rgb_video")
    if not info:
        return None
    import imageio_ffmpeg

    gen = imageio_ffmpeg.read_frames(str(p.parent / info["file"]))
    meta = next(gen)
    w, h = meta["size"]
    return np.stack([np.frombuffer(b, dtype=np.uint8).reshape(h, w, 3) for b in gen])


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
