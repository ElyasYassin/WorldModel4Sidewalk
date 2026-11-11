"""Episode recorder and the data-collection loop."""

from __future__ import annotations

import datetime as _dt
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

from ..common.frames import FRAME_CONVENTION, angle_diff, world_to_body_2d
from ..common.schema import (
    COMMAND_FIELDS,
    SCHEMA_VERSION,
    STATE_FIELDS,
    TERMINATION_CODES,
    Observation,
    StepResult,
    VelocityCommand,
)
from ..sim.base import SceneSpec, SimulatorAdapter
from .episodes import ARRAY_SPEC, Episode, save_episode


class EpisodeRecorder:
    """Accumulates aligned (o_t, a_t, o_t+1) transitions for one episode."""

    def __init__(self, meta: dict[str, Any], first_obs: Observation, goal_xy: np.ndarray | None = None):
        self.meta = dict(meta)
        self.goal = None if goal_xy is None else np.asarray(goal_xy, dtype=np.float64)
        self._states = [first_obs.state_vector()]
        self._obs_ts = [first_obs.timestamp_ns]
        self._dist = [self._goal_dist(first_obs)]
        self._clear = [first_obs.ground_truth.get("clearance_m", np.nan)]
        # extra ground truth (gt_<k>) and sensor arrays (scan_*) declared in ARRAY_SPEC, recorded when
        # the backend reports them in Observation.ground_truth
        self._gt_extra = {k: [v] for k, v in first_obs.ground_truth.items()
                          if k != "clearance_m" and (f"gt_{k}" in ARRAY_SPEC or (k.startswith(("scan_", "sensor_")) and k in ARRAY_SPEC))}
        self._rgb = [first_obs.rgb] if first_obs.rgb is not None else None
        self._cmds: list[np.ndarray] = []
        self._act_ts: list[int] = []
        self._flags: list[tuple[bool, bool, bool, int]] = []
        self._progress: list[float] = []
        self.done = False

    def _goal_dist(self, obs: Observation) -> float:
        if self.goal is None:
            return float("nan")
        return float(np.linalg.norm(obs.pos_w[:2] - self.goal))

    def add(self, command: VelocityCommand, result: StepResult) -> None:
        if self.done:
            raise RuntimeError("episode already finished")
        o1 = result.observation
        if o1.timestamp_ns <= self._obs_ts[-1]:
            raise ValueError(f"non-increasing observation timestamp {o1.timestamp_ns} <= {self._obs_ts[-1]}")
        if not (self._obs_ts[-1] <= result.action_timestamp_ns < o1.timestamp_ns):
            raise ValueError("action timestamp must lie in [t_obs, t_next_obs)")
        # Record the command actually applied by the backend if it reports one (clipping etc.).
        applied = result.info.get("applied_command")
        cmd = command.as_array()
        if applied is not None:
            cmd[:3] = np.asarray(applied, dtype=np.float64)[:3]
        self._cmds.append(cmd)
        self._act_ts.append(int(result.action_timestamp_ns))
        self._states.append(o1.state_vector())
        self._obs_ts.append(int(o1.timestamp_ns))
        d1 = self._goal_dist(o1)
        self._progress.append(self._dist[-1] - d1)
        self._dist.append(d1)
        self._clear.append(o1.ground_truth.get("clearance_m", np.nan))
        for k, v in self._gt_extra.items():
            v.append(o1.ground_truth.get(k, np.full_like(np.asarray(v[0], dtype=np.float64), np.nan)))
        if self._rgb is not None:
            self._rgb.append(o1.rgb if o1.rgb is not None else np.zeros_like(self._rgb[0]))
        code = TERMINATION_CODES.get(result.termination_reason, TERMINATION_CODES["error"])
        self._flags.append((result.collision, result.terminated, result.truncated, code))
        if result.terminated or result.truncated:
            self.done = True
            self.meta["termination_reason"] = result.termination_reason or ("timeout" if result.truncated else "")

    def finish(self, truncate_reason: str = "timeout") -> Episode:
        if not self._cmds:
            raise ValueError("empty episode")
        flags = np.array(self._flags, dtype=object)
        truncated = flags[:, 2].astype(bool)
        terminated = flags[:, 1].astype(bool)
        codes = flags[:, 3].astype(np.int8)
        if not self.done:  # ran out of steps: mark last transition truncated
            truncated[-1] = True
            codes[-1] = TERMINATION_CODES[truncate_reason]
            self.meta["termination_reason"] = truncate_reason
        obs_ts = np.array(self._obs_ts, dtype=np.int64)
        arrays = {
            "states": np.array(self._states, dtype=np.float64),
            "obs_timestamp_ns": obs_ts,
            "commands": np.array(self._cmds, dtype=np.float64),
            "action_timestamp_ns": np.array(self._act_ts, dtype=np.int64),
            "dt_s": np.diff(obs_ts).astype(np.float64) * 1e-9,
            "collision": flags[:, 0].astype(bool),
            "terminated": terminated,
            "truncated": truncated,
            "termination_code": codes,
            "progress_to_waypoint_m": np.array(self._progress, dtype=np.float64),
            "distance_to_goal_m": np.array(self._dist, dtype=np.float64),
            "gt_clearance_m": np.array(self._clear, dtype=np.float64),
        }
        for k, v in self._gt_extra.items():
            name = k if k.startswith(("scan_", "sensor_")) else f"gt_{k}"
            dt = np.dtype(ARRAY_SPEC[name][0])
            if dt.kind in "iub":  # integer arrays: missing values (NaN filler) become 0
                v = [np.nan_to_num(np.asarray(x, dtype=np.float64), nan=0.0) for x in v]
            arrays[name] = np.asarray(v, dtype=np.float64).astype(dt)
        self.meta["num_transitions"] = len(self._cmds)
        self.meta["duration_s"] = float(arrays["dt_s"].sum())
        rgb = np.stack(self._rgb).astype(np.uint8) if self._rgb is not None else None
        if rgb is not None:
            self.meta.setdefault("modalities", {})["rgb"] = True
        return Episode(meta=self.meta, arrays=arrays, rgb_frames=rgb)


class ExplorationCommands:
    """Diverse bounded body-frame commands for data collection.

    Piecewise segments of: uniform random commands, noisy goal pursuit, or stop,
    each held 0.5-2.5 s and low-pass filtered so commands are locomotion-friendly.
    """

    def __init__(self, seed: int, limits: dict[str, list[float]], control_dt: float, cfg: dict[str, Any] | None = None):
        c = {"p_random": 0.45, "p_goal": 0.45, "p_stop": 0.10, "hold_s": [0.5, 2.5], "smoothing": 0.5, "avoid_radius": 1.0}
        c.update(cfg or {})
        self.c = c
        self.rng = np.random.default_rng(seed)
        self.lim = limits
        self.dt = control_dt
        self._remaining = 0
        self._mode = "stop"
        self._target = np.zeros(3)
        self._prev = np.zeros(3)

    def _new_segment(self) -> None:
        c = self.c
        p = np.array([c["p_random"], c["p_goal"], c["p_stop"]], dtype=np.float64)
        self._mode = str(self.rng.choice(["random", "goal", "stop"], p=p / p.sum()))
        self._remaining = int(round(self.rng.uniform(*c["hold_s"]) / self.dt))
        self._target = np.array([self.rng.uniform(*self.lim[k]) for k in ("vx", "vy", "yaw_rate")])
        self._goal_noise = self.rng.normal(0, [0.15, 0.1, 0.3])

    def __call__(self, obs: Observation, scene: SceneSpec | None) -> VelocityCommand:
        if self._remaining <= 0:
            self._new_segment()
        self._remaining -= 1
        if self._mode == "random":
            raw = self._target
        elif self._mode == "goal" and scene is not None:
            pose = obs.pose2d
            d_b = world_to_body_2d(scene.goal - pose[:2], pose[2])
            heading_err = float(np.arctan2(d_b[1], d_b[0]))
            raw = np.array([0.8 * np.cos(heading_err), 0.0, 1.5 * heading_err]) + self._goal_noise
            # crude repulsion so goal segments don't all end in collisions
            obs_arr = np.asarray(scene.obstacles).reshape(-1, 3)
            for ox, oy, r in obs_arr:
                rel = world_to_body_2d(np.array([ox, oy]) - pose[:2], pose[2])
                dist = np.linalg.norm(rel) - r
                if dist < self.c["avoid_radius"] and rel[0] > 0:
                    raw[2] += -np.sign(rel[1] if abs(rel[1]) > 1e-3 else 1.0) * 1.2 * (self.c["avoid_radius"] - dist)
                    raw[0] *= 0.6
        else:
            raw = np.zeros(3)
        a = self.c["smoothing"]
        cmd = a * self._prev + (1 - a) * raw
        cmd = np.array([np.clip(cmd[i], *self.lim[k]) for i, k in enumerate(("vx", "vy", "yaw_rate"))])
        self._prev = cmd
        return VelocityCommand.from_action(cmd, self.dt)


class HeldOutCommandPatterns:
    """Command patterns NOT produced by ExplorationCommands (spec 10.1 held-out command test).

    ExplorationCommands makes piecewise-constant segments held 0.5-2.5 s, low-pass filtered
    (smoothing 0.5), with vx in [-0.4, 1.0], |vy| <= 0.4, |yaw| <= 1. Each episode uses one pattern:
      sine        continuous sinusoids on vx and yaw (periods 2-6 s), never piecewise constant
      long_arc    one (vx, yaw) arc held 4-6 s, repeated with new arcs; long steady turns
      fast_switch unsmoothed random commands held 0.2-0.3 s (faster than any training segment)
      lateral_ood lateral |vy| in [0.45, 0.7] (outside the training range) mixed with forward motion
    All stay inside the locomotion policy's trained ranges (|lin| <= 1, |yaw| <= 1).
    """

    PATTERNS = ("sine", "long_arc", "fast_switch", "lateral_ood")

    def __init__(self, seed: int, control_dt: float, pattern: str | None = None):
        self.rng = np.random.default_rng(seed)
        self.pattern = pattern or self.PATTERNS[seed % len(self.PATTERNS)]
        if self.pattern not in self.PATTERNS:
            raise ValueError(f"unknown pattern {self.pattern}")
        self.dt = control_dt
        self.t = 0.0
        self._remaining = 0
        self._target = np.zeros(3)
        r = self.rng
        self._sine = dict(a_vx=r.uniform(0.2, 0.4), c_vx=r.uniform(0.4, 0.6), T_vx=r.uniform(2, 6),
                          a_wz=r.uniform(0.4, 0.9), T_wz=r.uniform(2, 6), ph=r.uniform(0, 2 * np.pi))

    def _segment(self) -> None:
        r = self.rng
        if self.pattern == "long_arc":
            self._remaining = int(round(r.uniform(4.0, 6.0) / self.dt))
            self._target = np.array([r.uniform(0.4, 0.9), 0.0, r.choice([-1, 1]) * r.uniform(0.3, 0.9)])
        elif self.pattern == "fast_switch":
            self._remaining = int(round(r.uniform(0.2, 0.3) / self.dt))
            self._target = np.array([r.uniform(-0.3, 0.9), r.uniform(-0.3, 0.3), r.uniform(-1.0, 1.0)])
        else:  # lateral_ood
            self._remaining = int(round(r.uniform(1.0, 2.5) / self.dt))
            self._target = np.array([r.uniform(0.0, 0.5), r.choice([-1, 1]) * r.uniform(0.45, 0.7), r.uniform(-0.3, 0.3)])

    def __call__(self, obs: Observation, scene: SceneSpec | None) -> VelocityCommand:
        if self.pattern == "sine":
            s = self._sine
            cmd = np.array([s["c_vx"] + s["a_vx"] * np.sin(2 * np.pi * self.t / s["T_vx"]), 0.0,
                            s["a_wz"] * np.sin(2 * np.pi * self.t / s["T_wz"] + s["ph"])])
        else:
            if self._remaining <= 0:
                self._segment()
            self._remaining -= 1
            cmd = self._target.copy()
        self.t += self.dt
        cmd = np.clip(cmd, [-1.0, -1.0, -1.0], [1.0, 1.0, 1.0])
        return VelocityCommand.from_action(cmd, self.dt)


def collect_episodes(
    adapter: SimulatorAdapter,
    out_dir: str | Path,
    num_episodes: int,
    base_seed: int,
    max_steps: int,
    limits: dict[str, list[float]],
    exploration_cfg: dict[str, Any] | None = None,
    goal_tolerance_m: float = 0.5,
    scene_ids: list[str] | None = None,
    command_fn_factory: Callable[[int], Callable] | None = None,
    log: Callable[[str], None] = print,
    extra_meta: dict[str, Any] | None = None,
) -> list[str]:
    out_dir = Path(out_dir)
    ids = []
    t0 = time.perf_counter()
    total_steps = 0
    for i in range(num_episodes):
        seed = base_seed + i
        scene_id = scene_ids[i % len(scene_ids)] if scene_ids else None
        obs = adapter.reset(seed=seed, scene_id=scene_id)
        scene = adapter.scene
        episode_id = f"{adapter.backend}_{scene.scene_id if scene else 'noscene'}_s{seed:06d}"
        meta = {
            "schema_version": SCHEMA_VERSION,
            "episode_id": episode_id,
            "scene_id": scene.scene_id if scene else None,
            "scene_family": scene.family if scene else None,
            "seed": seed,
            "backend": adapter.backend,
            "created_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
            "frame_convention": FRAME_CONVENTION,
            "state_fields": {k: [v.start, v.stop] for k, v in STATE_FIELDS.items()},
            "command_fields": list(COMMAND_FIELDS),
            "modalities": {"rgb": False, "depth": False, "joints": False},
            "scene": scene.to_dict() if scene else None,
            "simulator": adapter.metadata(),
        }
        cmd_fn = command_fn_factory(seed) if command_fn_factory else ExplorationCommands(seed, limits, adapter.control_dt, exploration_cfg)
        meta["collector"] = {"type": type(cmd_fn).__name__, "cfg": exploration_cfg or {}, "limits": limits}
        if isinstance(cmd_fn, HeldOutCommandPatterns):
            meta["collector"] = {"type": type(cmd_fn).__name__, "pattern": cmd_fn.pattern}
            meta["command_pattern"] = cmd_fn.pattern
        meta.update(extra_meta or {})
        rec = EpisodeRecorder(meta, obs, goal_xy=scene.goal if scene else None)
        for _ in range(max_steps):
            cmd = cmd_fn(obs, scene)
            res = adapter.step(cmd)
            if scene is not None and not res.terminated and np.linalg.norm(res.observation.pos_w[:2] - scene.goal) < goal_tolerance_m:
                res.terminated, res.termination_reason = True, "goal"
            rec.add(cmd, res)
            obs = res.observation
            total_steps += 1
            if rec.done:
                break
        ep = rec.finish()
        save_episode(out_dir, ep)
        ids.append(episode_id)
        log(f"[collect] {episode_id}: {ep.num_transitions} transitions, end={ep.meta['termination_reason']}")
    wall = time.perf_counter() - t0
    log(f"[collect] {num_episodes} episodes, {total_steps} transitions in {wall:.1f}s wall")
    return ids
