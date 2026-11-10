"""E4 terrain sweep, one variant per Isaac process (the terrain is built when the env is created).

  scripts\\isaac_python.bat scripts\\isaac_terrain_sweep.py --variant slab_10

Runs the variant's probes from configs/e4_sweep.yaml, records every episode (with terrain ground truth)
to --data, and writes a short per-episode summary to outputs/e4_sweep/<variant>.json. The analysis
(command-kinematics error by terrain section, tracking error, outcomes) is offline:
scripts/analyze_terrain_sweep.py.
"""

from __future__ import annotations

import argparse
import copy
import sys

import numpy as np
import yaml

from go2wm.cli import write_json
from go2wm.common.config import load_config, resolve_path
from go2wm.common.schema import VelocityCommand


def _wrap(a: float) -> float:
    return float(np.arctan2(np.sin(a), np.cos(a)))


class LaneFollow:
    """Forward speed vx along +x with P-control on lateral offset; y_ref may change with x.
    schedule: list of (x_from, y_ref), applied in order; None = the scene goal's y."""

    def __init__(self, vx: float, dt: float, schedule=None, k_y: float = 0.8, k_yaw: float = 1.5):
        self.vx, self.dt, self.schedule, self.k_y, self.k_yaw = vx, dt, schedule, k_y, k_yaw

    def __call__(self, obs, scene) -> VelocityCommand:
        x, y, yaw = obs.pose2d
        y_ref = float(scene.goal[1])
        for x_from, yr in self.schedule or []:
            if x >= x_from:
                y_ref = yr
        yaw_ref = float(np.clip(-self.k_y * (y - y_ref), -0.5, 0.5))
        wz = float(np.clip(self.k_yaw * _wrap(yaw_ref - yaw), -1.0, 1.0))
        return VelocityCommand(self.vx, 0.0, wz, self.dt)


class Constant:
    def __init__(self, cmd, dt: float):
        self.cmd, self.dt = cmd, dt

    def __call__(self, obs, scene) -> VelocityCommand:
        return VelocityCommand(*self.cmd, self.dt)


def make_probe(p: dict, dt: float):
    if p["kind"] == "lane":
        return LaneFollow(p["vx"], dt), 0.5
    if p["kind"] == "straight":
        return Constant((p["vx"], 0.0, 0.0), dt), -1.0  # no goal stop
    if p["kind"] == "curb":
        return LaneFollow(p["vx"], dt, schedule=[(2.0, -2.5), (7.0, 0.0)]), -1.0
    raise ValueError(p["kind"])


def variant_spec(sweep: dict, name: str) -> tuple[dict, list[dict], int]:
    names = list(sweep["variants"])
    v = sweep["variants"][name] or {}
    if "terrain" in v or "probes" in v:
        terrain, probes = v.get("terrain", {}), v.get("probes", sweep["default_probes"])
    else:
        terrain, probes = v, sweep["default_probes"]
    expanded = []
    for p in probes:
        n = int(p.get("repeats", 1))
        expanded += [p] if n == 1 else [{**p, "name": f"{p['name']}_r{k}"} for k in range(n)]
    return dict(terrain), expanded, names.index(name)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", required=True)
    ap.add_argument("--sweep", default="configs/e4_sweep.yaml")
    ap.add_argument("--data", default="data/go2_e4_sweep")
    ap.add_argument("--out-dir", default="outputs/e4_sweep")
    args = ap.parse_args(argv)
    with open(args.sweep, encoding="utf-8") as f:
        sweep = yaml.safe_load(f)
    terrain, probes, vi = variant_spec(sweep, args.variant)
    cfg = copy.deepcopy(load_config(sweep["base_config"]))
    c = cfg["isaac"]
    c["families"] = ["sidewalk"]
    c["scene_obstacles"] = False
    c["scene_bounds"] = sweep["scene_bounds"]
    c["terrain"] = {**(c.get("terrain") or {}), **terrain}

    from go2wm.data.recorder import collect_episodes
    from go2wm.sim.isaac_go2 import IsaacGo2Adapter

    sim = IsaacGo2Adapter(cfg)
    report = {"variant": args.variant, "terrain": sim.terrain, "probes": probes, "episodes": []}
    try:
        root = resolve_path(args.data)
        for pi, p in enumerate(probes):
            seed = 60000 + 100 * vi + pi
            fn, tol = make_probe(p, sim.control_dt)
            ids = collect_episodes(
                sim, root, num_episodes=1, base_seed=seed, max_steps=int(p["steps"]),
                limits={"vx": [-1, 1], "vy": [-1, 1], "yaw_rate": [-1, 1]}, goal_tolerance_m=tol,
                scene_ids=[f"sidewalk_{seed:05d}"], command_fn_factory=lambda s, fn=fn: fn,
                extra_meta={"data_role": "e4_sweep", "e4_variant": args.variant, "e4_probe": p})
            report["episodes"] += ids
        p = write_json(resolve_path(args.out_dir) / f"{args.variant}.json", report)  # before close()
        print(f"[sweep] {args.variant}: {len(report['episodes'])} episodes -> {p}", flush=True)
    finally:
        sim.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
