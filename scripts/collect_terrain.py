"""E4 terrain data collection on the frozen tile sets (configs/e4_tiles.yaml), one Isaac process.

  scripts\isaac_python.bat scripts\collect_terrain.py --split train --episodes 96
  scripts\isaac_python.bat scripts\collect_terrain.py --split test --episodes 36

All tiles of the split are built side by side; episode seed s runs on tile s % n_tiles. Commands:
TerrainMixCommands (half exploration with the standard training limits, half lane-following at a
random 0.3-0.9 m/s with random lateral changes that cross the walkway edge or curb). Seeds:
train 70000+, test 80000+ (disjoint from every other data set). Episodes carry terrain ground truth
and meta.e4_tile / meta.e4_commands.
"""

from __future__ import annotations

import argparse
import copy
import sys

import yaml

from go2wm.common.config import load_config, resolve_path

SPLITS = {"train": (70000, "data/go2_e4_train"), "test": (80000, "data/go2_e4_test")}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", choices=list(SPLITS), required=True)
    ap.add_argument("--episodes", type=int, required=True)
    ap.add_argument("--first", type=int, default=0, help="episode offset (resume)")
    ap.add_argument("--tiles", default="configs/e4_tiles.yaml")
    ap.add_argument("--config", default="configs/isaac_go2_rough.yaml")
    ap.add_argument("--max-steps", type=int, default=300)
    ap.add_argument("--data", default=None)
    args = ap.parse_args(argv)
    base_seed, default_data = SPLITS[args.split]
    with open(args.tiles, encoding="utf-8") as f:
        tiles = yaml.safe_load(f)[args.split]
    cfg = copy.deepcopy(load_config(args.config))
    c = cfg["isaac"]
    c.update(families=["sidewalk"], scene_obstacles=False, scene_bounds=[-1.0, 14.0, -3.5, 3.5], terrain_tiles=tiles)
    cc = load_config("configs/isaac_go2.yaml")["collect"]

    from go2wm.data.recorder import collect_episodes
    from go2wm.data.terrain_commands import TerrainMixCommands
    from go2wm.sim.isaac_go2 import IsaacGo2Adapter

    sim = IsaacGo2Adapter(cfg)
    root = resolve_path(args.data or default_data)
    try:
        for k in range(args.first, args.first + args.episodes):
            seed = base_seed + k
            gen = TerrainMixCommands(seed, cc["limits"], sim.control_dt, cc.get("exploration"))
            collect_episodes(
                sim, root, num_episodes=1, base_seed=seed, max_steps=args.max_steps, limits=cc["limits"],
                goal_tolerance_m=cc["goal_tolerance_m"], scene_ids=[f"sidewalk_{seed:05d}"],
                command_fn_factory=lambda s, g=gen: g,
                extra_meta={"data_role": f"e4_{args.split}", "e4_tile": seed % len(tiles),
                            "e4_tiles_file": args.tiles, "e4_commands": gen.describe()})
        print(f"[e4collect] {args.split}: episodes {args.first}..{args.first + args.episodes - 1} -> {root}", flush=True)
    finally:
        sim.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
