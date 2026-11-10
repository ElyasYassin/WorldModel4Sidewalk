"""Collect episodes from a simulator backend.

  python scripts/collect.py --backend mock --episodes 20 --out data/mock
  scripts\\isaac_python.bat scripts/collect.py --backend isaac --config configs/isaac_go2.yaml --out data/go2_sim

Relative --out paths resolve under the data root (configs/paths.yaml / GO2WM_ROOT).
"""

from __future__ import annotations

import argparse
import sys
import traceback

from go2wm.common.config import load_config, resolve_path
from go2wm.data.recorder import HeldOutCommandPatterns, collect_episodes
from go2wm.sim.base import make_adapter


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", choices=["mock", "isaac"], required=True)
    ap.add_argument("--config", default=None, help="defaults to configs/<backend>.yaml (isaac -> isaac_go2.yaml)")
    ap.add_argument("--episodes", type=int, default=None)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None, help="base seed (episode i uses seed+i)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--commands", choices=["exploration", "heldout_patterns"], default="exploration",
                    help="heldout_patterns: spec 10.1 command patterns never used for training data")
    ap.add_argument("--scene-family", default=None,
                    help="force every episode into this family (scene id <family>_<seed>), e.g. open")
    ap.add_argument("--no-goal-stop", action="store_true", help="don't end episodes on reaching the goal")
    ap.add_argument("--role", default=None, help="recorded as meta.data_role, e.g. final_test")
    args = ap.parse_args(argv)

    cfg = load_config(args.config or ("mock.yaml" if args.backend == "mock" else "isaac_go2.yaml"))
    cc = cfg["collect"]
    out = resolve_path(args.out)
    n = args.episodes or cc["episodes"]
    base = cc["base_seed"] if args.seed is None else args.seed
    scene_ids = cc.get("scene_ids")
    if args.scene_family:
        scene_ids = [f"{args.scene_family}_{base + i:05d}" for i in range(n)]
    factory = None
    if args.commands == "heldout_patterns":
        factory = lambda seed: HeldOutCommandPatterns(seed, adapter.control_dt)  # noqa: E731  (adapter bound below)
    adapter = make_adapter(args.backend, cfg)
    status = 0
    try:
        collect_episodes(
            adapter,
            out,
            num_episodes=n,
            base_seed=base,
            max_steps=args.max_steps or cc["max_steps"],
            limits=cc["limits"],
            exploration_cfg=cc.get("exploration"),
            goal_tolerance_m=-1.0 if args.no_goal_stop else cc.get("goal_tolerance_m", 0.5),
            scene_ids=scene_ids,
            command_fn_factory=factory,
            extra_meta={"data_role": args.role} if args.role else None,
        )
        print(f"[collect] wrote episodes to {out}", flush=True)
        if args.backend == "mock":
            print("[collect] NOTE: mock data is for software smoke tests only, not Go2 physics.")
    except BaseException:
        # print before close(): Isaac's app.close() ends the process and would swallow it
        traceback.print_exc()
        sys.stderr.flush()
        status = 1
    finally:
        adapter.close()
    return status


if __name__ == "__main__":
    sys.exit(main())
