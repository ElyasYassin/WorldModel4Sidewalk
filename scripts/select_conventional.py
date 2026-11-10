"""Pick the tuned conventional baseline from development-scene runs (P0.5) and freeze it.

  python scripts/select_conventional.py            # dry run: print the ranking
  python scripts/select_conventional.py --write    # also write configs/planner_isaac_conv_tuned.yaml

Selection rule (fixed 2026-11-09, before any tuning run finished), over obstacle + sidewalk
development scenes 0-16 (34 scenes): highest success count; ties -> fewer collisions -> fewer
safety interventions per episode -> shorter mean time to goal on successes. The default config
(E3b, outputs/navigation*/isaac) is a candidate too. Refuses to read any final-test scene.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

from go2wm.common.config import load_config, resolve_path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_RUNS = ["outputs/navigation/isaac/conventional_episodes.json",
                "outputs/navigation_ext/isaac/conventional_episodes.json"]


def dev_scene_ok(sid: str, proto: dict) -> bool:
    idx = int(sid.rpartition("_")[2]) - proto["base_seed"]
    if proto["final_test"]["first_index"] <= idx <= proto["final_test"]["last_index"]:
        raise SystemExit(f"[ERROR] final-test scene {sid} found in tuning results")
    return proto["development"]["first_index"] <= idx <= proto["development"]["last_index"]


def score(res: list[dict]) -> tuple:
    n_s = sum(r["success"] for r in res)
    n_c = sum(r["collision"] for r in res)
    interv = sum(r["safety_interventions"] for r in res) / len(res)
    t = [r["time_s"] for r in res if r["success"]]
    return (-n_s, n_c, interv, sum(t) / len(t) if t else 1e9)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tuning-dir", default="outputs/p0_tuning")
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args(argv)
    proto = load_config(REPO / "configs" / "eval_protocol.yaml")["navigation"]
    cands: dict[str, tuple[str, list[dict]]] = {}
    default = []
    for f in DEFAULT_RUNS:
        default += json.loads(resolve_path(f).read_text(encoding="utf-8"))
    cands["default"] = ("configs/planner_isaac.yaml", default)
    for d in sorted(resolve_path(args.tuning_dir).glob("conv_*")):
        f = d / "isaac" / "conventional_episodes.json"
        if f.exists():
            cands[d.name] = (f"configs/tuning/{d.name}.yaml", json.loads(f.read_text(encoding="utf-8")))
    rows = []
    for name, (cfg, res) in cands.items():
        res = [r for r in res if r["family"] in ("obstacles", "sidewalk") and dev_scene_ok(r["scene_id"], proto)]
        complete = len({r["scene_id"] for r in res}) == 34
        rows.append((score(res), name, cfg, len(res), complete))
    rows.sort()
    print(f"{'config':24s}{'n':>4s}{'success':>8s}{'coll':>5s}{'interv':>7s}{'t_goal':>7s}  complete")
    for sc, name, _, n, ok in rows:
        print(f"{name:24s}{n:4d}{-sc[0]:8d}{sc[1]:5d}{sc[2]:7.1f}{sc[3]:7.1f}  {ok}")
    best = next((r for r in rows if r[4]), None)
    if best is None:
        print("[select] no candidate has all 34 dev scenes yet")
        return 1
    print(f"[select] chosen: {best[1]} ({best[2]})")
    if args.write:
        chosen = load_config(REPO / best[2])
        cfg = {"_base_": "planner_isaac.yaml", "reactive": chosen["reactive"]}
        out = REPO / "configs" / "planner_isaac_conv_tuned.yaml"
        header = (f"# Tuned conventional baseline, FROZEN for E3f. Chosen by scripts/select_conventional.py from\n"
                  f"# {len(rows)} configs on obstacle+sidewalk development scenes 0-16 only. Winner: {best[1]}\n"
                  f"# (success {-best[0][0]}/34, collisions {best[0][1]}, interventions/ep {best[0][2]:.1f}).\n")
        out.write_text(header + yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        print(f"[select] wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
