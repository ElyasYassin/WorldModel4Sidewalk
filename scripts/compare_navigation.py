"""Paired comparison of labeled navigation runs on shared scenes, with a failure taxonomy.

  python scripts/compare_navigation.py conv=outputs/e3f/conventional_default/isaac/conventional_episodes.json \\
      kin=outputs/e3f/kinematic_mpc/isaac --out outputs/e3f/compare

Each input is LABEL=PATH, PATH an *_episodes.json file or a folder holding exactly one.
Reports per label: success (Wilson CI), collisions, falls, failure taxonomy, time to goal,
route efficiency, min clearance, safety interventions, command smoothness, latency. Pairwise on
shared scenes: exact McNemar on success, and the paired time-to-goal difference over scenes both
solved (bootstrap CI over scenes).
"""

from __future__ import annotations

import argparse
import json
import sys
from math import comb
from pathlib import Path

import numpy as np

from go2wm.cli import write_json
from go2wm.common.config import resolve_path
from go2wm.eval.navigation import aggregate
from go2wm.eval.prediction import bootstrap_ci


def mcnemar_exact(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(comb(n, i) for i in range(min(b, c) + 1)) / 2**n)


def failure_class(r: dict, control_dt: float = 0.1) -> str:
    """success | collision | fall | no_global_path | freeze (timeout, <0.1 m moved in last 10 s) |
    wander (timeout while still moving) | other."""
    o = r["outcome"]
    if o in ("success", "collision", "fall", "no_global_path"):
        return o
    if o == "timeout":
        tr = np.asarray(r["trajectory"])
        k = min(len(tr) - 1, int(round(10.0 / control_dt)))
        return "freeze" if np.linalg.norm(tr[-1, :2] - tr[-1 - k, :2]) < 0.1 else "wander"
    return "other"


def load(path: str) -> list[dict]:
    p = resolve_path(path)
    if p.is_dir():
        files = sorted(p.glob("*_episodes.json"))
        if len(files) != 1:
            raise SystemExit(f"[ERROR] {p} has {len(files)} *_episodes.json files; name one")
        p = files[0]
    return json.loads(p.read_text(encoding="utf-8"))


def summarize(res: list[dict]) -> dict:
    s = aggregate(res)
    tax: dict[str, int] = {}
    for r in res:
        c = failure_class(r)
        tax[c] = tax.get(c, 0) + 1
    s["failure_taxonomy"] = tax
    s["failed_scenes"] = sorted(r["scene_id"] for r in res if not r["success"])
    return s


def compare(runs: dict[str, list[dict]]) -> dict:
    by = {k: {r["scene_id"]: r for r in v} for k, v in runs.items()}
    out = {}
    names = list(by)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            shared = sorted(set(by[a]) & set(by[b]))
            if not shared:
                continue
            a_only = [s for s in shared if by[a][s]["success"] and not by[b][s]["success"]]
            b_only = [s for s in shared if by[b][s]["success"] and not by[a][s]["success"]]
            both = [s for s in shared if by[a][s]["success"] and by[b][s]["success"]]
            dt = np.array([by[b][s]["time_s"] - by[a][s]["time_s"] for s in both])
            out[f"{a}_vs_{b}"] = {
                "n_shared": len(shared), "a_only_success": a_only, "b_only_success": b_only,
                "mcnemar_p": mcnemar_exact(len(a_only), len(b_only)),
                "n_both_success": len(both),
                "time_to_goal_diff_s_b_minus_a": float(dt.mean()) if len(dt) else None,
                "time_to_goal_diff_ci95": bootstrap_ci(dt) if len(dt) > 1 else None,
            }
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="LABEL=PATH")
    ap.add_argument("--families", nargs="*", default=None, help="restrict to these scene families")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    runs = {}
    for spec in args.runs:
        label, _, path = spec.partition("=")
        res = load(path)
        if args.families:
            res = [r for r in res if r["family"] in args.families]
        runs[label] = res
    summ = {k: summarize(v) for k, v in runs.items()}
    pairs = compare(runs)
    print(f"{'run':24s}{'n':>4s}{'success':>18s}{'coll':>6s}{'fall':>5s}{'freeze':>7s}{'wander':>7s}{'t_goal':>7s}{'minclr':>7s}{'interv':>7s}{'dcmd':>7s}")
    for k, s in summ.items():
        t, ci = s["failure_taxonomy"], s["success_ci95"]
        tg = s["time_to_goal_s_mean_successes"]
        print(f"{k:24s}{s['n_episodes']:4d}{s['success_rate']:7.2f} [{ci[0]:.2f},{ci[1]:.2f}]{t.get('collision', 0):6d}"
              f"{t.get('fall', 0):5d}{t.get('freeze', 0):7d}{t.get('wander', 0):7d}"
              f"{(tg if tg is not None else float('nan')):7.1f}{s['min_clearance_m_mean']:7.2f}{s['safety_interventions_per_episode']:7.1f}"
              f"{(s['command_change_mean'] if s['command_change_mean'] is not None else float('nan')):7.3f}")
    for k, c in pairs.items():
        d = c["time_to_goal_diff_s_b_minus_a"]
        dci = c["time_to_goal_diff_ci95"]
        dtxt = f"  dt(b-a)={d:+.2f}s [{dci[0]:+.2f},{dci[1]:+.2f}]" if dci else ""
        print(f"  {k}: n={c['n_shared']} a-only={len(c['a_only_success'])} b-only={len(c['b_only_success'])} "
              f"p={c['mcnemar_p']:.3f}{dtxt}")
    if args.out:
        out = resolve_path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        write_json(out / "comparison.json", {"runs": summ, "paired": pairs, "sources": args.runs,
                                             "families": args.families})
        print(f"[compare] wrote {out / 'comparison.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
