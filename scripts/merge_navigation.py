"""Merge navigation result folders and compare planners on the scenes they share.

  python scripts/merge_navigation.py outputs/navigation/isaac outputs/navigation_kin/isaac ... --out outputs/nav_merged

Reads every `<planner>_episodes.json` in the given folders. If the same (planner, scene_id)
appears twice, the later folder wins. Aggregates per planner, then runs paired comparisons
(exact McNemar on success) restricted to the scenes both planners ran.
"""

from __future__ import annotations

import argparse
import json
import sys
from math import comb
from pathlib import Path

from go2wm.cli import write_json
from go2wm.common.config import resolve_path
from go2wm.eval.navigation import aggregate


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value from discordant counts b, c."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = 2 * sum(comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, p)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    by_planner: dict[str, dict[str, dict]] = {}
    for d in args.dirs:
        for f in sorted(resolve_path(d).glob("*_episodes.json")):
            name = f.name[: -len("_episodes.json")]
            for r in json.loads(f.read_text()):
                by_planner.setdefault(name, {})[r["scene_id"]] = r
    if not by_planner:
        print("[merge] no *_episodes.json found", file=sys.stderr)
        return 1

    summary = {}
    print(f"{'planner':16s}{'n':>4s}{'success':>18s}{'collide':>9s}{'falls':>6s}{'timeout':>8s}{'t_goal':>8s}{'minclr':>8s}{'interv':>8s}{'p50ms':>7s}")
    for name, eps in by_planner.items():
        res = [eps[k] for k in sorted(eps)]
        s = aggregate(res)
        summary[name] = s
        ci = s["success_ci95"]
        t = s["time_to_goal_s_mean_successes"]
        print(f"{name:16s}{s['n_episodes']:4d}{s['success_rate']:7.2f} [{ci[0]:.2f},{ci[1]:.2f}]"
              f"{s['collision_rate']:9.2f}{s['falls']:6d}{s['timeouts']:8d}{(t if t is not None else float('nan')):8.1f}"
              f"{s['min_clearance_m_mean']:8.2f}{s['safety_interventions_per_episode']:8.1f}{s['latency_ms_p50']:7.1f}")
        for fam, fs in s["by_family"].items():
            print(f"    {fam:12s} n={fs['n']:3d} success={fs['success_rate']:.2f} collision={fs['collision_rate']:.2f}")

    names = list(by_planner)
    paired = {}
    print("\npaired (shared scenes): A-only successes / B-only successes, exact McNemar p")
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            shared = sorted(set(by_planner[a]) & set(by_planner[b]))
            if not shared:
                continue
            ab = sum(by_planner[a][k]["success"] and not by_planner[b][k]["success"] for k in shared)
            ba = sum(by_planner[b][k]["success"] and not by_planner[a][k]["success"] for k in shared)
            p = mcnemar_exact(ab, ba)
            paired[f"{a}_vs_{b}"] = {"n_shared": len(shared), "a_only": ab, "b_only": ba, "mcnemar_p": p}
            print(f"  {a} vs {b}: n={len(shared)}  {ab} / {ba}  p={p:.3f}")
    summary["_paired"] = paired
    summary["_sources"] = args.dirs
    if args.out:
        out = resolve_path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        write_json(out / "merged_summary.json", summary)
        print(f"[merge] wrote {out / 'merged_summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
