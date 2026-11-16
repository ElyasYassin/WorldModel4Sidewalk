"""Go2 vs. Go2-W (or any set of Isaac mission runs): one markdown table per tile.

  python scripts/robot_comparison.py --runs "Go2 stop & look=D:/.../approach_robust_tile0.json" \
      "Go2-W stop & look + crouch=D:/.../approach_posture_tile0.json" ... --out outputs/go2w/comparison.md

Per run: outcome, mission time, path, close inspections, crouches done / refused / aborted, yields, false findings,
and every ground-truth defect's measured value against the true size (upheaval height / hole depth), with median and
worst absolute error over the measured upheavals and holes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="label=mission.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="Isaac physics missions: robot / strategy comparison")
    a = ap.parse_args()
    runs = []
    for spec in a.runs:
        lab, f = spec.split("=", 1)
        runs.append((lab, json.loads(Path(f).read_text())))
    L = [f"# {a.title}", "",
         "Realistic sensor errors + odometry drift + camera latency unless noted. Errors in mm on upheaval height and "
         "hole depth; puddles are flagged (no depth), cracks are not scored.", "",
         "| run | outcome | time [s] | path [m] | inspections | crouch done / refused / aborted | yields | "
         "found | median / worst error [mm] | false findings |", "|" + "---|" * 10]
    for lab, r in runs:
        ev = r["events"]
        sc = r["score"]
        n_insp = sum(e["event"] == "inspected" for e in ev)
        n_pose = sum(e["event"] == "posture_stand" for e in ev)
        n_skip = sum(e["event"] == "posture_skipped" for e in ev)
        n_abort = max(0, int(r.get("posture_aborts", 0)) - n_skip)
        n_yield = sum(e["event"] == "yield_pedestrian" for e in ev)
        errs = [abs(abs(x["value_m"]) - abs(x["true_value_m"])) * 1000 for x in sc["per_defect"]
                if x.get("value_m") is not None and x["type"] in ("upheaval", "hole")]
        found = sum(x["found"] for x in sc["per_defect"])
        L.append(f"| {lab} | {r['outcome']} | {r['time_s']:.1f} | {r['path_m']:.1f} | {n_insp} | "
                 + (f"{n_pose} / {n_skip} / {n_abort}" if r.get("posture") else "-") + f" | {n_yield} | "
                 f"{found}/{len(sc['per_defect'])} | "
                 + (f"{np.median(errs):.1f} / {max(errs):.1f}" if errs else "-") + f" | {sc['false_findings']} |")
    L += ["", "## Every defect: measured (error) [mm]", ""]
    first = runs[0][1]["score"]["per_defect"]
    L.append("| defect | true | " + " | ".join(lab for lab, _ in runs) + " |")
    L.append("|" + "---|" * (2 + len(runs)))
    for i, d in enumerate(first):
        tv = abs(d["true_value_m"]) * 1000
        cells = []
        for _, r in runs:
            x = r["score"]["per_defect"][i]
            if x.get("value_m") is not None:
                v = abs(x["value_m"]) * 1000
                cells.append(f"{v:.1f} ({v - tv:+.1f}) {(x.get('best_sensor') or '')[-4:].upper()}")
            else:
                cells.append("found" if x["found"] else "**missed**")
        L.append(f"| {d['type']} | {tv:.1f} | " + " | ".join(cells) + " |")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
