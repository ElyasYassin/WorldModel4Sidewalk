"""Summarize the scanned-sidewalk seed runs (scripts/scan_seeds_queue.ps1) per arm.

  python scripts/summarize_scan_seeds.py [--dir D:/go2wm/outputs/scan_missions/seeds]

Per arm: missions, goal reached, mean time, scan fault found (and how often measured), measured value and error
against the scan reference (13.4 mm), close inspections, false findings. Writes <dir>/summary.md.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default="D:/go2wm/outputs/scan_missions/seeds")
    a = ap.parse_args(argv)
    d = Path(a.dir)
    arms = defaultdict(list)
    for f in sorted(d.glob("*_tile0_rescored.json")):
        m = re.search(r"_(look_v2|look_base|walk_v2)_s(\d+)_tile0", f.name)
        if m:
            arms[m.group(1)].append(json.loads(f.read_text()))
    rows = ["| arm | missions | goal | mean time [s] | fault found | measured | value [mm] median (range) | "
            "abs error [mm] median / max | stops | false findings (total) |", "|" + "---|" * 10]
    for arm in ("look_v2", "look_base", "walk_v2"):
        rr = arms.get(arm, [])
        if not rr:
            continue
        pd = [r["score"]["per_defect"][0] for r in rr]
        vals = [abs(x["value_m"]) * 1000 for x in pd if x.get("value_m") is not None]
        errs = [x["abs_err_m"] * 1000 for x in pd if "abs_err_m" in x]
        stops = sum(sum(e["event"] == "inspected" for e in r["events"]) for r in rr)
        rows.append(
            f"| {arm} | {len(rr)} | {sum(r['outcome'] == 'goal' for r in rr)} | {np.mean([r['time_s'] for r in rr]):.1f} | "
            f"{sum(x['found'] for x in pd)}/{len(pd)} | {len(vals)}/{len(pd)} | "
            + (f"{np.median(vals):.1f} ({min(vals):.1f}-{max(vals):.1f})" if vals else "-") + " | "
            + (f"{np.median(errs):.1f} / {max(errs):.1f}" if errs else "-")
            + f" | {stops} | {sum(r['score']['false_findings'] for r in rr)} |")
    ref = next((r.get("score_reference") for rr in arms.values() for r in rr), "")
    text = "\n".join(["# Scanned sidewalk, Go2-W in Isaac Lab, repeated over seeds", "", f"Reference: {ref}", ""] + rows)
    (d / "summary.md").write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
