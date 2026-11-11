"""Seed-mean summary of a prediction_metrics.json with several seeds per model.

Models are grouped by run name with the trailing `_s<seed>` removed (e.g. `terrain_gru:e4c_l1_gru_s1` -> `e4c_l1_gru`).
Prints mean ± sd over seeds per horizon, and the per-segment (`by_segment_cmd`) gain over a reference group at
each horizon (seed-mean error minus the reference's seed-mean error, with the [min, max] over paired seeds).

  python scripts/summarize_seeds.py outputs/e4c/final_test/prediction_metrics.json --ref e4c_gru
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from go2wm.common.config import resolve_path  # noqa: E402


def group_name(model_key: str) -> tuple[str, int | None]:
    run = model_key.split(":", 1)[-1]
    m = re.match(r"(.*)_s(\d+)$", run)
    return (m.group(1), int(m.group(2))) if m else (run, None)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("metrics")
    ap.add_argument("--ref", default=None, help="reference group for segment gains")
    ap.add_argument("--horizons", nargs="+", default=["1.0", "2.0", "3.0"])
    ap.add_argument("--json", default=None, help="write the summary here")
    a = ap.parse_args()
    d = json.loads(resolve_path(a.metrics).read_text())
    groups: dict[str, dict] = defaultdict(dict)
    for k, v in d["models"].items():
        g, s = group_name(k)
        groups[g][s] = v["open_loop"]
    out: dict = {"metrics": a.metrics, "n_episodes": d.get("n_episodes"), "overall": {}, "segment_gain": {}}
    hs = a.horizons
    print(f"{a.metrics}  ({d.get('n_episodes')} episodes, {d.get('n_windows')} windows)\n")
    print("| model | n seeds | " + " | ".join(f"{h} s" for h in hs) + " |")
    print("|---|---|" + "---|" * len(hs))
    for g, seeds in groups.items():
        row = {}
        for h in hs:
            x = np.array([seeds[s][h]["pos_err_m_mean"] for s in seeds])
            row[h] = {"mean": float(x.mean()), "sd": float(x.std(ddof=1)) if len(x) > 1 else 0.0}
        out["overall"][g] = row
        cells = [f"{row[h]['mean']:.3f}" + (f" ± {row[h]['sd']:.3f}" if len(seeds) > 1 else "") for h in hs]
        print(f"| {g} | {len(seeds)} | " + " | ".join(cells) + " |")
    if a.ref and a.ref in groups:
        ref = groups[a.ref]
        segs = sorted({sg for s in ref for sg in ref[s][hs[-1]].get("by_segment_cmd", {})})
        for h in hs[1:]:
            print(f"\nGain over {a.ref} at {h} s by segment_cmd (seed-mean diff, m; [min, max] over paired seeds)\n")
            print("| model | " + " | ".join(segs) + " |")
            print("|---|" + "---|" * len(segs))
            for g, seeds in groups.items():
                if g == a.ref or len(seeds) < 2:
                    continue
                cells = []
                for sg in segs:
                    common = [s for s in seeds if s in ref and sg in seeds[s][h].get("by_segment_cmd", {})]
                    diffs = np.array([seeds[s][h]["by_segment_cmd"][sg]["pos_err_m_mean"]
                                      - ref[s][h]["by_segment_cmd"][sg]["pos_err_m_mean"] for s in common])
                    if not len(diffs):
                        cells.append("")
                        continue
                    out["segment_gain"].setdefault(h, {}).setdefault(g, {})[sg] = {
                        "mean": float(diffs.mean()), "min": float(diffs.min()), "max": float(diffs.max())}
                    cells.append(f"{diffs.mean():+.3f} [{diffs.min():+.3f}, {diffs.max():+.3f}]")
                print(f"| {g} | " + " | ".join(cells) + " |")
        n = {sg: ref[next(iter(ref))][hs[-1]]["by_segment_cmd"][sg].get("n_episodes") for sg in segs}
        print(f"\nepisodes per segment: {n}")
    if a.json:
        p = resolve_path(a.json)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
