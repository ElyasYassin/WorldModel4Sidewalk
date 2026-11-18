"""Paired per-tile comparison of mission arms (same tiles, same strategies), e.g. the curb-aware held-out check.

  python scripts/compare_arms.py --arms base=outputs/missions_final3_base curb=outputs/missions_final3_curb ... \
      --policies walkover approach --out outputs/curb_aware_final3.md

Per arm and strategy: goal rate, mean time, recall (found / ground-truth upheavals, holes, puddles), false findings
per mission, and, against the first arm, the
per-tile false-finding difference (tiles with fewer / more false findings) with a two-sided paired sign-flip permutation p and a bootstrap 95 % CI of the mean.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from go2wm.common.config import resolve_path  # noqa: E402


def load(d: Path, pol: str):
    return {int(f.stem.split("_tile")[1]): json.loads(f.read_text()) for f in sorted(d.glob(f"{pol}_tile*.json"))}


def boot_ci(x, n=10000, seed=0):
    r = np.random.default_rng(seed)
    m = r.choice(x, (n, len(x))).mean(1)
    return np.percentile(m, [2.5, 97.5])


def perm_p(d, n=20000, seed=0):
    """Two-sided paired sign-flip permutation test of mean(d) = 0."""
    d = np.asarray(d, float)
    r = np.random.default_rng(seed)
    m = (r.choice([-1.0, 1.0], (n, len(d))) * d).mean(1)
    return float((np.sum(np.abs(m) >= abs(d.mean()) - 1e-12) + 1) / (n + 1))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--policies", nargs="+", default=["walkover", "approach"])
    ap.add_argument("--title", default="Paired arm comparison")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    arms = [(s.split("=", 1)[0], resolve_path(s.split("=", 1)[1])) for s in a.arms]
    L = [f"# {a.title}", "", "| strategy | arm | goal | time [s] | recall | found up / hole / puddle | false / mission | "
         "Δ false vs " + arms[0][0] + " [95 % CI] | tiles fewer / more false | perm. p |", "|" + "---|" * 10]
    for pol in a.policies:
        runs = {name: load(d, pol) for name, d in arms}
        tiles = sorted(set.intersection(*[set(r) for r in runs.values()]))
        ref = np.array([runs[arms[0][0]][t]["score"]["false_findings"] for t in tiles], float)
        for name, _ in arms:
            rr = [runs[name][t] for t in tiles]
            per = [x for r in rr for x in r["score"]["per_defect"] if x["type"] in ("upheaval", "hole", "puddle")]
            ft = {k: f"{sum(x['found'] for x in per if x['type'] == k)}/{sum(x['type'] == k for x in per)}"
                  for k in ("upheaval", "hole", "puddle")}
            fa = np.array([r["score"]["false_findings"] for r in rr], float)
            if name == arms[0][0]:
                dcol, pcol, bw = "–", "–", "–"
            else:
                d = fa - ref
                lo, hi = boot_ci(d)
                dcol = f"{d.mean():+.2f} [{lo:+.2f}, {hi:+.2f}]"
                bw = f"{int(np.sum(d < 0))} / {int(np.sum(d > 0))}"
                pcol = f"{perm_p(d):.3g}" if np.any(d != 0) else "1"
            L.append(f"| {pol} | {name} | {sum(r['outcome'] == 'goal' for r in rr)}/{len(rr)} | "
                     f"{np.mean([r['time_s'] for r in rr]):.1f} | {np.mean([x['found'] for x in per]) * 100:.0f} % | "
                     f"{ft['upheaval']} / {ft['hole']} / {ft['puddle']} | {fa.mean():.2f} | {dcol} | {bw} | {pcol} |")
    Path(a.out).write_text("\n".join(L) + "\n", encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
