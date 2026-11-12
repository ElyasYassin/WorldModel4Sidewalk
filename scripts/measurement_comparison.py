"""Phase 1: measured vs. true defect sizes across mission result sets (every defect, not medians only).

  python scripts/measurement_comparison.py --sets realistic=outputs/missions_final2 \
      pessimistic=outputs/missions_final2_pess ... --out outputs/measurement_comparison

Writes <out>/measured_vs_true.png (one panel per error setting: true size on x, measured on y, walk past vs stop &
look, ADA 6.4 / 13 mm lines), <out>/measured_vs_true.md (error statistics per setting, strategy and type: median,
90th percentile, max, ADA class agreement) and <out>/all_measurements.csv (one row per defect and mission).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from go2wm.common.config import resolve_path  # noqa: E402

ADA = (6.4, 13.0)
NAMES = {"walkover": "walk past", "approach": "stop & look (D405)", "slow": "slow down"}
COL = {"walkover": "#8a96a3", "approach": "#e8590c", "slow": "#2b8a3e"}
MARK = {"upheaval": "^", "hole": "o"}


def ada(v):
    v = abs(v)
    return 0 if v <= ADA[0] / 1000 else 1 if v <= ADA[1] / 1000 else 2


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", nargs="+", required=True, help="label=dir")
    ap.add_argument("--out", default="outputs/measurement_comparison")
    a = ap.parse_args()
    sets = [s.split("=", 1) for s in a.sets]
    out = resolve_path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for label, d in sets:
        for f in sorted(resolve_path(d).glob("*_tile*.json")):
            r = json.loads(f.read_text())
            for x in r["score"]["per_defect"]:
                if x["type"] not in ("upheaval", "hole"):
                    continue
                if r["policy"] not in ("walkover", "approach"):
                    continue
                rows.append({"setting": label, "strategy": r["policy"], "tile": r.get("tile"), "type": x["type"],
                             "true_mm": abs(x["true_value_m"]) * 1000, "found": x["found"],
                             "measured_mm": abs(x["value_m"]) * 1000 if x.get("value_m") is not None else None,
                             "sensor": x.get("best_sensor"), "status": x.get("status")})
    with open(out / "all_measurements.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    fig, axs = plt.subplots(1, len(sets), figsize=(5.2 * len(sets), 5.6), sharex=True, sharey=True)
    axs = np.atleast_1d(axs)
    lim = 155
    L = ["# Measured vs. true defect size (upheaval height, hole depth)", "",
         "Every found-and-measured defect on the fresh seed-300 layouts. Error = |measured| - |true|. "
         "ADA agreement: same class (<= 6.4 mm ok, <= 13 mm bevel, above: violation).", "",
         "| setting | strategy | type | measured / found / total | median abs err [mm] | p90 [mm] | max [mm] | "
         "within 2 mm | ADA class agrees |", "|" + "---|" * 9]
    for ax, (label, _) in zip(axs, sets):
        ax.plot([0, lim], [0, lim], color="#333", lw=0.8)
        ax.fill_between([0, lim], [-2, lim - 2], [2, lim + 2], color="#2b8a3e", alpha=0.12, lw=0)
        for v in ADA:
            ax.axhline(v, color="#c92a2a", lw=0.6, ls=":")
            ax.axvline(v, color="#c92a2a", lw=0.6, ls=":")
        for pol in ("walkover", "approach"):
            for typ in ("upheaval", "hole"):
                rr = [x for x in rows if x["setting"] == label and x["strategy"] == pol and x["type"] == typ]
                m = [x for x in rr if x["measured_mm"] is not None]
                if m:
                    ax.scatter([x["true_mm"] for x in m], [x["measured_mm"] for x in m], s=28, marker=MARK[typ],
                               facecolor=COL[pol] if pol == "approach" else "none", edgecolor=COL[pol], lw=1.2,
                               alpha=0.85, label=f"{typ}, {NAMES[pol]}")
                    e = np.array([x["measured_mm"] - x["true_mm"] for x in m])
                    ae = np.abs(e)
                    ok = np.mean([ada(x["measured_mm"] / 1000) == ada(x["true_mm"] / 1000) for x in m]) * 100
                    L.append(f"| {label} | {NAMES[pol]} | {typ} | {len(m)} / {sum(x['found'] for x in rr)} / {len(rr)} | "
                             f"{np.median(ae):.1f} | {np.percentile(ae, 90):.1f} | {ae.max():.1f} | "
                             f"{np.mean(ae <= 2) * 100:.0f} % | {ok:.0f} % |")
        ax.set_title(label, fontsize=12)
        ax.set_xlabel("true size [mm]")
        ax.set_xlim(0, lim)
        ax.set_ylim(0, 210)
        ax.grid(alpha=0.2)
    axs[0].set_ylabel("measured by the robot [mm]")
    axs[0].legend(fontsize=8, loc="upper left")
    fig.suptitle("Measured vs. true defect size: every defect, fresh test layouts (green band = within 2 mm, "
                 "red dotted = ADA 6.4 / 13 mm)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out / "measured_vs_true.png", dpi=130)
    # worst cases
    L += ["", "## Largest errors", "", "| setting | strategy | tile | type | true [mm] | measured [mm] | sensor |",
          "|" + "---|" * 7]
    m = sorted([x for x in rows if x["measured_mm"] is not None and x["strategy"] in ("walkover", "approach")], key=lambda x: -abs(x["measured_mm"] - x["true_mm"]))
    for x in m[:12]:
        L.append(f"| {x['setting']} | {NAMES[x['strategy']]} | {x['tile']} | {x['type']} | {x['true_mm']:.1f} | "
                 f"{x['measured_mm']:.1f} | {x['sensor']} |")
    (out / "measured_vs_true.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
