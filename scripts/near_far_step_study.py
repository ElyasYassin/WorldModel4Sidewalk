"""'D555 finds, D405 measures': how well does each sensor measure an ADA-scale step? (offline, no Isaac)

A transverse step of height h (3 to 50 mm, chosen inside the ADA classes, not on their 6.4 / 13 mm boundaries) lies across flat ground, d metres ahead of the robot base. Each trial is a
fresh sensor session (its own calibration, depth-scale error and fixed-pattern noise under the chosen profile) and
measures the step from RAW points (no grid): one plane is fitted on each side of the joint (x in [j - 0.20, j - 0.02]
and [j + 0.02, j + 0.20], |y| < 0.25 m) and the step is their difference at the joint. A dwell of N frames averages
N independent frames from the same stand-off (same session, so systematic errors stay).

  D555 while walking: joint at 0.8, 1.2, 1.6, 2.0 m;  D405 at its stand-off: joint at 0.53 m (centre of its view)

Reports per sensor / distance / profile: |error| p50 / p90 [mm], ADA class agreement (<= 6.4 mm ok, <= 13 mm bevel,
> 13 mm violation), violation recall and false flags on steps that are OK.

  python scripts/near_far_step_study.py --trials 60 --out outputs/near_far_steps
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from go2wm.common.config import resolve_path  # noqa: E402
from go2wm.scene.graph import ada_level_class  # noqa: E402
from go2wm.sim.offline_sensors import PROFILES, OfflineRig  # noqa: E402

HEIGHTS_MM = [3, 5, 9, 11, 18, 50]  # inside the ADA classes (ok, ok, bevel, bevel, violation, violation)
BASE_Z = 0.33
CASES = [("depth_d555", d, 1) for d in (0.8, 1.2, 1.6, 2.0)] + [("depth_d405", 0.53, 1), ("depth_d405", 0.53, 5)]


def measure(pts: np.ndarray, j: float) -> float | None:
    """Step at x = j from base-frame points: plane either side, difference at (j, 0)."""
    p = pts[np.all(np.isfinite(pts), axis=1)]
    p = p[np.abs(p[:, 1]) < 0.25]
    z_at = []
    for lo, hi in ((j - 0.20, j - 0.02), (j + 0.02, j + 0.20)):
        q = p[(p[:, 0] >= lo) & (p[:, 0] <= hi)]
        if len(q) < 12:
            return None
        A = np.c_[np.ones(len(q)), q[:, 0] - j, q[:, 1]]
        coef, *_ = np.linalg.lstsq(A, q[:, 2], rcond=None)
        z_at.append(coef[0])
    return float(z_at[1] - z_at[0])


def run(profile: str, trials: int, seed: int) -> list[dict]:
    rows = []
    for h_mm in HEIGHTS_MM:
        h = h_mm / 1000
        for sensor, d, dwell in CASES:
            errs, est_cls, true_cls = [], [], []
            for k in range(trials):
                rig = OfflineRig([sensor], profile, seed=seed + 7919 * k + int(h_mm * 10) + int(d * 100) + dwell)
                H = lambda x, y, d=d, h=h: np.where(np.asarray(x) > d, h, 0.0)  # noqa: E731 (base at x = 0)
                vals = [measure(rig.frame(sensor, np.array([0.0, 0.0, BASE_Z]), np.eye(3), H), d) for _ in range(dwell)]
                vals = [v for v in vals if v is not None]
                if not vals:
                    continue
                m = float(np.mean(vals))
                errs.append(abs(m - h))
                est_cls.append(ada_level_class(m))
                true_cls.append(ada_level_class(h))
            e = np.array(errs) * 1000
            rows.append({"profile": profile, "height_mm": h_mm, "sensor": sensor, "distance_m": d, "dwell": dwell,
                         "n": len(e), "p50_mm": float(np.median(e)) if len(e) else None,
                         "p90_mm": float(np.percentile(e, 90)) if len(e) else None,
                         "ada_agree": float(np.mean(np.array(est_cls) == np.array(true_cls))) if len(e) else None,
                         "true_class": ada_level_class(h)})
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=60)
    ap.add_argument("--profiles", nargs="+", default=list(PROFILES))
    ap.add_argument("--out", default="outputs/near_far_steps")
    a = ap.parse_args()
    out = resolve_path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for prof in a.profiles:
        rows += run(prof, a.trials, seed=11)
        print(f"[{prof}] done", flush=True)
    (out / "near_far_steps.json").write_text(json.dumps(rows, indent=1))
    L = ["# Step-height measurement: D555 while walking vs D405 at stand-off", "",
         f"{a.trials} sensor sessions per cell (own calibration / scale / pattern errors). Cell = |error| p90 [mm] "
         "/ ADA class agreement.", ""]
    cases = [(s, d, w) for s, d, w in CASES]
    hdr = " | ".join(f"{'D555' if s == 'depth_d555' else 'D405'} {d:.2f} m" + (f" x{w}" if w > 1 else "") for s, d, w in cases)
    for prof in a.profiles:
        L += [f"## {prof}", "", f"| step | true ADA class | {hdr} |", "|" + "---|" * (2 + len(cases))]
        for h_mm in HEIGHTS_MM:
            cells = []
            for s, d, w in cases:
                r = next(r for r in rows if r["profile"] == prof and r["height_mm"] == h_mm and r["sensor"] == s
                         and r["distance_m"] == d and r["dwell"] == w)
                cells.append("–" if r["p90_mm"] is None else f"{r['p90_mm']:.1f} / {r['ada_agree'] * 100:.0f} %")
            L.append(f"| {h_mm} mm | {ada_level_class(h_mm / 1000)} | " + " | ".join(cells) + " |")
        L.append("")
    (out / "near_far_steps.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
