"""How well can cracks be measured from the D555 colour camera (walking) and the D405 colour imagers (stand-off)?

Synthetic images (go2wm.sim.synthetic_camera: true-width crack texture, supersampled, blurred, noisy, 8-bit) of a
random crack (~0.5 m long, mostly across the walk) of width w on plain concrete, mid-panel on a 1.8 m walk (no centre
joint). The image is orthorectified with the known camera geometry and the crack is measured WITHOUT ground truth:
width = median over 5 cm tiles of (dark area / local length), length = sum of local lengths.

Reports per camera / distance / width: detection rate (largest dark structure covers >= 50 % of the true length),
median |width error| [mm], median |length error| [%], and agreement with the ADA 1/2 in (13 mm) opening limit.

  python scripts/crack_measurement_study.py --trials 8 --out outputs/crack_study
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from go2wm.common.config import resolve_path  # noqa: E402
from go2wm.scene.graph import ADA_OPENING_M  # noqa: E402
from go2wm.sim.defects import polyline_length  # noqa: E402
from go2wm.sim.synthetic_camera import CAMERAS, measure_crack, orthorectify, project, render  # noqa: E402

WIDTHS_MM = [1, 2, 3, 5, 8, 13, 20]
CASES = [("d555_rgb", 0.8), ("d555_rgb", 1.2), ("d555_rgb", 1.6), ("d555_rgb", 2.0), ("d405_rgb", 0.52)]
BASE_Z = 0.33
CX = 0.75  # crack centre: middle of a 1.5 m panel
# Blur (0.8 px), bilinear orthorectification and the dark threshold widen a crack by ~2.2 image pixels at every width
# (fitted on the 8-trial run, seeds 0-7, cases <= 1.2 m); calibrated width = raw - BIAS_PX * mm/px.
BIAS_PX = 2.2


def random_crack(rng, w):
    ang = np.pi / 2 + rng.normal(0, 0.35)
    pt = np.array([CX, -0.25 * np.sin(ang)])
    pts = [pt.copy()]
    for _ in range(10):
        ang += rng.normal(0, 0.25)
        pt = pt + 0.05 * np.array([np.cos(ang), np.sin(ang)])
        pts.append(pt.copy())
    pts = np.array(pts)
    pts -= pts.mean(0) - [CX, 0.0]
    # depth 0: appearance only (the synthetic camera has no occlusion, a narrow groove would mis-place hits)
    return {"type": "crack", "points": pts.round(4).tolist(), "width": w, "depth": 0.0}


def one(cam_name, dist, w, seed):
    rng = np.random.default_rng(seed)
    crack = random_crack(rng, w)
    terrain = {"width": 1.8, "size": [6.0, 4.0], "defects": [crack], "horizontal_scale": 0.02, "vertical_scale": 0.001}
    base = np.array([CX - dist, 0.0, BASE_Z])
    R = np.eye(3)
    pts = np.asarray(crack["points"])
    lo, hi = pts.min(0) - 0.08, pts.max(0) + 0.08
    corners = np.array([[x, y, 0.0] for x in (lo[0], hi[0]) for y in (lo[1], hi[1])])
    u, v = project(CAMERAS[cam_name], corners, base, R)
    cam = CAMERAS[cam_name]
    if not np.all(np.isfinite(u)):
        return None
    win = (int(max(0, np.floor(u.min()))), int(min(cam["width"], np.ceil(u.max()))),
           int(max(0, np.floor(v.min()))), int(min(cam["height"], np.ceil(v.max()))))
    if win[1] - win[0] < 5 or win[3] - win[2] < 5:
        return None
    img, _, mpp = render(cam_name, base, R, terrain, window=win, ss=2, seed=seed)
    px = float(np.nanmedian(mpp))
    res = max(0.0003, px / 1.5)
    ortho, _, _ = orthorectify(img, cam_name, base, R, terrain, win, (lo[0], hi[0]), (lo[1], hi[1]), res)
    m = measure_crack(ortho, res)
    true_len = polyline_length(pts)
    out = {"mm_per_px": px * 1000, "true_len_m": true_len}
    if m is None or m["length_m"] < 0.5 * true_len:
        return {**out, "detected": False}
    wc = max(m["width_m"] - BIAS_PX * px, 0.0)
    return {**out, "detected": True, "width_err_mm": abs(m["width_m"] - w) * 1000, "width_mm": m["width_m"] * 1000,
            "width_cal_mm": wc * 1000, "width_cal_err_mm": abs(wc - w) * 1000,
            "ada_cal_ok": (wc > ADA_OPENING_M) == (w > ADA_OPENING_M),
            "len_err_pct": abs(m["length_m"] - true_len) / true_len * 100,
            "ada_ok": (m["width_m"] > ADA_OPENING_M) == (w > ADA_OPENING_M)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=8)
    ap.add_argument("--out", default="outputs/crack_study")
    ap.add_argument("--seed-offset", type=int, default=0, help="e.g. 100: trials disjoint from the calibration run")
    a = ap.parse_args()
    out = resolve_path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows, t0 = [], time.time()
    for cam_name, dist in CASES:
        for w_mm in WIDTHS_MM:
            rs = [one(cam_name, dist, w_mm / 1000, 1000 * (k + a.seed_offset) + int(w_mm * 10)) for k in range(a.trials)]
            rs = [r for r in rs if r is not None]
            det = [r for r in rs if r["detected"]]
            rows.append({"camera": cam_name, "distance_m": dist, "width_mm": w_mm, "n": len(rs),
                         "mm_per_px": float(np.median([r["mm_per_px"] for r in rs])) if rs else None,
                         "detect_rate": len(det) / len(rs) if rs else None,
                         "width_err_mm": float(np.median([r["width_err_mm"] for r in det])) if det else None,
                         "width_est_mm": float(np.median([r["width_mm"] for r in det])) if det else None,
                         "len_err_pct": float(np.median([r["len_err_pct"] for r in det])) if det else None,
                         "ada_agree": float(np.mean([r["ada_ok"] for r in det])) if det else None,
                         "width_cal_mm": float(np.median([r["width_cal_mm"] for r in det])) if det else None,
                         "width_cal_err_mm": float(np.median([r["width_cal_err_mm"] for r in det])) if det else None,
                         "ada_cal_agree": float(np.mean([r["ada_cal_ok"] for r in det])) if det else None})
        print(f"[{cam_name} {dist} m] done ({time.time() - t0:.0f}s)", flush=True)
    (out / "crack_study.json").write_text(json.dumps(rows, indent=1))
    L = ["# Crack measurement from synthetic images (no lighting / glare / motion blur: optimistic)", "",
         "Cell: detected % · median width estimate [mm] (|error|) · median length error %", ""]
    cases = [f"{'D555 RGB' if c == 'd555_rgb' else 'D405 RGB'} {d:.2f} m ({next(r['mm_per_px'] for r in rows if r['camera'] == c and r['distance_m'] == d):.2f} mm/px)"
             for c, d in CASES]
    L += ["| crack width | " + " | ".join(cases) + " |", "|" + "---|" * (1 + len(CASES))]
    for w_mm in WIDTHS_MM:
        cells = []
        for c, d in CASES:
            r = next(r for r in rows if r["camera"] == c and r["distance_m"] == d and r["width_mm"] == w_mm)
            if not r["detect_rate"]:
                cells.append("0 % · –")
            else:
                cells.append(f"{r['detect_rate'] * 100:.0f} % · {r['width_est_mm']:.1f} ({r['width_err_mm']:.1f}) · {r['len_err_pct']:.0f} %")
        L.append(f"| {w_mm} mm | " + " | ".join(cells) + " |")
    L += ["", f"Calibrated width (raw - {BIAS_PX} px x mm/px): median estimate [mm] (|error|)", "",
          "| crack width | " + " | ".join(cases) + " |", "|" + "---|" * (1 + len(CASES))]
    for w_mm in WIDTHS_MM:
        cells = []
        for c, d in CASES:
            r = next(r for r in rows if r["camera"] == c and r["distance_m"] == d and r["width_mm"] == w_mm)
            cells.append("–" if not r["detect_rate"] else f"{r['width_cal_mm']:.1f} ({r['width_cal_err_mm']:.1f})")
        L.append(f"| {w_mm} mm | " + " | ".join(cells) + " |")
    (out / "crack_study.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
