"""How much of the < 1 cm sidewalk accuracy survives realistic sensor errors? (offline, no Isaac)

For recorded E4-C test episodes (robot path + tile), re-simulates the D555 and L1 with
go2wm.sim.offline_sensors under the profiles ideal / realistic / pessimistic (one random session per episode:
its own calibration error, depth-scale error and odometry drift) and measures, per sensor (and D555 + L1):

  single   one frame: p90 |height error| per distance band, share of cells < 1 cm   (heights relative to the base)
  fused    last 10 frames (1 s) registered with the robot's DRIFTING pose estimate, evaluated in the current
           robot frame: p90 |error| per band
  steps    joint step height measured in one frame (plane fit either side, as evaluate_sensors.py): p90 error,
           ADA class agreement, violation recall and false flags on flush joints

  python scripts/sensor_error_study.py --episodes 12 --out outputs/sensor_errors
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluate_sensors import EDGE_MARGIN, edge_mask, mesh_height, step_fit  # noqa: E402

from go2wm.common.config import resolve_path  # noqa: E402
from go2wm.data.episodes import load_episode  # noqa: E402
from go2wm.scene.graph import ada_level_class  # noqa: E402
from go2wm.sim.offline_sensors import PROFILES, simulate_path  # noqa: E402
from go2wm.sim.sensors import grid_centres  # noqa: E402
from go2wm.sim.sidewalk_texture import joint_positions  # noqa: E402
from go2wm.sim.terrain import params_with_defaults, sidewalk_height  # noqa: E402

BANDS = [(0.3, 0.8), (0.8, 1.5), (1.5, 2.5)]
GROUPS = {"calib": ("calib_pitch_deg", "calib_roll_deg", "calib_z_m"), "scale": ("depth_scale",),
          "odom": ("odom_xy", "odom_yaw", "odom_z", "odom_z_jitter"), "stereo": ("static_frac", "mixed_pixels"),
          "lidar": ("lidar_bias_m",)}


def resolve_profile(name: str) -> dict:
    """'realistic' or an ablation 'realistic-calib' (= realistic with that error group switched off)."""
    base, _, off = name.partition("-")
    prof = dict(PROFILES[base])
    for g in filter(None, off.split("+")):
        for k in GROUPS[g]:
            prof.pop(k, None)
    return prof
RES = 0.05
SETS = {"D555": ["depth_d555"], "L1": ["l1"], "D555+L1": ["depth_d555", "l1"]}


def merge(grids, names, t):
    """Count-weighted mean of several sensors' grids at frame t."""
    num, den = 0.0, 0.0
    for n in names:
        e, c = grids[n]["elev"][t], grids[n]["count"][t].astype(float)
        num = num + np.where(np.isfinite(e), e, 0) * c
        den = den + np.where(np.isfinite(e), c, 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / np.maximum(den, 1e-9), np.nan), den


def analyse(ep, profile, seed, stride=5, K=10, t0=20):
    terrain = params_with_defaults(ep.meta["scene"]["params"]["terrain"])
    H = lambda x, y: mesh_height(x, y, terrain)  # noqa: E731  (what Isaac meshes)
    S = ep["states"].copy()
    # recorded base z is in the env frame; the surface (and the sensor grids Isaac recorded) are in the tile frame,
    # offset by a per-tile constant (see E4-C (2)): shift the base into the surface frame
    pose0 = S[:, :2]
    S[:, 2] += float(np.nanmedian(sidewalk_height(pose0[:, 0], pose0[:, 1], terrain) - ep["gt_ground_z_m"]))
    T = len(S)
    eval_frames = list(range(t0, T, stride))
    need = sorted({u for t in eval_frames for u in range(max(0, t - K + 1), t + 1)})
    sim = simulate_path(S, H, ["depth_d555", "l1"], resolve_profile(profile), seed, frames=need)
    g, pose, est, est_z, true_z = sim["grids"], sim["pose"], sim["est_pose"], sim["est_z"], sim["true_z"]
    cx, cy = grid_centres()
    CX, CY = np.meshgrid(cx, cy, indexing="ij")
    half = 0.5 * terrain["width"]
    out = {k: {"single": [], "fused": [], "steps": []} for k in SETS}

    def to_world(p, xr, yr):
        c, s = np.cos(p[2]), np.sin(p[2])
        return p[0] + c * xr - s * yr, p[1] + s * xr + c * yr

    def gt_rel(t, xr, yr):  # true surface height relative to the true base, at robot-frame (xr, yr)
        X, Y = to_world(pose[t], xr, yr)
        return H(X, Y) - true_z[t], X, Y

    joints = joint_positions(terrain, -1.0, 14.0)
    for name, sens in SETS.items():
        for t in eval_frames:
            E, C = merge(g, sens, t)
            ref, X, Y = gt_rel(t, CX, CY)
            em = edge_mask(X, Y, terrain)
            hit = np.isfinite(E) & ~em & (CX >= 0.3)
            out[name]["single"].append(np.stack([CX[hit], E[hit] - ref[hit]], 1))
            # fused: cells of frames t-K+1..t moved into frame t with the ESTIMATED poses and base heights
            keys, vals, ws = [], [], []
            for u in range(max(0, t - K + 1), t + 1):
                Eu, Cu = merge(g, sens, u)
                h = np.isfinite(Eu)
                wx, wy = to_world(est[u], CX[h], CY[h])  # where the robot thinks frame u's cells are
                c, s = np.cos(est[t, 2]), np.sin(est[t, 2])
                dx, dy = wx - est[t, 0], wy - est[t, 1]
                xr, yr = c * dx + s * dy, -s * dx + c * dy  # ... in its current frame
                keys.append(np.floor(xr / RES).astype(np.int64) * 10000 + np.floor(yr / RES).astype(np.int64))
                vals.append(Eu[h] + est_z[u] - est_z[t])
                ws.append(Cu[h])
            k, v, w = np.concatenate(keys), np.concatenate(vals), np.concatenate(ws)
            uk, inv = np.unique(k, return_inverse=True)
            fz = np.bincount(inv, w * v) / np.bincount(inv, w)
            kx = np.floor_divide(uk + 5000, 10000)
            ky = uk - kx * 10000
            xr, yr = (kx + 0.5) * RES, (ky + 0.5) * RES
            ref, X, Y = gt_rel(t, xr, yr)
            keep = (xr >= 0.3) & ~edge_mask(X, Y, terrain)
            out[name]["fused"].append(np.stack([xr[keep], fz[keep] - ref[keep]], 1))
            # steps at joints, single frame
            Xw, Yw = to_world(pose[t], CX, CY)
            for j in joints:
                d = (j - pose[t, 0]) / max(np.cos(pose[t, 2]), 0.5)
                if not (0.5 <= d <= 2.5):
                    continue
                lat = (np.abs(Yw) <= half - 0.1) & (np.abs(Yw - pose[t, 1]) <= 0.6)
                other = np.zeros(Xw.shape, bool)
                for x0, length, hh in terrain["slabs"]:
                    for xe in (x0, x0 + length):
                        if abs(xe - j) > 1e-6 and abs(hh) > 1e-4:
                            other |= np.abs(Xw - xe) < EDGE_MARGIN
                base = np.isfinite(E) & lat & ~other
                after = base & (Xw >= j + EDGE_MARGIN) & (Xw <= j + 0.4)
                before = base & (Xw <= j - EDGE_MARGIN) & (Xw >= j - 0.4)
                if after.sum() < 8 or before.sum() < 8:
                    continue
                m = after | before
                est_step = step_fit(Xw[m], Yw[m], E[m], np.maximum(C[m], 1), j)
                yc = float(np.clip(pose[t, 1], -half + 0.05, half - 0.05))
                true = float(sidewalk_height(j + 1e-3, yc, terrain) - sidewalk_height(j - 1e-3, yc, terrain))
                out[name]["steps"].append((d, est_step, true))
    return out, sim["session"]


def summarise(recs):
    res = {}
    for name in SETS:
        r = {}
        for kind in ("single", "fused"):
            a = np.concatenate([x for rec in recs for x in rec[name][kind]]) if recs else np.zeros((0, 2))
            r[kind] = {f"{lo}-{hi} m": {"p90_cm": float(np.percentile(np.abs(a[(a[:, 0] >= lo) & (a[:, 0] < hi), 1]), 90) * 100),
                                        "median_cm": float(np.median(a[(a[:, 0] >= lo) & (a[:, 0] < hi), 1]) * 100),
                                        "under_1cm": float(np.mean(np.abs(a[(a[:, 0] >= lo) & (a[:, 0] < hi), 1]) < 0.01))}
                       for lo, hi in BANDS if np.any((a[:, 0] >= lo) & (a[:, 0] < hi))}
        st = np.array([s for rec in recs for s in rec[name]["steps"]]).reshape(-1, 3)
        if len(st):
            err = np.abs(st[:, 1] - st[:, 2])
            cls_est = np.array([ada_level_class(v) for v in st[:, 1]])
            cls_true = np.array([ada_level_class(v) for v in st[:, 2]])
            viol = cls_true == "violation"
            flush = np.abs(st[:, 2]) < 1e-3
            r["steps"] = {"n": int(len(st)), "p90_err_mm": float(np.percentile(err, 90) * 1000),
                          "ada_agree": float(np.mean(cls_est == cls_true)),
                          "violation_recall": float(np.mean(cls_est[viol] == "violation")) if viol.any() else None,
                          "false_flag_flush": float(np.mean(cls_est[flush] != "ok")) if flush.any() else None}
        res[name] = r
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/go2_e4c_test")
    ap.add_argument("--episodes", type=int, default=12)
    ap.add_argument("--profiles", nargs="+", default=list(PROFILES))
    ap.add_argument("--out", default="outputs/sensor_errors")
    ap.add_argument("--stride", type=int, default=20, help="evaluate every n-th frame (each with its 10-frame window)")
    a = ap.parse_args()
    files = sorted(glob.glob(str(resolve_path(a.data) / "episodes" / "*.npz")))
    files = files[:: max(1, len(files) // a.episodes)][: a.episodes]
    eps = [load_episode(f) for f in files]
    out = resolve_path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    results = {"data": a.data, "episodes": [e.episode_id for e in eps], "profiles": {k: resolve_profile(k) for k in a.profiles},
               "results": {}, "sessions": {}}
    for prof in a.profiles:
        t0 = time.time()
        recs, sess = [], []
        for i, ep in enumerate(eps):
            r, s = analyse(ep, prof, seed=1000 + i, stride=a.stride)
            recs.append(r)
            sess.append(s)
        results["results"][prof] = summarise(recs)
        results["sessions"][prof] = sess
        print(f"[{prof}] {len(eps)} episodes in {time.time() - t0:.0f}s", flush=True)
        (out / "sensor_error_study.json").write_text(json.dumps(results, indent=1, default=float))
    write_table(results, out / "sensor_error_study.md")
    print((out / "sensor_error_study.md").read_text(encoding="utf-8"))
    return 0


def write_table(res, path):
    L = [f"# Sensor accuracy under realistic errors ({len(res['episodes'])} episodes of {res['data']})", ""]
    L += ["p90 |height error| [cm] (share of cells < 1 cm) by distance ahead; fused = 1 s of frames registered with "
          "the drifting pose estimate", ""]
    L += ["| profile | sensors | single 0.3-0.8 | single 0.8-1.5 | single 1.5-2.5 | fused 0.3-0.8 | fused 0.8-1.5 | "
          "fused 1.5-2.5 | step p90 err [mm] | ADA agree | violation recall | false flags (flush) |", "|" + "---|" * 12]
    for prof, rr in res["results"].items():
        for name, r in rr.items():
            def cell(kind, band):
                b = r[kind].get(band)
                return f"{b['p90_cm']:.2f} ({b['under_1cm'] * 100:.0f} %)" if b else "–"
            st = r.get("steps") or {}
            fmt = lambda v, pc=True: "–" if v is None else (f"{v * 100:.0f} %" if pc else f"{v:.1f}")  # noqa: E731
            L.append(f"| {prof} | {name} | " + " | ".join(cell(k, f"{lo}-{hi} m") for k in ("single", "fused") for lo, hi in BANDS)
                     + f" | {fmt(st.get('p90_err_mm'), False)} | {fmt(st.get('ada_agree'))} | {fmt(st.get('violation_recall'))} | "
                       f"{fmt(st.get('false_flag_flush'))} |")
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
