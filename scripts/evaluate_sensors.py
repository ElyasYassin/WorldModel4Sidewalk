"""E4-C: which exteroceptive sensor measures the sidewalk well enough (target: < 1 cm height error)?

Per sensor (and the D555 + L1 union), on recorded E4-C episodes:

  1. single-frame elevation accuracy vs distance ahead: coverage, |error| p50/p90, share of cells under
     1 cm, hits per cell, and the "1 cm range" (farthest distance up to which every 10 cm bin has
     p90 |error| < 1 cm);
  2. multi-frame fusion over the last K frames (0.1 s each) with perfect registration and with an
     independent per-frame vertical registration error sigma_reg (sensitivity, not an odometry model);
  3. step height at every transverse joint (slab edges and panel joints, 0-step joints included), measured
     within ONE frame by fitting a plane on each side of the joint (so slope and common offsets cancel):
     error, and ADA class agreement (<= 6.4 mm ok, <= 13 mm bevel, > 13 mm violation).

Ground truth is the Isaac terrain mesh the rays hit: the tile height field (5 cm grid, heights rounded to
5 mm) interpolated bilinearly, averaged over 3 x 3 points per 5 cm cell. Cells within 7.5 cm of a height
discontinuity (slab edge, curb) are excluded from (1)-(2) because a 5 cm cell there mixes two levels.
Robot poses are simulator ground truth.

  python scripts/evaluate_sensors.py --data data/go2_e4c_test --out outputs/e4c/test
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from go2wm.common.config import resolve_path  # noqa: E402
from go2wm.data.episodes import RANGE_SENSORS  # noqa: E402
from go2wm.scene.graph import ada_level_class  # noqa: E402
from go2wm.sim.sensors import ELEV_GRID, grid_centres  # noqa: E402
from go2wm.sim.sidewalk_texture import joint_positions  # noqa: E402
from go2wm.sim.terrain import params_with_defaults, sidewalk_height  # noqa: E402

SENSORS = list(RANGE_SENSORS) + ["d555+l1"]
BANDS = [(0.3, 0.8), (0.8, 1.5), (1.5, 2.5), (2.5, 3.5)]
BIN_EDGES = np.round(np.arange(0.0, 3.5001, 0.1), 3)
RES = ELEV_GRID["res"]
EDGE_MARGIN = 0.075
TOL = 0.01  # the requirement: 1 cm
FUSE_K = [1, 5, 10, 20]
SIGMA_REG = [0.0, 0.003, 0.005]


def mesh_height(X, Y, p):
    """Bilinear interpolation of the quantized tile height field (scene frame)."""
    hs, vs = p["horizontal_scale"], p["vertical_scale"]
    sx, sy = p["size"]
    fx, fy = (np.asarray(X) + 0.5 * sx) / hs, (np.asarray(Y) + 0.5 * sy) / hs
    i0, j0 = np.floor(fx), np.floor(fy)
    tx, ty = fx - i0, fy - j0
    q = lambda i, j: np.round(sidewalk_height(i * hs - 0.5 * sx, j * hs - 0.5 * sy, p) / vs) * vs  # noqa: E731
    return ((1 - tx) * (1 - ty) * q(i0, j0) + tx * (1 - ty) * q(i0 + 1, j0) + (1 - tx) * ty * q(i0, j0 + 1)
            + tx * ty * q(i0 + 1, j0 + 1))


def cell_gt(X, Y, p):
    o = np.array([-1, 0, 1]) * RES / 3
    return np.mean([mesh_height(X + a, Y + b, p) for a in o for b in o], axis=0)


def edge_mask(X, Y, p):
    m = np.zeros(np.shape(X), bool)
    for x0, length, h in p["slabs"]:
        if abs(h) > 1e-4:
            m |= (np.abs(X - x0) < EDGE_MARGIN) | (np.abs(X - x0 - length) < EDGE_MARGIN)
    if p["curb_drop"] > 0:
        m |= np.abs(Y + 0.5 * p["width"]) < EDGE_MARGIN
    return m


def yaw_of(q):  # (w, x, y, z)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def load_episode(npz):
    z = np.load(npz)
    meta = json.load(open(npz[:-4] + ".json"))
    p = params_with_defaults(meta["scene"]["params"]["terrain"])
    st = z["states"]
    pose = np.stack([st[:, 0], st[:, 1], yaw_of(st[:, 3:7])], 1)
    grids = {}
    for s in RANGE_SENSORS:
        grids[s] = (z[f"sensor_{s}_elev_m"].astype(np.float64), z[f"sensor_{s}_count"].astype(np.float64))
    e1, c1 = grids["depth_d555"]
    e2, c2 = grids["l1"]
    c = c1 + c2
    with np.errstate(invalid="ignore", divide="ignore"):
        e = (np.nan_to_num(e1) * c1 + np.nan_to_num(e2) * c2) / c
    grids["d555+l1"] = (np.where(c > 0, e, np.nan), c)
    return meta, p, pose, grids


def frame_world(pose_t, cx, cy):
    c, s = np.cos(pose_t[2]), np.sin(pose_t[2])
    CX, CY = np.meshgrid(cx, cy, indexing="ij")
    return pose_t[0] + c * CX - s * CY, pose_t[1] + s * CX + c * CY, CX, CY


def step_fit(X, Y, Z, W, j):
    """Step at x = j from one weighted least-squares fit over both sides of the joint:
    z = a + b (x - j) + c y + h [x > j] (shared running and cross slope); returns h."""
    A = np.stack([np.ones_like(X), X - j, Y, (X > j).astype(np.float64)], 1) * np.sqrt(W)[:, None]
    sol, *_ = np.linalg.lstsq(A, Z * np.sqrt(W), rcond=None)
    return sol[3]


def analyse_episode(npz, frame_stride=5, fuse_stride=10, t0=20, seed=0):
    meta, p, pose, grids = load_episode(npz)
    cx, cy = grid_centres()
    T = len(pose)
    half = 0.5 * p["width"]
    rng = np.random.default_rng(seed)
    out = {"episode": meta["episode_id"], "single": {}, "fused": {}, "steps": {}}

    # per-frame geometry (shared by all sensors)
    geo = {}
    for t in range(T):
        X, Y, CX, CY = frame_world(pose[t], cx, cy)
        geo[t] = (X, Y, CX, edge_mask(X, Y, p), None)
    gt_cache = {}

    def gt(t):
        if t not in gt_cache:
            X, Y = geo[t][0], geo[t][1]
            gt_cache[t] = cell_gt(X, Y, p)
        return gt_cache[t]

    single_frames = list(range(t0, T, frame_stride))
    fuse_frames = list(range(t0, T, fuse_stride))
    joints = [j for j in joint_positions(p, -1.0, 14.0)]

    for s in SENSORS:
        E, C = grids[s]
        # 1. single frame: per-cell records (x_r, err, count) of hit cells + per-bin cell totals
        xr_l, err_l, cnt_l = [], [], []
        tot_bins = np.zeros(len(BIN_EDGES) - 1)
        for t in single_frames:
            X, Y, CX, em, _ = geo[t]
            ok = ~em
            tot_bins += np.histogram(CX[ok], BIN_EDGES)[0]
            hit = ok & (C[t] > 0) & np.isfinite(E[t])
            err = E[t][hit] - gt(t)[hit]
            xr_l.append(CX[hit]); err_l.append(err); cnt_l.append(C[t][hit])
        out["single"][s] = {"xr": np.concatenate(xr_l), "err": np.concatenate(err_l), "cnt": np.concatenate(cnt_l),
                            "tot_bins": tot_bins}

        # 2. fusion over the last K frames (world 5 cm cells keyed in the scene frame)
        fused = {}
        reg = {sig: rng.normal(size=T) * sig for sig in SIGMA_REG}
        for K in FUSE_K:
            for sig in SIGMA_REG:
                if K == 1 and sig > 0:
                    continue
                xr_f, err_f, n_f, cov = [], [], [], np.zeros(len(BANDS))
                for t in fuse_frames:
                    keys, errs, ws = [], [], []
                    for u in range(max(0, t - K + 1), t + 1):
                        X, Y, CX, em, _ = geo[u]
                        hit = ~em & (C[u] > 0) & np.isfinite(E[u])
                        if not hit.any():
                            continue
                        e = E[u][hit] - gt(u)[hit] + reg[sig][u]
                        k = np.floor(X[hit] / RES).astype(np.int64) * 100000 + np.floor(Y[hit] / RES).astype(np.int64)
                        keys.append(k); errs.append(e); ws.append(C[u][hit])
                    if not keys:
                        continue
                    k, e, w = np.concatenate(keys), np.concatenate(errs), np.concatenate(ws)
                    uk, inv = np.unique(k, return_inverse=True)
                    sw = np.bincount(inv, w)
                    fe = np.bincount(inv, w * e) / sw
                    kx, ky = np.floor_divide(uk + 50000, 100000), None
                    ky = uk - kx * 100000
                    wx, wy = (kx + 0.5) * RES, (ky + 0.5) * RES
                    c, s_ = np.cos(pose[t, 2]), np.sin(pose[t, 2])
                    dx, dy = wx - pose[t, 0], wy - pose[t, 1]
                    xr, yr = c * dx + s_ * dy, -s_ * dx + c * dy
                    inside = (np.abs(yr) <= ELEV_GRID["y1"]) & (xr >= 0)
                    xr_f.append(xr[inside]); err_f.append(fe[inside]); n_f.append(sw[inside])
                    for b, (a, bb) in enumerate(BANDS):
                        cov[b] += np.sum(inside & (xr >= a) & (xr < bb))
                denom = np.array([(bb - a) * (ELEV_GRID["y1"] - ELEV_GRID["y0"]) / RES**2 for a, bb in BANDS]) * len(fuse_frames)
                fused[f"K{K}_reg{sig * 1000:.0f}mm"] = {"xr": np.concatenate(xr_f) if xr_f else np.zeros(0),
                                                        "err": np.concatenate(err_f) if err_f else np.zeros(0),
                                                        "n": np.concatenate(n_f) if n_f else np.zeros(0),
                                                        "cov_cells": cov, "cov_denom": denom}
        out["fused"][s] = fused

        # 3. step height at joints, single frame
        rows = []
        for t in single_frames:
            X, Y, CX, em, _ = geo[t]
            hit = (C[t] > 0) & np.isfinite(E[t])
            if not hit.any():
                continue
            g = gt(t)
            for j in joints:
                d = (j - pose[t, 0]) / max(np.cos(pose[t, 2]), 0.5)
                if not (0.5 <= d <= 3.2):
                    continue
                lat = (np.abs(Y) <= half - 0.1) & (np.abs(Y - pose[t, 1]) <= 0.6)
                other = np.zeros_like(em)
                for x0, length, h in p["slabs"]:
                    for xe in (x0, x0 + length):
                        if abs(xe - j) > 1e-6 and abs(h) > 1e-4:
                            other |= np.abs(X - xe) < EDGE_MARGIN
                base = hit & lat & ~other
                after = base & (X >= j + EDGE_MARGIN) & (X <= j + 0.4)
                before = base & (X <= j - EDGE_MARGIN) & (X >= j - 0.4)
                if after.sum() < 8 or before.sum() < 8:
                    continue
                m = after | before
                est = step_fit(X[m], Y[m], E[t][m], C[t][m], j)
                ref = step_fit(X[m], Y[m], g[m], C[t][m], j)
                yc = float(np.clip(pose[t, 1], -half + 0.05, half - 0.05))
                true = float(sidewalk_height(j + 1e-3, yc, p) - sidewalk_height(j - 1e-3, yc, p))
                rows.append((d, est, ref, true))
        out["steps"][s] = np.array(rows) if rows else np.zeros((0, 4))
    return out


def pct(a, q):
    return float(np.percentile(a, q)) if len(a) else float("nan")


def summarise_single(recs, sensor):
    xr = np.concatenate([r["single"][sensor]["xr"] for r in recs])
    err = np.abs(np.concatenate([r["single"][sensor]["err"] for r in recs]))
    cnt = np.concatenate([r["single"][sensor]["cnt"] for r in recs])
    tot = np.sum([r["single"][sensor]["tot_bins"] for r in recs], axis=0)
    bands = []
    for a, b in BANDS:
        m = (xr >= a) & (xr < b)
        tb = tot[(BIN_EDGES[:-1] >= a - 1e-9) & (BIN_EDGES[1:] <= b + 1e-9)].sum()
        bands.append({"band_m": [a, b], "coverage": float(m.sum() / max(tb, 1)), "p50_cm": 100 * pct(err[m], 50),
                      "p90_cm": 100 * pct(err[m], 90), "under_1cm": float(np.mean(err[m] < TOL)) if m.any() else float("nan"),
                      "hits_per_cell": float(np.mean(cnt[m])) if m.any() else float("nan"), "n_cells": int(m.sum())})
    curve = []
    for a, b, tb in zip(BIN_EDGES[:-1], BIN_EDGES[1:], tot):
        m = (xr >= a) & (xr < b)
        curve.append({"x": float(a + 0.05), "coverage": float(m.sum() / max(tb, 1)), "p90_cm": 100 * pct(err[m], 90),
                      "under_1cm": float(np.mean(err[m] < TOL)) if m.sum() >= 50 else float("nan"), "n": int(m.sum())})
    return {"bands": bands, "curve": curve, "range_1cm_m": range_1cm(curve)}


def range_1cm(curve, min_cov=0.1):
    """[near, far]: the contiguous stretch, starting at the first 10 cm bin with >= min_cov coverage,
    over which every bin has p90 |error| < 1 cm (far = near when the first covered bin already fails)."""
    ok = [c["n"] >= 50 and c["coverage"] >= min_cov for c in curve]
    if not any(ok):
        return [float("nan"), float("nan")]
    k = ok.index(True)
    near = far = curve[k]["x"] - 0.05
    for c, o in zip(curve[k:], ok[k:]):
        if o and c["p90_cm"] < 100 * TOL:
            far = c["x"] + 0.05
        else:
            break
    return [round(near, 2), round(far, 2)]


def summarise_fused(recs, sensor, key):
    xr = np.concatenate([r["fused"][sensor][key]["xr"] for r in recs])
    err = np.abs(np.concatenate([r["fused"][sensor][key]["err"] for r in recs]))
    n = np.concatenate([r["fused"][sensor][key]["n"] for r in recs])
    cov = np.sum([r["fused"][sensor][key]["cov_cells"] for r in recs], 0)
    den = np.sum([r["fused"][sensor][key]["cov_denom"] for r in recs], 0)
    out = []
    for b, (a, bb) in enumerate(BANDS):
        m = (xr >= a) & (xr < bb)
        out.append({"band_m": [a, bb], "coverage": float(min(cov[b] / max(den[b], 1), 1.0)), "p90_cm": 100 * pct(err[m], 90),
                    "under_1cm": float(np.mean(err[m] < TOL)) if m.any() else float("nan"),
                    "hits_per_cell": float(np.mean(n[m])) if m.any() else float("nan")})
    return out


def summarise_steps(recs, sensor, rng, n_boot=1000):
    per_ep = [r["steps"][sensor] for r in recs]
    rows = np.concatenate([a for a in per_ep if len(a)]) if any(len(a) for a in per_ep) else np.zeros((0, 4))
    out = []
    for a, b in [(0.5, 1.5), (1.5, 2.5), (2.5, 3.2)]:
        m = (rows[:, 0] >= a) & (rows[:, 0] < b)
        r = rows[m]
        if not len(r):
            out.append({"band_m": [a, b], "n": 0})
            continue
        err = np.abs(r[:, 1] - r[:, 2])
        tc = np.array([ada_level_class(abs(x)) for x in r[:, 2]])
        ec = np.array([ada_level_class(abs(x)) for x in r[:, 1]])
        viol = tc == "violation"
        flat = np.abs(r[:, 2]) < 0.002
        # bootstrap over episodes for the share of measurements under 1 cm
        ep_m = [(a_ <= e[:, 0]) & (e[:, 0] < b) if len(e) else np.zeros(0, bool) for a_, e in [(a, e) for e in per_ep]]
        ep_err = [np.abs(e[mm, 1] - e[mm, 2]) for e, mm in zip(per_ep, ep_m) if len(e) and mm.any()]
        boots = []
        for _ in range(n_boot):
            idx = rng.integers(0, len(ep_err), len(ep_err))
            boots.append(np.mean(np.concatenate([ep_err[i] for i in idx]) < TOL))
        out.append({"band_m": [a, b], "n": int(len(r)), "p50_mm": 1000 * pct(err, 50), "p90_mm": 1000 * pct(err, 90),
                    "under_1cm": float(np.mean(err < TOL)), "under_1cm_ci95": [float(np.percentile(boots, 2.5)),
                                                                              float(np.percentile(boots, 97.5))],
                    "ada_class_agreement": float(np.mean(tc == ec)),
                    "violation_recall": float(np.mean(ec[viol] == "violation")) if viol.any() else float("nan"),
                    "n_violation": int(viol.sum()),
                    "false_flag_rate_flat": float(np.mean(ec[flat] != "ok")) if flat.any() else float("nan"),
                    "n_flat": int(flat.sum())})
    return out


def boot_ci_band(recs, sensor, band, rng, n_boot=1000):
    a, b = band
    per = []
    for r in recs:
        xr, err = r["single"][sensor]["xr"], np.abs(r["single"][sensor]["err"])
        m = (xr >= a) & (xr < b)
        if m.any():
            per.append(err[m])
    if not per:
        return [float("nan")] * 2, [float("nan")] * 2
    p90s, unders = [], []
    for _ in range(n_boot):
        e = np.concatenate([per[i] for i in rng.integers(0, len(per), len(per))])
        p90s.append(np.percentile(e, 90)); unders.append(np.mean(e < TOL))
    return ([100 * float(np.percentile(p90s, 2.5)), 100 * float(np.percentile(p90s, 97.5))],
            [float(np.percentile(unders, 2.5)), float(np.percentile(unders, 97.5))])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/go2_e4c_test")
    ap.add_argument("--out", default="outputs/e4c/test")
    ap.add_argument("--max-episodes", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    files = sorted(glob.glob(os.path.join(str(resolve_path(a.data)), "episodes", "*.npz")))
    if a.max_episodes:
        files = files[: a.max_episodes]
    print(f"{len(files)} episodes from {a.data}")
    recs = []
    for i, f in enumerate(files):
        recs.append(analyse_episode(f, seed=a.seed + i))
        if (i + 1) % 8 == 0:
            print(f"  {i + 1}/{len(files)}", flush=True)

    # sanity: median signed single-frame error should be ~0 if frames and GT line up
    bias = {s: 100 * float(np.median(np.concatenate([r["single"][s]["err"] for r in recs]))) for s in SENSORS}
    rng = np.random.default_rng(a.seed)
    res = {"data": a.data, "n_episodes": len(files), "tolerance_m": TOL, "median_signed_error_cm": bias,
           "ground_truth": "terrain mesh (5 cm height field, 5 mm quantization), bilinear; cells within 7.5 cm of slab "
                           "edges/curb excluded from elevation metrics; poses are simulator ground truth",
           "single_frame": {}, "fused": {}, "steps": {}}
    for s in SENSORS:
        ss = summarise_single(recs, s)
        for bd in ss["bands"]:
            bd["p90_cm_ci95"], bd["under_1cm_ci95"] = boot_ci_band(recs, s, bd["band_m"], rng)
        res["single_frame"][s] = ss
        res["fused"][s] = {k: summarise_fused(recs, s, k) for k in recs[0]["fused"][s]}
        res["steps"][s] = summarise_steps(recs, s, rng)
    out = resolve_path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "sensor_metrics.json").write_text(json.dumps(res, indent=1))
    write_tables(res, out / "sensor_tables.md")
    plot(res, out / "sensor_accuracy.png")
    print((out / "sensor_tables.md").read_text())


def write_tables(res, path):
    L = [f"# E4-C sensor comparison ({res['data']}, {res['n_episodes']} episodes)", "",
         f"median signed error [cm] (sanity, should be ~0): " + ", ".join(f"{k} {v:+.2f}" for k, v in res["median_signed_error_cm"].items()),
         "", "## Single frame: p90 |height error| [cm] / share of cells < 1 cm / coverage, by distance ahead", "",
         "| sensor | p90 < 1 cm from-to [m] | " + " | ".join(f"{a}-{b} m" for a, b in BANDS) + " |",
         "|---|---|" + "---|" * len(BANDS)]
    for s, d in res["single_frame"].items():
        L.append(f"| {s} | {d['range_1cm_m'][0]:.1f}-{d['range_1cm_m'][1]:.1f} | " + " | ".join(
            f"{b['p90_cm']:.2f} / {100 * b['under_1cm']:.0f} % / {100 * b['coverage']:.0f} %" for b in d["bands"]) + " |")
    L += ["", "## Fused over the last K frames (0.1 s each): p90 |error| [cm] / coverage", "",
          "| sensor | fusion | " + " | ".join(f"{a}-{b} m" for a, b in BANDS) + " |", "|---|---|" + "---|" * len(BANDS)]
    for s, fd in res["fused"].items():
        for k, bands in fd.items():
            L.append(f"| {s} | {k} | " + " | ".join(f"{b['p90_cm']:.2f} / {100 * b['coverage']:.0f} %" for b in bands) + " |")
    L += ["", "## Step height at joints (single frame, plane fit each side)", "",
          "| sensor | joint distance | n | p90 err [mm] | < 1 cm [95 % CI] | ADA class agree | violation recall | false flag on flat joints |",
          "|---|---|---|---|---|---|---|---|"]
    for s, rows in res["steps"].items():
        for r in rows:
            if not r["n"]:
                L.append(f"| {s} | {r['band_m'][0]}-{r['band_m'][1]} m | 0 | | | | | |")
                continue
            ci = r["under_1cm_ci95"]
            L.append(f"| {s} | {r['band_m'][0]}-{r['band_m'][1]} m | {r['n']} | {r['p90_mm']:.1f} | "
                     f"{100 * r['under_1cm']:.0f} % [{100 * ci[0]:.0f}, {100 * ci[1]:.0f}] | {100 * r['ada_class_agreement']:.0f} % | "
                     f"{100 * r['violation_recall']:.0f} % (n={r['n_violation']}) | {100 * r['false_flag_rate_flat']:.0f} % (n={r['n_flat']}) |")
    Path(path).write_text("\n".join(L) + "\n")


def plot(res, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    for s, d in res["single_frame"].items():
        x = [c["x"] for c in d["curve"]]
        ax[0].plot(x, [c["p90_cm"] for c in d["curve"]], label=s)
        ax[1].plot(x, [100 * c["under_1cm"] for c in d["curve"]], label=s)
        ax[2].plot(x, [100 * c["coverage"] for c in d["curve"]], label=s)
    ax[0].axhline(1.0, color="k", ls="--", lw=0.8)
    ax[0].set(xlabel="distance ahead [m]", ylabel="p90 |height error| [cm]", ylim=(0, 3), title="single-frame error")
    ax[1].set(xlabel="distance ahead [m]", ylabel="cells under 1 cm [%]", ylim=(0, 101), title="share under 1 cm")
    ax[2].set(xlabel="distance ahead [m]", ylabel="cells with a hit [%]", ylim=(0, 101), title="coverage per frame")
    ax[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)


if __name__ == "__main__":
    main()
