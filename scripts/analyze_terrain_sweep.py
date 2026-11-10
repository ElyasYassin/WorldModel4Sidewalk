"""Offline analysis of the E4 terrain sweep (data/go2_e4_sweep, configs/e4_sweep.yaml). No simulator.

  python scripts/analyze_terrain_sweep.py [--data data/go2_e4_sweep] [--out outputs/e4_sweep/analysis.json]

Per episode:
  outcome     goal / end (reached the far end of the tile) / stuck (moved < 0.1 m in the last 10 s)
              / fall / timeout / out_of_bounds
  kin1s       command-kinematics 1 s position error [m]: from the true pose at t, integrate the
              commanded body velocities for 10 steps and compare with the true pose at t + 1 s.
              This is what a hand-written model gets wrong, i.e. the room a learned model has.
              Reported on windows whose midpoint lies on the swept feature, and on the plain rest.
  track       mean realized - commanded body velocity (vx, vy, wz) on the feature
  slip        mean foot slip [m/s] on the feature
  drift       straight probes only: lateral drift [m per m travelled] and heading change [deg]
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict

import numpy as np

from go2wm.cli import write_json
from go2wm.common.config import resolve_path
from go2wm.common.schema import state_pose2d

H = 10  # 1 s at 10 Hz


def feature_mask(xy: np.ndarray, terrain: dict) -> np.ndarray:
    """True where a position lies on the swept terrain feature."""
    x, y = xy[:, 0], xy[:, 1]
    m = np.zeros(len(x), bool)
    for x0, L, _h in terrain.get("slabs", []):
        m |= (x >= x0 - 0.5) & (x < x0 + L + 0.5)
    for x0, x1, _r in terrain.get("grade", []):
        m |= (x >= x0 - 0.5) & (x < x1 + 0.5)
    if terrain.get("curb_drop", 0) > 0:
        m |= np.abs(y + 0.5 * terrain.get("width", 3.0)) < 0.6
    if terrain.get("cross_slope", 0) != 0 or terrain.get("friction", 1.0) != 1.0 or not (
            terrain.get("slabs") or terrain.get("grade") or terrain.get("curb_drop", 0) > 0):
        m |= x > 1.0  # whole-surface features (and the flat reference): everything after the start
    return m


def kin_errors(pose: np.ndarray, cmds: np.ndarray, dt: float) -> np.ndarray:
    """1 s command-kinematics error for every window start t (len T - H + 1)."""
    T = len(cmds)
    out = np.full(T - H + 1, np.nan)
    for t in range(T - H + 1):
        x, y, th = pose[t]
        for k in range(H):
            vx, vy, wz = cmds[t + k, :3]
            c, s = np.cos(th), np.sin(th)
            x, y, th = x + (vx * c - vy * s) * dt, y + (vx * s + vy * c) * dt, th + wz * dt
        out[t] = np.hypot(x - pose[t + H, 0], y - pose[t + H, 1])
    return out


def outcome(ep, pose) -> str:
    r = ep.meta["termination_reason"]
    if r == "goal":
        return "goal"
    if r == "out_of_bounds" and pose[-1, 0] > 13.0:
        return "end"
    if r in ("fall", "collision"):
        return "fall" if r == "fall" else r
    if len(pose) > 100 and np.linalg.norm(pose[-1, :2] - pose[-101, :2]) < 0.1:
        return "stuck"
    return r


def analyze(ep) -> dict:
    a = ep.arrays
    s, cmds = a["states"], a["commands"]
    pose = state_pose2d(s)
    terrain = ep.meta["scene"]["params"]["terrain"]
    probe = ep.meta["e4_probe"]
    dt = float(np.median(cmds[:, 3])) if cmds.shape[1] > 3 else 0.1
    r = {"variant": ep.meta["e4_variant"], "probe": probe["name"], "episode": ep.episode_id,
         "outcome": outcome(ep, pose), "x_max": float(pose[:, 0].max()), "time_s": float(len(cmds) * dt)}
    feat = feature_mask(pose[:, :2], terrain)
    if len(cmds) > H:
        e = kin_errors(pose, cmds, dt)
        mid = feat[np.arange(len(e)) + H // 2]
        r["kin1s_feature"] = float(np.mean(e[mid])) if mid.any() else None
        r["kin1s_plain"] = float(np.mean(e[~mid])) if (~mid).any() else None
        r["kin1s_feature_p90"] = float(np.percentile(e[mid], 90)) if mid.any() else None
        r["n_feature_windows"] = int(mid.sum())
    f1 = feat[1:]
    if f1.any():
        real = np.stack([s[1:, 7], s[1:, 8], s[1:, 12]], 1)
        d = (real - cmds[:, :3])[f1]
        r["track_feature"] = {"vx": float(d[:, 0].mean()), "vy": float(d[:, 1].mean()), "wz": float(d[:, 2].mean())}
        if "gt_foot_slip_m_s" in a:
            r["slip_feature"] = float(np.nanmean(a["gt_foot_slip_m_s"][1:][f1]))
    if probe["kind"] == "straight":
        dist = float(np.hypot(*(pose[-1, :2] - pose[0, :2])))
        r["drift_lat_per_m"] = float((pose[-1, 1] - pose[0, 1]) / max(dist, 1e-6))
        r["drift_heading_deg"] = float(np.degrees(np.arctan2(np.sin(pose[-1, 2] - pose[0, 2]), np.cos(pose[-1, 2] - pose[0, 2]))))
    return r


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/go2_e4_sweep")
    ap.add_argument("--out", default="outputs/e4_sweep/analysis.json")
    args = ap.parse_args(argv)
    from go2wm.data.episodes import list_episodes, load_episode

    rows = [analyze(load_episode(p)) for p in list_episodes(resolve_path(args.data))]
    by_v = defaultdict(list)
    for r in rows:
        by_v[r["variant"]].append(r)

    def f(v, fmt="{:+.2f}"):
        return "  -  " if v is None else fmt.format(v)

    print(f"{'variant':13s} {'probe':13s} {'outcome':8s} {'x_max':>5s} {'kin1s feat':>10s} {'plain':>6s} {'p90':>6s} "
          f"{'dvx':>6s} {'dvy':>6s} {'dwz':>6s} {'slip':>5s} {'drift/m':>7s} {'dhead':>6s}")
    for v in by_v:  # insertion order = file order
        for r in sorted(by_v[v], key=lambda r: r["probe"]):
            t = r.get("track_feature", {})
            print(f"{v:13s} {r['probe']:13s} {r['outcome']:8s} {r['x_max']:5.1f} "
                  f"{f(r.get('kin1s_feature'), '{:.3f}'):>10s} {f(r.get('kin1s_plain'), '{:.3f}'):>6s} "
                  f"{f(r.get('kin1s_feature_p90'), '{:.3f}'):>6s} {f(t.get('vx')):>6s} {f(t.get('vy')):>6s} {f(t.get('wz')):>6s} "
                  f"{f(r.get('slip_feature'), '{:.2f}'):>5s} {f(r.get('drift_lat_per_m'), '{:+.3f}'):>7s} "
                  f"{f(r.get('drift_heading_deg'), '{:+.0f}'):>6s}")
    p = write_json(resolve_path(args.out), {"episodes": rows})
    print(f"[analyze] {len(rows)} episodes -> {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
