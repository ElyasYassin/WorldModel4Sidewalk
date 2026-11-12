"""Phase 1: top-down replay video of inspection missions, strategies stacked and time-synced (no Isaac needed).

  python scripts/mission_video.py --missions outputs/missions_final2 --tiles outputs/defect_tiles_final2/tiles.json \
      --tile 8 --policies walkover approach --out outputs/videos/13_inspection_walkover_vs_approach_tile8.mp4

Replays saved mission results (run_mission.py): the tile's true height field (relative to the walk's plane),
the robot, the D555 measuring zone and the D405 zone, the trail, and the findings in memory as they appear
(hollow = detected by the D555, filled = measured with the D405, label = measured value). Approach / inspect
events are shown as a banner. Ends on a ground-truth scorecard per strategy.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import imageio.v2 as imageio  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Circle, Ellipse, Polygon, Rectangle  # noqa: E402

from go2wm.common.config import resolve_path  # noqa: E402
from go2wm.sim.terrain import params_with_defaults, sidewalk_height  # noqa: E402

NAMES = {"walkover": "Walk past", "slow": "Slow down", "approach": "Stop & look closely (D405)"}
KIND = {"depression": ("hole / pit", "#2b6cff"), "edge": ("step / upheaval", "#ff7a1a"),
        "dropout": ("possible water", "#18c3d6")}
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

SURF = LinearSegmentedColormap.from_list("concrete", ["#0b3a8f", "#3d74d6", "#6b7178", "#6b7178", "#e05a2a", "#ffcf4a"])
BG, FG, MUTED = "#101418", "#e8ecf0", "#8a96a3"


def rect(pose, x0, x1, hw):
    x, y, th = pose
    c, s = np.cos(th), np.sin(th)
    loc = np.array([[x0, -hw], [x1, -hw], [x1, hw], [x0, hw]])
    return np.stack([x + c * loc[:, 0] - s * loc[:, 1], y + s * loc[:, 0] + c * loc[:, 1]], 1)


def finding_timeline(f):
    """(time it becomes a confirmed finding, time first measured by the D405, D405 value)."""
    obs = sorted(f.get("observations", []), key=lambda o: o["t"])
    det = [o["t"] for o in obs if o.get("provenance") == "detected"]
    t_show = det[1] if len(det) > 1 else (det[0] if det else f.get("first_seen_t", 0.0))
    d405 = [o for o in obs if o["sensor"] == "depth_d405" and o.get("provenance") != "detected"
            and o.get("value_m") is not None]
    return t_show, (d405[0]["t"] if d405 else None), f.get("value_m")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--missions", default="outputs/missions_final2")
    ap.add_argument("--tiles", default="outputs/defect_tiles_final2/tiles.json")
    ap.add_argument("--tile", type=int, default=8)
    ap.add_argument("--policies", nargs="+", default=["walkover", "approach"])
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--speed", type=float, default=2.0, help="playback speed vs mission time")
    ap.add_argument("--still", type=float, nargs="*", help="only save PNG stills at these mission times")
    ap.add_argument("--out", default="outputs/videos/13_inspection_walkover_vs_approach_tile8.mp4")
    a = ap.parse_args()

    tile = json.loads(resolve_path(a.tiles).read_text())[a.tile]
    p = params_with_defaults(tile)
    L, half = float(p["size"][0]), 0.5 * float(p["width"])
    runs = [json.loads((resolve_path(a.missions) / f"{pol}_tile{a.tile}.json").read_text()) for pol in a.policies]

    # true defect relief: surface minus the same sidewalk without defects (removes cross slope and grade)
    xs, ys = np.arange(-L / 2, L / 2, 0.02), np.arange(-half - 0.15, half + 0.15, 0.02)
    X, Y = np.meshgrid(xs, ys)
    Z = sidewalk_height(X, Y, p, grid=0.02) - sidewalk_height(X, Y, {**p, "defects": []}, grid=0.02)
    Z = np.where(np.abs(Y) <= half, Z, np.nan)
    x_lo, x_hi = -L / 2 + 0.3, L / 2 - 0.3

    n = len(runs)
    fig = plt.figure(figsize=(19.2, 10.8), dpi=100, facecolor=BG)
    fig.text(0.02, 0.955, "Sidewalk inspection mission: walk past vs. stop and look closely",
             color=FG, fontsize=24, weight="bold")
    fig.text(0.02, 0.925, f"Simulated Go2 (kinematic mock), D555 + D405 ray-cast with realistic errors · "
             f"fresh evaluation layout (seed 300, tile {a.tile}) · {a.speed:g}x speed",
             color=MUTED, fontsize=13)
    lane_h = 0.31
    lanes = []
    for i, r in enumerate(runs):
        bottom = 0.52 - i * (lane_h + 0.12)
        ax = fig.add_axes([0.03, bottom, 0.94, lane_h], facecolor=BG)
        ax.imshow(Z * 1000, extent=(xs[0], xs[-1], ys[0], ys[-1]), origin="lower", cmap=SURF,
                  vmin=-60, vmax=60, interpolation="bilinear", aspect="equal")
        for d in p["defects"]:
            if d["type"] == "puddle":
                ax.add_patch(Ellipse((d["x"], d["y"]), 2 * d["rx"], 2 * d["ry"], angle=np.degrees(d["angle"]),
                                     fc="#18c3d633", ec="#18c3d6", lw=1, ls="--"))
            elif d["type"] == "crack":
                q = np.asarray(d["points"])
                ax.plot(q[:, 0], q[:, 1], color="#222", lw=1.2)
            elif d["type"] == "obstruction":
                ax.add_patch(Circle((d["x"], d["y"]), d["r"], fc="#555", ec="#aaa", lw=1))
        ax.axhline(half, color="#666", lw=1)
        ax.axhline(-half, color="#666", lw=1)
        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(-half - 0.15, half + 0.15)
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        title = ax.text(0.0, 1.04, NAMES.get(r["policy"], r["policy"]), transform=ax.transAxes, color=FG,
                        fontsize=19, weight="bold", va="bottom")
        clock = ax.text(1.0, 1.04, "", transform=ax.transAxes, color=FG, fontsize=17, ha="right", va="bottom",
                        family="monospace")
        stats = ax.text(0.0, -0.04, "", transform=ax.transAxes, color=MUTED, fontsize=14, va="top")
        banner = ax.text(0.5, 1.04, "", transform=ax.transAxes, color="#ffd23f", fontsize=17, ha="center",
                         va="bottom", weight="bold")
        trail, = ax.plot([], [], color="#ffffffaa", lw=1.5)
        z555 = ax.add_patch(Polygon(np.zeros((4, 2)), fc="#ffffff18", ec="#ffffff80", lw=1))
        z405 = ax.add_patch(Polygon(np.zeros((4, 2)), fc="#ffd23f40", ec="#ffd23f", lw=1.5))
        body = ax.add_patch(Polygon(np.zeros((4, 2)), fc="#f2f2f2", ec="#000", lw=1.2))
        head = ax.add_patch(Polygon(np.zeros((3, 2)), fc="#000"))
        fnd = []
        for fid, f in r["memory"]["findings"].items():
            ts, t405, val = finding_timeline(f)
            lab, col = KIND.get(f["kind"], (f["kind"], "#fff"))
            m = ax.add_patch(Circle(f["xy"], 0.22, fc="none", ec=col, lw=2.5, visible=False))
            txt = ax.text(f["xy"][0], f["xy"][1] + (0.32 if f["xy"][1] < half - 0.5 else -0.32), "", color=FG,
                          fontsize=12, ha="center", va="center", visible=False,
                          bbox=dict(boxstyle="round,pad=0.2", fc="#000000b0", ec=col, lw=1))
            fnd.append(dict(f=f, ts=ts, t405=t405, val=val, lab=lab, col=col, m=m, txt=txt))
        traj = np.asarray(r["trajectory"])
        dt = r["time_s"] / max(len(traj) - 1, 1)
        lanes.append(dict(r=r, ax=ax, traj=traj, dt=dt, clock=clock, stats=stats, banner=banner, trail=trail,
                          z555=z555, z405=z405, body=body, head=head, fnd=fnd, title=title))

    # legend
    lx = 0.03
    for k in ("depression", "edge", "dropout"):
        lab, col = KIND[k]
        fig.text(lx, 0.035, "●", color=col, fontsize=20, va="center")
        fig.text(lx + 0.015, 0.035, lab, color=FG, fontsize=13, va="center")
        lx += 0.1
    fig.text(lx, 0.035, "○ detected (D555, far)   ● measured close up (D405)   "
             "white box: D555 zone   yellow: D405 zone   "
             "surface: orange = raised, blue = sunken", color=MUTED, fontsize=12, va="center")

    def draw(t):
        for ln in lanes:
            r, traj = ln["r"], ln["traj"]
            k = min(int(round(t / ln["dt"])), len(traj) - 1)
            done = t >= r["time_s"]
            pose = traj[k]
            ln["trail"].set_data(traj[: k + 1, 0], traj[: k + 1, 1])
            ln["z555"].set_xy(rect(pose, 0.6, 1.2, 0.6))
            ln["z405"].set_xy(rect(pose, 0.42, 0.62, 0.2))
            ln["body"].set_xy(rect(pose, -0.35, 0.35, 0.155))
            ln["head"].set_xy(np.vstack([rect(pose, 0.12, 0.12, 0.12)[[0, 3]], rect(pose, 0.33, 0.33, 0.0)[:1]]))
            n_f = n_405 = 0
            for d in ln["fnd"]:
                vis = t >= d["ts"]
                d["m"].set_visible(vis)
                d["txt"].set_visible(vis)
                if not vis:
                    continue
                n_f += 1
                measured = d["t405"] is not None and t >= d["t405"]
                n_405 += measured
                d["m"].set_facecolor(d["col"] + ("cc" if measured else "00"))
                if measured and d["val"] is not None:
                    d["txt"].set_text(f"{d['lab']} · {abs(d['val']) * 1000:.0f} mm (D405)")
                elif d["f"]["kind"] == "dropout":
                    d["txt"].set_text(f"{d['lab']}?")
                else:
                    d["txt"].set_text(f"{d['lab']} (D555)")
            ln["clock"].set_text(("GOAL ✓  " if done else "") + f"t = {min(t, r['time_s']):5.1f} s")
            ln["clock"].set_color("#6ee36e" if done else FG)
            ln["stats"].set_text(f"findings in memory: {n_f}    measured close up with the D405: {n_405}")
            ev = [e for e in r["events"] if e["t"] <= t]
            msg = ""
            if ev and not done:
                e = ev[-1]
                if e["event"] == "approach":
                    msg = "▶ approaching a finding for a close look"
                elif e["event"] == "inspected" and t - e["t"] < 1.5:
                    msg = f"✓ inspected: {e['status']}"
                elif e["event"] == "approach_aborted" and t - e["t"] < 1.5:
                    msg = "✗ approach aborted"
            ln["banner"].set_text(msg)

    out = resolve_path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if a.still:
        for t in a.still:
            draw(t)
            fig.savefig(out.with_name(out.stem + f"_t{t:.0f}.png"), facecolor=BG)
        return 0
    T = max(r["time_s"] for r in runs)
    w = imageio.get_writer(out, fps=a.fps, codec="libx264", quality=8, macro_block_size=8,
                           pixelformat="yuv420p")

    def grab():
        fig.canvas.draw()
        return np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()

    draw(0.0)
    f0 = grab()
    for _ in range(int(1.5 * a.fps)):
        w.append_data(f0)
    for t in np.arange(0.0, T + 1e-9, a.speed / a.fps):
        draw(float(t))
        w.append_data(grab())
    draw(T)
    # scorecard against ground truth
    for ln in lanes:
        sc = ln["r"]["score"]
        parts = []
        for typ, v in sc["by_type"].items():
            parts.append(f"{typ}: {v['found']}/{v['n']} found, {v['measured_d405']} by D405")
        errs = [f"{x['type']} err {x['abs_err_m'] * 1000:.1f} mm" for x in sc["per_defect"] if "abs_err_m" in x]
        ln["banner"].set_text("")
        ln["stats"].set_text("Ground truth: " + " · ".join(parts) + f" · false findings {sc['false_findings']}"
                             + ("\nmeasurement error: " + ", ".join(errs) if errs else ""))
        ln["stats"].set_color(FG)
    fe = grab()
    for _ in range(int(5 * a.fps)):
        w.append_data(fe)
    w.close()
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
