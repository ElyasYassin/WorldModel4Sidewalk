"""Top-down MP4 replays of logged Isaac runs (no simulator needed).

Navigation: one panel per labeled run, same scene, played in sync:
  python scripts/render_replays.py nav --scene sidewalk_10002 \\
      "conventional=outputs/navigation/isaac/conventional_episodes.json" \\
      "kinematic MPC=outputs/navigation_kin/isaac" --out outputs/videos/nav_sidewalk_10002.mp4

Prediction: walk through one recorded episode; at each step draw the true next 3 s and each
model's open-loop 3 s prediction from that point:
  python scripts/render_replays.py pred --data data/go2_cmdpattern --episode isaac_open_40003_s040003 \\
      --checkpoint outputs/state_go2_mlp/best.pt --out outputs/videos/pred_lateral.mp4
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.animation import FFMpegWriter  # noqa: E402

from go2wm.common.config import resolve_path  # noqa: E402

COLORS = ["#2a6fdb", "#d9822b", "#2e9e5b", "#c23b5b", "#7a4fd0"]


def _writer(fps: int) -> FFMpegWriter:
    try:
        import imageio_ffmpeg

        matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        pass
    return FFMpegWriter(fps=fps, codec="libx264", extra_args=["-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-pix_fmt", "yuv420p", "-crf", "23"])


def draw_scene(ax, sc: dict, robot_r: float = 0.3) -> None:
    xmin, xmax, ymin, ymax = sc["bounds"]
    ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, ec="#555", lw=1.5))
    for ox, oy, r in sc["obstacles"]:
        ax.add_patch(plt.Circle((ox, oy), r, color="#777"))
        ax.add_patch(plt.Circle((ox, oy), r + robot_r, fill=False, ec="#bbb", ls=":", lw=0.8))
    ax.plot(*sc["goal"], marker="*", color="#2e9e5b", ms=16, mec="k")
    ax.set_xlim(xmin - 0.3, xmax + 0.3)
    ax.set_ylim(ymin - 0.3, ymax + 0.3)
    ax.set_aspect("equal")
    ax.tick_params(labelsize=7)


def draw_sidewalk(ax, terrain: dict, xlim=(-2.0, 15.0), ylim=(-4.0, 4.0), res: float = 0.02) -> None:
    """Top-down 'map' of a sidewalk tile: concrete walkway with score joints, grass verge, road and
    curb (if the tile has a curb drop), hill-shaded from the true height field so slab edges, ramps
    and cross-slope are visible, and labelled with the feature heights. Purely a visualisation of
    the same geometry the simulator uses (go2wm.sim.terrain.sidewalk_height)."""
    from go2wm.sim.terrain import params_with_defaults, sidewalk_height

    t = params_with_defaults(terrain)
    half = 0.5 * t["width"]
    xs = np.arange(xlim[0], xlim[1], res)
    ys = np.arange(ylim[0], ylim[1], res)
    X, Y = np.meshgrid(xs, ys)
    Z = sidewalk_height(X, Y, t)
    rng = np.random.default_rng(0)
    noise = rng.normal(0, 1, X.shape)
    concrete = np.array([0.78, 0.77, 0.74])
    grass = np.array([0.42, 0.60, 0.30])
    asphalt = np.array([0.27, 0.27, 0.29])
    granite = np.array([0.62, 0.62, 0.64])
    img = np.empty(X.shape + (3,))
    walk = np.abs(Y) <= half
    road = (Y < -half) if t["curb_drop"] > 0 else np.zeros_like(walk)
    img[:] = grass
    img[road] = asphalt
    img[walk] = concrete
    for x0, length, h in t["slabs"]:  # lifted panels: slightly different (older) concrete
        img[walk & (X >= x0) & (X < x0 + length)] = concrete * 0.94 + np.array([0.03, 0.02, 0.0])
    if t["curb_drop"] > 0:  # granite curb stone along the walkway edge
        img[(Y < -half + 0.15) & (Y >= -half)] = granite
    img *= (1 + 0.035 * noise)[..., None]
    img[(Y > half) | (~walk & ~road)] *= (1 + 0.06 * rng.normal(0, 1, X.shape))[(Y > half) | (~walk & ~road)][..., None]
    # score joints every 1.5 m across the walkway, plus panel edges at slab boundaries
    joints = set(np.round(np.arange(np.floor(xlim[0] / 1.5) * 1.5, xlim[1], 1.5), 3))
    for x0, length, _ in t["slabs"]:
        joints |= {round(x0, 3), round(x0 + length, 3)}
    for xj in joints:
        img[walk & (np.abs(X - xj) < res * 0.9)] *= 0.72
    img[walk & (np.abs(np.abs(Y) - half) < res * 0.9)] *= 0.6
    if t["curb_drop"] > 0:  # road edge line
        lane = -half - 0.35
        img[road & (np.abs(Y - lane) < 0.05) & ((X % 3.0) < 1.8)] = [0.93, 0.93, 0.90]
    # hill shading from the height field, light from the upper left: ramps get a gentle tint,
    # step edges (slab sides, curb) a sharp highlight/shadow line
    gy, gx = np.gradient(Z, res)
    shade = np.clip(1.0 + 0.6 * (-gx * 0.7 + gy * 0.7), 0.5, 1.4)
    img = np.clip(img * shade[..., None], 0, 1)
    ax.imshow(img, extent=(xs[0], xs[-1] + res, ys[0], ys[-1] + res), origin="lower", zorder=0, interpolation="bilinear")
    lab = dict(fontsize=8, ha="center", va="center", zorder=3, bbox=dict(fc="white", ec="none", alpha=0.8, pad=1.5))
    for k, (x0, length, h) in enumerate(t["slabs"]):
        ax.text(x0 + 0.5 * length, half - 0.3 - 0.35 * (k % 2), f"slab +{h * 100:.0f} cm", **lab)
    for x0, x1, rise in t["grade"]:
        ax.annotate("", xy=(x1, half - 1.2), xytext=(x0, half - 1.2), zorder=3,
                    arrowprops=dict(arrowstyle="->" if rise > 0 else "<-", color="k", lw=1.2))
        ax.text(0.5 * (x0 + x1), half - 1.0, f"ramp {abs(rise) * 100:.0f} % {'up' if rise > 0 else 'down'}", **lab)
    if t["curb_drop"] > 0:
        ax.text(xlim[0] + 2.2, -half - 1.0, f"road  (curb drop {t['curb_drop'] * 100:.0f} cm)", **lab)
    if abs(t["cross_slope"]) > 0.005:
        ax.text(xlim[0] + 2.2, half + 0.35, f"cross-slope {t['cross_slope'] * 100:+.0f} % (left high)" if t["cross_slope"] > 0
                else f"cross-slope {abs(t['cross_slope']) * 100:.0f} % (right high)", **lab)


def load_runs(specs: list[str], scene: str) -> list[tuple[str, dict]]:
    out = []
    for spec in specs:
        label, _, path = spec.partition("=")
        p = resolve_path(path)
        if p.is_dir():
            p = next(iter(sorted(p.glob("*_episodes.json"))))
        eps = [r for r in json.loads(p.read_text(encoding="utf-8")) if r["scene_id"] == scene]
        if not eps:
            raise SystemExit(f"[ERROR] {scene} not in {p}")
        out.append((label, eps[-1]))
    return out


def render_nav(args) -> None:
    runs = load_runs(args.runs, args.scene)
    sc = runs[0][1]["scene"]
    wide = (sc["bounds"][1] - sc["bounds"][0]) / (sc["bounds"][3] - sc["bounds"][2]) > 2.5
    n = len(runs)
    fig, axes = plt.subplots(n, 1, figsize=(11, 2.1 * n + 0.6)) if wide else plt.subplots(1, n, figsize=(4.6 * n, 4.6))
    axes = np.atleast_1d(axes)
    T = max(len(r["trajectory"]) for _, r in runs)
    arts = []
    for ax, (label, r), col in zip(axes, runs, COLORS):
        draw_scene(ax, r["scene"])
        wp = np.asarray(r["waypoints"])
        ax.plot(wp[:, 0], wp[:, 1], ".", color="#999", ms=3)
        tr = np.asarray(r["trajectory"])
        trail, = ax.plot([], [], "-", color=col, lw=1.8)
        body = plt.Circle((0, 0), 0.3, fc=col, ec="k", alpha=0.85)
        ax.add_patch(body)
        head, = ax.plot([], [], "-", color="k", lw=2)
        txt = ax.text(0.01, 0.97, "", transform=ax.transAxes, va="top", fontsize=8, family="monospace",
                      bbox=dict(fc="white", ec="none", alpha=0.8))
        ax.set_title(f"{label}  →  {r['outcome']}  ({r['time_s']:.1f} s)", fontsize=10, color=col, loc="left")
        arts.append((tr, np.asarray(r.get("commands") or []).reshape(-1, 3), trail, body, head, txt, r))
    fig.suptitle(f"{args.scene}  [Isaac Go2, top-down replay, {args.speed:g}× speed]  "
                 "grey dotted = obstacle + robot radius", fontsize=10)
    fig.tight_layout()
    stride = max(1, int(round(args.speed * 10 / args.fps)))
    out = resolve_path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    w = _writer(args.fps)
    with w.saving(fig, str(out), dpi=args.dpi):
        for k in list(range(0, T, stride)) + [T - 1] * args.fps:
            for tr, cmds, trail, body, head, txt, r in arts:
                i = min(k, len(tr) - 1)
                x, y, yaw = tr[i]
                trail.set_data(tr[: i + 1, 0], tr[: i + 1, 1])
                body.center = (x, y)
                head.set_data([x, x + 0.45 * np.cos(yaw)], [y, y + 0.45 * np.sin(yaw)])
                c = cmds[min(i, len(cmds) - 1)] if len(cmds) else None
                ctxt = f"cmd vx {c[0]:+.2f} vy {c[1]:+.2f} wz {c[2]:+.2f}" if c is not None else "cmd (not logged)"
                done = "" if i < len(tr) - 1 else f"  [{r['outcome']}]"
                txt.set_text(f"t={i * 0.1:5.1f}s  {ctxt}{done}")
            w.grab_frame()
    plt.close(fig)
    print(f"[render] wrote {out}")


def render_pred(args) -> None:
    import torch

    from go2wm.data.dataset import build_windows
    from go2wm.data.episodes import list_episodes, load_episode
    from go2wm.eval.prediction import _rollout_np
    from go2wm.models.state_dynamics import KinematicBaseline, load_checkpoint

    root = resolve_path(args.data)
    path = next(p for p in list_episodes(root) if load_episode(p, load_arrays=False).meta["episode_id"] == args.episode)
    ep = load_episode(path)
    models = {}
    hist = 4
    for c in args.checkpoint:
        m, ck = load_checkpoint(resolve_path(c))
        hist = ck["config"]["data"]["history"]
        models[f"{m.model_type.upper()} ({resolve_path(c).parent.name.replace('state_go2_', '')})"] = m
        m.eval()
    models["command kinematics"] = KinematicBaseline(hist, "command")
    H = 30
    w = build_windows([ep], hist, H, terrain_map=any(getattr(m, "uses_terrain", False) for m in models.values()))
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    preds = {k: _rollout_np(m, w, dev)["poses"] for k, m in models.items()}
    from go2wm.common.schema import state_pose2d

    pose = state_pose2d(ep["states"])
    cmds = ep["commands"][:, :3]

    def to_world(p0, rel):
        c, s = np.cos(p0[2]), np.sin(p0[2])
        return np.stack([p0[0] + c * rel[:, 0] - s * rel[:, 1], p0[1] + s * rel[:, 0] + c * rel[:, 1]], 1)

    fig, ax = plt.subplots(figsize=(10, 7))
    sc = ep.meta["scene"]
    draw_scene(ax, sc)
    terrain = (sc.get("params") or {}).get("terrain")
    if terrain:  # the sidewalk as a top-down map (concrete, joints, verge, curb), hill-shaded by height
        draw_sidewalk(ax, terrain, xlim=(pose[:, 0].min() - 4, pose[:, 0].max() + 4),
                      ylim=(pose[:, 1].min() - 3, pose[:, 1].max() + 3))
    ax.plot(pose[:, 0], pose[:, 1], "-", color="#ddd", lw=1)
    trail, = ax.plot([], [], "-", color="k", lw=1.5)
    body = plt.Circle((0, 0), 0.3, fc="#ccc", ec="k")
    ax.add_patch(body)
    truth, = ax.plot([], [], "-", color="k", lw=3, label="true next 3 s")
    lines = {k: ax.plot([], [], "--", color=COLORS[i], lw=2.2, label=f"{k} prediction")[0] for i, k in enumerate(models)}
    ends = {k: ax.plot([], [], "o", color=COLORS[i], ms=6)[0] for i, k in enumerate(models)}
    ax.legend(loc="lower right", fontsize=8)
    txt = ax.text(0.01, 0.98, "", transform=ax.transAxes, va="top", fontsize=9, family="monospace",
                  bbox=dict(fc="white", ec="none", alpha=0.85))
    pat = ep.meta.get("command_pattern") or (f"tile {ep.meta['e4_tile']}, {ep.meta.get('termination_reason')}"
                                             if "e4_tile" in ep.meta else ep.meta["scene_family"])
    ax.set_title(f"{args.episode}  pattern={pat}  [Isaac Go2; open-loop 3 s predictions from each step]", fontsize=10)
    fig.tight_layout()
    out = resolve_path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    wr = _writer(args.fps)
    with wr.saving(fig, str(out), dpi=args.dpi):
        for n in range(len(w)):
            i = int(w.start_index[n])
            p0 = pose[i]
            trail.set_data(pose[: i + 1, 0], pose[: i + 1, 1])
            body.center = p0[:2]
            tw = to_world(p0, w.fut_pose[n])
            truth.set_data(tw[:, 0], tw[:, 1])
            errs = []
            for k in models:
                pw = to_world(p0, preds[k][n])
                lines[k].set_data(pw[:, 0], pw[:, 1])
                ends[k].set_data([pw[-1, 0]], [pw[-1, 1]])
                errs.append(f"{k[:22]:22s} 3 s error {np.linalg.norm(pw[-1] - tw[-1]):.2f} m")
            c = cmds[i]
            txt.set_text(f"t={i * 0.1:5.1f}s  cmd vx {c[0]:+.2f} vy {c[1]:+.2f} wz {c[2]:+.2f}\n" + "\n".join(errs))
            ax.set_xlim(p0[0] - 3.3, p0[0] + 3.3)  # follow camera: 6.6 m x 4.6 m window
            ax.set_ylim(p0[1] - 2.3, p0[1] + 2.3)
            wr.grab_frame()
    plt.close(fig)
    print(f"[render] wrote {out}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)
    a = sub.add_parser("nav")
    a.add_argument("runs", nargs="+", help="LABEL=PATH")
    a.add_argument("--scene", required=True)
    a.add_argument("--speed", type=float, default=2.0)
    b = sub.add_parser("pred")
    b.add_argument("--data", required=True)
    b.add_argument("--episode", required=True)
    b.add_argument("--checkpoint", nargs="+", required=True)
    for p in (a, b):
        p.add_argument("--out", required=True)
        p.add_argument("--fps", type=int, default=20)
        p.add_argument("--dpi", type=int, default=100)
    args = ap.parse_args(argv)
    render_nav(args) if args.mode == "nav" else render_pred(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
