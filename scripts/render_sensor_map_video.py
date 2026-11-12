"""Side-by-side video of a recorded E4-C episode: front camera | map built from fused sensors | ground truth.

The middle panel is a world-frame elevation map accumulated from every frame up to the current one (the observed
cells of the chosen sensors' 5 cm grids moved into the tile frame with the recorded poses: perfect registration,
as in fused_sensor_maps). The right panel is the tile's exact surface. Both use the same colour scale with
hill-shading so slab steps show as edges. Grey = not yet observed. The robot was driving a recorded command
sequence; the map was not used for control.

Scene graph (go2wm.scene.graph, built from the tile parameters = ground truth): its panel joints and hazards are drawn
on the ground-truth panel; on the map panel a hazard is drawn once the sensors have covered at least half of its
area ("in sensor view", not a detection: there is no detector yet).

  python scripts/render_sensor_map_video.py --episode data/go2_e4c_test/episodes/isaac_sidewalk_80039_s080039.npz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from go2wm.common.config import resolve_path  # noqa: E402
from go2wm.data.dataset import state_pose2d  # noqa: E402
from go2wm.data.episodes import load_episode, load_rgb  # noqa: E402
from go2wm.models.terrain_map import episode_height_fn  # noqa: E402
from go2wm.scene.graph import graph_from_episode_meta  # noqa: E402
from go2wm.sim.sensors import ELEV_GRID, grid_centres  # noqa: E402

RES = 0.05
COLORS = {"vertical_displacement": "#E03131", "running_slope": "#D6336C", "crack": "#FCC419", "curb_edge": "#22B8CF",
          "upheaval": "#FF6B6B", "hole": "#9775FA", "puddle": "#4DABF7", "obstruction": "#FF922B"}


def graph_items(g) -> list[dict]:
    """Drawable scene-graph hazards (cross slope covers the whole walk and is reported in the title instead)."""
    items = []
    defects = (g["meta"].get("terrain") or {}).get("defects") or []
    for h in g["hazards"]:
        t, b = h["type"], h["box"]
        n = g["nodes"].get(h["id"], {})
        if n.get("kind") == "defect" and n.get("defect"):  # explicit defects (sim/defects.py)
            d = defects[int(h["id"].split("_")[1])]
            if t in ("hole", "puddle", "obstruction"):
                a = np.linspace(0, 2 * np.pi, 48)
                rx, ry, ang = (d["rx"], d["ry"], d.get("angle", 0.0)) if t != "obstruction" else (d["r"], d["r"], 0.0)
                u, v = rx * np.cos(a), ry * np.sin(a)
                xy = (d["x"] + u * np.cos(ang) - v * np.sin(ang), d["y"] + u * np.sin(ang) + v * np.cos(ang))
                lab = {"hole": f"hole {h.get('depth_m', 0) * 100:.0f} cm deep", "puddle": "puddle",
                       "obstruction": d.get("kind", "object")}[t]
            elif t == "upheaval":
                xy = ([b[0], b[1], b[1], b[0], b[0]], [b[2], b[2], b[3], b[3], b[2]])
                lab = f"upheaval +{h['lift_m'] * 100:.1f} cm"
            else:  # open crack
                pts = n["polyline"]
                xy = ([q[0] for q in pts], [q[1] for q in pts])
                lab = f"crack {h['width_m'] * 1000:.0f} mm"
            items.append({"t": t, "box": b, "xy": xy, "label": lab, "ada": h["ada"], "lx": float(np.mean(xy[0]))})
            continue
        if t == "vertical_displacement":
            items.append({"t": t, "box": b, "xy": ([h["xy"][0]] * 2, [b[2], b[3]]),
                          "label": f"{h['value_m'] * 100:+.1f} cm", "ada": h["ada"]})
        elif t == "running_slope":
            items.append({"t": t, "box": b, "xy": ([b[0], b[1], b[1], b[0], b[0]], [b[2], b[2], b[3], b[3], b[2]]),
                          "label": f"ramp {abs(h['value']) * 100:.0f} %", "ada": h["ada"]})
        elif t == "crack":
            pts = g["nodes"][h["id"]]["polyline"]
            items.append({"t": t, "box": b, "xy": ([q[0] for q in pts], [q[1] for q in pts]), "label": "", "ada": h["ada"]})
        elif t == "curb_edge":
            items.append({"t": t, "box": [b[0], b[1], b[3] - 0.05, b[3]], "xy": ([b[0], b[1]], [b[3], b[3]]),
                          "label": f"curb {h['value_m'] * 100:.0f} cm", "ada": h["ada"]})
    return items


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", required=True)
    ap.add_argument("--sensors", default="depth_d555,l1")
    ap.add_argument("--out", default="outputs/videos/10_fused_map_vs_truth.mp4")
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--stride", type=int, default=1, help="render every n-th frame")
    a = ap.parse_args()

    import imageio.v2 as imageio
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LightSource, Normalize

    p = resolve_path(a.episode)
    ep = load_episode(p)
    frames = load_rgb(p)
    pose = state_pose2d(ep["states"])
    hf = episode_height_fn(ep.meta)
    names = a.sensors.split(",")
    elevs = [ep[f"sensor_{n}_elev_m"] for n in names]
    T = len(pose)

    # world-frame grid covering the path and what the sensors could see from it
    pad = 2.0
    x0, x1 = pose[:, 0].min() - 1.0, pose[:, 0].max() + 3.5
    y0, y1 = pose[:, 1].min() - pad, pose[:, 1].max() + pad
    xs = np.arange(x0, x1, RES) + RES / 2
    ys = np.arange(y0, y1, RES) + RES / 2
    XX, YY = np.meshgrid(xs, ys)  # [ny, nx]
    truth = hf(XX, YY)
    acc = np.zeros(XX.size)
    cnt = np.zeros(XX.size)

    cx, cy = grid_centres(ELEV_GRID)
    GX, GY = np.meshgrid(cx, cy, indexing="ij")
    vmin, vmax = np.percentile(truth, [1, 99])
    norm = Normalize(vmin, vmax)
    cmap = plt.get_cmap("viridis")
    ls = LightSource(azdeg=315, altdeg=35)

    def shade(z):
        zf = np.where(np.isfinite(z), z, np.nanmedian(truth))
        rgb = ls.shade(zf, cmap=cmap, norm=norm, blend_mode="soft", vert_exag=8, dx=RES, dy=RES)
        rgb[~np.isfinite(z)] = (0.78, 0.78, 0.78, 1.0)
        return rgb

    truth_rgb = shade(truth)
    g = graph_from_episode_meta(ep.meta, x_range=(x0, x1))
    items = graph_items(g) if g else []
    for it in items:  # grid cells of each hazard's box, for the coverage test
        b = it["box"]
        jx = slice(max(0, int((b[0] - x0) / RES)), max(1, int(np.ceil((b[1] - x0) / RES))))
        jy = slice(max(0, int((b[2] - y0) / RES)), max(1, int(np.ceil((b[3] - y0) / RES))))
        it["cells"] = (jy, jx)
    joints = sorted({n["box"][0] for n in (g["nodes"].values() if g else []) if n["kind"] == "panel"})[1:]
    cross = g["nodes"]["walkway"] if g else None

    def draw_graph(ax, show, label_y):
        for it, on in zip(items, show):
            if not on:
                continue
            c = COLORS[it["t"]]
            ls_ = "--" if it["t"] == "running_slope" else "-"
            ax.plot(*it["xy"], color=c, lw=1.6 if it["t"] == "crack" and "lx" not in it else 2.2, ls=ls_)
            if it["label"] and "lx" in it:  # defects: label just above the defect
                ax.text(it["lx"], it["box"][3] + 0.12, it["label"], color=c, fontsize=7.5, fontweight="bold", ha="center",
                        bbox={"facecolor": "#14213D", "alpha": 0.75, "pad": 1.2, "edgecolor": "none"})
            elif it["label"]:
                lx = it["xy"][0][0] if it["t"] != "curb_edge" else x0 + 0.3
                ly = label_y if it["t"] != "curb_edge" else it["xy"][1][0] - 0.35
                if it["t"] == "running_slope":  # centre of the ramp box, below the step labels
                    lx, ly = 0.5 * (it["box"][0] + it["box"][1]), label_y - 0.55
                ax.text(lx, ly, it["label"], color=c, fontsize=8, fontweight="bold", ha="center" if it["t"] != "curb_edge" else "left",
                        bbox={"facecolor": "#14213D", "alpha": 0.75, "pad": 1.5, "edgecolor": "none"})
    out = resolve_path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(out), fps=a.fps / a.stride * a.stride, codec="libx264", quality=8,
                                macro_block_size=8)
    extent = (x0, x1, y0, y1)
    if frames is None:  # no camera in this run: show the top-down appearance (the texture Isaac uses) instead
        from go2wm.sim.sidewalk_texture import sidewalk_albedo

        terrain = ((ep.meta.get("scene") or {}).get("params") or {}).get("terrain") or {}
        top = sidewalk_albedo(XX, YY, terrain, seed=0)
        span = (x1 - x0) / (y1 - y0)
        fig = plt.figure(figsize=(16, min(12.0, 3 * 16 / span + 1.6)), dpi=100)
    else:
        fig = plt.figure(figsize=(16, 7.2), dpi=100)
    for t in range(T):
        for e in elevs:  # accumulate this frame's observed cells into the world map
            z = e[t].astype(np.float64)
            ok = np.isfinite(z)
            if ok.any():
                c, s = np.cos(pose[t, 2]), np.sin(pose[t, 2])
                wx = pose[t, 0] + c * GX[ok] - s * GY[ok]
                wy = pose[t, 1] + s * GX[ok] + c * GY[ok]
                ix = np.floor((wx - x0) / RES).astype(int)
                iy = np.floor((wy - y0) / RES).astype(int)
                m = (ix >= 0) & (ix < len(xs)) & (iy >= 0) & (iy < len(ys))
                flat = iy[m] * len(xs) + ix[m]
                acc += np.bincount(flat, weights=z[ok][m], minlength=acc.size)
                cnt += np.bincount(flat, minlength=acc.size)
        if t % a.stride:
            continue
        fused = np.where(cnt > 0, acc / np.maximum(cnt, 1), np.nan).reshape(XX.shape)
        seen = [bool(np.isfinite(fused[it["cells"]]).mean() >= 0.5) for it in items]
        fig.clf()
        if frames is not None:
            gs = fig.add_gridspec(2, 2, width_ratios=[0.75, 1.25], wspace=0.06, hspace=0.32,
                                  left=0.02, right=0.98, top=0.88, bottom=0.08)
            ax0 = fig.add_subplot(gs[:, 0])
            ax0.imshow(frames[min(t, len(frames) - 1)])
            ax0.set_title("Front camera (simulated)", fontsize=12)
            ax0.axis("off")
            slot = lambda k: gs[k - 1, 1]  # noqa: E731
        else:
            gs = fig.add_gridspec(3, 1, hspace=0.35, left=0.05, right=0.98, top=0.86, bottom=0.07)
            ax0 = fig.add_subplot(gs[0])
            ax0.imshow(top, origin="lower", extent=extent, interpolation="bilinear")
            for ox, oy, orad in (ep.meta["scene"].get("obstacles") or []):
                ax0.add_patch(plt.Circle((ox, oy), orad, color="#495057", alpha=0.9))
            ax0.plot(pose[: t + 1, 0], pose[: t + 1, 1], color="white", lw=1.5)
            ax0.arrow(pose[t, 0], pose[t, 1], 0.5 * np.cos(pose[t, 2]), 0.5 * np.sin(pose[t, 2]), width=0.06,
                      color="#E8590C", length_includes_head=True)
            ax0.set_title("Top-down appearance (texture; grey discs = obstacles; no camera in this run)", fontsize=12)
            ax0.set_ylabel("y [m]", fontsize=9)
            ax0.tick_params(labelsize=8)
            slot = lambda k: gs[k]  # noqa: E731
        for k, (img, title) in enumerate(((shade(fused), f"Map from fused {' + '.join(n.replace('depth_', '').upper() for n in names)}"),
                                          (truth_rgb, "Ground truth surface + scene graph")), start=1):
            ax = fig.add_subplot(slot(k))
            ax.imshow(img, origin="lower", extent=extent, interpolation="nearest")
            ax.plot(pose[: t + 1, 0], pose[: t + 1, 1], color="white", lw=1.5)
            hx, hy = np.cos(pose[t, 2]), np.sin(pose[t, 2])
            ax.arrow(pose[t, 0], pose[t, 1], 0.5 * hx, 0.5 * hy, width=0.06, color="#E8590C", length_includes_head=True)
            if k == 2:  # scene graph on the ground truth: faint panel joints + every hazard
                for jx_ in joints:
                    ax.axvline(jx_, color="white", lw=0.6, alpha=0.35)
                draw_graph(ax, [True] * len(items), y1 - 0.35)
            else:
                draw_graph(ax, seen, y1 - 0.35)
            ax.set_title(title, fontsize=12)
            ax.set_ylabel("y [m]", fontsize=9)
            if k == 2:
                ax.set_xlabel("x along the walk [m]", fontsize=9)
            ax.tick_params(labelsize=8)
        obs = np.isfinite(fused)
        err = np.abs(fused - truth)[obs]
        cs = f" · cross slope {abs(cross['cross_slope']) * 100:.1f} % (ADA {cross['ada']})" if cross else ""
        fig.suptitle(f"t = {t * 0.1:4.1f} s  ·  mapped {obs.mean() * 100:4.1f} % of the area  ·  "
                     f"median |map − truth| {np.median(err) * 100 if err.size else float('nan'):.1f} cm  ·  "
                     f"scene-graph hazards in sensor view {sum(seen)}/{len(items)}{cs}\n"
                     f"height colour, shaded (grey = not seen) · red = slab step (ADA level change) · pink = ramp · "
                     f"yellow = crack · cyan = curb · coral = upheaval · purple = hole · blue = puddle · orange = obstruction · white = path",
                     fontsize=11)
        fig.canvas.draw()
        img = np.asarray(fig.canvas.buffer_rgba())[..., :3]
        writer.append_data(img)
    writer.close()
    plt.close(fig)
    print(f"wrote {out} ({T // a.stride} frames)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
