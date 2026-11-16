"""Figures that accompany the demo video: system architecture, scene graph, mapping.

  PYTHONPATH=src D:/go2wm/render_env/Scripts/python.exe scripts/make_demo_figures.py \
      --mission D:/go2wm/outputs/demo_mission/isaac/approach_tile0.json --out D:/go2wm/outputs/figures [--only arch|graph|map]

  arch   architecture of the system as it runs in the demo (learned vs hand-written parts marked; planned parts dashed)
  graph  the demo street's scene graph: ground-truth objects and relations (scene/graph.py + site objects + people)
         plus the robot's own findings from the mission, linked to what they matched
  map    the map the robot builds while walking: the recorded run is replayed through the same simulated D555 / D405
         (sim/offline_sensors.py, realistic profile) and every frame is placed with the robot's ESTIMATED pose
         (odometry drift + latency). Odometry-based mapping, no loop closure: not full SLAM.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mp  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

FRAMES = Path("D:/go2wm/outputs/demo_mission/rtx_standing")
INK, MUTED = "#1f2a30", "#5b666c"
C = {  # fill, edge
    "purple": ("#f1ecfb", "#7a5cc4"), "green": ("#e8f6ec", "#3f9a5a"), "blue": ("#e7f0fb", "#3f73c0"),
    "yellow": ("#fff6dd", "#d4a01c"), "red": ("#fdeaea", "#d0504f"), "grey": ("#f3f5f6", "#9aa5ab"),
    "lav": ("#f4f2fb", "#b8b0dc"),
}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "text.color": INK})


# ------------------------------------------------------------------ helpers
def box(ax, x, y, w, h, col, lw=1.6, r=1.2, ls="-", z=1):
    f, e = C[col]
    ax.add_patch(mp.FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=f, ec=e, lw=lw,
                                   ls=ls, zorder=z))


def title(ax, x, y, t, sub=None, size=14, col=INK):
    ax.text(x, y, t, fontsize=size, fontweight="bold", color=col, va="top", zorder=5)
    if sub:
        ax.text(x, y - size * 0.2, sub, fontsize=size * 0.72, color=MUTED, va="top", zorder=5)


def bullets(ax, x, y, items, size=9.6, dy=2.35, col=INK):
    for k, it in enumerate(items):
        ax.text(x, y - k * dy, "\u2022 " + it, fontsize=size, color=col, va="top", zorder=5)


def pill(ax, x, y, w, h, t, col="purple", size=9.2):
    f, e = C[col]
    ax.add_patch(mp.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.8", fc="white", ec=e, lw=1.1,
                                   zorder=4))
    ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=size, zorder=5)


def arrow(ax, a, b, col=INK, lw=1.8, both=False, ls="-"):
    ax.annotate("", b, a, arrowprops=dict(arrowstyle="<|-|>" if both else "-|>", color=col, lw=lw, ls=ls,
                                          shrinkA=0, shrinkB=0, mutation_scale=14), zorder=6)


def image(ax, path, x, y, w, h, crop=None):
    im = Image.open(path).convert("RGB")
    if crop:
        im = im.crop(crop)
    ax.imshow(np.asarray(im), extent=(x, x + w, y, y + h), zorder=3, aspect="auto")
    ax.add_patch(mp.Rectangle((x, y), w, h, fill=False, ec="#c9cfd2", lw=1, zorder=4))


def load(mission):
    r = json.loads(Path(mission).read_text())
    tiles = json.loads(Path(r["tiles_file"]).read_text())
    return r, tiles[int(r.get("tile", 0))]


def mini_map(ax, r, p, x0, y0, w, h):
    """Route strip with the robot's findings, drawn into the given rectangle of the architecture canvas."""
    L = p["size"][0]
    sx, sy = w / L, h / 7.0
    X = lambda x: x0 + (x + L / 2) * sx  # noqa: E731
    Y = lambda y: y0 + (y + 3.5) * sy  # noqa: E731
    half = p["width"] / 2
    ax.add_patch(mp.Rectangle((X(-L / 2), Y(-3.5)), w, h, fc="#dfe6e2", ec="#c9cfd2", zorder=3))
    ax.add_patch(mp.Rectangle((X(-L / 2), Y(-3.5)), w, (-half - 0.15 + 3.5) * sy, fc="#9aa2a6", zorder=3))
    ax.add_patch(mp.Rectangle((X(-L / 2), Y(-half)), w, 2 * half * sy, fc="#f4f4f2", zorder=3))
    ax.add_patch(mp.Rectangle((X(-4), Y(0.1)), 7 * sx, 1.4 * sy, fc="#ffd27a", ec="#d4a01c", lw=0.6, zorder=3))
    tr = np.asarray(r["trajectory"])
    ax.plot(X(tr[:, 0]), Y(tr[:, 1]), color="#3f73c0", lw=1.4, zorder=4)
    for f in r["memory"]["findings"].values():
        real = f["id"] not in r["score"]["false_ids"]
        col = "#d0504f" if real else "#9aa5ab"
        ax.plot(X(f["xy"][0]), Y(f["xy"][1]), "o", ms=4.2, mfc=col if f["status"] == "measured" else "white",
                mec=col, mew=1.2, zorder=5)


# ------------------------------------------------------------------ architecture
def fig_arch(r, p, out):
    fig = plt.figure(figsize=(18, 9.6), dpi=170)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 180)
    ax.set_ylim(0, 96)
    ax.axis("off")
    fig.patch.set_facecolor("white")
    ax.text(90, 94.6, "Autonomous Sidewalk Inspection", fontsize=19, fontweight="bold", ha="right", va="top")
    ax.text(90.8, 94.6, "(modular architecture, as run in the demo)", fontsize=15, color=MUTED, ha="left", va="top")

    # robot & environment
    box(ax, 1, 31, 31, 58, "lav")
    title(ax, 3, 87.5, "Robot & Environment", size=13.5)
    image(ax, FRAMES / "00560.png", 3, 62, 27, 21.5, crop=(60, 110, 560, 650))
    ax.text(16.5, 60.2, "Unitree Go2 / Go2-W", ha="center", fontsize=11.5, fontweight="bold", va="top")
    ax.text(16.5, 57.6, "Isaac Sim physics, RTX replay", ha="center", fontsize=9, color=MUTED, va="top")
    sens = [("D555 depth camera", "forward, 0.5-6 m: finds candidates"),
            ("D405 depth camera", "chin, 7-50 cm: close measurement"),
            ("Proprioception", "IMU, joint states, odometry"),
            ("Realistic errors", "noise, drift, latency, dropouts")]
    for k, (a, b) in enumerate(sens):
        yy = 53.5 - k * 5.4
        ax.add_patch(mp.Circle((5.2, yy - 1.3), 1.2, fc="white", ec=C["lav"][1], lw=1.2, zorder=4))
        ax.text(5.2, yy - 1.3, ["D", "D", "⊕", "±"][k], fontsize=9, fontweight="bold", color="#5b4aa0",
                ha="center", va="center", zorder=5)
        ax.text(8, yy, a, fontsize=10, fontweight="bold", va="top")
        ax.text(8, yy - 2.1, b, fontsize=8.6, color=MUTED, va="top")
    arrow(ax, (32.2, 62), (36.8, 62))
    ax.text(34.6, 66.2, "obser-\nvations", fontsize=8.4, ha="center")
    ax.text(34.6, 59.4, "0.1 s", fontsize=8, ha="center", color=MUTED)

    # core
    ax.add_patch(mp.FancyBboxPatch((37, 31), 99, 58, boxstyle="round,pad=0,rounding_size=2", fc="white",
                                   ec="#2b3a4a", lw=1.8, zorder=0))
    # perception
    box(ax, 40, 59, 41, 27, "purple")
    title(ax, 42, 84.6, "Perception & Measurement", size=12.5, col="#4b3592")
    image(ax, FRAMES / "00150.png", 42, 61.5, 15.5, 18, crop=(150, 220, 690, 600))
    ax.add_patch(mp.Rectangle((44.0, 68.3), 6.6, 4.4, fill=False, ec="#7a5cc4", lw=1.6, zorder=5))
    for k, t in enumerate(["Candidate detection", "Depth / height", "Shape vote", "Confidence"]):
        pill(ax, 59.5, 76.4 - k * 4.1, 19.5, 3.2, t)
    ax.text(69.25, 61.2, "D555 height grid, plane fits\non raw points (hand-written)", fontsize=7.8, color=MUTED,
            ha="center", va="bottom")
    # memory / scene graph
    box(ax, 87, 68, 46, 18, "green")
    title(ax, 99, 84.6, "Scene Graph & Spatial Memory", size=12.5, col="#2e7a45")
    gx, gy = np.array([90.5, 93, 95.5, 93, 96.5]), np.array([78.5, 82, 78.5, 74.5, 74.5])
    for a, b in [(0, 1), (1, 2), (2, 3), (3, 0), (2, 4)]:
        ax.plot(gx[[a, b]], gy[[a, b]], color="#3f9a5a", lw=1.2, zorder=4)
    for k, (x, y) in enumerate(zip(gx, gy)):
        ax.add_patch(mp.Circle((x, y), 0.95, fc="#d0504f" if k == 2 else "#9fd3ae", ec="#2e7a45", lw=1, zorder=5))
    bullets(ax, 99, 81.2, ["Findings map: merge, confirm from 2+ views", "Robust value: median, outliers dropped",
                           "Coverage per sensor", "Ground-truth graph (ADA classes) for scoring"], size=9.1, dy=2.6)
    # world model (planned)
    box(ax, 87, 51, 46, 14.5, "blue", ls=(0, (4, 3)))
    title(ax, 89, 64.2, "World Model", size=12.5, col="#2c5c9f")
    ax.text(105.5, 64.2, "PLANNED", fontsize=8.5, fontweight="bold", color="white", va="top",
            bbox=dict(boxstyle="round,pad=0.25", fc="#3f73c0", ec="none"), zorder=6)
    bullets(ax, 89, 60.6, ["Predict next state, stall / traversal risk", "Score actions before taking them",
                           "Now: fixed risk thresholds from a terrain sweep"], size=9.1, dy=2.6, col="#34485c")
    # active perception
    box(ax, 40, 33, 33, 23, "yellow")
    title(ax, 42, 54.6, "Active Perception", size=12.5, col="#8a6400")
    ax.text(42, 51.8, "stop & look (hand-written rules)", fontsize=8.8, color=MUTED, va="top")
    bullets(ax, 42, 48.4, ["Needs a close look: real, no D405 value", "Walk to stand-off, face it, crouch",
                           "Hold and measure, then resume", "One attempt per finding, time budget"], size=9.1, dy=2.9)
    # planner
    box(ax, 77, 33, 56, 16, "red")
    title(ax, 79, 47.8, "Hierarchical Planner & Control", size=12.5, col="#a33a39")
    ax.add_patch(mp.FancyBboxPatch((79, 40.2), 25.5, 5.2, boxstyle="round,pad=0,rounding_size=0.8", fc="white",
                                   ec=C["red"][1], lw=1, zorder=4))
    ax.text(80, 44.6, "High level (rules)", fontsize=9.6, fontweight="bold", va="top", zorder=5)
    ax.text(80, 42.4, "A* route, avoid holes, yield to people", fontsize=8.1, va="top", zorder=5)
    ax.add_patch(mp.FancyBboxPatch((106, 40.2), 25.5, 5.2, boxstyle="round,pad=0,rounding_size=0.8", fc="white",
                                   ec=C["red"][1], lw=1, zorder=4))
    ax.text(107, 44.6, "Low level (learned)", fontsize=9.6, fontweight="bold", va="top", zorder=5)
    ax.text(107, 42.4, "PPO walking policy + crouch IK", fontsize=8.1, va="top", zorder=5)
    ax.text(79, 38.6, "speed / turn command  \u2192  joint & wheel targets at 50 Hz", fontsize=8.6, color=MUTED, va="top")
    ax.text(79, 36.0, "Go2: NVIDIA pretrained policy   \u2022   Go2-W: trained by us (PPO, 2000 iterations)",
            fontsize=8.6, color=MUTED, va="top")
    # arrows in core
    arrow(ax, (81.2, 77), (86.8, 77))
    arrow(ax, (84, 77), (84, 58.3))
    arrow(ax, (84, 58.3), (86.8, 58.3))
    arrow(ax, (60.5, 58.8), (60.5, 56.2))
    arrow(ax, (110, 67.8), (110, 65.7))
    arrow(ax, (110, 50.8), (110, 49.2), ls=(0, (3, 2)))
    arrow(ax, (73.2, 41), (76.8, 41), both=True)
    arrow(ax, (136.2, 62), (140.8, 62))

    # outputs
    box(ax, 141, 31, 38, 58, "lav")
    title(ax, 149, 87.5, "Mission Output", size=13.5)
    ev = r["events"]
    sc = r["score"]
    n_insp = sum(e["event"] == "inspected" for e in ev)
    n_y = sum(e["event"] == "yield_pedestrian" for e in ev)
    meas = [d for d in sc["per_defect"] if d.get("best_sensor") == "depth_d405" and d["type"] == "hole"]
    box(ax, 143, 67, 34, 17, "grey", lw=1.1, r=1)
    title(ax, 145, 82.8, "Inspection Report", size=11.5)
    bullets(ax, 145, 79.6, ["Defects with type, location, size", "ADA class (1/4 in, 1/2 in rules)",
                            "Sensor + number of readings", "Confidence / spread per value"], size=8.9, dy=2.6)
    box(ax, 143, 48.5, 34, 17, "grey", lw=1.1, r=1)
    title(ax, 145, 64.3, "Updated Spatial Map", size=11.5)
    mini_map(ax, r, p, 144.5, 51.5, 31, 8.4)
    ax.text(145, 50.3, "route   \u25cf measured   \u25cb candidate   grey: false", fontsize=7.6, color=MUTED, va="top")
    box(ax, 143, 33, 34, 14, "grey", lw=1.1, r=1)
    title(ax, 145, 45.8, "Mission Metrics (Go2 run)", size=11.5)
    hole_err = np.median([d["abs_err_m"] for d in meas]) * 1000 if meas else float("nan")
    bullets(ax, 145, 42.6, [f"{r['outcome']} in {r['time_s']:.0f} s, {r['path_m']:.1f} m, no falls",
                            f"{n_insp} close inspections, {n_y} yield to a person",
                            f"found {sum(d['found'] for d in sc['per_defect'])}/{len(sc['per_defect'])} defects, "
                            f"{sc['false_findings']} false",
                            f"hole depth error {hole_err:.1f} mm (median)"], size=8.9, dy=2.4)

    # bottom row
    box(ax, 1, 2, 56, 26, "purple", lw=1.2)
    title(ax, 3, 26.6, "Walking Policy", size=12.5)
    ax.text(25.5, 26.2, "learned with PPO in Isaac Lab", fontsize=10, color=MUTED, va="top")
    image(ax, FRAMES / "00330.png", 3, 6.5, 13, 12, crop=(470, 250, 790, 540))
    ax.text(9.5, 5.2, "state s\u209c + command c\u209c", fontsize=8.8, ha="center", va="top")
    arrow(ax, (16.6, 12.5), (19.4, 12.5))
    ax.add_patch(mp.FancyBboxPatch((19.6, 8.5), 14, 8, boxstyle="round,pad=0,rounding_size=1", fc="white",
                                   ec=C["purple"][1], lw=1.4, zorder=4))
    ax.text(26.6, 14.4, "Policy", fontsize=11, fontweight="bold", ha="center", zorder=5)
    ax.text(26.6, 11.2, r"$\pi(a_t \mid s_t, c_t)$", fontsize=11, ha="center", zorder=5)
    arrow(ax, (33.8, 12.5), (36.6, 12.5))
    bullets(ax, 37.2, 20.6, ["+ track speed (3.0)", "+ track turn rate (1.5)", "\u2212 leg pose, jerk, tilt",
                             "\u2212 body contacts (falls)", "4096 robots in parallel"], size=8.8, dy=2.7)
    ax.text(37.2, 6.0, "outputs joint / wheel targets", fontsize=8.6, color=MUTED, va="top")

    box(ax, 60, 2, 50, 26, "green", lw=1.2)
    title(ax, 62, 26.6, "Scene Graph:", size=12.5)
    ax.text(82, 26.4, "persistent spatial memory", fontsize=11, color=MUTED, va="top")
    nx_ = {"p1": (65, 16), "p2": (71, 19), "p3": (77, 16), "d": (71, 11.5), "o": (79, 9.5), "t": (64.5, 8.5)}
    for a, b in [("p1", "p2"), ("p2", "p3"), ("d", "p2"), ("o", "d"), ("t", "d")]:
        ax.plot(*zip(nx_[a], nx_[b]), color="#55656c", lw=1.2, zorder=3)
    cols = {"p1": "#cfe3f7", "p2": "#cfe3f7", "p3": "#cfe3f7", "d": "#e86a69", "o": "#c9b8ee", "t": "#9fd3ae"}
    for k, (x, y) in nx_.items():
        ax.add_patch(mp.Circle((x, y), 1.5, fc=cols[k], ec="#3c4a52", lw=1, zorder=4))
    for k, (c, t) in enumerate([("#cfe3f7", "Panel (sidewalk slab)"), ("#e86a69", "Defect (hole, upheaval...)"),
                                ("#c9b8ee", "Finding / observation"), ("#9fd3ae", "Asset (tree, work zone)")]):
        ax.add_patch(mp.Circle((85, 20.2 - k * 3.6), 1.1, fc=c, ec="#3c4a52", lw=1, zorder=4))
        ax.text(87.2, 20.2 - k * 3.6, t, fontsize=9, va="center")
    ax.plot([84, 86.2], [5.8, 5.8], color="#55656c", lw=1.2)
    ax.text(87.2, 5.8, "relation (on, next, causes...)", fontsize=9, va="center")

    box(ax, 113, 2, 66, 26, "lav", lw=1.2)
    title(ax, 115, 26.6, "Decision Loop", size=12.5)
    ax.text(131, 26.4, "(every 0.1 s)", fontsize=11, color=MUTED, va="top")
    steps = [("1. Sense", "depth + odometry", "purple"), ("2. Detect &\nmeasure", "perception", "purple"),
             ("3. Update\nmemory", "scene graph", "green"), ("4. Decide", "walk / inspect /\nyield", "yellow"),
             ("5. Execute", "PPO policy", "red")]
    for k, (a, b, c) in enumerate(steps):
        x = 115 + k * 12.6
        ax.add_patch(mp.FancyBboxPatch((x, 10.5), 10.8, 10, boxstyle="round,pad=0,rounding_size=1", fc=C[c][0],
                                       ec=C[c][1], lw=1.3, zorder=4))
        ax.text(x + 5.4, 18.6, a, fontsize=9.2, fontweight="bold", ha="center", va="top", zorder=5)
        ax.text(x + 5.4, 13.6, b, fontsize=7.8, ha="center", va="center", color=MUTED, zorder=5)
        if k < 4:
            arrow(ax, (x + 10.9, 15.5), (x + 12.5, 15.5), lw=1.4)
    ax.plot([175.6, 175.6, 120.4, 120.4], [10.4, 6.2, 6.2, 10.0], color=INK, lw=1.4, zorder=3)
    arrow(ax, (120.4, 8), (120.4, 10.3), lw=1.4)
    ax.text(148, 4.2, "Ground truth is only used to score the mission afterwards", fontsize=8.6, color=MUTED,
            ha="center", va="center")
    fig.savefig(out, facecolor="white")
    plt.close(fig)


# ------------------------------------------------------------------ scene graph
ADA_COL = {"violation": "#d0504f", "bevel_required": "#e8913a", "monitor": "#8a969c", "ok": "#3f9a5a", "n/a": "#8a969c"}
ADA_TXT = {"violation": "ADA violation", "bevel_required": "ADA: bevel", "monitor": "ADA: monitor",
           "ok": "ADA ok", "n/a": "no ADA rule"}
REL = {"next": "#5d7d99", "on": "#7d8a90", "matches": "#2e8b4e", "occupies": "#c08a14", "causes": "#3f9a5a",
       "people": "#c9548a", "borders": "#9aa5ab"}


def gbox(ax, cx, cy, w, h, fc, ec, lines, stripe=None, ls="-", lw=1.2, z=4):
    """Rounded node box with its text inside: lines = [(text, size, weight, colour), ...]."""
    ax.add_patch(mp.FancyBboxPatch((cx - w / 2, cy - h / 2), w, h, boxstyle="round,pad=0,rounding_size=0.9",
                                   fc=fc, ec=ec, lw=lw, ls=ls, zorder=z))
    if stripe:
        ax.add_patch(mp.FancyBboxPatch((cx - w / 2, cy - h / 2), 1.0, h, boxstyle="round,pad=0,rounding_size=0.5",
                                       fc=stripe, ec="none", zorder=z + 0.1))
    tot = sum(s for _, s, _, _ in lines) * 0.2
    y = cy + tot / 2
    for t, s, wgt, c in lines:
        y -= s * 0.2 / 2
        ax.text(cx + (0.4 if stripe else 0), y, t, fontsize=s, fontweight=wgt, color=c, ha="center", va="center",
                zorder=z + 0.2)
        y -= s * 0.2 / 2


def gline(ax, a, b, col, lw=1.3, ls="-", arrow_=False, z=2):
    if arrow_:
        ax.annotate("", b, a, arrowprops=dict(arrowstyle="-|>", color=col, lw=lw, ls=ls, shrinkA=0, shrinkB=0,
                                              mutation_scale=11), zorder=z)
    else:
        ax.plot([a[0], b[0]], [a[1], b[1]], color=col, lw=lw, ls=ls, zorder=z, solid_capstyle="round")


def fig_graph(r, p, out):
    from go2wm.scene.graph import build_scene_graph

    g = build_scene_graph(p, x_range=(-p["size"][0] / 2, p["size"][0] / 2), name="demo street")
    N = g["nodes"]
    L0, SEG = -p["size"][0] / 2, 7.0
    nseg = int(round(p["size"][0] / SEG))
    seg_of = lambda x: int(np.clip((x - L0) // SEG, 0, nseg - 1))  # noqa: E731
    defects = sorted([(i, n) for i, n in N.items() if n["kind"] == "defect" and i.startswith("defect_")],
                     key=lambda kv: kv[1]["xy"][0])
    trees = {next(e["dst"] for e in g["edges"] if e["src"] == i and e["rel"] == "causes"): (i, n)
             for i, n in N.items() if n["kind"] == "asset"}
    sc = r["score"]
    found_by = {d["gt"]: d for d in sc["per_defect"]}
    fnd = r["memory"]["findings"]
    false_ids = sorted(sc["false_ids"], key=lambda f: fnd[f]["xy"][0])
    # columns: every defect, a tree next to the upheaval it lifts, every false finding
    slots = []
    for i, n in defects:
        slots.append(("defect", i, n["xy"][0]))
        if i in trees:
            slots.append(("tree", trees[i][0], n["xy"][0] + 1e-3))
    slots += [("false", f, fnd[f]["xy"][0]) for f in false_ids]
    slots.sort(key=lambda s: s[2])
    X0, X1 = 33.0, 199.0
    per_seg = [sum(seg_of(s[2]) == k for s in slots) or 1 for k in range(nseg)]
    dx = (X1 - X0) / sum(per_seg)
    seg_x, sx, cursor = [], {}, X0
    for k in range(nseg):
        mine = [s for s in slots if seg_of(s[2]) == k]
        seg_x.append(cursor + per_seg[k] * dx / 2)
        for j, s in enumerate(mine):
            sx[s[1]] = cursor + (j + 0.5) * dx
        cursor += per_seg[k] * dx
    W = dx - 0.6
    segw = [min(W * n_, 14.0) if n_ > 1 else W for n_ in per_seg]

    fig = plt.figure(figsize=(26, 14.5), dpi=150)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 200)
    ax.set_ylim(0, 112)
    ax.axis("off")
    ax.text(2, 110, "Demo street scene graph", fontsize=24, fontweight="bold", va="top")
    ax.text(2, 105.6, "Objects and relations on the 56 m demo street (ground truth), and what the Go2 found on its "
            "stop & look mission. Columns run along the street from start (left) to goal (right).",
            fontsize=13, color=MUTED, va="top")
    bands = [("People & work site", 79, 99, "#fff8e6", "#ecd9a6"),
             ("Places (street segments)", 57, 75, "#eef4fb", "#c9d9ea"),
             ("Defects & assets (ground truth)", 31, 53, "#fdf0f0", "#efcaca"),
             ("Robot's findings (its belief)", 7, 27, "#f3f0fb", "#d6cdee")]
    for t, y0, y1, fc, ec in bands:
        ax.add_patch(mp.FancyBboxPatch((1, y0), 198.5, y1 - y0, boxstyle="round,pad=0,rounding_size=1.5", fc=fc,
                                       ec=ec, lw=1.2, zorder=0))
        ax.text(2.6, y1 - 1.4, t, fontsize=13, fontweight="bold", color=INK, va="top", zorder=1)

    # --- places: chain of segments + regions
    yP = 65
    for k in range(nseg):
        a, b = L0 + k * SEG, L0 + (k + 1) * SEG
        gbox(ax, seg_x[k], yP, segw[k], 7, "#dbe9f7", "#5d7d99",
             [(f"Segment {k + 1}", 10, "bold", INK), (f"x {a:+.0f} to {b:+.0f} m", 8.6, "normal", MUTED)])
        if k:
            gline(ax, (seg_x[k - 1] + segw[k - 1] / 2, yP), (seg_x[k] - segw[k] / 2, yP), REL["next"], lw=1.8,
                  arrow_=True)
    gbox(ax, 9.5, 66.0, 14, 5.6, "#e9edf0", "#8a969c", [("Walkway", 9.6, "bold", INK),
                                                       ("3 m wide, 1.2 % slope", 7.8, "normal", MUTED)])
    for yy, t in ((70.0, "Grass verge"), (65.5, "Curb 15 cm"), (61.0, "Road")):
        gbox(ax, 24.0, yy, 9.5, 3.4, "#e9edf0", "#8a969c", [(t, 8.4, "normal", INK)])
    for yy in (70.0, 65.5):
        gline(ax, (16.5, 66.0), (19.25, yy), REL["borders"], lw=1.0)
    gline(ax, (24.0, 63.8), (24.0, 62.7), REL["borders"], lw=1.0)
    ax.text(9.5, 61.6, "borders verge and curb;\ncurb borders the road", fontsize=7.6, color=MUTED, ha="center",
            va="center", style="italic")
    ax.text(seg_x[0] - segw[0] / 2, yP + 5.2, "each segment is part of the walkway", fontsize=8.2, color="#5d7d99",
            style="italic", va="center")

    # --- defects & assets row, findings row
    yD, yF = 41, 16
    for kind, i, x in slots:
        cx = sx[i]
        if kind == "defect":
            n = N[i]
            t = n["defect"]
            name = {"upheaval": "Upheaval", "hole": "Hole", "puddle": "Puddle", "crack": "Open crack",
                    "obstruction": "Pole"}[t]
            val = {"upheaval": f"+{n.get('lift_m', 0) * 100:.1f} cm lift",
                   "hole": f"{n.get('depth_m', 0) * 100:.0f} cm deep",
                   "puddle": f"{n.get('area_m2', 0):.2f} m²",
                   "crack": f"{n.get('width_m', 0) * 1000:.0f} mm × {n.get('length_m', 0):.1f} m",
                   "obstruction": f"{n.get('clear_width_m', 0):.1f} m clear width"}[t]
            ada = n.get("ada", "n/a")
            gbox(ax, cx, yD, W, 10, "white", "#c98585",
                 [(name, 10.5, "bold", INK), (val, 8.4, "normal", INK),
                  (ADA_TXT.get(ada, ada), 8.4, "bold", ADA_COL.get(ada, MUTED))], stripe=ADA_COL.get(ada, MUTED))
            gline(ax, (cx, yD + 5), (seg_x[seg_of(n["xy"][0])], yP - 3.5), REL["on"], lw=1.1)
            d = found_by.get(i)
            if d is None:  # not scored: cracks need an RGB camera; the pole is an obstacle the planner avoids
                note = "not scored\n(cracks need RGB)" if t == "crack" else "obstacle:\nplanner avoids it"
                ax.text(cx, yF, note, fontsize=8.4, color=MUTED, ha="center", va="center", style="italic", zorder=5)
            elif not d["found"]:
                gbox(ax, cx, yF, W, 10, "#fff5f5", "#d0504f",
                     [("MISSED", 10.5, "bold", "#d0504f"), ("too small for", 8.2, "normal", INK),
                      ("the D555 detector", 8.2, "normal", INK)], ls=(0, (3, 2)))
                gline(ax, (cx, yF + 5), (cx, yD - 5), "#d0504f", lw=1.1, ls=(0, (3, 2)))
            else:
                f = fnd[d["finding"]]
                v = f.get("value_m")
                kind_txt = {"depression": "Depression", "edge": "Raised edge", "dropout": "No depth data"}[f["kind"]]
                sens = (f.get("best_sensor") or "depth_d555")[-4:].upper()
                err = d.get("abs_err_m")
                ok = err is None or err < 0.005
                ec = "#2e8b4e" if ok else "#e8913a"
                lines = [(kind_txt, 9.6, "bold", INK),
                         (f"{abs(v) * 1000:.1f} mm" if v is not None else "flagged as water", 9, "normal", INK),
                         (f"{sens} · {f['status']}", 8.2, "normal", MUTED)]
                if err is not None:
                    lines.append((f"error {err * 1000:.1f} mm", 8.4, "bold", ec))
                gbox(ax, cx, yF, W, 11.5, "white", ec, lines, lw=1.5)
                gline(ax, (cx, yF + 5.75), (cx, yD - 5), REL["matches"], lw=2.0)
        elif kind == "tree":
            gbox(ax, cx, yD, W, 6.5, "#e3f4e8", "#3f9a5a", [("Street tree", 9.8, "bold", INK),
                                                            ("roots lift the slab", 8.0, "normal", MUTED)])
            gline(ax, (cx - W / 2, yD), (cx - dx + W / 2 + 0.05, yD), REL["causes"], lw=1.8, arrow_=True)
        else:  # false finding
            f = fnd[i]
            v = f.get("value_m")
            gbox(ax, cx, yF, W, 11.5, "#f4f5f6", "#9aa5ab",
                 [("Depression", 9.6, "bold", MUTED), (f"{abs(v) * 1000:.1f} mm" if v else "-", 9, "normal", MUTED),
                  (f"D555 · {f['status']}", 8.2, "normal", MUTED), ("FALSE: clutter", 8.4, "bold", "#d0504f")])
            ax.text(cx, yD, "no defect\nhere", fontsize=8.4, color="#a9b2b6", ha="center", va="center", style="italic")
            gline(ax, (cx, yF + 5.75), (cx, yD - 3), "#c3cacd", lw=1.0, ls=(0, (2, 2)))

    gbox(ax, 15, 15.5, 22, 11, "#3f73c0", "#2c5c9f",
         [("Go2 robot", 11.5, "bold", "white"), (f"{r['outcome']} in {r['time_s']:.0f} s, {r['path_m']:.0f} m", 8.8, "normal", "white"),
          (f"{len(fnd)} findings, {sc['false_findings']} false", 8.8, "normal", "white")])

    # --- people & work site
    site = p["site"]
    objs = site["objects"]
    zone = site["zones"][0]
    zx = [k for k in range(nseg) if L0 + k * SEG < zone["x"][1] and L0 + (k + 1) * SEG > zone["x"][0]]
    xa = seg_x[zx[0]] - per_seg[zx[0]] * dx / 2 + 0.6
    xb = seg_x[zx[-1]] + per_seg[zx[-1]] * dx / 2 - 0.6
    yS = 88
    ax.add_patch(mp.FancyBboxPatch((xa, yS - 7), xb - xa, 15.5, boxstyle="round,pad=0,rounding_size=1",
                                   fc="#ffeec4", ec="#c08a14", lw=1.4, zorder=3))
    ax.text((xa + xb) / 2, yS + 6.0, "Work zone  (left half of the walk closed)", fontsize=10.5, fontweight="bold",
            ha="center", va="center", zorder=5)
    cnt = {}
    for o in objs:
        cnt[o["kind"]] = cnt.get(o["kind"], 0) + 1
    chips = [f"{cnt.get('barrier', 0)} barriers", f"{cnt.get('cone', 0)} cones", "soil pile", "generator",
             "worker, digging", "worker, standing", "worker, walking", "closed sign"]
    cw = (xb - xa - 2) / 4
    for k, t in enumerate(chips):
        cx = xa + 1 + cw * (k % 4) + cw / 2
        cy = yS + 1.6 - (k // 4) * 4.6
        person = t.startswith("worker")
        gbox(ax, cx, cy, cw - 0.8, 3.6, "#fde3ef" if person else "white", "#c9548a" if person else "#d4a01c",
             [(t, 8.6, "normal", INK)], lw=1.0, z=5)
    for k in zx:
        gline(ax, (seg_x[k], yS - 7), (seg_x[k], yP + 3.5), REL["occupies"], lw=1.8, arrow_=True)
        ax.text(seg_x[k] + 0.8, 77, "occupies", fontsize=8.2, color=REL["occupies"], style="italic", va="center")
    yields = [e for e in r["events"] if e["event"] == "yield_pedestrian"]
    peds = [m for m in objs if "path" in m and not (zone["x"][0] <= m["path"][0][0] <= zone["x"][1])]
    for m in peds:
        x0_ = m["path"][0][0]
        cross = abs(m["path"][0][0] - m["path"][-1][0]) < 1.0
        kseg = seg_of(min(x0_, L0 + nseg * SEG - 0.1))
        cx = min(seg_x[kseg], X1 - max(W, 14) / 2)
        lines = [("Pedestrian", 10.5, "bold", INK),
                 ("crosses the walk" if cross else "walks toward the robot", 8.6, "normal", INK)]
        if cross and yields:
            lines.append((f"robot yielded at t = {yields[0]['t']:.0f} s", 8.4, "bold", "#c9548a"))
        elif not cross:
            lines.append(("steps aside to pass", 8.4, "normal", MUTED))
        gbox(ax, cx, yS, max(W, 14), 10.5, "#fde3ef", "#c9548a", lines)
        gline(ax, (cx, yS - 5.25), (seg_x[kseg], yP + 3.5), REL["people"], lw=1.6, arrow_=True)
        ax.text(cx + 0.8, 77, "crosses" if cross else "walks along", fontsize=8.2, color=REL["people"],
                style="italic", va="center")

    # --- legend
    items = [("next (along the street)", REL["next"]), ("on (defect is on a segment)", REL["on"]),
             ("matches (finding = defect)", REL["matches"]), ("occupies", REL["occupies"]),
             ("crosses / walks along", REL["people"]), ("causes", REL["causes"])]
    xs = [3, 29, 59, 89, 106, 131]
    for x, (t, c) in zip(xs, items):
        ax.plot([x, x + 3.5], [3.2, 3.2], color=c, lw=2.4)
        ax.text(x + 4.3, 3.2, t, fontsize=10.5, va="center")
    for k, (t, c) in enumerate([("ADA violation", ADA_COL["violation"]), ("bevel needed", ADA_COL["bevel_required"]),
                                ("monitor / no rule", ADA_COL["monitor"])]):
        x = 147 + k * 17
        ax.add_patch(mp.Rectangle((x, 2.1), 1.2, 2.2, fc=c, ec="none"))
        ax.text(x + 2.0, 3.2, t, fontsize=10.5, va="center")
    fig.savefig(out, facecolor="white")
    plt.close(fig)


# ------------------------------------------------------------------ mapping
def fig_map(r, p, out, stride=2):
    from go2wm.sim.offline_sensors import OfflineRig, quat_to_R
    from go2wm.sim.terrain import params_with_defaults, sidewalk_height

    p = params_with_defaults(p)
    objs = [o for o in p["site"]["objects"] if o.get("r", 0) > 0 or o["kind"] in ("pile", "generator", "sign")]
    OBJ_H = {"barrier": 1.0, "cone": 0.7, "worker": 1.75, "pile": 0.6, "generator": 0.8, "sign": 1.2}
    OBJ_R = {"pile": 0.45, "generator": 0.4, "sign": 0.25}
    trees = [d["tree"] for d in p["defects"] if d.get("tree")]
    statics = [(o["xy"][0], o["xy"][1], o.get("r") or OBJ_R[o["kind"]], OBJ_H[o["kind"]]) for o in objs if "xy" in o]
    statics += [(t[0], t[1], 0.25, 3.0) for t in trees]
    statics += [(d["x"], d["y"], d["r"], 1.5) for d in p["defects"] if d["type"] == "obstruction"]
    S = np.array(statics)
    movers = [m for m in p["site"]["objects"] if "path" in m]

    links, est = r["links"], np.asarray(r["pose_estimate"])
    tr = np.asarray(r["trajectory"])
    track = r["site_track"]

    def Hfun(k):
        mv = np.array([[q[0], q[1]] for q in track[min(k, len(track) - 1)]]) if track else np.zeros((0, 2))

        def H(x, y):
            z = np.asarray(sidewalk_height(x, y, p), dtype=np.float64).copy()
            for (ox, oy, rr, hh) in S:
                z = np.where((x - ox) ** 2 + (y - oy) ** 2 < rr ** 2, z + hh, z)
            for (ox, oy) in mv:
                z = np.where((x - ox) ** 2 + (y - oy) ** 2 < 0.3 ** 2, z + 1.7, z)
            return z
        return H

    rig = OfflineRig(["depth_d555", "depth_d405"], "realistic", seed=7, defects=p.get("defects"))
    res = 0.05
    x0, x1, y0, y1 = -28.5, 28.5, -5.0, 4.0
    nx, ny = int((x1 - x0) / res), int((y1 - y0) / res)
    zsum = np.zeros((ny, nx))
    zcnt = np.zeros((ny, nx))
    d405 = np.zeros((ny, nx), bool)
    snaps, snap_t = {}, (30.0, 65.0)
    for k in range(0, len(links), stride):
        L = links[k]
        bp = np.asarray(L["pos"][0])
        R = quat_to_R(L["quat"][0])
        H = Hfun(k)
        ex, ey, eyaw = est[k]
        tyaw = tr[k][2]
        dyaw = eyaw - tyaw  # the robot believes it is rotated by the drift
        Rz = np.array([[np.cos(dyaw), -np.sin(dyaw), 0], [np.sin(dyaw), np.cos(dyaw), 0], [0, 0, 1]])
        Rb = Rz @ R
        for s in ("depth_d555", "depth_d405"):
            pts = rig.frame(s, bp, R, H)
            ok = np.all(np.isfinite(pts), 1)
            w = pts[ok] @ Rb.T + np.array([ex, ey, bp[2]])
            i = ((w[:, 0] - x0) / res).astype(int)
            j = ((w[:, 1] - y0) / res).astype(int)
            m = (i >= 0) & (i < nx) & (j >= 0) & (j < ny)
            np.add.at(zsum, (j[m], i[m]), w[m, 2])
            np.add.at(zcnt, (j[m], i[m]), 1)
            if s == "depth_d405":
                d405[j[m], i[m]] = True
        t = L["t"]
        for ts in snap_t:
            if ts not in snaps and t >= ts:
                snaps[ts] = (zcnt > 0).copy(), k
    zm = np.where(zcnt > 0, zsum / np.maximum(zcnt, 1), np.nan)
    # ground reference: the true surface without objects (the map is shown as height above the street level)
    gx = x0 + (np.arange(nx) + 0.5) * res
    gy = y0 + (np.arange(ny) + 0.5) * res
    GX, GY = np.meshgrid(gx, gy)
    Z0 = 0.012 * np.clip(GY, -1.5, 1.5)  # cross slope only (the map's colour = local relief)
    rel = zm - Z0

    fig = plt.figure(figsize=(22, 12), dpi=150)
    fig.text(0.012, 0.978, "Mapping the street on the mission", fontsize=24, fontweight="bold", va="top")
    fig.text(0.012, 0.94, "Height map built from the robot's depth cameras as it walks. Each frame is placed with the "
             "robot's own pose estimate (odometry with drift and latency). There is no loop closure, so this is "
             "mapping with odometry, not full SLAM.", fontsize=12.5, color=MUTED, va="top")
    ext = (x0, x1, y0, y1)
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

    cmap = LinearSegmentedColormap.from_list("relief", ["#173f8a", "#5d8fd6", "#eef0ef", "#f2b561", "#c4471c",
                                                        "#5a1d0c"])
    norm = TwoSlopeNorm(vmin=-0.16, vcenter=0.0, vmax=0.6)
    # snapshots
    for n, ts in enumerate(snap_t):
        ax = fig.add_axes([0.03, 0.75 - n * 0.16, 0.94, 0.135])
        mask, k = snaps[ts]
        ax.imshow(np.where(mask, rel, np.nan), origin="lower", extent=ext, cmap=cmap, norm=norm, interpolation="nearest", aspect="auto")
        ax.plot(tr[:k, 0], tr[:k, 1], color="#1f2a30", lw=1.0, alpha=0.6)
        ax.plot(*tr[k, :2], "o", ms=8, mfc="#3f73c0", mec="white", mew=1.5)
        ax.set_facecolor("#20262a")
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.text(0.003, 0.93, f"t = {ts:.0f} s", transform=ax.transAxes, fontsize=12, fontweight="bold", color="white",
                va="top")
    # final
    ax = fig.add_axes([0.03, 0.30, 0.94, 0.27])
    ax.set_facecolor("#20262a")
    im = ax.imshow(rel, origin="lower", extent=ext, cmap=cmap, norm=norm, interpolation="nearest", aspect="auto")
    ax.contour(GX, GY, d405.astype(float), levels=[0.5], colors="#7fe0ff", linewidths=0.6)
    ax.plot(tr[:, 0], tr[:, 1], color="white", lw=1.6, label="true path")
    ax.plot(est[:, 0], est[:, 1], color="#ffd84a", lw=1.3, ls=(0, (4, 2)), label="robot's estimate (odometry)")
    for f in r["memory"]["findings"].values():
        false = f["id"] in r["score"]["false_ids"]
        ax.plot(*f["xy"], marker="s", ms=10, mfc="none", mec="#9aa5ab" if false else "#ff4d6d", mew=2)
    for (ox, oy) in [m["path"][0] for m in movers]:
        pass
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_xlabel("along the street [m]", fontsize=11)
    ax.set_ylabel("across [m]", fontsize=11)
    ax.text(0.003, 0.95, f"end of mission, t = {r['time_s']:.0f} s", transform=ax.transAxes, fontsize=12,
            fontweight="bold", color="white", va="top")
    for (x, y, t) in [(-18.75, -2.9, "tree"), (-0.5, 2.0, "work zone: barriers, cones, workers"), (-31, 0, ""),
                      (-24, -3.7, "road (15 cm below the curb)"), (-24, 2.6, "grass verge")]:
        if t:
            ax.text(x, y, t, fontsize=9.5, color="white", ha="center",
                    bbox=dict(boxstyle="round,pad=0.2", fc="#000000", alpha=0.45, ec="none"))
    leg = ax.legend(loc="lower right", fontsize=9.5, facecolor="#20262a", edgecolor="#55656c", labelcolor="white")
    leg.set_zorder(10)
    cax = fig.add_axes([0.03, 0.235, 0.3, 0.014])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.set_label("height relative to the street surface [m]  (holes blue, obstacles red)", fontsize=10)
    # drift + coverage panels
    ax2 = fig.add_axes([0.40, 0.055, 0.27, 0.15])
    t = np.array([L["t"] for L in links])
    err = np.linalg.norm(est[:, :2] - tr[:, :2], axis=1) * 100
    ax2.plot(t[: len(err)], err, color="#c4471c", lw=1.6)
    ax2.set_xlabel("mission time [s]")
    ax2.set_ylabel("pose error [cm]")
    ax2.set_title("Pose error of the robot's estimate (drift + camera latency)", fontsize=11, loc="left")
    ax2.grid(alpha=0.3)
    ax3 = fig.add_axes([0.72, 0.055, 0.25, 0.15])
    cov = r["memory"]["coverage"]
    lab = [f"{c['x0']:.0f}..{c['x1']:.0f}" for c in cov["depth_d555"]]
    xx = np.arange(len(lab))
    ax3.bar(xx - 0.2, [c["seen"] * 100 for c in cov["depth_d555"]], 0.4, color="#5d8fd6", label="D555 (forward)")
    ax3.bar(xx + 0.2, [c["seen"] * 100 for c in cov["depth_d405"]], 0.4, color="#7fd0e6", label="D405 (close)")
    ax3.set_xticks(xx, lab, fontsize=8.5)
    ax3.set_ylabel("walkway seen [%]")
    ax3.set_xlabel("street section [m]")
    ax3.set_title("Coverage per sensor (robot's own record)", fontsize=11, loc="left")
    ax3.legend(fontsize=8.5, loc="center right")
    ax3.set_ylim(0, 105)
    fig.text(0.03, 0.17, "\u25a1 red: robot finding     \u25a1 grey: false finding\n"
             "cyan outline: ground seen by the close-range D405\nwhite: true path   yellow dashed: robot's estimate",
             fontsize=10.5, va="top", color=INK)
    fig.text(0.03, 0.09, f"Replayed from the recorded Isaac run ({len(links)} steps, every {stride}nd frame) through "
             "the same simulated cameras\n(realistic noise profile). People are moving and appear as short streaks.",
             fontsize=9.5, va="top", color=MUTED)
    fig.savefig(out, facecolor="white")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mission", default="D:/go2wm/outputs/demo_mission/isaac/approach_tile0.json")
    ap.add_argument("--out", default="D:/go2wm/outputs/figures")
    ap.add_argument("--only", default=None, choices=["arch", "graph", "map"])
    ap.add_argument("--stride", type=int, default=2)
    a = ap.parse_args()
    r, p = load(a.mission)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.only in (None, "arch"):
        fig_arch(r, p, out / "architecture.png")
        print("architecture.png")
    if a.only in (None, "graph"):
        fig_graph(r, p, out / "scene_graph.png")
        print("scene_graph.png")
    if a.only in (None, "map"):
        fig_map(r, p, out / "mapping.png", stride=a.stride)
        print("mapping.png")


if __name__ == "__main__":
    main()
