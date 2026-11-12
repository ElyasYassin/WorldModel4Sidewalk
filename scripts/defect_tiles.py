"""Generate random defect-rich sidewalk tiles and check them offline (no Isaac).

For each tile writes, under --out:
  tile_<i>.json        tile params (terrain + defects), ready for configs (terrain_tiles)
  graph_<i>.json       ground-truth scene graph (defect nodes, hazards with ADA class and robot risk)
  tile_<i>.png         three views on the same axes: appearance (the texture Isaac uses), height field as the
                       simulator meshes it (fine grid, hill-shaded), and the scene-graph hazards
plus tiles.json (all params) and summary.json (defect / hazard counts).

  python scripts/defect_tiles.py --n 4 --seed 0 --out outputs/defect_tiles
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from go2wm.common.config import resolve_path  # noqa: E402
from go2wm.scene.graph import build_scene_graph, hazard_summary, save_graph  # noqa: E402
from go2wm.sim.sidewalk_texture import sidewalk_albedo  # noqa: E402
from go2wm.sim.terrain import params_with_defaults, random_defect_tile, sidewalk_heightfield  # noqa: E402

TYPE_COLORS = {"upheaval": "#E03131", "crack": "#FCC419", "hole": "#7048E8", "puddle": "#1C7ED6",
               "obstruction": "#F76707", "vertical_displacement": "#C92A2A", "running_slope": "#D6336C",
               "cross_slope": "#868E96", "curb_edge": "#22B8CF"}


def render(p, g, path, title):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LightSource
    from matplotlib.patches import Circle, Rectangle

    sx, sy = p["size"]
    x0, x1, y0, y1 = -sx / 2, sx / 2, -min(3.0, sy / 2), min(3.0, sy / 2)
    res = 0.01
    xs, ys = np.arange(x0, x1, res), np.arange(y0, y1, res)
    X, Y = np.meshgrid(xs, ys)
    alb = sidewalk_albedo(X, Y, p, seed=0)
    hs, vs = p["horizontal_scale"], p["vertical_scale"]
    hf = sidewalk_heightfield(tuple(p["size"]), hs, vs, p).astype(np.float64) * vs  # [nx, ny], scene origin centred
    gx = np.arange(hf.shape[0]) * hs - 0.5 * sx
    gy = np.arange(hf.shape[1]) * hs - 0.5 * sy
    mx, my = (gx >= x0) & (gx < x1), (gy >= y0) & (gy < y1)
    H = hf[np.ix_(mx, my)].T
    shade = LightSource(315, 40).shade(H, cmap=plt.get_cmap("viridis"), blend_mode="soft", vert_exag=6, dx=hs, dy=hs,
                                       vmin=np.percentile(H, 1), vmax=np.percentile(H, 99))
    fig, axes = plt.subplots(3, 1, figsize=(18, 10), sharex=True)
    ext = (x0, x1, y0, y1)
    axes[0].imshow(alb, origin="lower", extent=ext, interpolation="bilinear")
    axes[0].set_title("Appearance (texture baked onto the Isaac terrain)", fontsize=11)
    axes[1].imshow(shade, origin="lower", extent=ext, interpolation="nearest")
    axes[1].set_title(f"Height field as meshed ({hs * 100:.0f} cm grid, {vs * 1000:.0f} mm steps; hill-shaded; "
                      f"range {np.percentile(H, 1) * 100:.0f} to {np.percentile(H, 99) * 100:.0f} cm)", fontsize=11)
    ax = axes[2]
    ax.imshow(alb * 0.45 + 0.55, origin="lower", extent=ext)
    n_lab = 0
    for h in g["hazards"]:
        t = h["type"]
        if t in ("cross_slope", "curb_edge"):
            continue
        c = TYPE_COLORS.get(t, "k")
        b = h["box"]
        n = g["nodes"].get(h["id"], {})
        if t == "crack" and "polyline" in n:
            pts = np.asarray(n["polyline"])
            ax.plot(pts[:, 0], pts[:, 1], color=c, lw=2.5 if n.get("defect") else 1.0)
        elif t == "obstruction":
            ax.add_patch(Circle(h["xy"], 0.5 * (b[1] - b[0]), color=c, alpha=0.8))
        else:
            ax.add_patch(Rectangle((b[0], b[2]), b[1] - b[0], b[3] - b[2], fill=False, ec=c, lw=2))
        if n.get("kind") == "defect" and n.get("defect"):
            n_lab += 1
            ax.text(h["xy"][0], b[3] + 0.08 + 0.45 * (n_lab % 2), n["label"] + (" (ADA)" if h["ada"] == "violation" else ""), fontsize=7.5,
                    ha="center", va="bottom", color="white",
                    bbox={"facecolor": c, "alpha": 0.9, "pad": 1.5, "edgecolor": "none"})
    for i, nd in g["nodes"].items():
        if nd["kind"] == "asset":
            ax.add_patch(Circle(nd["xy"], 0.2, color="#2B8A3E"))
            ax.text(nd["xy"][0], nd["xy"][1] + 0.25, "tree", fontsize=7.5, ha="center", color="#2B8A3E")
    s = hazard_summary(g)
    ax.set_title("Scene graph: defects and hazards (label = true size; ADA = violation) · "
                 + ", ".join(f"{k} {v}" for k, v in sorted(s["by_type"].items())), fontsize=11)
    for a in axes:
        a.set_ylabel("y [m]")
        a.set_aspect("equal")
    axes[2].set_xlabel("x along the walk [m]")
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--length", type=float, default=24.0)
    ap.add_argument("--out", default="outputs/defect_tiles")
    ap.add_argument("--no-render", action="store_true")
    a = ap.parse_args()
    out = resolve_path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tiles, summary = [], []
    for i in range(a.n):
        rng = np.random.default_rng(a.seed * 1000 + i)
        p = params_with_defaults(random_defect_tile(rng, a.length))
        tiles.append(p)
        g = build_scene_graph(p, tile_index=i, x_range=(-a.length / 2 + 0.5, a.length / 2 - 0.5), name=f"defect_tile_{i}")
        (out / f"tile_{i}.json").write_text(json.dumps(p, indent=1))
        save_graph(g, out / f"graph_{i}.json")
        types = [d["type"] for d in p["defects"]]
        summary.append({"tile": i, "defects": {t: types.count(t) for t in sorted(set(types))}, **hazard_summary(g)})
        if not a.no_render:
            render(p, g, out / f"tile_{i}.png", f"defect tile {i} (seed {a.seed}): {len(types)} defects, width "
                   f"{p['width']:.2f} m, grade {p['grade']}, curb {p['curb_drop']:.2f} m")
        print(f"tile {i}: {summary[-1]['defects']}  hazards {summary[-1]['n_hazards']}  ADA violations "
              f"{summary[-1]['ada_violations']}  high robot risk {summary[-1]['robot_high_risk']}")
    (out / "tiles.json").write_text(json.dumps(tiles, indent=1))
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
