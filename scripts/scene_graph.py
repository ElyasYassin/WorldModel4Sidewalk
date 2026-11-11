"""Build ground-truth sidewalk scene graphs for a tile set; save JSON, a text description and a figure.

  python scripts/scene_graph.py --split test [--tiles configs/e4_tiles.yaml] [--out outputs/scene_graphs]

Per tile: <out>/<split>_tile<i>.json (graph), .txt (description + hazard list), .png (graph drawn
over the top-down sidewalk: panels, features, hazards coloured by robot risk, cracks, keep-out boxes,
edges between panels labelled with the joint step).
"""

from __future__ import annotations

import argparse
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402

from go2wm.common.config import resolve_path  # noqa: E402
from go2wm.eval.plots import draw_sidewalk  # noqa: E402
from go2wm.scene.graph import build_scene_graph, describe, hazard_summary, keep_out_boxes, save_graph  # noqa: E402

RISK_COLOR = {"none": "#7f7f7f", "low": "#2ca02c", "medium": "#ff7f0e", "high": "#d62728"}


def draw_graph(g: dict, out) -> None:
    t = g["meta"]["terrain"]
    fig, ax = plt.subplots(figsize=(15, 6.5))
    draw_sidewalk(ax, t, xlim=(-1.5, 14.5), ylim=(-4.0, 4.0), seed=g["meta"]["tile_index"])
    for i, n in g["nodes"].items():  # panel outlines and centres (graph nodes)
        if n["kind"] == "panel":
            x0, x1, y0, y1 = n["box"]
            ax.add_patch(mpatches.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec="#1f77b4", lw=0.6, alpha=0.6))
    cen = {i: ((n["box"][0] + n["box"][1]) / 2, (n["box"][2] + n["box"][3]) / 2) for i, n in g["nodes"].items()
           if n["kind"] == "panel"}
    for e in g["edges"]:
        if e["rel"] == "next" and e["src"] in cen:
            (xa, ya), (xb, yb) = cen[e["src"]], cen[e["dst"]]
            col = {"ok": "#1f77b4", "bevel_required": "#ff7f0e", "violation": "#d62728"}[e["ada"]]
            ax.plot([xa, xb], [ya, yb], "-", color=col, lw=1.2, zorder=4)
            if e["ada"] != "ok":
                ax.text(e["joint_x"], ya + 0.18, f"{e['step_m'] * 100:+.0f}", fontsize=7, color=col, ha="center", zorder=6,
                        bbox=dict(fc="white", ec="none", alpha=0.7, pad=0.5))
    ax.scatter(*np.array(list(cen.values())).T, s=10, c="#1f77b4", zorder=5)
    for k in keep_out_boxes(g, "high"):
        x0, x1, y0, y1 = k["box"]
        ax.add_patch(mpatches.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=True, fc="#d62728", alpha=0.12, ec="#d62728",
                                        lw=1.5, ls="--", zorder=3))
    for i, n in g["nodes"].items():
        if n["kind"] == "defect":
            pts = np.array(n["polyline"])
            ax.plot(pts[:, 0], pts[:, 1], "-", color="#9467bd", lw=1.8, zorder=6)
    for lm in ("start", "end"):
        x, y = g["nodes"][lm]["xy"]
        ax.plot(x, y, marker="*", ms=14, color="gold", mec="k", zorder=7)
        ax.text(x, y - 0.35, lm, ha="center", fontsize=8, zorder=7)
    handles = [mpatches.Patch(color=RISK_COLOR[r], label=f"robot risk {r}") for r in ("low", "medium", "high")]
    handles += [plt.Line2D([], [], color="#1f77b4", label="panel node / next edge"),
                plt.Line2D([], [], color="#d62728", label="joint: ADA violation (cm)"),
                plt.Line2D([], [], color="#9467bd", label="crack"),
                mpatches.Patch(fc="#d62728", alpha=0.2, ec="#d62728", ls="--", label="keep-out (high risk)")]
    for h in g["hazards"]:
        if h["type"] in ("vertical_displacement", "running_slope", "curb_edge"):
            x0, x1, y0, y1 = h["box"]
            ax.add_patch(mpatches.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec=RISK_COLOR[h["robot_risk"]], lw=2.2,
                                            zorder=5))
    s = hazard_summary(g)
    ax.set(xlim=(-1.5, 14.5), ylim=(-4.0, 4.0), aspect="equal", xlabel="x along walk [m]", ylabel="y [m]",
           title=f"{g['meta']['name']}: {len(g['nodes'])} nodes, {len(g['edges'])} edges, {s['n_hazards']} hazards "
                 f"({s['ada_violations']} ADA violations, {s['robot_high_risk']} high robot risk)")
    ax.legend(handles=handles, loc="upper right", fontsize=7, ncol=2, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tiles", default="configs/e4_tiles.yaml")
    ap.add_argument("--split", choices=["train", "test"], default="test")
    ap.add_argument("--out", default="outputs/scene_graphs")
    args = ap.parse_args(argv)
    with open(args.tiles, encoding="utf-8") as f:
        tiles = yaml.safe_load(f)[args.split]
    out = resolve_path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for i, t in enumerate(tiles):
        g = build_scene_graph(t, tile_index=i, name=f"{args.split}_tile{i}")
        base = out / f"{args.split}_tile{i}"
        save_graph(g, base.with_suffix(".json"))
        txt = describe(g) + "\n\nHazards:\n" + "\n".join(
            f"- {h['id']}: {h['type']}, ADA {h['ada']}, robot risk {h['robot_risk']}: {h['note']}" for h in g["hazards"])
        base.with_suffix(".txt").write_text(txt, encoding="utf-8")
        draw_graph(g, base.with_suffix(".png"))
        s = hazard_summary(g)
        print(f"[graph] {base.name}: {len(g['nodes'])} nodes, {len(g['edges'])} edges, hazards {s['by_type']}, "
              f"ADA violations {s['ada_violations']}, high robot risk {s['robot_high_risk']}", flush=True)
    print(f"[graph] wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
