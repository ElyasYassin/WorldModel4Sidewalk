"""Ground-truth sidewalk scene graph, built from a terrain tile's parameters (P4).

One graph serves two uses:
  * inspection: every panel, joint, slab, ramp, curb and crack, with ADA checks (vertical
    displacement, running slope, cross slope) and a robot-traversability risk;
  * mission grounding (voice / HITL): named regions and landmarks that a mission spec can refer to
    (goal / via / keep_out), plus keep-out boxes for high-risk features.

Frame: the tile's scene frame (terrain.py): x along the walk, y across (+y left), robot starts at
(0, 0), metres. Built only from the tile params (and the texture seed for cracks), so it is exact
ground truth; a sensor-built graph can later be scored against it.

Explicit defects in the tile params (sim/defects.py: tree-root upheavals, open cracks, holes, puddles,
obstructions) become `kind="defect"` nodes with their true sizes and one hazard each; a street tree that caused
an upheaval is an `asset` node with a `causes` edge. A joint step created by an upheaval keeps its own hazard
(it is what a sensor measures) and names the upheaval as its `cause`.

Plain dicts (JSON-serialisable): {"nodes": {id: {...}}, "edges": [{"src", "rel", "dst", ...}],
"hazards": [...], "meta": {...}}.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from ..sim.defects import defect_box, polyline_length
from ..sim.sidewalk_texture import CURB_W, GUTTER_W, crack_segments, joint_positions
from ..sim.terrain import params_with_defaults, sidewalk_height

# 2010 ADA Standards for Accessible Design, 303 (changes in level) and 403.3 (slope)
ADA_LEVEL_OK_M = 0.0064  # <= 1/4 in: no treatment
ADA_LEVEL_BEVEL_M = 0.013  # 1/4 to 1/2 in: must be bevelled 1:2; above: must be ramped
ADA_RUNNING_SLOPE = 1 / 20  # above this a walk is a ramp
ADA_RAMP_MAX = 1 / 12
ADA_CROSS_SLOPE = 1 / 48
ADA_OPENING_M = 0.013  # 302.3: openings must not pass a 1/2 in sphere (cracks, holes, gaps)
ADA_CLEAR_WIDTH_M = 0.915  # 403.5.1: 36 in minimum clear width of a walking surface

# Go2 + rough-terrain policy, from the E4 sweep (docs/EXPERIMENTS.md): step-ups >= 15 cm stall or
# fail (15 cm stochastic), ~10 cm passes; 30 % grade stalls at 0.4 m/s, 25 % passes. A placeholder
# until the world model provides a predicted stall probability per feature.
ROBOT_STEP_RISK = ((0.10, "low"), (0.15, "medium"), (np.inf, "high"))
ROBOT_GRADE_RISK = ((0.20, "low"), (0.28, "medium"), (np.inf, "high"))
RISK_ORDER = {"none": 0, "low": 1, "medium": 2, "high": 3}


def ada_level_class(dz: float) -> str:
    dz = abs(dz)
    return "ok" if dz <= ADA_LEVEL_OK_M else "bevel_required" if dz <= ADA_LEVEL_BEVEL_M else "violation"


def ada_running_class(s: float) -> str:
    s = abs(s)
    return "ok" if s <= ADA_RUNNING_SLOPE else "ramp" if s <= ADA_RAMP_MAX else "violation"


def _risk(v: float, table) -> str:
    return next(r for lim, r in table if abs(v) < lim)


def build_scene_graph(terrain: dict[str, Any], tile_index: int = 0, x_range=(-1.0, 14.0),
                      name: str | None = None) -> dict[str, Any]:
    """Scene graph of one sidewalk tile over the walked stretch x_range."""
    p = params_with_defaults(terrain)
    half = 0.5 * p["width"]
    x_lo, x_hi = map(float, x_range)
    nodes: dict[str, dict[str, Any]] = {}
    edges: list[dict[str, Any]] = []
    hazards: list[dict[str, Any]] = []

    def node(i, **a):
        nodes[i] = a
        return i

    def edge(s, rel, d, **a):
        edges.append({"src": s, "rel": rel, "dst": d, **a})

    z = lambda x, y=0.0: float(sidewalk_height(x, y, p))  # noqa: E731
    eps = 1e-3

    # --- regions
    node("walkway", kind="region", label="sidewalk", box=[x_lo, x_hi, -half, half], width_m=p["width"],
         cross_slope=p["cross_slope"], ada=("ok" if abs(p["cross_slope"]) <= ADA_CROSS_SLOPE else "violation"))
    node("verge_left", kind="region", label="grass verge (left)", box=[x_lo, x_hi, half, half + 2.0])
    edge("walkway", "borders", "verge_left")
    if p["curb_drop"] > 0:
        node("curb", kind="feature", label=f"curb ({p['curb_drop'] * 100:.0f} cm drop)", box=[x_lo, x_hi, -half - CURB_W, -half],
             drop_m=p["curb_drop"], robot_risk=_risk(p["curb_drop"], ROBOT_STEP_RISK))
        node("road", kind="region", label="road", box=[x_lo, x_hi, -half - CURB_W - GUTTER_W - 3.0, -half - CURB_W])
        edge("walkway", "borders", "curb")
        edge("curb", "borders", "road", drop_m=p["curb_drop"])
        hazards.append({"id": "curb", "type": "curb_edge", "value_m": p["curb_drop"], "box": nodes["curb"]["box"],
                        "ada": "n/a", "robot_risk": nodes["curb"]["robot_risk"], "note": "drop-off along the walkway edge"})
    else:
        node("verge_right", kind="region", label="grass verge (right)", box=[x_lo, x_hi, -half - 2.0, -half])
        edge("walkway", "borders", "verge_right")
    if abs(p["cross_slope"]) > ADA_CROSS_SLOPE:
        hazards.append({"id": "walkway", "type": "cross_slope", "value": p["cross_slope"], "box": nodes["walkway"]["box"],
                        "ada": "violation", "robot_risk": "low",
                        "note": f"cross slope {abs(p['cross_slope']) * 100:.1f} % > 2.08 % ({'left' if p['cross_slope'] > 0 else 'right'} side high)"})

    # --- landmarks (mission grounding)
    node("start", kind="landmark", label="start", xy=[0.0, 0.0])
    node("end", kind="landmark", label="end of the walk", xy=[x_hi - 1.0, 0.0])
    for lm in ("start", "end"):
        edge(lm, "on", "walkway")

    # --- slabs and ramps
    for k, (x0, length, h) in enumerate(p["slabs"]):
        sid = node(f"slab_{k}", kind="feature", label=f"raised slab +{h * 100:.0f} cm", box=[x0, x0 + length, -half, half],
                   height_m=h, robot_risk=_risk(h, ROBOT_STEP_RISK))
        edge(sid, "on", "walkway")
    for k, (x0, x1, rise) in enumerate(p["grade"]):
        rid = node(f"ramp_{k}", kind="feature", label=f"ramp {abs(rise) * 100:.0f} % {'up' if rise > 0 else 'down'}",
                   box=[x0, x1, -half, half], running_slope=rise, ada=ada_running_class(rise),
                   robot_risk=_risk(rise, ROBOT_GRADE_RISK) if rise > 0 else "low")
        edge(rid, "on", "walkway")
        if nodes[rid]["ada"] != "ok":
            hazards.append({"id": rid, "type": "running_slope", "value": rise, "box": nodes[rid]["box"], "ada": nodes[rid]["ada"],
                            "robot_risk": nodes[rid]["robot_risk"], "note": f"running slope {abs(rise) * 100:.1f} %"})

    defects = list(p.get("defects") or [])
    faults = {k: (d["x1"] if d.get("high_end", "x1") == "x1" else d["x0"]) for k, d in enumerate(defects)
              if d["type"] == "upheaval"}

    # --- panels between joints (two per row when the walk has a centre joint), joint displacements
    js = [j for j in joint_positions(p, x_lo, x_hi) if x_lo < j < x_hi]
    bounds = [x_lo] + js + [x_hi]
    sides = [("L", 0.0, half), ("R", -half, 0.0)] if p["width"] >= 2.0 else [("C", -half, half)]
    for k, (a, b) in enumerate(zip(bounds[:-1], bounds[1:])):
        for s, y0, y1 in sides:
            yc = 0.5 * (y0 + y1)
            pid = node(f"panel_{k}{s}", kind="panel", label=f"panel {k}{s}", box=[a, b, y0, y1],
                       z_mean_m=float(np.mean(sidewalk_height(np.linspace(a, b, 9), yc, p))),
                       running_slope=(z(b - eps, yc) - z(a + eps, yc)) / max(b - a - 2 * eps, 1e-6),
                       cross_slope=p["cross_slope"])
            edge(pid, "part_of", "walkway")
            for sk, (x0, length, _) in enumerate(p["slabs"]):
                if a >= x0 - eps and b <= x0 + length + eps:
                    edge(f"slab_{sk}", "covers", pid)
            for rk, (x0, x1, _) in enumerate(p["grade"]):
                if b > x0 and a < x1:
                    edge(f"ramp_{rk}", "spans", pid)
            if k > 0:
                dz = z(a + eps, yc) - z(a - eps, yc)
                prev = f"panel_{k - 1}{s}"
                edge(prev, "next", pid, joint_x=a, step_m=dz, ada=ada_level_class(dz))
        if k > 0:  # one hazard per joint line (both panel rows share it)
            dzs = [z(a + eps, y) - z(a - eps, y) for y in (-half + 0.05, 0.0, half - 0.05)]
            dz = max(dzs, key=abs)
            if ada_level_class(dz) != "ok":
                hz = {"id": f"joint_{k}", "type": "vertical_displacement", "value_m": dz, "xy": [a, 0.0],
                      "box": [a - 0.05, a + 0.05, -half, half], "ada": ada_level_class(dz),
                      "robot_risk": _risk(dz, ROBOT_STEP_RISK) if dz > 0 else "low",
                      "note": f"{'step up' if dz > 0 else 'step down'} {abs(dz) * 100:.1f} cm at joint x = {a:.2f} m"}
                cause = [f"defect_{i}_upheaval" for i, xf in faults.items() if abs(xf - a) < 1e-3]
                if cause:
                    hz["cause"] = cause[0]
                hazards.append(hz)
        if len(sides) == 2:
            edge(f"panel_{k}L", "beside", f"panel_{k}R")

    # --- cracks (the same ones the texture paints, so an RGB sensor can be scored on them)
    for c, pts in enumerate(crack_segments(p, tile_index, x_lo, x_hi)):
        L = float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))
        cid = node(f"crack_{c}", kind="defect", label=f"crack {L:.2f} m", polyline=np.round(pts, 3).tolist(), length_m=L,
                   box=[float(pts[:, 0].min()), float(pts[:, 0].max()), float(pts[:, 1].min()), float(pts[:, 1].max())])
        mid = pts[len(pts) // 2]
        host = next((i for i, n in nodes.items() if n["kind"] == "panel" and n["box"][0] <= mid[0] < n["box"][1]
                     and n["box"][2] <= mid[1] <= n["box"][3]), "walkway")
        edge(cid, "on", host)
        hazards.append({"id": cid, "type": "crack", "value_m": L, "box": nodes[cid]["box"], "ada": "monitor",
                        "robot_risk": "none", "note": f"hairline crack, {L:.2f} m long"})

    # --- explicit defects (sim/defects.py)
    def host_of(x, y):
        return next((i for i, n in nodes.items() if n["kind"] == "panel" and n["box"][0] <= x < n["box"][1]
                     and n["box"][2] <= y <= n["box"][3]), "walkway")

    for k, d in enumerate(defects):
        t = d["type"]
        b = [float(v) for v in defect_box(d)]
        did = f"defect_{k}_{t}"
        cx, cy = 0.5 * (b[0] + b[1]), 0.5 * (b[2] + b[3])
        if t == "upheaval":
            lift, xe = float(d["lift"]), float(faults[k])
            a = {"label": f"tree-root upheaval +{lift * 100:.1f} cm", "lift_m": lift, "fault_x": xe,
                 "ada": ada_level_class(lift), "robot_risk": _risk(lift, ROBOT_STEP_RISK),
                 "note": f"panel lifted {lift * 100:.1f} cm at x = {xe:.2f} m (tree roots)"}
        elif t == "crack":
            w, L = float(d["width"]), polyline_length(d["points"])
            a = {"label": f"open crack {w * 1000:.0f} mm x {L:.2f} m", "width_m": w, "depth_m": float(d["depth"]),
                 "length_m": L, "polyline": d["points"], "ada": "violation" if w > ADA_OPENING_M else "monitor",
                 "robot_risk": "low" if w > 0.03 else "none",
                 "note": f"open crack {w * 1000:.0f} mm wide, {d['depth'] * 1000:.0f} mm deep, {L:.2f} m long"}
            mid = np.asarray(d["points"])[len(d["points"]) // 2]
            cx, cy = float(mid[0]), float(mid[1])
        elif t == "hole":
            dia, dep = 2 * max(d["rx"], d["ry"]), float(d["depth"])
            risk = "high" if dep >= 0.08 and min(d["rx"], d["ry"]) >= 0.10 else "medium" if dep >= 0.04 else "low"
            a = {"label": f"hole {dia * 100:.0f} cm, {dep * 100:.0f} cm deep", "diameter_m": dia, "depth_m": dep,
                 "ada": "violation" if dia > ADA_OPENING_M else "monitor", "robot_risk": risk,
                 "note": f"pothole / missing concrete {dia * 100:.0f} cm across, {dep * 100:.0f} cm deep (foot can drop in)"}
        elif t == "puddle":
            area = float(np.pi * d["rx"] * d["ry"])
            a = {"label": f"puddle {area:.2f} m2", "area_m2": area, "depth_m": float(d["depth"]), "ada": "n/a",
                 "robot_risk": "low",
                 "note": f"standing water {area:.2f} m2 (ponding): slip risk; depth sensors lose returns on water"}
        else:  # obstruction
            clear = float(max(half - (d["y"] + d["r"]), (d["y"] - d["r"]) + half))
            a = {"label": f"{d.get('kind', 'object')} on the walk", "clear_width_m": clear,
                 "ada": "violation" if clear < ADA_CLEAR_WIDTH_M else "ok", "robot_risk": "medium",
                 "note": f"{d.get('kind', 'object')} leaves {clear:.2f} m clear width (ADA min 0.915 m)"}
        note = a.pop("note")
        node(did, kind="defect", defect=t, box=b, xy=[cx, cy], **a)
        edge(did, "on", host_of(cx, cy))
        if t == "upheaval" and d.get("tree"):
            tid = node(f"tree_{k}", kind="asset", label="street tree", xy=[float(v) for v in d["tree"]])
            edge(tid, "causes", did)
        hazards.append({"id": did, "type": t, "box": b, "xy": [cx, cy], "ada": a["ada"], "robot_risk": a["robot_risk"],
                        "note": note, **{k_: v for k_, v in a.items() if k_.endswith(("_m", "_m2"))}})

    return {"meta": {"name": name or f"tile_{tile_index}", "tile_index": tile_index, "x_range": [x_lo, x_hi],
                     "frame": "tile scene frame: x along walk, y across (+y left), start at (0, 0), metres",
                     "terrain": p, "risk_source": "E4 sweep heuristic thresholds (placeholder for world-model stall risk)"},
            "nodes": nodes, "edges": edges, "hazards": hazards}


def graph_from_episode_meta(meta: dict[str, Any], x_range=(-1.0, 14.0)) -> dict[str, Any] | None:
    params = (meta.get("scene") or {}).get("params") or {}
    if not params.get("terrain"):
        return None
    return build_scene_graph(params["terrain"], int(params.get("terrain_tile", meta.get("e4_tile", 0))), x_range,
                             name=meta.get("episode_id"))


# --- queries for mission grounding / inspection

def nodes_of(g: dict[str, Any], kind: str) -> dict[str, dict[str, Any]]:
    return {i: n for i, n in g["nodes"].items() if n["kind"] == kind}


def keep_out_boxes(g: dict[str, Any], min_risk: str = "high") -> list[dict[str, Any]]:
    """Axis-aligned boxes [x0, x1, y0, y1] of features whose robot risk is at least min_risk."""
    lim = RISK_ORDER[min_risk]
    return [{"id": i, "box": n["box"], "robot_risk": n["robot_risk"]} for i, n in g["nodes"].items()
            if RISK_ORDER.get(n.get("robot_risk", "none"), 0) >= lim]


def hazard_summary(g: dict[str, Any]) -> dict[str, Any]:
    by_type: dict[str, int] = {}
    for h in g["hazards"]:
        by_type[h["type"]] = by_type.get(h["type"], 0) + 1
    return {"n_hazards": len(g["hazards"]), "by_type": by_type,
            "ada_violations": sum(h["ada"] == "violation" for h in g["hazards"]),
            "robot_high_risk": sum(h["robot_risk"] == "high" for h in g["hazards"])}


def describe(g: dict[str, Any]) -> str:
    """Compact text description (for an LLM mission compiler or an inspection report)."""
    lines = [f"Scene {g['meta']['name']} ({g['meta']['frame']})."]
    w = g["nodes"]["walkway"]
    lines.append(f"walkway: {w['width_m']:.2f} m wide, x {w['box'][0]:.1f} to {w['box'][1]:.1f} m, cross slope "
                 f"{w['cross_slope'] * 100:+.1f} % (ADA {w['ada']}).")
    for i, n in g["nodes"].items():
        if n["kind"] in ("feature", "landmark") or (n["kind"] == "region" and i != "walkway"):
            where = f"at ({n['xy'][0]:.1f}, {n['xy'][1]:.1f})" if "xy" in n else \
                f"x {n['box'][0]:.2f}-{n['box'][1]:.2f}, y {n['box'][2]:.2f}-{n['box'][3]:.2f}"
            extra = f", robot risk {n['robot_risk']}" if "robot_risk" in n else ""
            lines.append(f"{i}: {n['label']} {where}{extra}.")
    s = hazard_summary(g)
    lines.append(f"hazards: {s['n_hazards']} ({', '.join(f'{k} {v}' for k, v in sorted(s['by_type'].items()))}); "
                 f"ADA violations {s['ada_violations']}, high robot risk {s['robot_high_risk']}.")
    return "\n".join(lines)


def save_graph(g: dict[str, Any], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(g, indent=1, default=float), encoding="utf-8")
    return path
