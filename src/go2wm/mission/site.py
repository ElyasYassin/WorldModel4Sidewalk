"""Site context for demo missions: a construction work zone, workers and pedestrians on the sidewalk.

A tile's params may carry `site`:

  {"zones":   [{"name": "work zone", "x": [x0, x1], "y": [y0, y1]}],          # closed area (drawn, reported)
   "objects": [{"kind": "cone" | "barrier" | "worker" | "sign" | "pile" | "generator", "xy": [x, y], "r": r, ...},
               ...                                                              # static: planner obstacles
               {"kind": "pedestrian" | "worker", "path": [[x, y], ...], "speed": v, "r": r,
                "trigger_x": x or None, "loop": bool}]}                         # moving (no "xy")

Static objects with r > 0 become planner obstacles (mission/scene.py; Isaac spawns a kinematic cylinder for each).
Moving people are not planned around: the mission runner YIELDS (stops) while one is in the corridor ahead of the
robot and slows down near people. People are known to the robot from the simulator's ground truth, standing in for
a person detector (the depth simulation ray-casts the sidewalk surface only).
"""

from __future__ import annotations

from typing import Any

import numpy as np


def static_obstacles(site: dict[str, Any] | None) -> np.ndarray:
    if not site:
        return np.zeros((0, 3))
    o = [[*s["xy"], s["r"]] for s in site.get("objects", []) if "xy" in s and s.get("r", 0) > 0]
    return np.asarray(o, dtype=np.float64).reshape(-1, 3)


class SiteActors:
    """Moving people: positions over mission time. A mover with `trigger_x` starts walking when the robot passes
    that x; `loop` walks the path back and forth."""

    def __init__(self, site: dict[str, Any] | None):
        self.movers = [dict(m) for m in (site or {}).get("objects", []) if "path" in m]
        for m in self.movers:
            p = np.asarray(m["path"], dtype=np.float64)
            seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
            m["_p"], m["_cum"] = p, np.r_[0.0, np.cumsum(seg)]
            m["_t0"] = None if m.get("trigger_x") is not None else 0.0

    def __len__(self):
        return len(self.movers)

    def update(self, t: float, robot_xy) -> np.ndarray:
        """[n, 5]: x, y, r, heading, walking (1/0)."""
        out = []
        for m in self.movers:
            if m["_t0"] is None and robot_xy[0] >= m["trigger_x"]:
                m["_t0"] = t
            L = m["_cum"][-1]
            s = 0.0 if m["_t0"] is None else (t - m["_t0"]) * m["speed"]
            walking = m["_t0"] is not None
            if m.get("loop"):
                s = s % (2 * L)
                fwd = s <= L
                s = s if fwd else 2 * L - s
            else:
                fwd = True
                if s >= L:
                    s, walking = L, False
            k = int(np.clip(np.searchsorted(m["_cum"], s, side="right") - 1, 0, len(m["_p"]) - 2))
            a, b = m["_p"][k], m["_p"][k + 1]
            f = (s - m["_cum"][k]) / max(m["_cum"][k + 1] - m["_cum"][k], 1e-9)
            xy = a + f * (b - a)
            d = (b - a) if fwd else (a - b)
            if m.get("courteous") and walking:  # step aside for the robot (smoothly, kept on the walk)
                u = d / max(np.linalg.norm(d), 1e-9)
                rel = np.asarray(robot_xy, dtype=np.float64) - xy
                along, lat = float(rel @ u), float(rel @ np.array([-u[1], u[0]]))
                want = 0.0
                if -1.0 < along < 3.0 and abs(lat) < 1.1:
                    want = -np.sign(lat if abs(lat) > 1e-3 else 1.0) * (1.1 - abs(lat))
                m["_off"] = m.get("_off", 0.0) + 0.15 * (want - m.get("_off", 0.0))
                xy = xy + m["_off"] * np.array([-u[1], u[0]])
                xy[1] = float(np.clip(xy[1], -1.25, 1.35))
            out.append([xy[0], xy[1], m.get("r", 0.3), float(np.arctan2(d[1], d[0])), float(walking)])
        return np.asarray(out, dtype=np.float64).reshape(-1, 5)


def demo_tile() -> dict[str, Any]:
    """Hand-authored 56 m demo street: five 10 m blocks, ten defects of every type (two near the ADA 6.4 / 13 mm
    thresholds), a construction work zone closing the building-side half of the walk, workers, and two pedestrians
    (one crossing from a doorway to the road, one walking towards the robot). x along the walk, +y building side,
    curb and road at y < -1.5."""
    half = 1.5
    defects = [
        # block 1
        {"type": "crack", "points": [[-22.6, 0.55], [-22.45, 0.47], [-22.3, 0.42], [-22.12, 0.35], [-21.95, 0.33],
                                     [-21.8, 0.24], [-21.62, 0.18], [-21.45, 0.08]], "width": 0.006, "depth": 0.012},
        {"type": "upheaval", "x0": -19.5, "x1": -18.0, "y0": -half, "y1": 0.0, "lift": 0.011, "high_end": "x1",
         "tree": [-18.75, -2.1]},
        # block 2
        {"type": "puddle", "x": -14.2, "y": 0.62, "rx": 0.55, "ry": 0.38, "angle": 0.3, "depth": 0.015},
        {"type": "hole", "x": -10.6, "y": -0.38, "rx": 0.19, "ry": 0.15, "angle": 0.6, "depth": 0.045},
        {"type": "obstruction", "kind": "pole", "r": 0.08, "x": -7.8, "y": 1.3},
        # block 4 (block 3 is the work zone)
        {"type": "upheaval", "x0": 12.0, "x1": 13.5, "y0": 0.0, "y1": half, "lift": 0.045, "high_end": "x0",
         "tree": [12.75, 2.1]},
        {"type": "crack", "points": [[14.9, -1.1], [15.02, -0.95], [15.1, -0.78], [15.24, -0.62], [15.3, -0.45],
                                     [15.42, -0.3], [15.5, -0.12]], "width": 0.02, "depth": 0.025},
        # block 5
        {"type": "hole", "x": 18.6, "y": 0.32, "rx": 0.21, "ry": 0.18, "angle": 1.2, "depth": 0.11},
        {"type": "puddle", "x": 20.9, "y": -0.85, "rx": 0.45, "ry": 0.3, "angle": 2.4, "depth": 0.02},
        {"type": "upheaval", "x0": 22.5, "x1": 24.0, "y0": -half, "y1": 0.0, "lift": 0.028, "high_end": "x1"},
    ]
    zx0, zx1, by = -4.0, 3.0, 0.1  # work zone along x, barrier line at y = by
    objects: list[dict[str, Any]] = []
    for x in np.arange(zx0, zx1 + 1e-6, 0.5):
        objects.append({"kind": "barrier", "xy": [round(float(x), 2), by], "r": 0.12})
    for x, y in ((-5.6, 1.25), (-5.1, 0.85), (-4.6, 0.45), (3.6, 0.45), (4.1, 0.85)):  # taper cones
        objects.append({"kind": "cone", "xy": [x, y], "r": 0.15})
    objects += [
        {"kind": "worker", "xy": [-2.2, 0.95], "r": 0.3, "pose": "dig", "yaw": -1.2},
        {"kind": "worker", "xy": [1.4, 1.05], "r": 0.3, "pose": "stand", "yaw": 3.0},
        {"kind": "pile", "xy": [-0.6, 0.95], "r": 0.0},
        {"kind": "generator", "xy": [2.4, 0.9], "r": 0.0},
        {"kind": "sign", "xy": [-6.4, 2.0], "r": 0.0, "text": "SIDEWALK CLOSED AHEAD"},
        # moving people
        {"kind": "worker", "path": [[-3.3, 0.62], [0.4, 0.62]], "speed": 0.5, "r": 0.3, "loop": True},
        {"kind": "pedestrian", "path": [[9.0, 2.8], [9.0, -2.8]], "speed": 1.1, "r": 0.3, "trigger_x": 6.4},
        {"kind": "pedestrian", "path": [[27.5, 1.15], [-27.0, 1.15]], "speed": 1.2, "r": 0.3, "trigger_x": 14.0,
         "courteous": True},
    ]
    return {"size": [56.0, 7.0], "width": 2 * half, "cross_slope": 0.012, "grade": [], "slabs": [],
            "curb_drop": 0.15, "friction": 1.0, "horizontal_scale": 0.02, "vertical_scale": 0.001,
            "defects": defects, "n_waypoints": 5,
            "site": {"zones": [{"name": "work zone", "x": [zx0, zx1], "y": [by, half]}], "objects": objects},
            "blocks": ["Block 1", "Block 2", "Block 3 (work zone)", "Block 4", "Block 5"]}
