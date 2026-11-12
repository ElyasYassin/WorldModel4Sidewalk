"""Sidewalk defect library: one parametric description per defect drives geometry, appearance, sensor
effects and the ground-truth scene graph.

A tile's params may carry `"defects": [ {...}, ... ]` (scene frame of terrain.py: x along the walk, y across,
+y left, metres). Types and fields:

  upheaval     tree-root heave: a walkway panel tilted up from one joint to the next, leaving a vertical fault
               of `lift` at its high edge.  x0, x1, y0, y1 (panel box), lift [m], high_end ("x1" or "x0"),
               optional tree [x, y] (a trunk on the verge, also an obstacle)
  crack        an open crack.  points [[x, y], ...] (polyline), width [m], depth [m]
  hole         pothole / missing chunk: steep-walled depression.  x, y, rx, ry, angle [rad], depth [m]
  puddle       standing water in a shallow depression.  x, y, rx, ry, angle, depth [m]
  obstruction  object on the walk (no terrain change; it becomes a SceneSpec obstacle).  kind (bin, pole, debris),
               x, y, r [m]

Geometry is added to terrain.sidewalk_height (`defect_height`). Features narrower than the tile's grid
(horizontal_scale) are rasterised with the same cross-section area (a narrow deep crack becomes a grid-wide
shallower groove), so the height field has no aliasing speckle; the scene graph keeps the true sizes.
Appearance is painted by sidewalk_texture.sidewalk_albedo (`paint_defects`). Water makes range sensors drop
returns (`range_dropout`); per-region friction is NOT modelled (one physics material per terrain).

Pure numpy; deterministic from the params.
"""

from __future__ import annotations

from typing import Any

import numpy as np

DEFECT_TYPES = ("upheaval", "crack", "hole", "puddle", "obstruction")

HOLE_WALL = 0.15  # fraction of the radius over which a hole's wall drops to full depth
PUDDLE_DROPOUT = {"pinhole": 0.85, "lidar": 0.6}  # probability that a return from water is lost
WATER = np.array([0.20, 0.24, 0.28])
WET_CONCRETE = np.array([0.50, 0.49, 0.46])
SOIL = np.array([0.36, 0.29, 0.20])
ROOT = np.array([0.30, 0.22, 0.14])
BARK = np.array([0.28, 0.21, 0.15])


# --- shared geometry helpers

def dist_to_polyline(x, y, pts) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float64)
    d = np.full(np.shape(x), np.inf)
    for a, b in zip(pts[:-1], pts[1:]):
        ab = b - a
        t = np.clip(((x - a[0]) * ab[0] + (y - a[1]) * ab[1]) / max(ab @ ab, 1e-12), 0, 1)
        d = np.minimum(d, np.hypot(x - a[0] - t * ab[0], y - a[1] - t * ab[1]))
    return d


def ellipse_r(x, y, d: dict[str, Any]) -> np.ndarray:
    """Normalised elliptical radius (1 on the rim) in the defect's rotated frame."""
    c, s = np.cos(d.get("angle", 0.0)), np.sin(d.get("angle", 0.0))
    u = (x - d["x"]) * c + (y - d["y"]) * s
    v = -(x - d["x"]) * s + (y - d["y"]) * c
    return np.hypot(u / d["rx"], v / d["ry"])


def polyline_length(pts) -> float:
    pts = np.asarray(pts, dtype=np.float64)
    return float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())


def defect_box(d: dict[str, Any]) -> list[float]:
    t = d["type"]
    if t == "upheaval":
        return [d["x0"], d["x1"], d["y0"], d["y1"]]
    if t == "crack":
        p = np.asarray(d["points"])
        w = d["width"] / 2
        return [float(p[:, 0].min() - w), float(p[:, 0].max() + w), float(p[:, 1].min() - w), float(p[:, 1].max() + w)]
    if t in ("hole", "puddle"):
        r = max(d["rx"], d["ry"])
        return [d["x"] - r, d["x"] + r, d["y"] - r, d["y"] + r]
    if t == "obstruction":
        return [d["x"] - d["r"], d["x"] + d["r"], d["y"] - d["r"], d["y"] + d["r"]]
    raise ValueError(f"unknown defect type {t!r}")


# --- geometry

def defect_height(x, y, defects: list[dict[str, Any]], grid: float = 0.0) -> np.ndarray:
    """Height change [m] from all defects at scene coordinates (broadcast). `grid` = height-field spacing
    used to rasterise sub-grid features with the same cross-section area (0 = analytic, no widening)."""
    x, y = np.broadcast_arrays(np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64))
    dz = np.zeros(x.shape)
    for d in defects:
        t = d["type"]
        if t == "upheaval":
            inside = (x >= d["x0"]) & (x < d["x1"]) & (y >= d["y0"]) & (y < d["y1"])
            f = (x - d["x0"]) / max(d["x1"] - d["x0"], 1e-6)
            if d.get("high_end", "x1") == "x0":
                f = 1.0 - f
            dz = dz + np.where(inside, d["lift"] * np.clip(f, 0, 1), 0.0)
        elif t == "crack":
            w_eff = max(d["width"], grid)
            depth = d["depth"] * (d["width"] / w_eff)
            dz = dz - np.where(dist_to_polyline(x, y, d["points"]) < w_eff / 2, depth, 0.0)
        elif t == "hole":
            r = ellipse_r(x, y, d)
            dz = dz - d["depth"] * np.clip((1.0 - r) / HOLE_WALL, 0, 1)
        elif t == "puddle":
            r = ellipse_r(x, y, d)
            dz = dz - d["depth"] * np.clip(1.0 - r * r, 0, 1)
        elif t != "obstruction":
            raise ValueError(f"unknown defect type {t!r}")
    return dz


def obstacles_from_defects(defects: list[dict[str, Any]]) -> np.ndarray:
    """[N, 3] circles (x, y, r) for SceneSpec.obstacles: obstructions and upheaval trees."""
    out = [[d["x"], d["y"], d["r"]] for d in defects if d["type"] == "obstruction"]
    out += [[*d["tree"], 0.2] for d in defects if d["type"] == "upheaval" and d.get("tree")]
    return np.asarray(out, dtype=np.float64).reshape(-1, 3)


# --- sensors

def range_dropout(x, y, defects: list[dict[str, Any]], kind: str = "pinhole") -> np.ndarray:
    """Probability that a range return at scene (x, y) is lost (standing water is specular)."""
    p = np.zeros(np.shape(x))
    for d in defects:
        if d["type"] == "puddle":
            p = np.where(ellipse_r(x, y, d) < 1.0, PUDDLE_DROPOUT["lidar" if kind != "pinhole" else "pinhole"], p)
    return p


# --- appearance

def paint_defects(img: np.ndarray, x: np.ndarray, y: np.ndarray, defects: list[dict[str, Any]], half_width: float,
                  noise) -> np.ndarray:
    """Paint defects onto an albedo image [..., 3] (in place and returned). `noise(x, y, cell, seed)` is the
    texture module's value noise in [-1, 1]."""
    for k, d in enumerate(defects):
        t = d["type"]
        if t == "upheaval":
            # lifted panel slightly lighter (older, dusty); a dark shadow line at the raised fault edge
            on = (x >= d["x0"]) & (x < d["x1"]) & (y >= d["y0"]) & (y < d["y1"])
            img[on] *= 1.04
            xe = d["x1"] if d.get("high_end", "x1") == "x1" else d["x0"]
            edge = (np.abs(x - xe) < 0.012 + 0.002 * d["lift"] / 0.01) & (y >= d["y0"]) & (y < d["y1"])
            img[edge] *= 0.45
            if d.get("tree"):
                tx, ty = d["tree"]
                r = np.hypot(x - tx, y - ty)
                img[(r < 0.55) & (np.abs(y) > half_width)] = SOIL * (1 + 0.1 * noise(x, y, 0.05, 900 + k))[(r < 0.55) & (np.abs(y) > half_width)][..., None]
                img[r < 0.2] = BARK
                rng = np.random.default_rng(1000 + k)
                for _ in range(4):  # surface roots running from the trunk toward the walk
                    ang = np.arctan2(-np.sign(ty), 0.0) + rng.normal(0, 0.6)
                    pts = [np.array([tx, ty])]
                    for _ in range(8):
                        ang += rng.normal(0, 0.25)
                        pts.append(pts[-1] + 0.12 * np.array([np.cos(ang), np.sin(ang)]))
                    m = (dist_to_polyline(x, y, np.array(pts)) < 0.025) & (np.abs(y) > half_width - 0.02)
                    img[m] = ROOT
        elif t == "crack":
            dist = dist_to_polyline(x, y, d["points"])
            core = dist < max(d["width"] / 2, 0.005)
            img[core] *= 0.30 + 0.25 * np.clip(1 - d["width"] / 0.012, 0, 1)
            halo = (dist >= max(d["width"] / 2, 0.005)) & (dist < d["width"] / 2 + 0.015)
            img[halo] *= 0.85
        elif t == "hole":
            jag = 1.0 + 0.08 * noise(x, y, 0.04, 1200 + k)
            r = ellipse_r(x, y, d) / jag
            inside = r < 1.0
            img[inside] = SOIL * (0.75 + 0.35 * np.clip(r[inside], 0, 1))[..., None] * \
                (1 + 0.15 * noise(x[inside], y[inside], 0.03, 1300 + k))[..., None]
            img[(r >= 1.0) & (r < 1.08)] *= 0.7  # broken rim
        elif t == "puddle":
            wob = 1.0 + 0.06 * noise(x, y, 0.15, 1400 + k)
            r = ellipse_r(x, y, d) / wob
            img[(r >= 1.0) & (r < 1.12)] = WET_CONCRETE  # wet ring
            w = r < 1.0
            sky = np.clip(0.5 + 0.5 * noise(x[w], y[w], 0.6, 1500 + k), 0, 1)[..., None]
            img[w] = WATER * (0.9 + 0.5 * sky)
    return img


# --- random generation

def random_defects(rng: np.random.Generator, p: dict[str, Any], x_range=(2.0, 18.0), n: int | None = None,
                   types=DEFECT_TYPES, panel_len: float = 1.5) -> list[dict[str, Any]]:
    """Non-overlapping random defects on the walkway of tile params p (needs width)."""
    half = 0.5 * p["width"]
    n = int(rng.integers(4, 9)) if n is None else n
    out: list[dict[str, Any]] = []
    boxes: list[list[float]] = []

    def free(b, margin=0.25):
        return all(b[1] + margin < o[0] or o[1] + margin < b[0] or b[3] + margin < o[2] or o[3] + margin < b[2]
                   for o in boxes)

    for _ in range(n * 20):
        if len(out) >= n:
            break
        t = types[int(rng.integers(len(types)))]
        if t == "upheaval":
            k = int(rng.integers(int(np.ceil(x_range[0] / panel_len)), int(np.floor(x_range[1] / panel_len))))
            x0, x1 = k * panel_len, (k + 1) * panel_len
            side = rng.choice(["L", "R", "full"], p=[0.4, 0.4, 0.2])
            y0, y1 = {"L": (0.0, half), "R": (-half, 0.0), "full": (-half, half)}[side]
            d = {"type": "upheaval", "x0": x0, "x1": x1, "y0": y0, "y1": y1,
                 "lift": round(float(rng.uniform(0.01, 0.08)), 3), "high_end": str(rng.choice(["x0", "x1"]))}
            if rng.uniform() < 0.7:  # the tree whose roots did it, on the verge beside the panel
                ty = (half + 0.6) * (1 if side != "R" else -1)
                d["tree"] = [round(0.5 * (x0 + x1), 2), round(ty, 2)]
        elif t == "crack":
            pt = np.array([rng.uniform(*x_range), rng.uniform(-half + 0.25, half - 0.25)])
            ang, pts = rng.uniform(0, np.pi), [pt.copy()]
            for _ in range(int(rng.integers(5, 16))):
                ang += rng.normal(0, 0.4)
                pt = pt + 0.1 * np.array([np.cos(ang), np.sin(ang)])
                pt[1] = np.clip(pt[1], -half + 0.05, half - 0.05)
                pts.append(pt.copy())
            d = {"type": "crack", "points": np.round(pts, 3).tolist(),
                 "width": round(float(np.exp(rng.uniform(np.log(0.003), np.log(0.03)))), 4),
                 "depth": round(float(rng.uniform(0.005, 0.03)), 3)}
        elif t in ("hole", "puddle"):
            big = t == "puddle"
            rx = float(rng.uniform(0.3, 0.9) if big else rng.uniform(0.08, 0.35))
            ry = float(min(rng.uniform(0.2, 0.6) if big else rng.uniform(0.08, 0.3), half - 0.1))
            d = {"type": t, "x": round(float(rng.uniform(*x_range)), 2),
                 "y": round(float(rng.uniform(-half + ry, half - ry)), 2), "rx": round(rx, 3), "ry": round(ry, 3),
                 "angle": round(float(rng.uniform(0, np.pi)), 3),
                 "depth": round(float(rng.uniform(0.008, 0.025) if big else rng.uniform(0.03, 0.15)), 3)}
        else:
            kind = str(rng.choice(["bin", "pole", "debris"]))
            r = {"bin": 0.3, "pole": 0.08, "debris": 0.2}[kind]
            d = {"type": "obstruction", "kind": kind, "r": r, "x": round(float(rng.uniform(*x_range)), 2),
                 "y": round(float(rng.uniform(-half + r, half - r)), 2)}
        b = defect_box(d)
        if free(b):
            out.append(d)
            boxes.append(b)
    return out
