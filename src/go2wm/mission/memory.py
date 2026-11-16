"""Phase 1 spatial memory: the robot's own record of findings and coverage (never filled from ground truth).

A finding is created by the first observation of a candidate and absorbs later observations within
`assoc_radius` of the same kind. Its estimate is the confidence-weighted mean of the MEASURED values (detections
only locate it). Status:
  candidate   detected only (no measurement yet)
  uncertain   measured, but only from long range / low confidence, or measurements disagree
  measured    at least one high-confidence measurement (D405, or D555 within 1.2 m) and consistent values
Coverage is tracked per route segment as the share of walkway cells (10 cm) seen by each sensor.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from .perception import Observation


class SpatialMemory:
    def __init__(self, walk_bounds: tuple[float, float, float, float], route_x: list[float],
                 assoc_radius: float = 0.35, cell: float = 0.10, robust: bool = False):
        self.bounds, self.route_x, self.assoc, self.cell = walk_bounds, list(route_x), assoc_radius, cell
        # robust=True: a finding's value is the median of the best-sensor readings after dropping outliers
        # (> 3 MAD, min 4 mm, from the median) instead of their mean. Off by default (frozen evaluations).
        self.robust = robust
        self.partial_frac = 0.85  # robust depressions: readings below this share of the 2nd-largest are partial views
        x0, x1, y0, y1 = walk_bounds
        self.nx, self.ny = int(np.ceil((x1 - x0) / cell)), int(np.ceil((y1 - y0) / cell))
        self.seen: dict[str, np.ndarray] = {}
        self.findings: dict[str, dict[str, Any]] = {}
        self.aliases: dict[str, str] = {}
        self._next = 0

    # --- findings
    def add(self, ob: Observation) -> str:
        best, bd = None, np.inf
        for fid, f in self.findings.items():
            if f["kind"] != ob.kind:
                continue
            dx, dy = np.abs(np.asarray(f["xy"]) - ob.xy)
            if ob.kind == "edge":  # transverse edges are lines across the walk: same x, anywhere along y
                ok, d = dx < 0.2, dx  # (an upheaval fault can be seen in pieces far apart across a wide walk)
            else:
                d = float(np.hypot(dx, dy))
                ok = d < self._radius(f)
            if ok and d < bd:
                best, bd = fid, d
        if best is None:
            best = f"finding_{self._next:03d}"
            self._next += 1
            self.findings[best] = {"id": best, "kind": ob.kind, "xy": ob.xy.tolist(), "observations": [],
                                   "status": "candidate", "first_seen_t": ob.t}
        f = self.findings[best]
        rec = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in asdict(ob).items()}
        f["observations"].append(rec)
        self._update(f)
        return self._consolidate(best)

    def _radius(self, f: dict[str, Any]) -> float:
        ext = max([o.get("extent_m") or 0.0 for o in f["observations"]] + [0.0])
        if f["kind"] == "dropout":  # a puddle (up to ~1.7 m long) is seen in partial patches
            return max(0.9, ext)
        return max(self.assoc + 0.05, 0.75 * ext)

    def _consolidate(self, fid: str) -> str:
        """Merge findings of the same kind that turn out to overlap (one defect first seen in pieces). The older
        id survives; `aliases` maps merged ids to it."""
        f = self.findings[fid]
        if f["kind"] == "depression":  # pieces of one fault across the walk: same x, far apart in y -> an edge
            for oid, o in list(self.findings.items()):
                if (oid != fid and o["kind"] == "depression" and abs(o["xy"][0] - f["xy"][0]) < 0.2
                        and abs(o["xy"][1] - f["xy"][1]) >= 0.6 and min(o.get("n_detections", 0), f.get("n_detections", 0)) >= 2):
                    keep, drop = sorted((fid, oid), key=lambda k: self.findings[k]["first_seen_t"])
                    k = self.findings[keep]
                    k["observations"] += self.findings[drop]["observations"]
                    k["observations"].sort(key=lambda r: r["t"])
                    for ob in k["observations"]:  # the depression values were not step measurements
                        if ob["value_m"] is not None:
                            ob["value_m"], ob["provenance"] = None, "detected"
                    k["kind"], k["relabelled"] = "edge", "depressions aligned across the walk"
                    del self.findings[drop]
                    self.aliases[drop] = keep
                    self._update(k)
                    return self._consolidate(keep)
        for oid, o in list(self.findings.items()):
            if oid == fid or o["kind"] != f["kind"]:
                continue
            dx, dy = np.abs(np.asarray(o["xy"]) - np.asarray(f["xy"]))
            near = dx < 0.2 if f["kind"] == "edge" else np.hypot(dx, dy) < max(self._radius(f), self._radius(o))
            if near:
                keep, drop = sorted((fid, oid), key=lambda k: self.findings[k]["first_seen_t"])
                self.findings[keep]["observations"] += self.findings[drop]["observations"]
                self.findings[keep]["observations"].sort(key=lambda r: r["t"])
                del self.findings[drop]
                self.aliases[drop] = keep
                self._update(self.findings[keep])
                return self._consolidate(keep)
        return fid

    def _edge_span(self, e: dict[str, Any]) -> float:
        ys = [y for o in e["observations"] for y in (o.get("extra") or {}).get("y_span", [])]
        if ys:
            return float(np.ptp(ys))
        return max([o.get("extent_m") or 0.0 for o in e["observations"]] + [0.0])

    def _hole_wall(self, eid: str, max_span: float = 0.8) -> str | None:
        """A short edge (< max_span across the walk) inside a confirmed depression is that hole's far wall seen up
        close, not a fault across the walk. Returns the depression's id, or None."""
        e = self.findings[eid]
        if self._edge_span(e) >= max_span:
            return None
        for did, d in self.findings.items():
            # the hole was seen many times from afar; its wall only up close (an upheaval's low-side pit is the
            # other way round: fewer detections than the fault itself)
            v = d.get("shape_votes", {"step": 0, "pit": 0})
            if (d["kind"] == "depression" and d.get("n_detections", 0) >= max(2, e.get("n_detections", 0))
                    and v["step"] <= v["pit"]  # its shape says step: it is the fault's footprint, not a hole
                    and np.hypot(*(np.asarray(d["xy"]) - e["xy"])) < self._radius(d) + 0.1):
                return did
        return None

    def vote_shape(self, fid: str, shape: str | None, min_votes: int = 3, share: float = 0.7) -> str:
        """Record a step / pit shape vote (perception.classify_shape); relabel an edge <-> depression once the
        votes clearly disagree with the kind. Values measured under the old kind are discarded."""
        f = self.findings[fid]
        if shape is None or f["kind"] not in ("edge", "depression"):
            return fid
        v = f.setdefault("shape_votes", {"step": 0, "pit": 0})
        v[shape] += 1
        n = v["step"] + v["pit"]
        want = "edge" if v["step"] >= share * n else "depression" if v["pit"] >= share * n else None
        if n >= min_votes and want and want != f["kind"]:
            for ob in f["observations"]:
                if ob["value_m"] is not None:
                    ob["value_m"], ob["provenance"] = None, "detected"
                ob["kind"] = want
            f["kind"], f["relabelled"] = want, f"shape votes {v}"
            f["shape_votes"] = {"step": 0, "pit": 0}
            self._update(f)
            return self._consolidate(fid)
        return fid

    def resolve(self, fid: str | None) -> str | None:
        while fid is not None and fid not in self.findings and fid in self.aliases:
            fid = self.aliases[fid]
        return fid

    def _update(self, f: dict[str, Any]) -> None:
        obs = f["observations"]
        f["n_detections"] = sum(o["provenance"] == "detected" for o in obs)
        w = np.array([o["confidence"] for o in obs])
        xy = np.array([o["xy"] for o in obs])
        f["xy"] = (w[:, None] * xy).sum(0).__truediv__(w.sum()).tolist()
        meas = [o for o in obs if o["value_m"] is not None]
        if not meas:
            f["status"] = "candidate"
            f["confidence"] = float(max(w))
            return
        mw = np.array([o["confidence"] for o in meas])
        vals = np.array([o["value_m"] for o in meas])
        best_conf = float(mw.max())
        top = vals[mw >= best_conf - 1e-9]
        if self.robust and f.get("kind") == "depression" and len(top) >= 3:
            # a partial view (the close camera sees only the near part of a hole while approaching) can only read
            # a hole SHALLOW: keep the readings near the top, judged against the 2nd-largest so that one spurious
            # deep reading cannot set the bar
            top = np.sort(top)
            top = top[top >= self.partial_frac * top[-2]]
        if self.robust and len(top) >= 3:
            med = float(np.median(top))
            mad = float(np.median(np.abs(top - med)))
            keep = top[np.abs(top - med) <= max(3.0 * 1.4826 * mad, 0.004)]
            top = keep if len(keep) else top
            f["value_m"] = float(np.median(top))
        else:
            f["value_m"] = float(np.average(top))  # trust the best sensor class present
        f["value_spread_m"] = float(np.std(top)) if len(top) > 1 else 0.0
        ext = [o["extent_m"] for o in meas if o.get("extent_m")]
        f["extent_m"] = float(np.median(ext)) if ext else None
        f["n_measurements"] = len(meas)
        f["best_sensor"] = max(meas, key=lambda o: o["confidence"])["sensor"]
        f["confidence"] = best_conf
        f["status"] = "measured" if best_conf >= 0.8 and f["value_spread_m"] < 0.005 else "uncertain"

    def is_real(self, fid: str, min_baseline: float = 0.15) -> bool:
        """A finding counts once it was detected in at least two frames from viewpoints >= `min_baseline` apart
        (single detections are usually noise, and a stereo camera's fixed-pattern error repeats identically while the
        robot stands still) and is not a side effect of a step edge or a hole."""
        f = self.findings[fid]
        if f.get("n_detections", 0) < 2 or self.explained_by(fid) is not None:
            return False
        views = [o["extra"]["view_xy"] for o in f["observations"]
                 if o["provenance"] == "detected" and "view_xy" in o.get("extra", {})]
        if len(views) < 2:
            return len(views) == 0  # observations without viewpoints (older callers, tests): frame count only
        v = np.asarray(views)
        return float(np.max(np.linalg.norm(v - v[0], axis=1))) >= min_baseline

    def explained_by(self, fid: str, near: float = 0.5, low_side: float = 0.65) -> str | None:
        """A depression or dropout right at a confirmed transverse edge is the edge's own footprint: the low side
        of the step and the ground it occludes (no returns). It may lie up to `low_side` beyond the edge on its low
        side (from the measured step sign), `near` on the other. Returns that edge's id, or None."""
        f = self.findings[fid]
        if f["kind"] == "edge":
            return self._hole_wall(fid)
        if f["kind"] == "dropout":  # the ground a deep hole's near wall hides from the camera
            for did, d in self.findings.items():
                if (d["kind"] == "depression" and d.get("n_detections", 0) >= 2
                        and np.hypot(*(np.asarray(d["xy"]) - f["xy"])) < self._radius(d) + 0.2):
                    return did
        for eid, e in self.findings.items():
            if e["kind"] != "edge" or e.get("n_detections", 0) < 2 or self._hole_wall(eid) is not None:
                continue
            dx = f["xy"][0] - e["xy"][0]  # + = beyond the edge along the route
            v = e.get("value_m")
            lo, hi = -near, near
            if v is None or v < 0:  # step down along +x: the low side is beyond the edge
                hi = low_side
            if v is None or v > 0:
                lo = -low_side
            if lo < dx < hi:
                return eid
        return None

    def needs_close_look(self, fid: str) -> bool:
        f = self.findings[fid]
        return self.is_real(fid) and (f["status"] != "measured" or f.get("best_sensor") != "depth_d405")

    # --- coverage
    def mark_seen(self, sensor: str, world_xy: np.ndarray) -> None:
        g = self.seen.setdefault(sensor, np.zeros((self.nx, self.ny), bool))
        x0, _, y0, _ = self.bounds
        i = np.floor((world_xy[:, 0] - x0) / self.cell).astype(int)
        j = np.floor((world_xy[:, 1] - y0) / self.cell).astype(int)
        m = (i >= 0) & (i < self.nx) & (j >= 0) & (j < self.ny)
        g[i[m], j[m]] = True

    def coverage(self) -> dict[str, list[dict[str, float]]]:
        x0 = self.bounds[0]
        edges = [x0] + self.route_x
        out = {}
        for s, g in self.seen.items():
            segs = []
            for a, b in zip(edges[:-1], edges[1:]):
                ia, ib = int((a - x0) / self.cell), int(np.ceil((b - x0) / self.cell))
                segs.append({"x0": a, "x1": b, "seen": float(g[ia:ib].mean()) if ib > ia else 0.0})
            out[s] = segs
        return out

    def to_dict(self) -> dict[str, Any]:
        real = {k: v for k, v in self.findings.items() if self.is_real(k)}
        merged = {k: self.explained_by(k) for k in self.findings if self.explained_by(k)}
        return {"findings": real, "merged_into": merged,
                "n_single_detections_dropped": len(self.findings) - len(real) - len(merged),
                "coverage": self.coverage(), "frame": "tile scene frame (robot's estimate)"}

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=1, default=float))
        return path
