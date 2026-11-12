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
                 assoc_radius: float = 0.35, cell: float = 0.10):
        self.bounds, self.route_x, self.assoc, self.cell = walk_bounds, list(route_x), assoc_radius, cell
        x0, x1, y0, y1 = walk_bounds
        self.nx, self.ny = int(np.ceil((x1 - x0) / cell)), int(np.ceil((y1 - y0) / cell))
        self.seen: dict[str, np.ndarray] = {}
        self.findings: dict[str, dict[str, Any]] = {}
        self._next = 0

    # --- findings
    def add(self, ob: Observation) -> str:
        best, bd = None, np.inf
        for fid, f in self.findings.items():
            if f["kind"] != ob.kind:
                continue
            dx, dy = np.abs(np.asarray(f["xy"]) - ob.xy)
            if ob.kind == "edge":  # transverse edges are lines across the walk: same x, anywhere along y
                ok, d = dx < 0.2 and dy < 1.2, dx
            else:
                d = float(np.hypot(dx, dy))
                ext = max([o.get("extent_m") or 0.0 for o in f["observations"]] + [0.0])
                ok = d < max(self.assoc + 0.05, 0.75 * ext)
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
        return best

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
        f["value_m"] = float(np.average(top))  # trust the best sensor class present
        f["value_spread_m"] = float(np.std(top)) if len(top) > 1 else 0.0
        ext = [o["extent_m"] for o in meas if o.get("extent_m")]
        f["extent_m"] = float(np.median(ext)) if ext else None
        f["n_measurements"] = len(meas)
        f["best_sensor"] = max(meas, key=lambda o: o["confidence"])["sensor"]
        f["confidence"] = best_conf
        f["status"] = "measured" if best_conf >= 0.8 and f["value_spread_m"] < 0.005 else "uncertain"

    def is_real(self, fid: str) -> bool:
        """A finding counts once it was detected in at least two frames or measured (single detections are
        usually noise)."""
        return self.findings[fid].get("n_detections", 0) >= 2

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
        return {"findings": real, "n_single_detections_dropped": len(self.findings) - len(real),
                "coverage": self.coverage(), "frame": "tile scene frame (robot's estimate)"}

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=1, default=float))
        return path
