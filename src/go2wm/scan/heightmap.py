"""Measured terrain heightmap from a multi-station TDX scan, in the simulator's scene frame.

Scene frame: x along the sidewalk, y across it (+y to the left of +x), origin on the sidewalk centreline in
the middle of the tile, z = 0 at the origin. The frame is set by a world-XY centre point and heading.

Pipeline (pure numpy/scipy):
  1. per station, keep points inside the tile, drop everything more than `ground_band` above the lowest
     return in each 10 cm cell (removes canopy, people, poles; keeps the ground layer);
  2. per station, average onto a `res` grid (count, mean, std);
  3. align stations vertically: one constant offset per station, least squares over the median height
     difference in the cells two stations share on hard (smooth) ground. The TDX registration leaves
     2-10 mm vertical offsets, as large as the defects we care about, so without this the station seams
     would read as steps;
  4. blend stations weighted by point count (density falls off smoothly with range, so seams feather);
  5. classify cells: hard (std < hard_std), soft (grass etc.), or unmeasured; lightly median-filter soft
     cells (grass returns are spiky);
  6. fill unmeasured cells by multi-scale normalized convolution (smooth, no new features).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy import ndimage

from .tzf import iter_world_points

MEASURED_HARD, MEASURED_SOFT, FILLED = 0, 1, 2


@dataclass
class TileFrame:
    centre_xy: tuple[float, float]  # world XY of the scene origin
    heading: float  # world angle of scene +x [rad]
    size: tuple[float, float]  # tile size along x, y [m]

    def to_scene(self, P: np.ndarray) -> np.ndarray:
        c, s = np.cos(self.heading), np.sin(self.heading)
        d = P[:, :2] - np.asarray(self.centre_xy)
        return np.stack([d[:, 0] * c + d[:, 1] * s, -d[:, 0] * s + d[:, 1] * c, P[:, 2]], 1)

    def to_world_xy(self, x, y):
        c, s = np.cos(self.heading), np.sin(self.heading)
        return self.centre_xy[0] + x * c - y * s, self.centre_xy[1] + x * s + y * c


@dataclass
class StationGrid:
    n: np.ndarray
    mean: np.ndarray
    std: np.ndarray


@dataclass
class Heightmap:
    z: np.ndarray  # float32 [nx, ny], scene z [m]
    cls: np.ndarray  # uint8 [nx, ny]: MEASURED_HARD / MEASURED_SOFT / FILLED
    res: float
    x0: float  # scene x of z[0, :]
    y0: float  # scene y of z[:, 0]
    meta: dict = field(default_factory=dict)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, z=self.z, cls=self.cls, res=self.res, x0=self.x0, y0=self.y0,
                            meta=np.array(repr(self.meta)))
        return path


def station_grids(tdx, frame: TileFrame, res: float = 0.01, step: int = 1, ground_band: float = 0.08,
                  margin: float = 0.0, log=print) -> dict[int, StationGrid]:
    hx, hy = frame.size[0] / 2 + margin, frame.size[1] / 2 + margin
    nx, ny = int(round(2 * hx / res)), int(round(2 * hy / res))
    grids: dict[int, StationGrid] = {}
    cell = int(round(0.10 / res))

    def finish(k: int, parts: list[np.ndarray]) -> None:
        if not parts:
            return
        Q = np.concatenate(parts)
        parts.clear()
        ix = np.clip(((Q[:, 0] + hx) / res).astype(np.int64), 0, nx - 1)
        iy = np.clip(((Q[:, 1] + hy) / res).astype(np.int64), 0, ny - 1)
        cidx = (ix // cell) * (ny // cell + 1) + iy // cell
        cmin = np.full(int(cidx.max()) + 1, np.inf, np.float32)
        np.minimum.at(cmin, cidx, Q[:, 2])
        keep = Q[:, 2] < cmin[cidx] + ground_band
        idx = (ix * ny + iy)[keep]
        z = Q[keep, 2].astype(np.float64)
        n = np.bincount(idx, minlength=nx * ny).reshape(nx, ny).astype(np.float32)
        zs = np.bincount(idx, weights=z, minlength=nx * ny).reshape(nx, ny)
        zz = np.bincount(idx, weights=z * z, minlength=nx * ny).reshape(nx, ny)
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = np.where(n >= 3, zs / n, np.nan)
            std = np.where(n >= 3, np.sqrt(np.maximum(zz / n - mean ** 2, 0)), np.nan)
        grids[k] = StationGrid(n, mean.astype(np.float32), std.astype(np.float32))
        log(f"station {k}: {len(Q)} pts in tile, {keep.mean():.1%} ground, {np.isfinite(mean).mean():.1%} cells")

    current, parts = None, []
    for k, W in iter_world_points(tdx, step):  # station by station
        if k != current:
            finish(current, parts)
            current = k
        Q = frame.to_scene(W)
        m = (np.abs(Q[:, 0]) < hx) & (np.abs(Q[:, 1]) < hy)
        if m.any():
            parts.append(Q[m].astype(np.float32))
    finish(current, parts)
    return grids


def station_offsets(grids: dict[int, StationGrid], frame: TileFrame, res: float, hard_std: float = 0.0015,
                    min_cells: int = 2000, log=print) -> dict[int, np.ndarray]:
    """Per-station correction plane c0 + cx*x + cy*y [m] (to subtract), so stations agree on shared hard ground.
    Each overlapping pair gives a robust plane fit of their height difference; the per-station planes are the
    least-squares solution with the linked stations' planes summing to zero. Returns {station: (c0, cx, cy)}."""
    ks = sorted(grids)
    any_g = grids[ks[0]]
    xs = -frame.size[0] / 2 + (np.arange(any_g.n.shape[0]) + 0.5) * res
    ys = -frame.size[1] / 2 + (np.arange(any_g.n.shape[1]) + 0.5) * res
    rows, rhs, wts = [], [], []
    for a_i, a in enumerate(ks):
        for b in ks[a_i + 1:]:
            A, B = grids[a], grids[b]
            m = (A.n >= 5) & (B.n >= 5) & (A.std < hard_std) & (B.std < hard_std)
            if m.sum() < min_cells:
                continue
            ii, jj = np.nonzero(m)
            d = (B.mean[m] - A.mean[m]).astype(np.float64)
            X = np.stack([np.ones_like(d), xs[ii], ys[jj]], 1)
            keep = np.ones(len(d), bool)
            for _ in range(3):  # robust: refit without > 3 MAD outliers
                c, *_ = np.linalg.lstsq(X[keep], d[keep], rcond=None)
                r = d - X @ c
                keep = np.abs(r - np.median(r[keep])) < 3 * 1.4826 * np.median(np.abs(r[keep] - np.median(r[keep])))
            row = np.zeros(len(ks))
            row[ks.index(b)], row[ks.index(a)] = 1, -1
            rows.append(row), rhs.append(c), wts.append(np.sqrt(m.sum()))
            span = np.ptp(xs[ii])
            log(f"stations {a}-{b}: {int(m.sum())} shared hard cells over {span:.1f} m, "
                f"dz = {c[0] * 1000:+.1f} mm {c[1] * 1000:+.2f} mm/m (x) {c[2] * 1000:+.2f} mm/m (y), "
                f"residual {np.std(r[keep]) * 1000:.1f} mm")
    linked = np.abs(np.array(rows)).sum(0) > 0 if rows else np.zeros(len(ks), bool)
    n_pairs = len(rows)
    for i in np.flatnonzero(~linked):  # no overlap with any other station: leave it alone
        row = np.zeros(len(ks))
        row[i] = 1
        rows.append(row), rhs.append(np.zeros(3)), wts.append(1e3)
    rows.append(linked.astype(float)), rhs.append(np.zeros(3)), wts.append(1e3)  # gauge
    W = np.array(wts)[:, None]
    P, *_ = np.linalg.lstsq(np.array(rows) * W, np.array(rhs) * W, rcond=None)
    log("station planes [mm, mm/m, mm/m]: " + "; ".join(
        f"{k}: {p[0] * 1000:+.1f} {p[1] * 1000:+.2f} {p[2] * 1000:+.2f}" for k, p in zip(ks, P)))
    if n_pairs:
        resid = np.array(rhs[:n_pairs]) - np.array(rows[:n_pairs]) @ P
        log(f"pair residual RMS (constant term) {np.sqrt(np.mean(resid[:, 0] ** 2)) * 1000:.1f} mm")
    return {k: P[i] for i, k in enumerate(ks)}


def _fill(z: np.ndarray, valid: np.ndarray) -> np.ndarray:
    out = np.where(valid, z, 0.0)
    w = valid.astype(np.float64)
    filled = valid.copy()
    sigma = 1.0
    while not filled.all():
        num = ndimage.gaussian_filter(np.where(valid, z, 0.0), sigma, mode="nearest")
        den = ndimage.gaussian_filter(w, sigma, mode="nearest")
        new = (~filled) & (den > 1e-3)
        out[new] = num[new] / den[new]
        filled |= new
        sigma *= 2
        if sigma > 4 * max(z.shape):
            out[~filled] = np.nanmean(np.where(valid, z, np.nan))
            break
    return out


def build_heightmap(grids: dict[int, StationGrid], frame: TileFrame, grid_res: float, out_res: float = 0.02,
                    hard_std: float = 0.0015, offsets: dict[int, np.ndarray] | None = None,
                    max_above: float = 0.3, feather: float = 0.5, spike_height: float = 0.015,
                    spike_cells: int = 6, exclude: list[tuple[float, float, float]] | None = None) -> Heightmap:
    """`exclude`: (x, y, radius) discs in scene coordinates treated as unmeasured (e.g. scanner tripods)."""
    offsets = offsets or {k: np.zeros(3) for k in grids}
    shape = next(iter(grids.values())).n.shape
    gx = -frame.size[0] / 2 + (np.arange(shape[0]) + 0.5) * grid_res
    gy = -frame.size[1] / 2 + (np.arange(shape[1]) + 0.5) * grid_res
    num = np.zeros(shape)
    den = np.zeros(shape)
    for k, g in grids.items():
        c = np.asarray(offsets[k], dtype=np.float64)
        plane = c[0] + c[1] * gx[:, None] + c[2] * gy[None, :]
        ok = g.n >= 3
        # feathered weight: point density smoothed over `feather` m (raw counts flicker with the scan-line
        # pattern and occlusions, which would switch stations abruptly and expose any residual mismatch)
        w = ndimage.gaussian_filter(np.where(ok, g.n, 0).astype(np.float64), feather / grid_res, mode="constant")
        w = np.where(ok, w, 0)
        num += w * np.where(ok, np.nan_to_num(g.mean) - plane, 0)
        den += w
    den_pts = sum(np.where(g.n >= 3, g.n, 0) for g in grids.values())
    # roughness from the densest station only: mixing stations would add their residual offsets
    best = np.argmax(np.stack([g.n for g in grids.values()]), 0)
    std = np.choose(best, [np.nan_to_num(g.std, nan=1.0) for g in grids.values()])
    f = int(round(out_res / grid_res))
    nx, ny = (den.shape[0] // f) * f, (den.shape[1] // f) * f
    pool = lambda a: a[:nx, :ny].reshape(nx // f, f, ny // f, f).sum((1, 3))  # noqa: E731
    D = pool(den_pts)
    with np.errstate(invalid="ignore", divide="ignore"):
        z = pool(num) / pool(den)
        sd = pool(np.where(den_pts > 0, std * den_pts, 0)) / D
    valid = (D >= 3) & np.isfinite(z)
    hard = valid & (sd < hard_std)
    soft = valid & ~hard
    # despike soft ground (grass) with a 3x3 median over soft cells only
    zm = ndimage.median_filter(np.where(valid, z, np.nan), size=3, mode="nearest")
    z = np.where(soft & np.isfinite(zm), zm, z)
    # cells whose lowest return is canopy / a sign / a pole top (ground never seen there): reject anything more
    # than `max_above` over a robust ground trend (20th percentile per 0.5 m block, median-smoothed)
    b = max(1, int(round(0.5 / out_res)))
    for _ in range(2):
        zz = np.where(valid, z, np.nan)
        pb, qb = -(-zz.shape[0] // b) * b, -(-zz.shape[1] // b) * b
        blk = np.full((pb, qb), np.nan)
        blk[:zz.shape[0], :zz.shape[1]] = zz
        with np.errstate(all="ignore"):
            import warnings

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                lo = np.nanpercentile(blk.reshape(pb // b, b, qb // b, b).transpose(0, 2, 1, 3).reshape(pb // b, qb // b, -1),
                                      20, axis=2)
        lo = _fill(np.nan_to_num(lo), np.isfinite(lo))
        lo = ndimage.median_filter(lo, size=5, mode="nearest")
        trend = np.kron(lo, np.ones((b, b)))[:zz.shape[0], :zz.shape[1]]
        valid &= ~(z > trend + max_above)
    # small isolated spikes above the local surface (litter, leaves in a joint, mixed edge returns): not terrain
    xs_o = -frame.size[0] / 2 + (np.arange(z.shape[0]) + 0.5) * out_res
    ys_o = -frame.size[1] / 2 + (np.arange(z.shape[1]) + 0.5) * out_res
    for _ in range(2):
        med = ndimage.median_filter(_fill(z, valid), size=11, mode="nearest")
        spike = valid & (z - med > spike_height)
        lab, nl = ndimage.label(spike)
        if nl:
            sizes = ndimage.sum(spike, lab, range(1, nl + 1))
            valid &= ~np.isin(lab, 1 + np.flatnonzero(sizes <= spike_cells))
    # scanner / tripod footprints (and anything else listed): never trust what is measured there
    for ex, ey, er in exclude or []:
        valid &= ~(((xs_o[:, None] - ex) ** 2 + (ys_o[None, :] - ey) ** 2) < er * er)
    # isolated measured specks inside big holes are unreliable: drop components < 25 cells
    lab, nl = ndimage.label(valid)
    if nl:
        sizes = ndimage.sum(valid, lab, range(1, nl + 1))
        valid &= np.isin(lab, 1 + np.flatnonzero(sizes >= 25))
    z = _fill(z, valid)
    cls = np.full(z.shape, FILLED, np.uint8)
    cls[valid & hard] = MEASURED_HARD
    cls[valid & ~hard] = MEASURED_SOFT
    x0 = -frame.size[0] / 2 + out_res / 2
    y0 = -frame.size[1] / 2 + out_res / 2
    # z = 0 at the scene origin (median of measured hard ground within 0.5 m)
    xs = x0 + np.arange(z.shape[0]) * out_res
    ys = y0 + np.arange(z.shape[1]) * out_res
    near = (np.abs(xs)[:, None] < 0.5) & (np.abs(ys)[None, :] < 0.5) & (cls == MEASURED_HARD)
    z0 = float(np.median(z[near])) if near.any() else float(z[len(xs) // 2, len(ys) // 2])
    z = (z - z0).astype(np.float32)
    meta = {"frame": {"centre_xy": list(frame.centre_xy), "heading": frame.heading, "size": list(frame.size)},
            "z0_world": z0, "station_planes": {int(k): [float(c) for c in np.atleast_1d(v)] for k, v in offsets.items()},
            "fraction": {"hard": float((cls == 0).mean()), "soft": float((cls == 1).mean()),
                         "filled": float((cls == 2).mean())}}
    return Heightmap(z, cls, out_res, x0, y0, meta)
