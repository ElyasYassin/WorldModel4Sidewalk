"""Reader for Trimble X7 TDX exports (.tdx project + .tdf/*.tzf scans), no Trimble software needed.

The .tzf layout was reverse-engineered from our own 2026-11-18 sidewalk scan (lidar_data/sidewalk_TDX) and
checked against it: decoded point counts match the TDX metadata, and adjacent stations placed with the TDX
registration agree on ground height to 2-10 mm (MAD ~2 mm). Not an official spec; validate against one
Trimble export (E57/LAS) before relying on it for sub-centimetre claims.

A .tzf is a sequence of chunks. Each starts with a 60-byte header of little-endian u32 words:
  [0:2] magic c61efe2c cf572198   [4] payload length   [5] channel   [6] tile x   [7] tile y
  [9], [10] tile size (512 x 512)  [11] uncompressed bytes             byte 48: float32 scale
then the payload: channels 1-3 = u32 raw length + zlib stream of int32[512, 512] (read with a raw inflate:
the adler32 trailer does not verify); channels 4 (grey) and 6 (RGB) = JPEG tiles whose pixel order does not
match the range grid (not mapped yet).
  channel 1: range, 0.1 mm units (max 799999 = the X7's 80 m)
  channel 2: zenith angle, microradians, delta-coded along each tile row
  channel 3: azimuth, microradians, delta-coded along each tile row
A pixel with range 0 has no return.

The .tdx (SQLite) stores each station's pose: world = scanner_xyz @ R + p, where R's rows are the
orientation U, V, W vectors.
"""

from __future__ import annotations

import mmap
import sqlite3
import zlib
from pathlib import Path
from typing import Iterator

import numpy as np

MAGIC = bytes.fromhex("c61efe2ccf572198")
TILE = 512


class TzfScan:
    """Random access to the range / angle tiles of one .tzf file."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._f = open(self.path, "rb")
        self._m = mmap.mmap(self._f.fileno(), 0, access=mmap.ACCESS_READ)
        self.chunks: dict[tuple[int, int, int], tuple[int, int]] = {}
        i = 0
        while (i := self._m.find(MAGIC, i)) >= 0:
            h = np.frombuffer(self._m[i:i + 60], "<u4")
            if h[2] == 200 and h[9] == TILE and h[10] == TILE:
                self.chunks[(int(h[5]), int(h[6]), int(h[7]))] = (i, int(h[4]))
            i += 8
        if not self.chunks:
            raise ValueError(f"{self.path}: no TZF chunks found")
        self.tiles = sorted({(x, y) for c, x, y in self.chunks if c == 1})

    def close(self) -> None:
        self._m.close()
        self._f.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def channel(self, ch: int, tx: int, ty: int) -> np.ndarray:
        """int64[512, 512] for channels 1-3 (2 and 3 already integrated along rows)."""
        o, n = self.chunks[(ch, tx, ty)]
        payload = self._m[o + 60:o + 60 + n]
        a = np.frombuffer(zlib.decompressobj(-15).decompress(payload[6:]), "<i4").reshape(TILE, TILE)
        a = a.astype(np.int64)
        return np.cumsum(a, axis=1) if ch in (2, 3) else a

    def tile_points(self, tx: int, ty: int, step: int = 1, rmin: float = 0.3, rmax: float = 80.0) -> np.ndarray:
        """float64[N, 3] scanner-frame XYZ [m] of the valid returns in one tile."""
        r = self.channel(1, tx, ty)[::step, ::step] / 1e4
        v = (r > rmin) & (r < rmax)
        if not v.any():
            return np.zeros((0, 3))
        zen = self.channel(2, tx, ty)[::step, ::step][v] / 1e6
        az = self.channel(3, tx, ty)[::step, ::step][v] / 1e6
        r = r[v]
        s = r * np.sin(zen)
        return np.stack([s * np.cos(az), s * np.sin(az), r * np.cos(zen)], 1)


def station_poses(tdx: str | Path) -> dict[int, dict]:
    """{station number: {"p": (3,), "R": (3, 3), "scan": .tzf path, "points": count}} from a .tdx."""
    tdx = Path(tdx)
    con = sqlite3.connect(f"file:{tdx.as_posix()}?mode=ro", uri=True)
    try:
        rows = con.execute(
            "select s.name, ts.fileURI, ts.numberOfPoints, t.positionX, t.positionY, t.positionZ, "
            "t.orientationUx, t.orientationUy, t.orientationUz, t.orientationVx, t.orientationVy, t.orientationVz, "
            "t.orientationWx, t.orientationWy, t.orientationWz "
            "from station s join transformation t on t.rowid = s.transformation "
            "join terrestrialScan ts on ts.parentStation = s.rowid").fetchall()
    finally:
        con.close()
    out = {}
    for name, uri, npts, *t in rows:
        out[int(name)] = {"p": np.array(t[:3]), "R": np.array(t[3:]).reshape(3, 3),
                          "scan": tdx.parent / uri, "points": int(npts)}
    return out


def iter_world_points(tdx: str | Path, step: int = 1, stations=None) -> Iterator[tuple[int, np.ndarray]]:
    """Yield (station, float64[N, 3] world XYZ) tile by tile for every station of a .tdx export."""
    for k, st in sorted(station_poses(tdx).items()):
        if stations is not None and k not in stations:
            continue
        with TzfScan(st["scan"]) as scan:
            for tx, ty in scan.tiles:
                X = scan.tile_points(tx, ty, step)
                if len(X):
                    yield k, X @ st["R"] + st["p"]
