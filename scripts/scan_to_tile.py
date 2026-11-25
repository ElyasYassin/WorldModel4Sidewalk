"""Recreate a scanned sidewalk as a simulator terrain tile (Trimble X7 TDX export -> heightmap tile).

  python scripts/scan_to_tile.py --tdx lidar_data/sidewalk_TDX/sidewalk.tdx --out scan/sidewalk_tdx

Writes to <out> (relative paths resolve under the data root, configs/paths.yaml):
  heightmap.npz   measured terrain z[nx, ny] (scene frame, 2 cm grid) + per-cell class (hard / soft / filled)
  tiles.json      a one-tile list, usable wherever a tile list is (scripts/isaac_mission.py --tiles ... --tile 0,
                  make_isaac_terrain_generator); the tile carries no analytic features, only the heightmap
  preview.png     elevation, local relief and data-provenance maps
  report.json     station alignment, coverage and the frame

Default frame = the main stretch of the 2026-11-18 sidewalk scan: the concrete centreline fitted in world XY
is straight (heading -0.09 deg, 3.8 cm RMS) through (23.98, -0.21); the concrete was found along
x_world 6.3-42. The 27 m gap to station 6 (x_world ~50-62) was not covered densely enough, so the second
stretch near station 6 would need its own tile (--centre 67.6 -4.4 --heading -7.4 --size 14 7).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from go2wm.cli import write_json
from go2wm.common.config import resolve_path
from go2wm.scan.heightmap import FILLED, MEASURED_HARD, TileFrame, build_heightmap, station_grids, station_offsets
from go2wm.scan.tzf import station_poses


def preview(hm, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy import ndimage

    ext = [hm.x0, hm.x0 + hm.z.shape[0] * hm.res, hm.y0, hm.y0 + hm.z.shape[1] * hm.res]
    relief = hm.z - ndimage.gaussian_filter(hm.z, 20)
    fig, ax = plt.subplots(3, 1, figsize=(18, 9))
    for a, img, kw, title in [
        (ax[0], hm.z, dict(cmap="terrain"), "elevation [m] (z = 0 at scene origin)"),
        (ax[1], relief * 1000, dict(cmap="RdBu_r", vmin=-15, vmax=15), "local relief [mm] (minus 0.4 m Gaussian)"),
        (ax[2], hm.cls, dict(cmap="viridis", vmin=0, vmax=2), "data: 0 measured hard, 1 measured soft, 2 filled"),
    ]:
        im = a.imshow(img.T, origin="lower", extent=ext, aspect="equal", interpolation="nearest", **kw)
        a.set_title(title, loc="left")
        a.set_xlabel("scene x along sidewalk [m]")
        a.set_ylabel("y [m]")
        fig.colorbar(im, ax=a, fraction=0.02)
    fig.tight_layout()
    fig.savefig(path, dpi=80)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tdx", default="lidar_data/sidewalk_TDX/sidewalk.tdx")
    ap.add_argument("--out", default="scan/sidewalk_tdx")
    ap.add_argument("--centre", type=float, nargs=2, default=[24.1, -0.21], metavar=("X", "Y"),
                    help="world XY of the scene origin [m]")
    ap.add_argument("--heading", type=float, default=-0.09, help="world heading of scene +x [deg]")
    ap.add_argument("--size", type=float, nargs=2, default=[40.0, 7.0], metavar=("LX", "LY"))
    ap.add_argument("--width", type=float, default=2.2, help="walkway width for planners/routes [m] (measured 2.2)")
    ap.add_argument("--res", type=float, default=0.02, help="output grid [m]")
    ap.add_argument("--step", type=int, default=1, help="decode every n-th scan pixel (1 = full resolution)")
    ap.add_argument("--tripod-radius", type=float, default=0.7, help="unmeasured disc around each scanner [m]")
    args = ap.parse_args(argv)

    out = resolve_path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    frame = TileFrame(tuple(args.centre), float(np.radians(args.heading)), tuple(args.size))
    t0 = time.time()
    grids = station_grids(Path(args.tdx), frame, res=0.01, step=args.step)
    offsets = station_offsets(grids, frame, 0.01)
    # scanner tripod footprints: legs survive the ground filter and the ground under the head is never seen
    tripods = [(float(q[0]), float(q[1]), args.tripod_radius)
               for q in frame.to_scene(np.array([st["p"] for st in station_poses(args.tdx).values()]))]
    hm = build_heightmap(grids, frame, grid_res=0.01, out_res=args.res, offsets=offsets, exclude=tripods)
    hm.meta["source"] = str(Path(args.tdx).resolve())
    hm_path = hm.save(out / "heightmap.npz")
    tile = {"size": list(args.size), "width": args.width, "cross_slope": 0.0, "grade": [], "slabs": [],
            "curb_drop": 0.0, "friction": 1.0, "horizontal_scale": args.res, "vertical_scale": 0.001,
            "defects": [], "heightmap": {"file": str(hm_path)}, "n_waypoints": 5,
            "name": "scanned sidewalk (Trimble X7, 2026-11-18)"}
    write_json(out / "tiles.json", [tile])
    preview(hm, out / "preview.png")
    zr = hm.z[hm.cls == MEASURED_HARD]
    report = {"frame": hm.meta["frame"], "station_planes_mm_mm_per_m": {k: [round(float(c) * 1000, 2) for c in v] for k, v in offsets.items()},
              "fraction": hm.meta["fraction"], "grid": list(hm.z.shape), "res": hm.res,
              "z_range_hard_m": [float(zr.min()), float(zr.max())] if zr.size else None,
              "filled_fraction_on_walkway": float((hm.cls[:, np.abs(hm.y0 + np.arange(hm.z.shape[1]) * hm.res)
                                                          < args.width / 2] == FILLED).mean()),
              "seconds": round(time.time() - t0, 1)}
    write_json(out / "report.json", report)
    print(report)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
