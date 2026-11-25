"""Measured-heightmap terrain (go2wm.sim.terrain heightmap feature) and the scan -> heightmap builder."""

import numpy as np
import pytest

from go2wm.sim.terrain import heightmap_height, params_with_defaults, sidewalk_height, sidewalk_heightfield


def _write_hm(tmp_path, z, res=0.1, x0=-1.0, y0=-0.5):
    path = tmp_path / "hm.npz"
    np.savez(path, z=z.astype(np.float32), cls=np.zeros(z.shape, np.uint8), res=res, x0=x0, y0=y0)
    return path


def test_heightmap_bilinear_and_edges(tmp_path):
    xs = -1.0 + 0.1 * np.arange(21)
    ys = -0.5 + 0.1 * np.arange(11)
    z = 0.02 * xs[:, None] - 0.01 * ys[None, :]  # a plane: bilinear must reproduce it exactly
    path = _write_hm(tmp_path, z)
    x, y = np.array([0.0, 0.33, -0.77]), np.array([0.0, 0.12, -0.41])
    assert np.allclose(heightmap_height(x, y, str(path)), 0.02 * x - 0.01 * y, atol=1e-6)
    assert np.isclose(heightmap_height(5.0, 0.0, {"file": str(path)}), 0.02 * 1.0)  # edge value outside


def test_heightmap_in_tile_and_heightfield(tmp_path):
    z = np.zeros((21, 11))
    z[12:, :] = 0.013  # a 13 mm step at x = 0.2
    path = _write_hm(tmp_path, z)
    p = params_with_defaults({"size": [2.0, 1.0], "heightmap": {"file": str(path)},
                              "horizontal_scale": 0.02, "vertical_scale": 0.001})
    assert np.isclose(sidewalk_height(0.5, 0.0, p), 0.013) and np.isclose(sidewalk_height(-0.5, 0.0, p), 0.0)
    hf = sidewalk_heightfield((2.0 - 0.04, 1.0 - 0.04), 0.02, 0.001, p)
    assert hf.dtype == np.int16 and hf.max() == 13 and hf.min() == 0


def test_station_alignment_removes_offset_and_tilt():
    pytest.importorskip("scipy")
    from go2wm.scan.heightmap import FILLED, StationGrid, TileFrame, build_heightmap, station_offsets

    frame = TileFrame((0.0, 0.0), 0.0, (4.0, 1.0))
    res = 0.01
    nx, ny = 400, 100
    xs = -2.0 + (np.arange(nx) + 0.5) * res
    ys = -0.5 + (np.arange(ny) + 0.5) * res
    truth = 0.02 * xs[:, None] + 0.0 * ys[None, :]
    truth = truth + np.where(xs[:, None] > 0.5, 0.008, 0.0)  # a real 8 mm step both stations see
    rng = np.random.default_rng(0)

    def grid(lo, hi, offset, tilt):
        n = np.where((xs[:, None] > lo) & (xs[:, None] < hi), 20.0, 0.0) * np.ones((1, ny))
        mean = truth + offset + tilt * xs[:, None] + rng.normal(0, 2e-4, (nx, ny))
        return StationGrid(n.astype(np.float32), np.where(n > 0, mean, np.nan).astype(np.float32),
                           np.where(n > 0, 3e-4, np.nan).astype(np.float32))

    grids = {1: grid(-2.0, 1.0, 0.006, 0.001), 2: grid(-1.0, 2.0, -0.004, -0.0005)}
    planes = station_offsets(grids, frame, res, min_cells=100, log=lambda *a: None)
    rel = planes[1] - planes[2]
    assert np.isclose(rel[0], 0.010, atol=3e-4) and np.isclose(rel[1], 0.0015, atol=3e-4)
    hm = build_heightmap(grids, frame, grid_res=res, out_res=0.02, offsets=planes)
    assert (hm.cls != FILLED).mean() > 0.95
    hx = hm.x0 + np.arange(hm.z.shape[0]) * hm.res
    prof = np.median(hm.z, 1)
    # the real step survives (~8 mm) and the station overlap (x in -1..1) has no seam
    step = np.median(prof[(hx > 0.55) & (hx < 0.7)] - 0.02 * hx[(hx > 0.55) & (hx < 0.7)]) - \
        np.median(prof[(hx > 0.3) & (hx < 0.45)] - 0.02 * hx[(hx > 0.3) & (hx < 0.45)])
    assert np.isclose(step, 0.008, atol=1e-3)
    resid = prof - 0.02 * hx - np.where(hx > 0.5, 0.008, 0.0)
    assert np.ptp(resid[(hx > -1.5) & (hx < 1.5) & (np.abs(hx - 0.5) > 0.06)]) < 2e-3
