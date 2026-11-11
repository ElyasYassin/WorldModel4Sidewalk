import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import evaluate_sensors as ev  # noqa: E402
from go2wm.sim.terrain import params_with_defaults, sidewalk_height  # noqa: E402


def test_step_fit_recovers_step_on_sloped_ground():
    rng = np.random.default_rng(0)
    j = 2.0
    X = np.r_[rng.uniform(j - 0.4, j - 0.075, 40), rng.uniform(j + 0.075, j + 0.4, 40)]
    Y = rng.uniform(-0.5, 0.5, 80)
    Z = 0.1 + 0.2 * (X - j) - 0.03 * Y + 0.012 * (X > j)
    assert abs(ev.step_fit(X, Y, Z, np.ones(80), j) - 0.012) < 1e-9


def test_mesh_height_matches_quantized_grid_at_nodes_and_surface_nearby():
    p = params_with_defaults({"slabs": [[3.0, 1.5, 0.05]], "grade": [[6.0, 8.0, 0.1]], "cross_slope": 0.02})
    hs, vs = p["horizontal_scale"], p["vertical_scale"]
    x, y = np.array([1.0, 3.5, 7.0]), np.array([0.3, -0.4, 0.2])
    xn, yn = np.round(x / hs) * hs, np.round(y / hs) * hs  # node positions (grid starts at -size/2)
    assert np.allclose(ev.mesh_height(xn, yn, p), np.round(sidewalk_height(xn, yn, p) / vs) * vs)
    assert np.all(np.abs(ev.mesh_height(x, y, p) - sidewalk_height(x, y, p)) <= 0.5 * vs + 1e-9 + 0.1 * hs)


def test_edge_mask_and_range():
    p = params_with_defaults({"slabs": [[3.0, 1.5, 0.05]]})
    m = ev.edge_mask(np.array([2.95, 3.2, 4.48, 5.0]), np.zeros(4), p)
    assert m.tolist() == [True, False, True, False]
    curve = [{"x": x, "n": 100, "coverage": c, "p90_cm": e}
             for x, c, e in [(0.05, 0.0, 0.0), (0.15, 0.5, 0.4), (0.25, 0.5, 0.9), (0.35, 0.5, 1.2), (0.45, 0.5, 0.5)]]
    assert ev.range_1cm(curve) == [0.1, 0.3]
