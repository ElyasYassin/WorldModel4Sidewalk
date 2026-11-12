import json

import numpy as np

from go2wm.scene.graph import ADA_CLEAR_WIDTH_M, build_scene_graph
from go2wm.sim.defects import (DEFECT_TYPES, defect_height, obstacles_from_defects, random_defects, range_dropout)
from go2wm.sim.sidewalk_texture import sidewalk_albedo
from go2wm.sim.terrain import params_with_defaults, random_defect_tile, sidewalk_height, sidewalk_heightfield

UPH = {"type": "upheaval", "x0": 3.0, "x1": 4.5, "y0": 0.0, "y1": 1.5, "lift": 0.05, "high_end": "x1", "tree": [3.75, 2.1]}
CRACK = {"type": "crack", "points": [[6.0, -0.5], [6.5, -0.4], [7.0, -0.6]], "width": 0.008, "depth": 0.02}
HOLE = {"type": "hole", "x": 8.0, "y": 0.5, "rx": 0.2, "ry": 0.15, "angle": 0.0, "depth": 0.10}
PUDDLE = {"type": "puddle", "x": 10.0, "y": -0.5, "rx": 0.6, "ry": 0.4, "angle": 0.3, "depth": 0.02}
BIN = {"type": "obstruction", "kind": "bin", "x": 12.0, "y": 0.9, "r": 0.3}
TILE = {"width": 3.0, "defects": [UPH, CRACK, HOLE, PUDDLE, BIN], "horizontal_scale": 0.02, "vertical_scale": 0.001,
        "size": [30.0, 7.0]}


def test_defect_geometry():
    p = params_with_defaults(TILE)
    z = lambda x, y, g=0.0: float(sidewalk_height(x, y, p, grid=g))  # noqa: E731
    # upheaval: 0 at the low joint, +lift at the high joint -> a 5 cm fault onto the next panel
    assert abs(z(3.0, 0.7)) < 1e-9 and abs(z(4.499, 0.7) - 0.05) < 1e-3 and abs(z(4.501, 0.7)) < 1e-9
    assert abs(z(4.0, -0.7)) < 1e-9  # only the left half-panel lifts
    # hole: full depth inside, steep wall, flat outside; puddle: shallow bowl
    assert abs(z(8.0, 0.5) + 0.10) < 1e-9 and abs(z(8.0, 0.5 + 0.15 * 0.9) + 0.10 * 0.1 / 0.15) < 1e-6
    assert abs(z(8.3, 0.5)) < 1e-9
    assert abs(z(10.0, -0.5) + 0.02) < 1e-9 and z(10.0 + 0.3, -0.5 + 0.3 * np.tan(0.3)) > -0.02
    # narrow crack: analytic groove at true width; on a 2 cm grid the same cross-section area
    on = (6.5, -0.4)
    assert abs(z(*on) + 0.02) < 1e-9 and abs(z(6.5, -0.4 + 0.006)) < 1e-9
    assert abs(z(*on, 0.02) + 0.02 * 0.008 / 0.02) < 1e-9 and abs(z(6.5, -0.4 + 0.009, 0.02) + 0.008) < 1e-9
    # obstruction: no terrain change, but an obstacle (plus the upheaval's tree)
    assert abs(z(12.0, 0.9)) < 1e-9
    obs = obstacles_from_defects(TILE["defects"])
    assert obs.shape == (2, 3) and np.allclose(obs[0], [12.0, 0.9, 0.3]) and np.allclose(obs[1, :2], [3.75, 2.1])
    # height field on the fine grid keeps 1 mm steps and the hole depth
    hf = sidewalk_heightfield(tuple(p["size"]), 0.02, 0.001, p)
    assert hf.dtype == np.int16 and hf.min() <= -95


def test_puddle_dropout_and_albedo():
    p = params_with_defaults(TILE)
    assert range_dropout(np.array([10.0]), np.array([-0.5]), p["defects"])[0] > 0.5
    assert range_dropout(np.array([8.0]), np.array([0.5]), p["defects"])[0] == 0.0
    X, Y = np.meshgrid(np.arange(2.0, 13.0, 0.02), np.arange(-2.0, 2.5, 0.02))
    img = sidewalk_albedo(X, Y, p)
    assert img.shape == X.shape + (3,) and np.isfinite(img).all() and img.min() >= 0 and img.max() <= 1
    at = lambda x, y: img[np.argmin(np.abs(Y[:, 0] - y)), np.argmin(np.abs(X[0] - x))]  # noqa: E731
    plain = at(11.0, -1.0)
    assert at(10.0, -0.5)[2] > at(10.0, -0.5)[0] and at(10.0, -0.5).mean() < plain.mean()  # water: dark, bluish
    assert at(8.0, 0.5).mean() < plain.mean()  # hole: dark soil


def test_scene_graph_defects():
    g = build_scene_graph(TILE, x_range=(-1.0, 14.0))
    hz = {h["id"]: h for h in g["hazards"]}
    up, cr, ho, pu, ob = (f"defect_{k}_{t}" for k, t in enumerate(["upheaval", "crack", "hole", "puddle", "obstruction"]))
    assert hz[up]["ada"] == "violation" and abs(hz[up]["lift_m"] - 0.05) < 1e-9 and hz[up]["robot_risk"] == "low"
    assert hz[cr]["ada"] == "monitor" and abs(hz[cr]["width_m"] - 0.008) < 1e-9  # 8 mm < 1/2 in opening
    assert hz[ho]["ada"] == "violation" and hz[ho]["robot_risk"] == "high"  # 40 cm wide, 10 cm deep
    assert hz[pu]["ada"] == "n/a" and abs(hz[pu]["area_m2"] - np.pi * 0.24) < 1e-9
    clear = g["nodes"][ob]["clear_width_m"]
    assert abs(clear - (0.9 - 0.3 + 1.5)) < 1e-9 and hz[ob]["ada"] == ("violation" if clear < ADA_CLEAR_WIDTH_M else "ok")
    # the tree is an asset that causes the upheaval; the fault joint names its cause
    assert any(e["src"] == "tree_0" and e["rel"] == "causes" and e["dst"] == up for e in g["edges"])
    joint = [h for h in g["hazards"] if h["type"] == "vertical_displacement" and abs(h["xy"][0] - 4.5) < 1e-6]
    assert joint and joint[0]["cause"] == up and abs(joint[0]["value_m"] + 0.05) < 1e-3
    # every defect sits on a panel or the walkway
    on = {e["src"]: e["dst"] for e in g["edges"] if e["rel"] == "on"}
    assert all(on[i].startswith(("panel_", "walkway")) for i in (up, cr, ho, pu, ob))
    json.dumps(g, default=float)


def test_random_defect_tiles_deterministic_and_disjoint():
    a = random_defect_tile(np.random.default_rng(3))
    b = random_defect_tile(np.random.default_rng(3))
    assert a == b and 4 <= len(a["defects"]) <= 8 and a["horizontal_scale"] == 0.02
    L = a["size"][0]
    for d in a["defects"]:
        assert d["type"] in DEFECT_TYPES
        x = d.get("x", d.get("x0", d.get("points", [[0, 0]])[0][0]))
        assert -L / 2 + 2.5 <= x <= L / 2 - 1.0
    p = params_with_defaults({"width": 3.0})
    ds = random_defects(np.random.default_rng(0), p, (0.0, 30.0), n=12)
    from go2wm.sim.defects import defect_box

    boxes = [defect_box(d) for d in ds]
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            bi, bj = boxes[i], boxes[j]
            assert bi[1] < bj[0] or bj[1] < bi[0] or bi[3] < bj[2] or bj[3] < bi[2]
