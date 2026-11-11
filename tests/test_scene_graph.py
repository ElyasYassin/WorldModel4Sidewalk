import json

import numpy as np

from go2wm.scene.graph import (ada_level_class, build_scene_graph, describe, graph_from_episode_meta, hazard_summary,
                               keep_out_boxes)

TILE = {"width": 3.0, "cross_slope": -0.05, "slabs": [[4.0, 1.5, 0.16]], "grade": [[8.0, 10.0, 0.30]], "curb_drop": 0.12}


def test_ada_level_thresholds():
    assert ada_level_class(0.005) == "ok" and ada_level_class(-0.01) == "bevel_required" and ada_level_class(0.02) == "violation"


def test_graph_structure_and_hazards():
    g = build_scene_graph(TILE, tile_index=3)
    n, kinds = g["nodes"], {}
    for v in n.values():
        kinds[v["kind"]] = kinds.get(v["kind"], 0) + 1
    assert {"walkway", "road", "curb", "verge_left", "slab_0", "ramp_0", "start", "end"} <= set(n)
    assert "verge_right" not in n  # the curb side is road
    assert kinds["panel"] % 2 == 0 and kinds["panel"] >= 20  # two rows of panels on a 3 m walk
    # slab joints: one hazard per joint line, step up then step down, both ADA violations
    vd = sorted((h for h in g["hazards"] if h["type"] == "vertical_displacement"), key=lambda h: h["xy"][0])
    assert [round(h["xy"][0], 2) for h in vd] == [4.0, 5.5]
    assert vd[0]["value_m"] > 0.15 and vd[1]["value_m"] < -0.15 and all(h["ada"] == "violation" for h in vd)
    assert vd[0]["robot_risk"] == "high"  # 16 cm step up: the rough policy stalls (E4 sweep)
    # panels on the slab are covered by it; next edges carry the joint step
    covered = {e["dst"] for e in g["edges"] if e["rel"] == "covers"}
    assert covered and all(4.0 - 1e-6 <= n[p]["box"][0] and n[p]["box"][1] <= 5.5 + 1e-6 for p in covered)
    steps = [e["step_m"] for e in g["edges"] if e["rel"] == "next" and abs(e["joint_x"] - 4.0) < 1e-6]
    assert len(steps) == 2 and all(abs(s - 0.16) < 1e-6 for s in steps)
    # ramp 30 %: ADA violation and high robot risk; cross slope 5 % violation
    assert n["ramp_0"]["ada"] == "violation" and n["ramp_0"]["robot_risk"] == "high"
    assert any(h["type"] == "cross_slope" for h in g["hazards"])
    ko = {k["id"] for k in keep_out_boxes(g, "high")}
    assert {"slab_0", "ramp_0"} <= ko and "curb" not in ko  # 12 cm curb is medium
    s = hazard_summary(g)
    assert s["by_type"]["crack"] >= 2 and s["robot_high_risk"] >= 2
    assert "slab_0" in describe(g)
    json.dumps(g, default=float)  # serialisable


def test_graph_cracks_match_texture_and_episode_meta():
    from go2wm.sim.sidewalk_texture import crack_segments
    from go2wm.sim.terrain import params_with_defaults

    g = build_scene_graph(TILE, tile_index=2)
    cr = [np.array(v["polyline"]) for v in g["nodes"].values() if v["kind"] == "defect"]
    ref = crack_segments(params_with_defaults(TILE), 2)
    assert len(cr) == len(ref) and all(np.allclose(a, b, atol=1e-3) for a, b in zip(cr, ref))
    meta = {"episode_id": "e", "scene": {"params": {"terrain": TILE, "terrain_tile": 2}}}
    assert graph_from_episode_meta(meta)["meta"]["tile_index"] == 2
    assert graph_from_episode_meta({"scene": {"params": {}}}) is None
