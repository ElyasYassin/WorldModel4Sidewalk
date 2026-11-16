import numpy as np

from go2wm.mission.memory import SpatialMemory
from go2wm.mission.perception import Observation


def _mem():
    return SpatialMemory((-5.0, 5.0, -1.5, 1.5), [0.0, 5.0])


def _det(t, kind, x, y, ext=0.2):
    return Observation(t, "depth_d555", kind, np.array([x, y]), 1.5, extent_m=ext, provenance="detected")


def test_single_detection_is_not_real_and_two_are():
    m = _mem()
    a = m.add(_det(0.0, "depression", 1.0, 0.0))
    assert not m.is_real(a)
    m.add(_det(0.1, "depression", 1.05, 0.02))
    assert m.is_real(a) and len(m.findings) == 1


def test_overlapping_pieces_are_consolidated_and_alias_resolves():
    m = _mem()
    a = m.add(_det(0.0, "depression", 1.0, 0.0, ext=0.1))
    b = m.add(_det(0.1, "depression", 1.5, 0.0, ext=0.1))  # too far for the first association
    assert a != b and len(m.findings) == 2
    m.add(_det(0.2, "depression", 1.25, 0.0, ext=0.8))  # a wide view of the same hole bridges both
    assert len(m.findings) == 1 and m.resolve(b) == a


def test_depressions_aligned_across_the_walk_become_an_edge():
    m = _mem()
    for t in (0.0, 0.1):
        m.add(_det(t, "depression", 2.0, -0.4))
        m.add(_det(t + 0.05, "depression", 2.05, 0.5))
    (f,) = m.findings.values()
    assert f["kind"] == "edge" and f.get("relabelled")


def test_edge_footprint_and_hole_shadow_are_merged():
    m = _mem()
    for t in (0.0, 0.1):
        m.add(_det(t, "edge", 3.0, 0.0, ext=1.2))
        m.add(_det(t, "depression", 3.4, 0.3))  # low side of the step (sign unknown yet: both sides allowed)
        m.add(_det(t, "depression", -2.0, 0.0, ext=0.3))  # a real hole
        m.add(_det(t, "dropout", -1.9, 0.1))  # the ground its near wall hides
    real = m.to_dict()["findings"]
    kinds = sorted(f["kind"] for f in real.values())
    assert kinds == ["depression", "edge"], kinds


def test_classify_shape_step_vs_pit_vs_flat():
    from go2wm.mission.perception import classify_shape
    from go2wm.sim.offline_sensors import OfflineRig

    terrains = {
        "step": lambda x, y: np.where(np.asarray(x) > 2.0, 0.02, 0.0),
        "pit": lambda x, y: np.where(np.hypot(np.asarray(x) - 2.0, np.asarray(y)) < 0.2, -0.05, 0.0),
        "flat": lambda x, y: np.zeros(np.shape(x)),
    }
    for want, H in terrains.items():
        rig = OfflineRig(["depth_d555"], "ideal", seed=0)
        q = rig.frame("depth_d555", np.array([1.0, 0.0, 0.33]), np.eye(3), H)
        got = classify_shape(q, np.array([1.0, 0.0, 0.0]), (2.0, 0.0), extent=0.4)
        assert got == (None if want == "flat" else want), (want, got)


def test_shape_votes_relabel_a_depression_as_edge():
    m = _mem()
    a = m.add(_det(0.0, "depression", 2.0, 0.3))
    m.add(_det(0.1, "depression", 2.0, 0.32))
    for _ in range(3):
        a = m.vote_shape(a, "step")
    assert m.findings[a]["kind"] == "edge"


def test_short_edge_inside_a_hole_is_its_wall_and_static_views_do_not_count():
    m = _mem()
    for t in (0.0, 0.1):
        m.add(_det(t, "depression", 3.0, 0.2, ext=0.5))
        e = _det(t, "edge", 3.05, 0.25, ext=0.6)
        e.extra["y_span"] = [0.0, 0.5]
        m.add(e)
    real = m.to_dict()["findings"]
    assert [f["kind"] for f in real.values()] == ["depression"]
    # the same blob detected repeatedly from one standing position is not confirmed
    m2 = _mem()
    for t in (0.0, 0.1, 0.2):
        o = _det(t, "depression", 1.0, 0.0)
        o.extra["view_xy"] = [0.0, 0.0]
        fid = m2.add(o)
    assert not m2.is_real(fid)
    o = _det(0.3, "depression", 1.0, 0.0)
    o.extra["view_xy"] = [0.3, 0.0]
    assert m2.is_real(m2.add(o))


def test_robust_hole_depth_ignores_partial_views_and_one_spurious_deep_reading():
    """Approaching a 110 mm hole, the D405 first sees only its near part (shallow readings); the robust depth uses
    the readings near the top, judged against the 2nd-largest, so a single spurious deep reading cannot set the bar."""
    for extra in ([], [0.634]):
        m = SpatialMemory((-5.0, 5.0, -1.5, 1.5), [0.0, 5.0], robust=True)
        fid = m.add(_det(0.0, "depression", 1.0, 0.0))
        for k, v in enumerate([0.0013, 0.0124, 0.0446, 0.0902, 0.1162, 0.1096, 0.1101] + extra):
            m.add(Observation(1.0 + 0.1 * k, "depth_d405", "depression", np.array([1.0, 0.0]), 0.5, value_m=v,
                              confidence=0.95))
        assert abs(m.findings[fid]["value_m"] - 0.110) < 0.004
    # non-robust memory keeps the plain mean of the best-sensor readings (frozen evaluations)
    m = _mem()
    fid = m.add(_det(0.0, "depression", 1.0, 0.0))
    for k, v in enumerate([0.02, 0.11, 0.11]):
        m.add(Observation(1.0 + 0.1 * k, "depth_d405", "depression", np.array([1.0, 0.0]), 0.5, value_m=v,
                          confidence=0.95))
    assert abs(m.findings[fid]["value_m"] - 0.08) < 1e-9
