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
