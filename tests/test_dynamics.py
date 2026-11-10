import numpy as np
import pytest
import torch

from go2wm.data.dataset import build_windows
from go2wm.data.episodes import load_all
from go2wm.models.state_dynamics import (
    GRUDynamics,
    KinematicBaseline,
    MLPDynamics,
    load_checkpoint,
    save_checkpoint,
)


@pytest.fixture(scope="module")
def windows(mock_dataset):
    return build_windows(load_all(mock_dataset), 4, 6)


@pytest.mark.parametrize("cls", [MLPDynamics, GRUDynamics])
def test_shapes_and_gradients(cls, windows):
    torch.manual_seed(0)
    m = cls(history=4)
    m.fit_normalizers(windows)
    b = lambda a: torch.as_tensor(a[:8])  # noqa: E731
    out = m.rollout(b(windows.hist_feat), b(windows.hist_act), b(windows.hist_dt), b(windows.fut_act), b(windows.fut_dt), return_normalized=True)
    assert out["deltas"].shape == (8, 6, 13) and out["poses"].shape == (8, 6, 3) and out["feats"].shape == (8, 6, 9)
    out["deltas_norm"].pow(2).mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
    # spec interface
    pred = m.predict(b(windows.hist_feat), b(windows.hist_act), b(windows.fut_act), 0.1)
    assert pred.shape == (8, 6, 13)


def test_checkpoint_roundtrip_reproducible(windows, tmp_path):
    torch.manual_seed(0)
    m = MLPDynamics(history=4, hidden=32, layers=2)
    m.fit_normalizers(windows)
    m.eval()
    args = [torch.as_tensor(getattr(windows, k)[:4]) for k in ("hist_feat", "hist_act", "hist_dt", "fut_act", "fut_dt")]
    y1 = m.rollout(*args)["poses"]
    save_checkpoint(tmp_path / "m.pt", m, {"config": {}})
    m2, ck = load_checkpoint(tmp_path / "m.pt")
    assert torch.allclose(y1, m2.rollout(*args)["poses"])
    assert torch.allclose(m2.rollout(*args)["poses"], m2.rollout(*args)["poses"])


def test_kinematic_command_straight_and_arc():
    m = KinematicBaseline(history=1, mode="command")
    hf = torch.zeros(1, 1, 9)
    ha = torch.zeros(1, 1, 3)
    hd = torch.full((1, 1), 0.1)
    acts = torch.tensor([[[1.0, 0.0, 0.0]] * 10])
    out = m.rollout(hf, ha, hd, acts, torch.full((1, 10), 0.1))
    assert torch.allclose(out["poses"][0, -1], torch.tensor([1.0, 0.0, 0.0]), atol=1e-5)
    # quarter circle: v=1, w=pi/2 for 1 s -> radius r=v/w=2/pi, end pose (r, r, pi/2)
    acts = torch.tensor([[[1.0, 0.0, np.pi / 2]] * 10])
    out = m.rollout(hf, ha, hd, acts, torch.full((1, 10), 0.1))
    r = 2 / np.pi
    assert torch.allclose(out["poses"][0, -1], torch.tensor([r, r, np.pi / 2], dtype=torch.float32), atol=1e-4)
