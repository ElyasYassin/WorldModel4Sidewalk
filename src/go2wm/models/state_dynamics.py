"""State-only action-conditioned dynamics: f(history, a_t, dt) -> delta targets.

All models share `step` (one transition) and `rollout` (open-loop autoregressive
over a candidate action sequence). Teacher-forced one-step prediction is
`rollout` with H=1 on ground-truth history.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from ..common.schema import ACTION_DIM
from .features import FEATURE_DIM, TARGET_DIM
from .normalization import Normalizer

# indices into target vector
DX, DY, DZ, DYAW = 0, 1, 2, 3
FEAT0 = 4
VX, VY, WZ = 0, 1, 5  # indices into feature vector


def compose_se2_torch(pose: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    """pose [..., 3] (x, y, yaw) composed with body-frame delta [..., 3]."""
    c, s = torch.cos(pose[..., 2]), torch.sin(pose[..., 2])
    x = pose[..., 0] + c * delta[..., 0] - s * delta[..., 1]
    y = pose[..., 1] + s * delta[..., 0] + c * delta[..., 1]
    yaw = torch.atan2(torch.sin(pose[..., 2] + delta[..., 2]), torch.cos(pose[..., 2] + delta[..., 2]))
    return torch.stack([x, y, yaw], dim=-1)


class DynamicsBase(nn.Module):
    model_type = "base"
    learned = True
    uses_terrain = False

    def __init__(self, history: int, feat_dim: int = FEATURE_DIM, act_dim: int = ACTION_DIM, target_dim: int = TARGET_DIM):
        super().__init__()
        self.history, self.feat_dim, self.act_dim, self.target_dim = history, feat_dim, act_dim, target_dim
        self.feat_norm = Normalizer(feat_dim)
        self.act_norm = Normalizer(act_dim)
        self.dt_norm = Normalizer(1)
        self.target_norm = Normalizer(target_dim)

    def fit_normalizers(self, w) -> None:
        self.feat_norm.fit(w.hist_feat)
        self.act_norm.fit(w.fut_act)
        self.dt_norm.fit(w.fut_dt.reshape(-1, 1))
        self.target_norm.fit(w.targets)

    def kwargs(self) -> dict[str, Any]:
        return {"history": self.history, "feat_dim": self.feat_dim, "act_dim": self.act_dim, "target_dim": self.target_dim}

    # --- to implement: normalized prediction -------------------------------
    def _step_norm(self, hf: torch.Tensor, ha: torch.Tensor, hd: torch.Tensor, a: torch.Tensor, dt: torch.Tensor,
                   extra: torch.Tensor | None = None) -> torch.Tensor:
        raise NotImplementedError

    def _context(self, pose: torch.Tensor, terrain_map: torch.Tensor | None) -> torch.Tensor | None:
        """Extra per-step input from the current predicted pose (terrain models); None otherwise."""
        return None

    def step_normalized(self, hist_feat, hist_act, hist_dt, act, dt, extra=None) -> torch.Tensor:
        """Returns normalized delta targets [B, T]."""
        return self._step_norm(
            self.feat_norm(hist_feat),
            self.act_norm(hist_act),
            self.dt_norm(hist_dt.unsqueeze(-1)),
            self.act_norm(act),
            self.dt_norm(dt.unsqueeze(-1)),
            extra,
        )

    def step(self, hist_feat, hist_act, hist_dt, act, dt) -> torch.Tensor:
        """Physical-unit delta targets [B, T]."""
        return self.target_norm.inverse(self.step_normalized(hist_feat, hist_act, hist_dt, act, dt))

    def rollout(self, hist_feat, hist_act, hist_dt, fut_act, fut_dt, return_normalized: bool = False,
                terrain_map: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        """Open-loop autoregressive rollout.

        hist_feat [B,K,F], hist_act [B,K,A], hist_dt [B,K], fut_act [B,H,A], fut_dt [B,H]
        -> deltas [B,H,T], poses [B,H,3] relative to start pose, feats [B,H,F]
        """
        B, H = fut_act.shape[0], fut_act.shape[1]
        pose = torch.zeros(B, 3, device=fut_act.device, dtype=fut_act.dtype)
        deltas, deltas_n, poses, feats = [], [], [], []
        hf, ha, hd = hist_feat, hist_act, hist_dt
        for h in range(H):
            a, dt = fut_act[:, h], fut_dt[:, h]
            dn = self.step_normalized(hf, ha, hd, a, dt, self._context(pose, terrain_map))
            d = self.target_norm.inverse(dn)
            new_feat = hf[:, -1] + d[:, FEAT0:]
            pose = compose_se2_torch(pose, d[:, [DX, DY, DYAW]])
            deltas.append(d)
            deltas_n.append(dn)
            poses.append(pose)
            feats.append(new_feat)
            hf = torch.cat([hf[:, 1:], new_feat[:, None]], dim=1)
            ha = torch.cat([ha[:, 1:], a[:, None]], dim=1)
            hd = torch.cat([hd[:, 1:], dt[:, None]], dim=1)
        out = {"deltas": torch.stack(deltas, 1), "poses": torch.stack(poses, 1), "feats": torch.stack(feats, 1)}
        if return_normalized:
            out["deltas_norm"] = torch.stack(deltas_n, 1)
        return out

    def predict(self, state_history, action_history, candidate_actions, dt) -> torch.Tensor:
        """Spec §7 interface: state_history [B,K,F] features, action_history [B,K,A],
        candidate_actions [B,H,A], dt [B,H] (or scalar) -> per-step predicted deltas [B,H,T]."""
        if not torch.is_tensor(dt) or dt.dim() == 0:
            dt = torch.full(candidate_actions.shape[:2], float(dt), device=candidate_actions.device)
        hd = dt[:, :1].expand(-1, state_history.shape[1])
        return self.rollout(state_history, action_history, hd, candidate_actions, dt)["deltas"]


class MLPDynamics(DynamicsBase):
    model_type = "mlp"

    def __init__(self, history: int, hidden: int = 256, layers: int = 3, **kw):
        super().__init__(history, **kw)
        self.hidden, self.layers = hidden, layers
        d_in = history * (self.feat_dim + self.act_dim + 1) + self.act_dim + 1
        mods: list[nn.Module] = []
        for i in range(layers):
            mods += [nn.Linear(d_in if i == 0 else hidden, hidden), nn.SiLU()]
        mods.append(nn.Linear(hidden, self.target_dim))
        self.net = nn.Sequential(*mods)

    def kwargs(self):
        return {**super().kwargs(), "hidden": self.hidden, "layers": self.layers}

    def _step_norm(self, hf, ha, hd, a, dt, extra=None):
        parts = [hf.flatten(1), ha.flatten(1), hd.flatten(1), a, dt] + ([extra] if extra is not None else [])
        return self.net(torch.cat(parts, dim=-1))


class GRUDynamics(DynamicsBase):
    model_type = "gru"

    def __init__(self, history: int, hidden: int = 128, layers: int = 1, **kw):
        super().__init__(history, **kw)
        self.hidden, self.layers = hidden, layers
        self.gru = nn.GRU(self.feat_dim + self.act_dim + 1, hidden, num_layers=layers, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden + self.act_dim + 1, hidden), nn.SiLU(), nn.Linear(hidden, self.target_dim))

    def kwargs(self):
        return {**super().kwargs(), "hidden": self.hidden, "layers": self.layers}

    def _step_norm(self, hf, ha, hd, a, dt, extra=None):
        _, h = self.gru(torch.cat([hf, ha, hd], dim=-1))
        return self.head(torch.cat([h[-1], a, dt], dim=-1))


class KinematicBaseline(DynamicsBase):
    """Non-learned baselines with the same interface.

    mode='constant_velocity': last realized body velocity persists.
    mode='command': realized velocity equals the commanded velocity (unicycle/holonomic kinematics).
    """

    learned = False

    def __init__(self, history: int = 1, mode: str = "command", **kw):
        super().__init__(history, **kw)
        if mode not in ("constant_velocity", "command"):
            raise ValueError(mode)
        self.mode = mode
        self.model_type = f"kinematic_{mode}"

    def kwargs(self):
        return {**super().kwargs(), "mode": self.mode}

    def step(self, hist_feat, hist_act, hist_dt, act, dt):
        last = hist_feat[:, -1]
        if self.mode == "command":
            v = torch.stack([act[:, 0], act[:, 1], act[:, 2]], -1)
        else:
            v = torch.stack([last[:, VX], last[:, VY], last[:, WZ]], -1)
        d = torch.zeros(act.shape[0], self.target_dim, device=act.device, dtype=act.dtype)
        # exact SE(2) integration of constant twist over dt
        w = v[:, 2]
        th = w * dt
        small = th.abs() < 1e-6
        safe_w = torch.where(small, torch.ones_like(w), w)
        sa = torch.where(small, dt, torch.sin(th) / safe_w)
        ca = torch.where(small, torch.zeros_like(dt), (1 - torch.cos(th)) / safe_w)
        d[:, DX] = sa * v[:, 0] - ca * v[:, 1]
        d[:, DY] = ca * v[:, 0] + sa * v[:, 1]
        d[:, DYAW] = th
        new_feat = last.clone()
        new_feat[:, VX], new_feat[:, VY], new_feat[:, WZ] = v[:, 0], v[:, 1], v[:, 2]
        d[:, FEAT0:] = new_feat - last
        return d

    def step_normalized(self, hist_feat, hist_act, hist_dt, act, dt, extra=None):
        return self.target_norm(self.step(hist_feat, hist_act, hist_dt, act, dt))


class TerrainMLPDynamics(MLPDynamics):
    """MLP + local terrain patch sampled at each predicted pose from the start-frame height map."""

    model_type = "terrain_mlp"
    uses_terrain = True

    def __init__(self, history: int, hidden: int = 256, layers: int = 3, terrain_source: str = "analytic", **kw):
        from .terrain_map import PATCH_DIM

        super().__init__(history, hidden, layers, **kw)
        assert terrain_source in ("analytic", "scan") or terrain_source.startswith("sensor:"), terrain_source
        self.terrain_source = terrain_source  # analytic = perfect tile geometry; scan = robot's height scanner
        self.patch_dim = PATCH_DIM
        d_in = history * (self.feat_dim + self.act_dim + 1) + self.act_dim + 1 + PATCH_DIM
        self.net[0] = nn.Linear(d_in, hidden)

    def kwargs(self):
        return {**super().kwargs(), "terrain_source": self.terrain_source}

    def _context(self, pose, terrain_map):
        from .terrain_map import sample_patch

        if terrain_map is None:
            return torch.zeros(pose.shape[0], self.patch_dim, device=pose.device, dtype=pose.dtype)
        return sample_patch(terrain_map, pose)


class TerrainGRUDynamics(GRUDynamics):
    """GRU + the same local terrain patch as TerrainMLPDynamics, fed to the output head."""

    model_type = "terrain_gru"
    uses_terrain = True

    def __init__(self, history: int, hidden: int = 128, layers: int = 1, terrain_source: str = "analytic", **kw):
        from .terrain_map import PATCH_DIM

        super().__init__(history, hidden, layers, **kw)
        assert terrain_source in ("analytic", "scan") or terrain_source.startswith("sensor:"), terrain_source
        self.terrain_source = terrain_source
        self.patch_dim = PATCH_DIM
        self.head[0] = nn.Linear(hidden + self.act_dim + 1 + PATCH_DIM, hidden)

    _context = TerrainMLPDynamics._context

    def kwargs(self):
        return {**super().kwargs(), "terrain_source": self.terrain_source}

    def _step_norm(self, hf, ha, hd, a, dt, extra=None):
        if extra is None:
            extra = torch.zeros(a.shape[0], self.patch_dim, device=a.device, dtype=a.dtype)
        _, h = self.gru(torch.cat([hf, ha, hd], dim=-1))
        return self.head(torch.cat([h[-1], a, dt, extra], dim=-1))


MODEL_TYPES = {"mlp": MLPDynamics, "gru": GRUDynamics, "terrain_mlp": TerrainMLPDynamics,
               "terrain_gru": TerrainGRUDynamics}


def build_model(cfg: dict[str, Any]) -> DynamicsBase:
    m = cfg["model"]
    t = m["type"]
    if t.startswith("kinematic_"):
        return KinematicBaseline(history=cfg["data"]["history"], mode=t[len("kinematic_") :])
    kw = {k: v for k, v in m.items() if k != "type"}
    return MODEL_TYPES[t](history=cfg["data"]["history"], **kw)


def save_checkpoint(path: str | Path, model: DynamicsBase, extra: dict[str, Any] | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_type": model.model_type, "kwargs": model.kwargs(), "state_dict": model.state_dict(), **(extra or {})}, path)


def load_checkpoint(path: str | Path, map_location: str = "cpu") -> tuple[DynamicsBase, dict[str, Any]]:
    ck = torch.load(path, map_location=map_location, weights_only=False)
    kw = dict(ck["kwargs"])
    t = ck["model_type"]
    if t.startswith("kinematic_"):
        model: DynamicsBase = KinematicBaseline(**kw)
    else:
        model = MODEL_TYPES[t](**kw)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, ck
