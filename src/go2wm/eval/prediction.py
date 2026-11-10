"""Prediction evaluation: teacher-forced one-step vs open-loop rollouts at fixed horizons."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from ..data.dataset import WindowArrays, build_windows
from ..data.episodes import Episode
from ..models.features import TARGET_NAMES
from ..models.state_dynamics import DynamicsBase, KinematicBaseline


def bootstrap_ci(x: np.ndarray, n: int = 1000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float]:
    x = np.asarray(x, dtype=np.float64)
    if len(x) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = rng.choice(x, size=(n, len(x)), replace=True).mean(axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


@torch.no_grad()
def _rollout_np(model: DynamicsBase, w: WindowArrays, device: torch.device, bs: int = 4096) -> dict[str, np.ndarray]:
    model.eval().to(device)
    outs: dict[str, list] = {"poses": [], "deltas": []}
    for i in range(0, len(w), bs):
        t = lambda a: torch.as_tensor(a[i : i + bs], device=device)  # noqa: E731
        o = model.rollout(t(w.hist_feat), t(w.hist_act), t(w.hist_dt), t(w.fut_act), t(w.fut_dt))
        outs["poses"].append(o["poses"].cpu().numpy())
        outs["deltas"].append(o["deltas"].cpu().numpy())
    return {k: np.concatenate(v) for k, v in outs.items()}


def evaluate_models(
    models: dict[str, DynamicsBase],
    episodes: list[Episode],
    history: int,
    horizons_s: list[float],
    device: torch.device | str = "cpu",
) -> dict[str, Any]:
    device = torch.device(device)
    dt_nom = float(np.median(np.concatenate([e["dt_s"] for e in episodes])))
    steps = {h: max(1, int(round(h / dt_nom))) for h in horizons_s}
    H = max(steps.values())
    lens = [e.num_transitions for e in episodes]
    if max(lens) < history + H:
        H = max(lens) - history
        steps = {h: s for h, s in steps.items() if s <= H}
    w = build_windows(episodes, history, H)
    w1 = build_windows(episodes, history, 1)
    fams = np.array([episodes[i].meta.get("scene_family", "unknown") for i in w.episode_index])
    result: dict[str, Any] = {
        "nominal_dt_s": dt_nom,
        "horizon_steps": {str(k): v for k, v in steps.items()},
        "n_windows": len(w),
        "n_windows_one_step": len(w1),
        "n_episodes": len(episodes),
        "models": {},
    }
    ex_idx = np.linspace(0, len(w) - 1, 6).astype(int)
    per_window: dict[str, np.ndarray] = {}
    result["examples"] = {"true": w.fut_pose[ex_idx].tolist(), "episode": [episodes[i].episode_id for i in w.episode_index[ex_idx]]}
    for name, model in models.items():
        r: dict[str, Any] = {}
        # teacher-forced one-step
        one = _rollout_np(model, w1, device)
        err = one["deltas"][:, 0] - w1.targets[:, 0]
        r["one_step"] = {
            "mae": dict(zip(TARGET_NAMES, np.abs(err).mean(0).round(6).tolist())),
            "rmse": dict(zip(TARGET_NAMES, np.sqrt((err**2).mean(0)).round(6).tolist())),
        }
        # open-loop autoregressive
        ro = _rollout_np(model, w, device)
        pos_err = np.linalg.norm(ro["poses"][..., :2] - w.fut_pose[..., :2], axis=-1)  # [N,H]
        dyaw = ro["poses"][..., 2] - w.fut_pose[..., 2]
        head_err = np.abs(np.arctan2(np.sin(dyaw), np.cos(dyaw)))
        r["open_loop"] = {}
        for h_s, s in steps.items():
            pe, he = pos_err[:, s - 1], head_err[:, s - 1]
            # per-episode means -> CI over episodes (windows within an episode are correlated)
            ep_means = np.array([pe[w.episode_index == e].mean() for e in np.unique(w.episode_index)])
            entry = {
                "steps": s,
                "pos_err_m_mean": float(pe.mean()),
                "pos_err_m_median": float(np.median(pe)),
                "pos_err_m_p90": float(np.quantile(pe, 0.9)),
                "pos_err_m_ci95_over_episodes": bootstrap_ci(ep_means),
                "heading_err_deg_mean": float(np.degrees(he.mean())),
                "heading_err_deg_median": float(np.degrees(np.median(he))),
                "by_family": {
                    f: {"pos_err_m_mean": float(pe[fams == f].mean()), "n_windows": int((fams == f).sum())}
                    for f in sorted(set(fams))
                },
            }
            r["open_loop"][str(h_s)] = entry
        per_window[name] = pos_err
        r["pos_err_curve_m"] = pos_err.mean(0).tolist()
        r["heading_err_curve_deg"] = np.degrees(head_err.mean(0)).tolist()
        result["models"][name] = r
        result["examples"][name] = ro["poses"][ex_idx].tolist()
    result["paired"] = paired_comparison(per_window, w.episode_index, steps)
    return result


def paired_comparison(per_window: dict[str, np.ndarray], episode_index: np.ndarray,
                      steps: dict[float, int]) -> dict[str, Any]:
    """Reference = first model. For each other model, at each horizon: difference in per-episode
    mean position error (other - ref; >0 means ref is better) with a bootstrap CI over episodes,
    and the fraction of windows where ref has the lower error. Same windows for every model."""
    names = list(per_window)
    if len(names) < 2:
        return {}
    ref = names[0]
    eps = np.unique(episode_index)
    out: dict[str, Any] = {"reference": ref, "comparisons": {}}
    for other in names[1:]:
        cmp = {}
        for h_s, s in steps.items():
            a, b = per_window[ref][:, s - 1], per_window[other][:, s - 1]
            d_ep = np.array([(b - a)[episode_index == e].mean() for e in eps])
            cmp[str(h_s)] = {
                "mean_diff_m": float(d_ep.mean()),
                "diff_ci95_over_episodes": bootstrap_ci(d_ep),
                "episodes_ref_better": int((d_ep > 0).sum()),
                "n_episodes": int(len(eps)),
                "window_frac_ref_better": float((a < b).mean()),
            }
        out["comparisons"][other] = cmp
    return out


def default_baselines(history: int) -> dict[str, DynamicsBase]:
    return {
        "kinematic_command": KinematicBaseline(history, "command"),
        "constant_velocity": KinematicBaseline(history, "constant_velocity"),
    }
