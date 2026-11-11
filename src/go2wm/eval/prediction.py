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
        tm = t(w.terrain_map) if (model.uses_terrain and w.terrain_map is not None) else None
        o = model.rollout(t(w.hist_feat), t(w.hist_act), t(w.hist_dt), t(w.fut_act), t(w.fut_dt), terrain_map=tm)
        outs["poses"].append(o["poses"].cpu().numpy())
        outs["deltas"].append(o["deltas"].cpu().numpy())
    return {k: np.concatenate(v) for k, v in outs.items()}


def commanded_path(pose0: np.ndarray, cmds: np.ndarray, dt: np.ndarray) -> np.ndarray:
    """Scene-frame (x, y) path [H+1, 2] if the robot followed body-frame commands (vx, vy, wz) exactly."""
    x, y, yaw = (float(v) for v in pose0)
    out = [(x, y)]
    for (vx, vy, wz), d in zip(cmds[:, :3], dt):
        c, s = np.cos(yaw), np.sin(yaw)
        x, y, yaw = x + (c * vx - s * vy) * d, y + (s * vx + c * vy) * d, yaw + wz * d
        out.append((x, y))
    return np.asarray(out)


def terrain_segments(episodes: list[Episode], w: WindowArrays, H: int, path: str = "true") -> np.ndarray:
    """Hard-segment label per window over the full horizon (terrain episodes only; '' otherwise):
    curb (crosses the walkway edge with a curb), step (crosses a height jump >= 3 cm, i.e. a slab
    edge), grade (on a running slope >= 10 %), plain. path='true' uses the realized future path
    (a robot that stalls in front of a step is then 'plain'); path='commanded' uses the path the
    commands would produce from the start pose, i.e. the terrain the robot was asked to cross."""
    from ..common.schema import state_pose2d
    from ..sim.terrain import params_with_defaults, sidewalk_height

    out = np.full(len(w), "", dtype=object)
    poses = {}
    for n, (ei, i) in enumerate(zip(w.episode_index, w.start_index)):
        terrain = ((episodes[ei].meta.get("scene") or {}).get("params") or {}).get("terrain")
        if not terrain:
            continue
        t = params_with_defaults(terrain)
        if ei not in poses:
            poses[ei] = state_pose2d(episodes[ei]["states"])
        if path == "commanded":
            ep = episodes[ei]
            xy = commanded_path(poses[ei][i], ep["commands"][i : i + H], ep["dt_s"][i : i + H])
        else:
            xy = poses[ei][i : i + H + 1, :2]
        x, y = xy[:, 0], xy[:, 1]
        edge = -0.5 * t["width"]
        if t["curb_drop"] > 0 and (np.any(y < edge) and np.any(y >= edge)):
            out[n] = "curb"
            continue
        t_nocurb = {**t, "curb_drop": 0.0, "cross_slope": 0.0, "grade": []}
        z = sidewalk_height(x, y, t_nocurb)
        if np.any(np.abs(np.diff(z)) >= 0.03):
            out[n] = "step"
            continue
        on_grade = any(np.any((x >= x0) & (x < x1)) and abs(r) >= 0.10 for x0, x1, r in t["grade"])
        out[n] = "grade" if on_grade else "plain"
    return out


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
    tmap = any(getattr(m, "uses_terrain", False) for m in models.values())
    w = build_windows(episodes, history, H, terrain_map=tmap)
    w1 = build_windows(episodes, history, 1, terrain_map=tmap)
    fams = np.array([episodes[i].meta.get("scene_family", "unknown") for i in w.episode_index])
    pats = np.array([episodes[i].meta.get("command_pattern", "") for i in w.episode_index])
    segs = terrain_segments(episodes, w, H)
    segs_cmd = terrain_segments(episodes, w, H, path="commanded")
    groups = {k: v for k, v in (("pattern", pats), ("segment", segs), ("segment_cmd", segs_cmd)) if any(v)}
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
            for gname, lab in groups.items():
                entry[f"by_{gname}"] = {}
                for pt in sorted(set(lab)):
                    sel = lab == pt
                    eps_pt = np.unique(w.episode_index[sel])
                    pm = np.array([pe[sel & (w.episode_index == e)].mean() for e in eps_pt])
                    entry[f"by_{gname}"][pt] = {"pos_err_m_mean": float(pe[sel].mean()),
                                                "pos_err_m_p90": float(np.quantile(pe[sel], 0.9)),
                                                "heading_err_deg_mean": float(np.degrees(he[sel].mean())),
                                                "pos_err_m_ci95_over_episodes": bootstrap_ci(pm),
                                                "n_windows": int(sel.sum()), "n_episodes": int(len(eps_pt))}
            r["open_loop"][str(h_s)] = entry
        per_window[name] = pos_err
        r["pos_err_curve_m"] = pos_err.mean(0).tolist()
        r["heading_err_curve_deg"] = np.degrees(head_err.mean(0)).tolist()
        result["models"][name] = r
        result["examples"][name] = ro["poses"][ex_idx].tolist()
    result["paired"] = paired_comparison(per_window, w.episode_index, steps)
    for gname, lab in groups.items():
        result[f"paired_by_{gname}"] = {
            pt: paired_comparison({k: v[lab == pt] for k, v in per_window.items()}, w.episode_index[lab == pt], steps)
            for pt in sorted(set(lab))}
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
