from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def plot_prediction_report(result: dict[str, Any], out_dir: str | Path, title_suffix: str = "") -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dt = result["nominal_dt_s"]
    paths = []

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for name, r in result["models"].items():
        t = dt * np.arange(1, len(r["pos_err_curve_m"]) + 1)
        axes[0].plot(t, r["pos_err_curve_m"], label=name)
        axes[1].plot(t, r["heading_err_curve_deg"], label=name)
    axes[0].set(xlabel="open-loop horizon [s]", ylabel="mean position error [m]")
    axes[1].set(xlabel="open-loop horizon [s]", ylabel="mean heading error [deg]")
    for ax in axes:
        ax.grid(alpha=0.3)
    axes[0].legend()
    fig.suptitle(f"Open-loop rollout error (n={result['n_windows']} windows, {result['n_episodes']} episodes){title_suffix}")
    fig.tight_layout()
    p = out_dir / "rollout_error_vs_horizon.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    paths.append(p)

    ex = result["examples"]
    n = len(ex["true"])
    fig, axes = plt.subplots(2, (n + 1) // 2, figsize=(4 * ((n + 1) // 2), 7))
    for i, ax in enumerate(np.ravel(axes)[:n]):
        tr = np.array(ex["true"][i])
        ax.plot([0, *tr[:, 0]], [0, *tr[:, 1]], "k-", lw=2, label="true")
        for name in result["models"]:
            pr = np.array(ex[name][i])
            ax.plot([0, *pr[:, 0]], [0, *pr[:, 1]], "--", label=name)
        ax.set_aspect("equal", "datalim")
        ax.set_title(ex["episode"][i], fontsize=7)
        ax.grid(alpha=0.3)
    np.ravel(axes)[0].legend(fontsize=7)
    fig.suptitle(f"Example open-loop rollouts in start body frame [m]{title_suffix}")
    fig.tight_layout()
    p = out_dir / "rollout_examples.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    paths.append(p)
    return paths


def draw_sidewalk(ax, terrain: dict, xlim=(-2.0, 15.0), ylim=(-4.0, 4.0), res: float = 0.02, seed: int = 0) -> None:
    """Top-down 'map' of a sidewalk tile: concrete walkway with score joints, grass verge, road and
    curb (if the tile has a curb drop), hill-shaded from the true height field so slab edges, ramps
    and cross-slope are visible, and labelled with the feature heights. Purely a visualisation of
    the same geometry the simulator uses (go2wm.sim.terrain.sidewalk_height)."""
    from go2wm.sim.sidewalk_texture import sidewalk_albedo
    from go2wm.sim.terrain import params_with_defaults, sidewalk_height

    t = params_with_defaults(terrain)
    half = 0.5 * t["width"]
    xs = np.arange(xlim[0], xlim[1], res)
    ys = np.arange(ylim[0], ylim[1], res)
    X, Y = np.meshgrid(xs, ys)
    Z = sidewalk_height(X, Y, t)
    img = sidewalk_albedo(X, Y, t, seed=int(seed))  # same painter as the Isaac terrain texture
    # hill shading from the height field, light from the upper left: ramps get a gentle tint,
    # step edges (slab sides, curb) a sharp highlight/shadow line
    gy, gx = np.gradient(Z, res)
    shade = np.clip(1.0 + 0.6 * (-gx * 0.7 + gy * 0.7), 0.5, 1.4)
    img = np.clip(img * shade[..., None], 0, 1)
    ax.imshow(img, extent=(xs[0], xs[-1] + res, ys[0], ys[-1] + res), origin="lower", zorder=0, interpolation="bilinear")
    lab = dict(fontsize=8, ha="center", va="center", zorder=3, bbox=dict(fc="white", ec="none", alpha=0.8, pad=1.5))
    for k, (x0, length, h) in enumerate(t["slabs"]):
        ax.text(x0 + 0.5 * length, half - 0.3 - 0.35 * (k % 2), f"slab +{h * 100:.0f} cm", **lab)
    for x0, x1, rise in t["grade"]:
        ax.annotate("", xy=(x1, half - 1.2), xytext=(x0, half - 1.2), zorder=3,
                    arrowprops=dict(arrowstyle="->" if rise > 0 else "<-", color="k", lw=1.2))
        ax.text(0.5 * (x0 + x1), half - 1.0, f"ramp {abs(rise) * 100:.0f} % {'up' if rise > 0 else 'down'}", **lab)
    if t["curb_drop"] > 0:
        ax.text(xlim[0] + 2.2, -half - 1.0, f"road  (curb drop {t['curb_drop'] * 100:.0f} cm)", **lab)
    if abs(t["cross_slope"]) > 0.005:
        ax.text(xlim[0] + 2.2, half + 0.35, f"cross-slope {t['cross_slope'] * 100:+.0f} % (left high)" if t["cross_slope"] > 0
                else f"cross-slope {abs(t['cross_slope']) * 100:.0f} % (right high)", **lab)
