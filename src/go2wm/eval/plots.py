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
