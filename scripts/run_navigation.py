"""Waypoint navigation benchmark on identical seeded scenarios.

  python scripts/run_navigation.py --backend mock --planner all --checkpoint outputs/state_mock_mlp/best.pt
  python scripts/run_navigation.py --backend mock --planner conventional
Planners: conventional (reactive), kinematic_mpc (CEM + command kinematics),
learned_mpc (CEM + learned state dynamics; needs --checkpoint), all.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from go2wm.cli import write_json
from go2wm.common.config import load_config, resolve_path
from go2wm.eval.navigation import aggregate, run_episode, scenarios_from_cfg
from go2wm.models.state_dynamics import KinematicBaseline, load_checkpoint
from go2wm.planning.baseline import ReactiveController
from go2wm.planning.local_mpc import ModelPredictor, MPCController
from go2wm.planning.safety import SafetyFilter
from go2wm.sim.base import make_adapter

PLANNERS = ["conventional", "kinematic_mpc", "learned_mpc"]
REPO = Path(__file__).resolve().parents[1]


def final_test_scenes(scen: list, backend: str) -> list[str]:
    """Scene ids in the frozen final-test index range (configs/eval_protocol.yaml)."""
    if backend == "mock":
        return []
    proto = load_config(REPO / "configs" / "eval_protocol.yaml")["navigation"]
    lo, hi = proto["final_test"]["first_index"], proto["final_test"]["last_index"]
    return [sid for seed, sid in scen if lo <= seed - proto["base_seed"] <= hi]


def ledger_entry(args, cfg, scen, planners) -> dict:
    def sh(cmd):
        try:
            return subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, timeout=20).stdout.strip()
        except Exception:
            return None
    ck = None
    if args.checkpoint:
        ck = hashlib.sha256(resolve_path(args.checkpoint).read_bytes()).hexdigest()
    diff = sh(["git", "diff", "HEAD"]) or ""
    return {"utc": _dt.datetime.now(_dt.timezone.utc).isoformat(), "planners": planners, "config": args.config,
            "config_resolved_sha256": hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest(),
            "checkpoint": args.checkpoint, "checkpoint_sha256": ck, "out": args.out,
            "scenes": [sid for _, sid in scen], "git_commit": sh(["git", "rev-parse", "HEAD"]),
            "git_dirty_diff_sha256": hashlib.sha256(diff.encode()).hexdigest() if diff else None}


def make_controller(name: str, cfg: dict, checkpoint: str | None, seed: int = 0):
    safety = SafetyFilter({**cfg["safety"], "limits": cfg["limits"], "robot_radius": cfg["robot_radius"]})
    dev = cfg["mpc"].get("device", "cpu")
    if name == "conventional":
        return ReactiveController(cfg, safety)
    if name == "kinematic_mpc":
        return MPCController(ModelPredictor(KinematicBaseline(1, "command"), dev), cfg, safety, seed, name)
    if name == "learned_mpc":
        if not checkpoint:
            raise SystemExit("[ERROR] learned_mpc needs --checkpoint (train with scripts/train.py)")
        p = resolve_path(checkpoint)
        if not p.exists():
            raise SystemExit(f"[ERROR] checkpoint not found: {p}")
        model, _ = load_checkpoint(p)
        return MPCController(ModelPredictor(model, dev), cfg, safety, seed, name)
    raise SystemExit(f"[ERROR] unknown planner {name}")


def plot(results_by_planner: dict, out: Path, backend: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    first = next(iter(results_by_planner.values()))
    n = min(8, len(first))
    idx = np.linspace(0, len(first) - 1, n).astype(int)
    fig, axes = plt.subplots(2, (n + 1) // 2, figsize=(4.5 * ((n + 1) // 2), 7))
    for ax, i in zip(np.ravel(axes), idx):
        r0 = first[i]
        sc = r0["scene"]
        xmin, xmax, ymin, ymax = sc["bounds"]
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, ec="gray"))
        for ox, oy, rr in sc["obstacles"]:
            ax.add_patch(plt.Circle((ox, oy), rr, color="gray", alpha=0.6))
        ax.plot(*sc["goal"], "g*", ms=12)
        wp = np.array(r0["waypoints"])
        ax.plot(wp[:, 0], wp[:, 1], "k.", ms=3)
        for name, res in results_by_planner.items():
            tr = np.array(res[i]["trajectory"])
            ax.plot(tr[:, 0], tr[:, 1], label=f"{name}: {res[i]['outcome']}")
        ax.set_aspect("equal")
        ax.set_title(r0["scene_id"], fontsize=8)
        ax.legend(fontsize=6)
    fig.suptitle(f"Navigation trajectories [{backend.upper()} backend]")
    fig.tight_layout()
    fig.savefig(out / "trajectories.png", dpi=110)
    plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", choices=["mock", "isaac"], required=True)
    ap.add_argument("--planner", choices=PLANNERS + ["all"], required=True)
    ap.add_argument("--config", default="configs/planner.yaml")
    ap.add_argument("--sim-config", default=None, help="backend config (default mock.yaml / isaac_go2.yaml)")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--per-family", type=int, default=None)
    ap.add_argument("--first-index", type=int, default=None,
                    help="start scenario index (to extend an earlier run without repeating scenes)")
    ap.add_argument("--out", default="outputs/navigation")
    ap.add_argument("--final-test", action="store_true",
                    help="allow frozen final-test scenes (configs/eval_protocol.yaml); the run is logged to "
                         "outputs/final_test_ledger.jsonl. Never use final-test results to tune anything.")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    if args.per_family:
        cfg["navigation"]["scenarios"]["per_family"] = args.per_family
    if args.first_index is not None:
        cfg["navigation"]["scenarios"]["first_index"] = args.first_index
    sim_cfg = load_config(args.sim_config or ("mock.yaml" if args.backend == "mock" else "isaac_go2.yaml"))
    planners = PLANNERS if args.planner == "all" else [args.planner]
    if args.planner == "all" and not args.checkpoint:
        planners = ["conventional", "kinematic_mpc"]
        print("[nav] no --checkpoint: skipping learned_mpc")
    scen = scenarios_from_cfg(cfg, args.backend)
    ft = final_test_scenes(scen, args.backend)
    if ft and not args.final_test:
        raise SystemExit(f"[ERROR] {len(ft)} scenes are in the frozen final-test range ({ft[0]}...). "
                         "Use development scenes for tuning, or pass --final-test for the confirmatory run.")
    if args.final_test and not ft:
        raise SystemExit("[ERROR] --final-test given but no scene is in the final-test range")
    out = resolve_path(args.out) / args.backend
    out.mkdir(parents=True, exist_ok=True)
    if ft:
        led = resolve_path("outputs/final_test_ledger.jsonl")
        led.parent.mkdir(parents=True, exist_ok=True)
        with open(led, "a", encoding="utf-8") as f:
            f.write(json.dumps(ledger_entry(args, cfg, scen, planners)) + "\n")
        print(f"[nav] FINAL TEST run logged to {led}", flush=True)
    adapter = make_adapter(args.backend, sim_cfg)
    results, summary = {}, {}
    try:
        for name in planners:
            ctrl = make_controller(name, cfg, args.checkpoint)
            res = []
            for seed, sid in scen:
                res.append(run_episode(adapter, ctrl, seed, sid, cfg))
                print(f"[nav] {name} {res[-1]['scene_id']}: {res[-1]['outcome']} ({res[-1]['time_s']:.1f}s)", flush=True)
                write_json(out / f"{name}_episodes.json", res)  # after every episode: survives a killed run
            results[name] = res
            summary[name] = aggregate(res)
        report(summary, results, planners, scen, args, cfg, out)  # before close: Isaac's close() exits
    except BaseException:
        import traceback

        traceback.print_exc()
        sys.stderr.flush()
        return 1
    finally:
        adapter.close()
    return 0


def report(summary, results, planners, scen, args, cfg, out) -> None:
    summary["_meta"] = {"backend": args.backend, "scenarios": scen, "checkpoint": args.checkpoint, "config": cfg}
    if args.backend == "mock":
        summary["_meta"]["WARNING"] = "mock backend: software smoke test, not Go2 navigation evidence"
    write_json(out / "summary.json", summary)
    plot(results, out, args.backend)

    print(f"[nav] backend={args.backend} scenarios={len(scen)} (identical for all planners)")
    print(f"{'planner':16s}{'success':>16s}{'collide':>9s}{'t_goal':>8s}{'eff':>6s}{'minclr':>8s}{'interv/ep':>10s}{'p50ms':>7s}{'p95ms':>7s}")
    for name in planners:
        s = summary[name]
        ci = s["success_ci95"]
        fmt = lambda v, f: (f % v) if v is not None else "-"  # noqa: E731
        print(f"{name:16s}{s['success_rate']:6.2f} [{ci[0]:.2f},{ci[1]:.2f}]{s['collision_rate']:9.2f}"
              f"{fmt(s['time_to_goal_s_mean_successes'], '%8.1f')}{fmt(s['route_efficiency_mean'], '%6.2f')}"
              f"{s['min_clearance_m_mean']:8.2f}{s['safety_interventions_per_episode']:10.1f}"
              f"{s['latency_ms_p50']:7.1f}{s['latency_ms_p95']:7.1f}")
    print(f"[nav] wrote {out / 'summary.json'} and trajectories.png", flush=True)


if __name__ == "__main__":
    sys.exit(main())
