"""E4 traversal smoke test: can the locomotion policy get along a terrain scenario at all?

  scripts\\isaac_python.bat scripts\\isaac_terrain_smoke.py --config configs/isaac_go2_rough_e4s1.yaml

Before any world-model experiment on terrain, check the policy itself (spec: don't confuse
locomotion-controller limits with world-model failures). Drives along the sidewalk centreline with
a simple lane-keeping law at several forward speeds, records full episodes (with terrain ground
truth: height above ground, foot contacts, slip) to --data, and reports per speed: fell / reached
the end, time, and per terrain section (flat, slab, grade) the mean forward-speed tracking error
and foot slip. Run scripts/isaac_smoke.py with configs/isaac_go2_rough.yaml first for the dead-zone
check (E1 repeated for the rough policy).
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from go2wm.cli import write_json
from go2wm.common.config import load_config, resolve_path
from go2wm.common.schema import VelocityCommand


class LaneFollow:
    """Forward speed vx along +x, P-control on lateral offset and heading to stay at y = y_ref."""

    def __init__(self, vx: float, y_ref: float = 0.0, k_y: float = 0.8, k_yaw: float = 1.5, dt: float = 0.1):
        self.vx, self.y_ref, self.k_y, self.k_yaw, self.dt = vx, y_ref, k_y, k_yaw, dt

    def __call__(self, obs, scene) -> VelocityCommand:
        x, y, yaw = obs.pose2d
        y_ref = float(scene.goal[1]) if scene is not None else self.y_ref  # end on the goal's line
        yaw_ref = float(np.clip(-self.k_y * (y - y_ref), -0.5, 0.5))
        wz = float(np.clip(self.k_yaw * np.arctan2(np.sin(yaw_ref - yaw), np.cos(yaw_ref - yaw)), -1.0, 1.0))
        return VelocityCommand(self.vx, 0.0, wz, self.dt)


def sections(terrain: dict) -> list[tuple[str, float, float]]:
    out = [("start_flat", -1.0, 3.5)]
    for x0, L, h in terrain.get("slabs", []):
        out.append((f"slab_{h * 100:.0f}cm", x0 - 0.5, x0 + L + 0.5))
    for x0, x1, rise in terrain.get("grade", []):
        out.append((f"grade_{rise * 100:.0f}pct", x0, x1))
    out.append(("end", 10.5, 13.0))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/isaac_go2_rough_e4s1.yaml")
    ap.add_argument("--speeds", type=float, nargs="+", default=[0.4, 0.6, 0.8])
    ap.add_argument("--repeats", type=int, default=2, help="episodes per speed (different seeds)")
    ap.add_argument("--data", default="data/go2_e4_smoke")
    ap.add_argument("--out", default="outputs/e4_smoke/report.json")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    from go2wm.data.episodes import load_all
    from go2wm.data.recorder import collect_episodes
    from go2wm.sim.isaac_go2 import IsaacGo2Adapter

    sim = IsaacGo2Adapter(cfg)
    report = {"config": args.config, "metadata": sim.metadata(), "runs": []}
    try:
        root = resolve_path(args.data)
        ids = []
        for j, vx in enumerate(args.speeds):
            base = 50000 + 100 * j
            ids += collect_episodes(
                sim, root, num_episodes=args.repeats, base_seed=base, max_steps=400,
                limits={"vx": [-1, 1], "vy": [-1, 1], "yaw_rate": [-1, 1]}, goal_tolerance_m=0.5,
                scene_ids=[f"sidewalk_{base + i:05d}" for i in range(args.repeats)],
                command_fn_factory=lambda s, vx=vx: LaneFollow(vx, dt=sim.control_dt),
                extra_meta={"data_role": "e4_smoke", "lane_follow_vx": vx})
        secs = sections(sim.terrain or {})
        for ep in load_all(root, ids):
            a = ep.arrays
            x = a["states"][:, 0]
            v = a["states"][1:, 7]  # realized body-frame vx
            vx_cmd = ep.meta["lane_follow_vx"]
            run = {"episode": ep.episode_id, "vx_cmd": vx_cmd, "end": ep.meta["termination_reason"],
                   "x_reached_m": float(x.max()), "time_s": ep.meta["duration_s"], "sections": {}}
            for name, x0, x1 in secs:
                m = (x[1:] >= x0) & (x[1:] < x1)
                if not m.any():
                    continue
                s = {"n_steps": int(m.sum()), "vx_err_mean": float(np.mean(v[m] - vx_cmd))}
                if "gt_foot_slip_m_s" in a:
                    s["slip_mean_m_s"] = float(np.nanmean(a["gt_foot_slip_m_s"][1:][m]))
                    s["height_above_ground_min_m"] = float(np.nanmin(a["gt_height_above_ground_m"][1:][m]))
                run["sections"][name] = s
            report["runs"].append(run)
            print(f"[e4] vx={vx_cmd:.1f} {ep.episode_id}: end={run['end']} x_max={run['x_reached_m']:.1f} m "
                  f"t={run['time_s']:.1f}s " + " ".join(f"{k}:verr={s['vx_err_mean']:+.2f}" for k, s in run["sections"].items()),
                  flush=True)
        report["falls"] = sum(r["end"] == "fall" for r in report["runs"])
        report["reached_end"] = sum(r["end"] == "goal" for r in report["runs"])
        report["pass"] = report["falls"] == 0 and report["reached_end"] == len(report["runs"])
        p = write_json(resolve_path(args.out), report)  # before close(): Kit's close ends the process
        print(f"[e4] falls={report['falls']} reached_end={report['reached_end']}/{len(report['runs'])} "
              f"pass={report['pass']} -> {p}", flush=True)
    finally:
        sim.close()
    return 0 if report.get("pass") else 1


if __name__ == "__main__":
    sys.exit(main())
