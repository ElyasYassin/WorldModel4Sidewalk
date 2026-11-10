"""Isaac Go2 smoke test: does the pretrained policy follow high-level velocity commands?

  scripts\\isaac_python.bat scripts\\isaac_smoke.py [--config configs/isaac_go2.yaml]

Runs an open scene, applies a fixed command schedule, and reports velocity
tracking error per segment + whether the robot fell. Writes a JSON report.
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from go2wm.cli import write_json
from go2wm.common.config import load_config, resolve_path
from go2wm.common.schema import VelocityCommand

SCHEDULE = [  # (vx, vy, yaw_rate, seconds)
    (0.0, 0.0, 0.0, 1.0),
    (0.5, 0.0, 0.0, 3.0),
    (1.0, 0.0, 0.0, 3.0),
    (0.5, 0.0, 0.5, 3.0),
    (0.0, 0.3, 0.0, 3.0),
    (0.0, 0.6, 0.0, 3.0),
    (0.0, -0.6, 0.0, 3.0),
    (0.0, 0.0, -0.8, 3.0),
    (0.0, 0.0, 0.5, 3.0),
    (0.0, 0.0, 1.0, 3.0),
    (0.3, 0.0, 0.8, 3.0),
    (0.0, 0.0, 0.0, 2.0),
]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/isaac_go2.yaml")
    ap.add_argument("--out", default="outputs/isaac_smoke/report.json")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    t_launch = time.perf_counter()
    from go2wm.sim.isaac_go2 import IsaacGo2Adapter

    sim = IsaacGo2Adapter(cfg)
    t_ready = time.perf_counter()
    report = {"metadata": sim.metadata(), "segments": [], "startup_s": t_ready - t_launch}
    try:
        obs = sim.reset(seed=0, scene_id="open_0000")
        start = obs.pose2d
        fell = False
        t_sim0 = time.perf_counter()
        steps = 0
        for vx, vy, wz, dur in SCHEDULE:
            n = int(round(dur / sim.control_dt))
            real = []
            for k in range(n):
                res = sim.step(VelocityCommand(vx, vy, wz, sim.control_dt))
                steps += 1
                o = res.observation
                if k >= n // 2:  # steady-state half of the segment
                    real.append([o.lin_vel_b[0], o.lin_vel_b[1], o.ang_vel_b[2]])
                if res.termination_reason == "fall":
                    fell = True
                    break
            real = np.array(real) if real else np.zeros((1, 3))
            seen = sim.cmd_term.command[0].detach().cpu().numpy().tolist()  # what the policy observed
            seg = {"command": [vx, vy, wz], "seconds": dur, "realized_mean": real.mean(0).tolist(),
                   "abs_err_mean": np.abs(real - np.array([vx, vy, wz])).mean(0).tolist(),
                   "command_term_at_end": seen}
            report["segments"].append(seg)
            print(f"[smoke] cmd vx={vx:+.2f} vy={vy:+.2f} wz={wz:+.2f} -> realized "
                  f"vx={seg['realized_mean'][0]:+.2f} vy={seg['realized_mean'][1]:+.2f} wz={seg['realized_mean'][2]:+.2f}"
                  f"   (command term seen by policy: {np.round(seen, 2).tolist()})")
            if fell:
                break
        wall = time.perf_counter() - t_sim0
        end = obs.pose2d if fell else res.observation.pose2d
        report.update({
            "fell": fell,
            "start_pose": start.tolist(),
            "end_pose": end.tolist(),
            "final_height_m": float(res.observation.pos_w[2]),
            "sim_seconds": steps * sim.control_dt,
            "wall_seconds": wall,
            "realtime_factor": steps * sim.control_dt / wall,
        })
        err = np.array([s["abs_err_mean"] for s in report["segments"][1:]])
        # loose acceptance: no fall and mean tracking error under 0.25 (m/s or rad/s)
        report["pass"] = (not fell) and bool(err.mean() < 0.25)
        # write BEFORE closing: Kit's app.close() terminates the process
        p = write_json(resolve_path(args.out), report)
        print(f"[smoke] fell={report.get('fell')} pass={report.get('pass')} startup={report['startup_s']:.0f}s "
              f"rtf={report.get('realtime_factor', 0):.2f} -> {p}", flush=True)
    finally:
        sim.close()
    return 0 if report.get("pass") else 1


if __name__ == "__main__":
    sys.exit(main())
