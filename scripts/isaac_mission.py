"""Phase 1: inspection missions with Isaac Go2 physics (rough-terrain walking policy), no Isaac cameras needed.

  scripts\\isaac_python.bat scripts\\isaac_mission.py --tiles outputs/defect_tiles_final2/tiles.json --tile 8 \
      --policies walkover approach --profile realistic --pose-errors --out outputs/isaac_missions

Same mission loop as run_mission.py (A* + MPC, D555 / D405 offline ray-cast rig, perception, memory, inspection
policy), but the robot is the simulated Go2 walking with NVIDIA's rough-terrain policy: its gait shakes the cameras
(pitch / roll / bounce of the real body pose), it overshoots and has to correct when approaching. With --pose-errors
the perception pipeline also gets odometry drift and camera-IMU latency (runner.PoseErrors). Every step records all
robot link poses so scripts/render_isaac_mission.py can draw the walk in 3D. Writes <out>/<policy>_tile<i>.json.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys

import numpy as np

from go2wm.common.config import load_config, resolve_path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tiles", default="outputs/defect_tiles_final2/tiles.json")
    ap.add_argument("--tile", type=int, default=8)
    ap.add_argument("--policies", nargs="+", default=["walkover", "approach"])
    ap.add_argument("--profile", default="realistic")
    ap.add_argument("--pose-errors", action="store_true", help="odometry drift + camera latency in perception")
    ap.add_argument("--config", default="configs/isaac_go2_rough.yaml")
    ap.add_argument("--planner", default="configs/planner_isaac.yaml")
    ap.add_argument("--reach", type=float, default=0.10, help="approach stand-off tolerance [m]")
    ap.add_argument("--max-steps", type=int, default=900)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="outputs/isaac_missions")
    a = ap.parse_args(argv)
    tile = json.loads(resolve_path(a.tiles).read_text())[a.tile]
    cfg = copy.deepcopy(load_config(a.config))
    cfg["isaac"].update(families=["sidewalk"], terrain_tiles=[tile], mission_scene=True, scene_obstacles=True,
                        sensor_rig=[], rgb_camera=False, terrain_texture=None, record_height_scan=False,
                        enable_cameras=False)
    pc = load_config(a.planner)
    out = resolve_path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    from go2wm.mission.runner import Policy, run_mission, score_mission
    from go2wm.models.state_dynamics import KinematicBaseline
    from go2wm.planning.local_mpc import ModelPredictor, MPCController
    from go2wm.planning.safety import SafetyFilter
    from go2wm.sim.isaac_go2 import IsaacGo2Adapter

    sim = IsaacGo2Adapter(cfg)
    names = list(sim.robot.body_names)
    try:
        for pol in a.policies:
            links: list = []

            def rec(step, t, obs, mem, policy, links=links):
                d = sim.robot.data
                o = sim._origin()
                pos = d.body_pos_w[0].detach().cpu().numpy() - o
                quat = d.body_quat_w[0].detach().cpu().numpy()
                links.append({"t": t, "pos": pos.round(4).tolist(), "quat": quat.round(4).tolist(),
                              "target": policy.target, "n_findings": len(mem.findings)})

            safety = SafetyFilter({**pc["safety"], "limits": pc["limits"], "robot_radius": pc["robot_radius"]})
            ctrl = MPCController(ModelPredictor(KinematicBaseline(1, "command"), pc["mpc"].get("device", "cpu")),
                                 pc, safety, a.seed, "kinematic_mpc")
            res = run_mission(sim, ctrl, tile, Policy(pol, reach=a.reach, align=True, budget_steps=150, avoid_holes=True), a.profile, seed=a.seed + a.tile, max_steps=a.max_steps,
                              nav_cfg=pc["navigation"], pose_errors=a.pose_errors, on_step=rec, log=lambda *x: None)
            res["score"] = score_mission(res, tile)
            res.update(tile=a.tile, tiles_file=a.tiles, backend="isaac_rough_policy", body_names=names, links=links)
            f = out / f"{pol}_tile{a.tile}.json"
            f.write_text(json.dumps(res, default=float))
            sc = res["score"]
            print(f"[isaac-mission] {pol} tile {a.tile}: {res['outcome']} in {res['time_s']:.1f} s; " + "; ".join(
                f"{x['type']} true {x['true_value_m'] * 1000:.1f} mm -> " +
                (f"{x['value_m'] * 1000:.1f} mm ({x.get('best_sensor')})" if x.get("value_m") is not None
                 else ("found" if x["found"] else "missed")) for x in sc["per_defect"]) +
                f"; false {sc['false_findings']}", flush=True)
    finally:
        sim.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
