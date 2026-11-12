"""Phase 1 step 2: Isaac smoke test on defect tiles (rough-terrain policy + A* + kinematic MPC).

  scripts\\isaac_python.bat scripts\\isaac_defect_smoke.py --tiles outputs/defect_tiles/tiles.json --use 3 4 5

Builds the chosen defect tiles side by side (2 cm / 1 mm height field, baked texture), spawns the mission scene
(start x = -L/2 + 1, goal x = L/2 - 1.5, obstructions as cylinders standing on the surface), and drives each tile
once with the existing navigation stack: global A* on the scene map + CEM-MPC with command kinematics + safety
filter. Holes, puddles, cracks and upheavals are deliberately NOT obstacles: the point is to see what the walking
policy does on them. Records full episodes (D555 + L1 grids with puddle dropout, front RGB, terrain truth) to
--data and writes a report: outcome per tile, and for every defect whether the robot walked over it, its speed and
height-above-ground there, and foot slip.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys

import numpy as np

from go2wm.cli import write_json
from go2wm.common.config import load_config, resolve_path
from go2wm.common.schema import VelocityCommand


class PlannerCommands:
    """Command function for collect_episodes: A* waypoints + any local controller (act(obs, waypoint))."""

    def __init__(self, nav_cfg: dict, controller, robot_radius: float, dt: float):
        self.nc, self.ctrl, self.rr, self.dt = nav_cfg, controller, robot_radius, dt
        self.tracker = None

    def __call__(self, obs, scene) -> VelocityCommand:
        from go2wm.planning.global_waypoints import GlobalPlanner, WaypointTracker

        if self.tracker is None:
            gp = GlobalPlanner(self.nc["grid_resolution"], self.rr + self.nc["inflation_margin"], self.nc["waypoint_spacing"])
            self.tracker = WaypointTracker(gp.plan(scene, obs.pos_w[:2], scene.goal), self.nc["waypoint_tolerance"])
            self.ctrl.reset(scene, obs)
        a, _ = self.ctrl.act(obs, self.tracker.update(obs.pos_w[:2]))
        return VelocityCommand.from_action(a, self.dt)


def defect_report(ep, defects) -> list[dict]:
    from go2wm.data.dataset import state_pose2d
    from go2wm.sim.defects import defect_box, dist_to_polyline, ellipse_r

    pose = state_pose2d(ep["states"])[:-1]
    speed = np.linalg.norm(ep["states"][:-1, 7:9], axis=1)
    hag = ep["gt_height_above_ground_m"][:-1] if "gt_height_above_ground_m" in ep.arrays else np.full(len(pose), np.nan)
    slip = ep["gt_foot_slip_m_s"][:-1].max(1) if "gt_foot_slip_m_s" in ep.arrays else np.full(len(pose), np.nan)
    out = []
    for k, d in enumerate(defects):
        x, y = pose[:, 0], pose[:, 1]
        t = d["type"]
        if t in ("hole", "puddle"):
            on = ellipse_r(x, y, d) < 1.0
        elif t == "crack":
            on = dist_to_polyline(x, y, d["points"]) < 0.25
        else:
            b = defect_box(d)
            on = (x >= b[0] - 0.25) & (x <= b[1] + 0.25) & (y >= b[2] - 0.25) & (y <= b[3] + 0.25)
        r = {"id": f"defect_{k}_{t}", "type": t, "base_over_it_steps": int(on.sum())}
        if on.any():
            r.update(speed_m_s=float(speed[on].mean()), min_height_above_ground_m=float(np.nanmin(hag[on])),
                     max_foot_slip_m_s=float(np.nanmax(slip[on])))
        if t == "obstruction":
            r["min_dist_to_centre_m"] = float(np.min(np.hypot(x - d["x"], y - d["y"])))
        out.append(r)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tiles", default="outputs/defect_tiles/tiles.json")
    ap.add_argument("--use", type=int, nargs="+", default=[3, 4, 5], help="tile indices from --tiles")
    ap.add_argument("--config", default="configs/isaac_go2_rough.yaml")
    ap.add_argument("--planner", default="configs/planner_isaac.yaml")
    ap.add_argument("--max-steps", type=int, default=600)
    ap.add_argument("--seed", type=int, default=90000)
    ap.add_argument("--data", default="data/go2_defect_smoke")
    ap.add_argument("--out", default="outputs/defect_smoke/report.json")
    ap.add_argument("--no-rgb", action="store_true", help="no camera / texture (headless app without rendering)")
    args = ap.parse_args(argv)
    all_tiles = json.loads(resolve_path(args.tiles).read_text())
    tiles = [all_tiles[i] for i in args.use]
    cfg = copy.deepcopy(load_config(args.config))
    cfg["isaac"].update(families=["sidewalk"], terrain_tiles=tiles, mission_scene=True, scene_obstacles=True,
                        sensor_rig=["depth_d555", "l1"], rgb_camera=not args.no_rgb,
                        terrain_texture=None if args.no_rgb else {"res": 0.02}, record_height_scan=False)
    pc = load_config(args.planner)

    from go2wm.data.episodes import load_episode
    from go2wm.data.recorder import collect_episodes
    from go2wm.models.state_dynamics import KinematicBaseline
    from go2wm.planning.local_mpc import ModelPredictor, MPCController
    from go2wm.planning.safety import SafetyFilter
    from go2wm.sim.isaac_go2 import IsaacGo2Adapter

    sim = IsaacGo2Adapter(cfg)
    root = resolve_path(args.data)
    report = {"tiles_file": args.tiles, "tiles_used": args.use, "planner": "A* + kinematic CEM-MPC", "episodes": []}
    try:
        for j, ti in enumerate(args.use):
            seed = args.seed + j  # seed % n_tiles = j -> tile j of the generator = all_tiles[ti]
            safety = SafetyFilter({**pc["safety"], "limits": pc["limits"], "robot_radius": pc["robot_radius"]})
            ctrl = MPCController(ModelPredictor(KinematicBaseline(1, "command"), pc["mpc"].get("device", "cpu")),
                                 pc, safety, seed, "kinematic_mpc")
            gen = PlannerCommands(pc["navigation"], ctrl, pc["robot_radius"], sim.control_dt)
            ids = collect_episodes(sim, root, num_episodes=1, base_seed=seed, max_steps=args.max_steps,
                                   limits=pc["limits"], goal_tolerance_m=pc["navigation"]["goal_tolerance"],
                                   scene_ids=[f"sidewalk_{seed:05d}"], command_fn_factory=lambda s, g=gen: g,
                                   extra_meta={"data_role": "defect_smoke", "defect_tile": ti, "tiles_file": args.tiles})
            ep = load_episode(root / "episodes" / f"{ids[0]}.npz")
            sc = ep.meta["scene"]
            xy = ep["states"][:, :2]
            rec = {"episode": ids[0], "tile": ti, "outcome": ep.meta["termination_reason"],
                   "steps": ep.num_transitions, "time_s": float(ep["dt_s"].sum()),
                   "progress_x_m": float(xy[-1, 0] - sc["start"][0]), "route_x_m": float(sc["goal"][0] - sc["start"][0]),
                   "n_obstacles": len(sc["obstacles"]), "obstacle_placement_max_err_m": sc["params"].get("obstacle_placement_max_err_m"),
                   "defects": defect_report(ep, tiles[j]["defects"])}
            report["episodes"].append(rec)
            print(f"[defect-smoke] tile {ti}: {rec['outcome']} after {rec['time_s']:.1f} s, "
                  f"progress {rec['progress_x_m']:.1f}/{rec['route_x_m']:.1f} m", flush=True)
            write_json(resolve_path(args.out), report)  # incremental: survives a killed run
    finally:
        sim.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
