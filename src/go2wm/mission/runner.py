"""Phase 1 mission runner: route following + live sensing + detection + memory + inspection policy.

One mission on a defect tile (mission/scene.py): the robot follows the route to the goal with the existing stack
(A* waypoints + any controller with act(obs, waypoint)), while every step the D555 and D405 are simulated
(sim/offline_sensors.py, chosen error profile), the D555 grid is searched for candidates, and candidates that fall
in a sensor's measuring zone are measured from raw points. The inspection policy decides what to do about
candidates that still need a close look:

  walkover   nothing extra: measure whatever passes through the sensors' zones
  slow       cap forward speed at `slow_speed` while such a candidate is 0.3-2 m ahead (more D405 frames on it)
  approach   go to a stand-off pose that puts the candidate in the D405 zone, stop for `dwell` steps, measure,
             then resume the route (one attempt per finding)

The mock backend (MockMissionSim) moves the robot kinematically over the tile at the right height; the sensors
ray-cast the tile's height field. Scoring against the ground-truth scene graph is in score_mission().
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from ..common.schema import VelocityCommand
from ..sim.mock import FAMILY_PARAMS, MockGo2Sim, _stable_seed
from ..sim.offline_sensors import OfflineRig, quat_to_R
from ..sim.terrain import params_with_defaults, sidewalk_height
from .memory import SpatialMemory
from .perception import detect_candidates, detect_dropouts, expected_hits, in_view, measure, to_robot, to_world
from .scene import sidewalk_mission_scene

D405_STANDOFF = 0.52  # m ahead of the base: centre of the D405 measuring zone


class MockMissionSim(MockGo2Sim):
    """Mock kinematics on a defect tile: mission scene, base height follows the surface (for the sensors)."""

    def __init__(self, cfg: dict[str, Any] | None, terrain: dict[str, Any]):
        super().__init__(cfg)
        self.terrain = params_with_defaults(terrain)

    def reset(self, seed: int, scene_id: str | None = None):
        self.scene = sidewalk_mission_scene(self.terrain, scene_id or f"mission_{seed:05d}")
        self.scene.params.update(FAMILY_PARAMS["sidewalk"])
        self._rng = np.random.default_rng(_stable_seed("episode", seed, self.scene.scene_id))
        self._pose = self.scene.start.astype(np.float64).copy()
        self._v, self._accel = np.zeros(3), np.zeros(2)
        self._t_ns, self._steps = 0, 0
        return self._observe()

    def _observe(self):
        o = super()._observe()
        o.pos_w[2] = float(sidewalk_height(o.pos_w[0], o.pos_w[1], self.terrain)) + self.cfg["base_height"]
        return o


class Policy:
    def __init__(self, name: str, slow_speed: float = 0.3, dwell: int = 5):
        assert name in ("walkover", "slow", "approach"), name
        self.name, self.slow_speed, self.dwell = name, slow_speed, dwell
        self.tried: set[str] = set()
        self.budget = 0
        self.target: str | None = None
        self.dwell_left = 0


def run_mission(sim, controller, terrain: dict[str, Any], policy: Policy, profile: str = "realistic",
                seed: int = 0, max_steps: int = 900, nav_cfg: dict[str, Any] | None = None,
                detect_every: int = 1, log=print) -> dict[str, Any]:
    from ..planning.global_waypoints import GlobalPlanner, WaypointTracker

    nc = {"grid_resolution": 0.1, "inflation_margin": 0.15, "waypoint_spacing": 1.0, "waypoint_tolerance": 0.5,
          "goal_tolerance": 0.4, **(nav_cfg or {})}
    p = params_with_defaults(terrain)
    half = 0.5 * p["width"]
    obs = sim.reset(seed)
    scene = sim.scene
    H = lambda x, y: sidewalk_height(x, y, p)  # noqa: E731
    rig = OfflineRig(["depth_d555", "depth_d405"], profile, seed=seed)
    route_x = [w[0] for w in scene.params["route"]]
    mem = SpatialMemory((scene.bounds[0], scene.bounds[1], -half, half), route_x)
    gp = GlobalPlanner(nc["grid_resolution"], sim.robot_radius + nc["inflation_margin"], nc["waypoint_spacing"])
    tracker = WaypointTracker(gp.plan(scene, obs.pos_w[:2], scene.goal), nc["waypoint_tolerance"])
    controller.reset(scene, obs)
    events, traj, t0 = [], [], time.perf_counter()
    outcome, t_s = "timeout", 0.0
    for step in range(max_steps):
        pose = obs.pose2d  # mock: the robot's estimate = truth (odometry error is in the sensor study)
        traj.append(pose.tolist())
        if np.linalg.norm(obs.pos_w[:2] - scene.goal) < nc["goal_tolerance"] and policy.target is None:
            outcome = "goal"
            break
        # --- sense
        R = quat_to_R(obs.quat_w)
        pts = {s: rig.frame(s, obs.pos_w, R, H) for s in ("depth_d555", "depth_d405")}
        for s, q in pts.items():
            ok = np.all(np.isfinite(q), 1)
            if ok.any():
                w = np.stack(to_world(pose, q[ok, 0], q[ok, 1]), 1)
                mem.mark_seen(s, w)
        # --- detect (D555 grid) and measure candidates in a sensor's zone
        if step % detect_every == 0:
            from ..sim.offline_sensors import bin_base_points

            e, c = bin_base_points(pts["depth_d555"], R)
            for ob in detect_candidates(e, c, pose, t_s, half):
                mem.add(ob)
            ground = float(obs.pos_w[2] - sidewalk_height(obs.pos_w[0], obs.pos_w[1], p))  # height above ground (legs)
            for ob in detect_dropouts(c, expected_hits("depth_d555", R, ground), pose, t_s, half):
                mem.add(ob)
        for fid, f in list(mem.findings.items()):
            if not mem.is_real(fid) or f["kind"] == "dropout":  # dropouts need a camera, not a depth measurement
                continue
            for s in ("depth_d405", "depth_d555"):
                if in_view(s, pose, np.asarray(f["xy"])):
                    m = measure(pts[s], pose, np.asarray(f["xy"]), f["kind"], s, t_s)
                    if m is not None:
                        mem.add(m)
                    break
        # --- inspection policy
        wp = tracker.update(obs.pos_w[:2])
        cap = None
        if policy.name == "slow":
            for fid, f in mem.findings.items():
                xr, yr = to_robot(pose, *f["xy"])
                if 0.3 <= xr <= 2.0 and abs(yr) < 0.6 and mem.needs_close_look(fid):
                    cap = policy.slow_speed
        elif policy.name == "approach":
            if policy.target is None:
                for fid, f in mem.findings.items():
                    ahead = f["xy"][0] - pose[0]  # along the route (+x), not along the current heading
                    if (fid not in policy.tried and f["kind"] != "dropout" and 0.9 <= ahead <= 3.0
                            and mem.needs_close_look(fid)):
                        policy.target, policy.dwell_left, policy.budget = fid, policy.dwell, 80
                        policy.tried.add(fid)
                        events.append({"t": t_s, "event": "approach", "finding": fid})
                        break
            policy.target = mem.resolve(policy.target)  # the target may have been merged into an older finding
            if policy.target is not None and policy.target not in mem.findings:
                policy.target = None
            if policy.target is not None:
                f = mem.findings[policy.target]
                sxy = np.array([f["xy"][0] - D405_STANDOFF, np.clip(f["xy"][1], -half + 0.35, half - 0.35)])
                policy.budget -= 1
                if policy.budget <= 0:  # give up after 8 s (unreachable stand-off, e.g. behind an obstacle)
                    events.append({"t": t_s, "event": "approach_aborted", "finding": policy.target})
                    policy.target = None
                    tracker = WaypointTracker(gp.plan(scene, obs.pos_w[:2], scene.goal), nc["waypoint_tolerance"])
                elif np.linalg.norm(obs.pos_w[:2] - sxy) > 0.10:
                    wp = sxy
                    cap = 0.4
                else:
                    policy.dwell_left -= 1
                    cap = 0.0
                    if policy.dwell_left <= 0 or not mem.needs_close_look(policy.target):
                        events.append({"t": t_s, "event": "inspected", "finding": policy.target,
                                       "status": mem.findings[policy.target]["status"]})
                        policy.target = None
                        tracker = WaypointTracker(gp.plan(scene, obs.pos_w[:2], scene.goal), nc["waypoint_tolerance"])
        a, _ = controller.act(obs, wp)
        if cap is not None:
            a = a.copy()
            a[0] = min(a[0], cap)
            if cap == 0.0:
                a[:] = 0.0
        res = sim.step(VelocityCommand.from_action(a, sim.control_dt))
        t_s += res.info.get("dt_s", sim.control_dt)
        obs = res.observation
        if res.terminated:
            outcome = res.termination_reason or "terminated"
            break
    traj = np.asarray(traj)
    return {"outcome": outcome, "time_s": t_s, "steps": len(traj), "policy": policy.name, "profile": profile,
            "path_m": float(np.linalg.norm(np.diff(traj[:, :2], axis=0), axis=1).sum()) if len(traj) > 1 else 0.0,
            "trajectory": traj.tolist(), "events": events, "memory": mem.to_dict(), "wall_s": time.perf_counter() - t0}


def score_mission(result: dict[str, Any], terrain: dict[str, Any]) -> dict[str, Any]:
    """Match findings to the ground-truth scene graph's geometric defects (upheaval, hole, puddle)."""
    from ..scene.graph import ada_level_class, build_scene_graph

    p = params_with_defaults(terrain)
    g = build_scene_graph(p, x_range=(-p["size"][0] / 2, p["size"][0] / 2))
    findings = result["memory"]["findings"]
    gt = []
    for k, d in enumerate(p["defects"]):
        if d["type"] == "upheaval":
            xe = d["x1"] if d.get("high_end", "x1") == "x1" else d["x0"]
            gt.append({"id": f"defect_{k}_upheaval", "type": "upheaval", "kind": "edge", "xy": [xe, 0.5 * (d["y0"] + d["y1"])],
                       "seg": (xe, d["y0"], d["y1"]), "true_value": -d["lift"]})  # walking +x: step down at x1
            if d.get("high_end", "x1") == "x0":
                gt[-1]["true_value"] = d["lift"]  # step up at x0
        elif d["type"] in ("hole", "puddle"):
            gt.append({"id": f"defect_{k}_{d['type']}", "type": d["type"], "kind": "depression", "xy": [d["x"], d["y"]],
                       "radius": max(d["rx"], d["ry"]), "true_value": d["depth"]})
    used, rows = set(), []
    for t in gt:
        best, bd = None, np.inf
        for fid, f in findings.items():
            kinds = (t["kind"], "dropout") if t["type"] == "puddle" else (t["kind"],)
            if f["kind"] not in kinds or fid in used:
                continue
            if t["kind"] == "edge":
                xe, ya, yb = t["seg"]
                dist = abs(f["xy"][0] - xe) + max(0.0, ya - f["xy"][1], f["xy"][1] - yb)
                ok = dist < 0.3
            else:
                dist = float(np.hypot(f["xy"][0] - t["xy"][0], f["xy"][1] - t["xy"][1]))
                ok = dist < max(0.35, t["radius"])
            if ok and dist < bd:
                best, bd = fid, dist
        r = {"gt": t["id"], "type": t["type"], "true_value_m": t["true_value"], "found": best is not None}
        if best is not None:
            used.add(best)
            f = findings[best]
            r.update(finding=best, kind_found=f["kind"], status=f["status"], best_sensor=f.get("best_sensor"),
                     value_m=f.get("value_m"))
            if f.get("value_m") is not None:
                if t["type"] == "upheaval":
                    r["abs_err_m"] = abs(f["value_m"] - t["true_value"])
                    r["ada_ok"] = ada_level_class(f["value_m"]) == ada_level_class(t["true_value"])
                else:
                    r["abs_err_m"] = abs(f["value_m"] - t["true_value"])
        rows.append(r)
    fp = [fid for fid in findings if fid not in used]
    by = {}
    for typ in ("upheaval", "hole", "puddle"):
        rr = [r for r in rows if r["type"] == typ]
        if rr:
            errs = [r["abs_err_m"] for r in rr if "abs_err_m" in r]
            by[typ] = {"n": len(rr), "found": sum(r["found"] for r in rr),
                       "measured_d405": sum(r.get("best_sensor") == "depth_d405" for r in rr),
                       "median_abs_err_mm": float(np.median(errs) * 1000) if errs else None}
    return {"per_defect": rows, "by_type": by, "false_findings": len(fp), "n_findings": len(findings),
            "hazards_in_graph": len(g["hazards"])}
