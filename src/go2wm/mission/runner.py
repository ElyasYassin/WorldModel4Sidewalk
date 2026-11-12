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
from .perception import (classify_shape, detect_candidates, detect_dropouts, expected_hits, in_view, measure,
                         to_robot, to_world)
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


class PoseErrors:
    """Online pose errors for the perception pipeline (off by default; the frozen evaluations ran without them).

    odometry   the robot's (x, y, yaw) estimate drifts as a random walk scaled by distance moved (profile odom_xy,
               odom_yaw per metre), like legged odometry without loop closure
    latency    each depth frame is captured `latency_s` before the pose it is stamped with (camera-IMU sync error):
               the rays leave the earlier true pose, the robot places the points with the current estimate
    """

    def __init__(self, prof: dict[str, float], seed: int, latency_s: float | None = None):
        self.p, self.rng = prof, np.random.default_rng(seed + 31337)
        self.latency_s = float(prof.get("latency_s", 0.02) if latency_s is None else latency_s)
        self.err = np.zeros(3)
        self.last = None

    def estimate(self, pose: np.ndarray) -> np.ndarray:
        if self.last is not None:
            d = float(np.linalg.norm(pose[:2] - self.last[:2]))
            self.err[:2] += self.rng.normal(size=2) * self.p.get("odom_xy", 0.0) * d
            self.err[2] += self.rng.normal() * self.p.get("odom_yaw", 0.0) * d
        self.last = pose.copy()
        c, s = np.cos(self.err[2]), np.sin(self.err[2])  # yaw error rotates the path about the start
        x, y = pose[0] - self.origin[0], pose[1] - self.origin[1]
        return np.array([self.origin[0] + c * x - s * y + self.err[0], self.origin[1] + s * x + c * y + self.err[1],
                         pose[2] + self.err[2]])

    def capture(self, prev, obs, dt: float):
        """True sensor pose at capture time: `latency_s` before the current observation (linear interpolation)."""
        if prev is None or dt <= 0:
            return obs.pos_w, obs.quat_w
        a = min(self.latency_s / dt, 1.0)
        q = (1 - a) * obs.quat_w + a * (prev.quat_w if np.dot(prev.quat_w, obs.quat_w) >= 0 else -prev.quat_w)
        return (1 - a) * obs.pos_w + a * prev.pos_w, q / np.linalg.norm(q)


class Policy:
    def __init__(self, name: str, slow_speed: float = 0.3, dwell: int = 5, reach: float = 0.10,
                 align: bool = False, budget_steps: int = 80, avoid_holes: bool = False):
        """approach: `reach` = stand-off tolerance [m]; `align` = turn in place to face the finding before the
        dwell (a legged robot arrives at the stand-off with any heading; the kinematic mock arrives facing +x);
        `budget_steps` = give up after this many control steps."""
        assert name in ("walkover", "slow", "approach"), name
        self.name, self.slow_speed, self.dwell = name, slow_speed, dwell
        self.reach, self.align, self.budget_steps = reach, align, budget_steps
        self.avoid_holes = avoid_holes  # route around confirmed holes (any strategy): a legged robot can fall in
        self.tried: set[str] = set()
        self.budget = 0
        self.target: str | None = None
        self.dwell_left = 0


def run_mission(sim, controller, terrain: dict[str, Any], policy: Policy, profile: str = "realistic",
                seed: int = 0, max_steps: int = 900, nav_cfg: dict[str, Any] | None = None,
                detect_every: int = 1, log=print, pose_errors: bool = False, on_step=None) -> dict[str, Any]:
    from ..planning.global_waypoints import GlobalPlanner, WaypointTracker

    nc = {"grid_resolution": 0.1, "inflation_margin": 0.15, "waypoint_spacing": 1.0, "waypoint_tolerance": 0.5,
          "goal_tolerance": 0.4, **(nav_cfg or {})}
    p = params_with_defaults(terrain)
    half = 0.5 * p["width"]
    obs = sim.reset(seed)
    scene = sim.scene
    H = lambda x, y: sidewalk_height(x, y, p)  # noqa: E731
    rig = OfflineRig(["depth_d555", "depth_d405"], profile, seed=seed, defects=p.get("defects"))
    route_x = [w[0] for w in scene.params["route"]]
    mem = SpatialMemory((scene.bounds[0], scene.bounds[1], -half, half), route_x)
    gp = GlobalPlanner(nc["grid_resolution"], sim.robot_radius + nc["inflation_margin"], nc["waypoint_spacing"])
    hazards: tuple = ()

    def plan_scene():
        """The mission scene plus confirmed holes as keep-out circles (planner only; the simulator's own scene,
        which decides collisions, is unchanged)."""
        if not hazards:
            return scene
        import copy

        sc = copy.copy(scene)
        extra = np.array([[*mem.findings[h]["xy"], max(0.5 * (mem.findings[h].get("extent_m") or 0.3), 0.15) + 0.1]
                          for h in hazards if h in mem.findings]).reshape(-1, 3)
        sc.obstacles = np.vstack([np.asarray(scene.obstacles).reshape(-1, 3), extra])
        sc.scene_id = scene.scene_id + "+holes:" + ",".join(hazards)  # the planner caches its grid per scene id
        return sc

    tracker = WaypointTracker(gp.plan(scene, obs.pos_w[:2], scene.goal), nc["waypoint_tolerance"])
    controller.reset(scene, obs)
    events, traj, est, t0 = [], [], [], time.perf_counter()
    outcome, t_s = "timeout", 0.0
    perr = PoseErrors(rig.prof, seed) if pose_errors else None
    if perr is not None:
        perr.origin = obs.pose2d[:2].copy()
    prev_obs, prev_dt = None, sim.control_dt
    for step in range(max_steps):
        true_pose = obs.pose2d
        pose = perr.estimate(true_pose) if perr is not None else true_pose  # the robot's belief (default: truth)
        traj.append(true_pose.tolist())
        est.append(pose.tolist())
        if on_step is not None:
            on_step(step, t_s, obs, mem, policy)
        if np.linalg.norm(obs.pos_w[:2] - scene.goal) < nc["goal_tolerance"] and policy.target is None:
            outcome = "goal"
            break
        # --- sense
        R = quat_to_R(obs.quat_w)
        if perr is not None:  # rays leave the true pose at capture time; the robot uses its current belief
            cpos, cq = perr.capture(prev_obs, obs, prev_dt)
            pts = {s: rig.frame(s, cpos, quat_to_R(cq), H) for s in ("depth_d555", "depth_d405")}
        else:
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
            if fid not in mem.findings:  # merged into another finding by a measurement earlier in this loop
                continue
            if not mem.is_real(fid) or f["kind"] == "dropout":  # dropouts need a camera, not a depth measurement
                continue
            for s in ("depth_d405", "depth_d555"):
                if in_view(s, pose, np.asarray(f["xy"])):
                    m = measure(pts[s], pose, np.asarray(f["xy"]), f["kind"], s, t_s)
                    if m is not None:
                        mem.add(m)
                    break
        # --- shape check (step vs pit) from the D555 while a finding is 0.6-2.0 m ahead
        if step % detect_every == 0:
            for fid in list(mem.findings):
                f = mem.findings.get(fid)
                if f is None or f["kind"] not in ("edge", "depression") or not mem.is_real(fid):
                    continue
                xr, yr = to_robot(pose, *f["xy"])
                if 0.6 <= xr <= 2.0 and abs(yr) < 0.8:
                    ext = max([o.get("extent_m") or 0.0 for o in f["observations"]] + [0.0])
                    mem.vote_shape(fid, classify_shape(pts["depth_d555"], pose, f["xy"], ext))
        # --- route around confirmed holes (replan when the set changes, not while approaching one)
        if policy.avoid_holes and policy.target is None:
            hz = tuple(sorted(h for h, f in mem.findings.items() if f["kind"] == "depression" and mem.is_real(h)
                              and abs(f.get("value_m") or 0.0) > 0.03))
            if hz != hazards:
                hazards = hz
                try:
                    tracker = WaypointTracker(gp.plan(plan_scene(), obs.pos_w[:2], scene.goal), nc["waypoint_tolerance"])
                    events.append({"t": t_s, "event": "replan_around_holes", "holes": list(hz)})
                except Exception as e:  # no path around (e.g. hole spans the walk): keep the old route
                    events.append({"t": t_s, "event": "replan_failed", "error": str(e)[:80]})
        # --- inspection policy
        wp = tracker.update(obs.pos_w[:2])
        cap, turn = None, None  # turn: direct [vx, vy, yaw_rate] override
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
                        policy.target, policy.dwell_left, policy.budget = fid, policy.dwell, policy.budget_steps
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
                    tracker = WaypointTracker(gp.plan(plan_scene(), obs.pos_w[:2], scene.goal), nc["waypoint_tolerance"])
                elif (dist := float(np.linalg.norm(obs.pos_w[:2] - sxy))) > policy.reach:
                    wp = sxy
                    cap = 0.4
                    if policy.align and dist < 0.4:
                        # last metres: direct body-frame command with a floor (the walking policy ignores tiny
                        # commands, so the MPC's slow-down near the waypoint would stall short of it)
                        dw = sxy - obs.pos_w[:2]
                        c, s = np.cos(true_pose[2]), np.sin(true_pose[2])
                        db = np.array([c * dw[0] + s * dw[1], -s * dw[0] + c * dw[1]])
                        turn = np.r_[db / dist * float(np.clip(dist, 0.35, 0.4)), 0.0]  # above the dead zone
                elif policy.align and abs(yerr := float((np.arctan2(f["xy"][1] - true_pose[1], f["xy"][0] - true_pose[0])
                                                        - true_pose[2] + np.pi) % (2 * np.pi) - np.pi)) > 0.15:
                    turn = np.array([0.0, 0.0, np.sign(yerr) * float(np.clip(1.5 * abs(yerr), 0.6, 0.9))])  # face it
                else:
                    policy.dwell_left -= 1
                    cap = 0.0
                    if policy.dwell_left <= 0 or not mem.needs_close_look(policy.target):
                        events.append({"t": t_s, "event": "inspected", "finding": policy.target,
                                       "status": mem.findings[policy.target]["status"]})
                        policy.target = None
                        tracker = WaypointTracker(gp.plan(plan_scene(), obs.pos_w[:2], scene.goal), nc["waypoint_tolerance"])
        a, _ = controller.act(obs, wp)
        if turn is not None:  # direct command from the approach (align mode)
            a = np.asarray(turn, dtype=a.dtype)
        elif cap is not None:
            a = a.copy()
            a[0] = min(a[0], cap)
            if cap == 0.0:
                a[:] = 0.0
        res = sim.step(VelocityCommand.from_action(a, sim.control_dt))
        prev_dt = res.info.get("dt_s", sim.control_dt)
        t_s += prev_dt
        prev_obs, obs = obs, res.observation
        if res.terminated:
            outcome = res.termination_reason or "terminated"
            break
    traj = np.asarray(traj)
    return {"outcome": outcome, "time_s": t_s, "steps": len(traj), "policy": policy.name, "profile": profile,
            "path_m": float(np.linalg.norm(np.diff(traj[:, :2], axis=0), axis=1).sum()) if len(traj) > 1 else 0.0,
            "trajectory": traj.tolist(), "pose_estimate": est if perr is not None else None,
            "pose_errors": perr is not None, "events": events, "memory": mem.to_dict(), "wall_s": time.perf_counter() - t0}


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
                if t["type"] == "upheaval":  # magnitude: the sign depends on the side it was measured from
                    r["abs_err_m"] = abs(abs(f["value_m"]) - abs(t["true_value"]))
                    r["ada_ok"] = ada_level_class(f["value_m"]) == ada_level_class(t["true_value"])
                else:
                    r["abs_err_m"] = abs(f["value_m"] - t["true_value"])
        rows.append(r)
    # depressions on a crack wide/deep enough for the depth sensors are real (not scored as a required defect)
    from ..sim.defects import dist_to_polyline

    cracks = [d for d in p["defects"] if d["type"] == "crack"]
    on_crack = [fid for fid in findings if fid not in used and findings[fid]["kind"] == "depression"
                and any(float(dist_to_polyline(np.array(findings[fid]["xy"][0]), np.array(findings[fid]["xy"][1]),
                                               c["points"])) < 0.2 for c in cracks)]
    fp = [fid for fid in findings if fid not in used and fid not in on_crack]
    by = {}
    for typ in ("upheaval", "hole", "puddle"):
        rr = [r for r in rows if r["type"] == typ]
        if rr:
            errs = [r["abs_err_m"] for r in rr if "abs_err_m" in r]
            by[typ] = {"n": len(rr), "found": sum(r["found"] for r in rr),
                       "measured_d405": sum(r.get("best_sensor") == "depth_d405" for r in rr),
                       "median_abs_err_mm": float(np.median(errs) * 1000) if errs else None}
    return {"per_defect": rows, "by_type": by, "false_findings": len(fp), "false_ids": fp, "crack_findings": len(on_crack),
            "n_findings": len(findings),
            "hazards_in_graph": len(g["hazards"])}
