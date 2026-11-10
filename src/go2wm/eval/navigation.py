"""Waypoint navigation benchmark: identical seeded scenarios across local controllers."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..common.schema import VelocityCommand
from ..planning.global_waypoints import GlobalPlanner, WaypointTracker
from ..sim.base import SimulatorAdapter


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z**2 / n
    c = (p + z**2 / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / den
    return (float(c - h), float(c + h))


def clear0(scene, obs, cfg) -> float:
    return float(scene.clearance(obs.pos_w[:2], cfg["robot_radius"]))


def run_episode(adapter: SimulatorAdapter, controller, seed: int, scene_id: str | None, cfg: dict[str, Any]) -> dict[str, Any]:
    nc = cfg["navigation"]
    obs = adapter.reset(seed=seed, scene_id=scene_id)
    scene = adapter.scene
    if scene is None:
        raise RuntimeError("navigation needs a scene with a known map/goal")
    gp = GlobalPlanner(nc["grid_resolution"], cfg["robot_radius"] + nc["inflation_margin"], nc["waypoint_spacing"])
    try:
        wps = gp.plan(scene, obs.pos_w[:2], scene.goal)
    except RuntimeError:
        return {"seed": seed, "scene_id": scene.scene_id, "family": scene.family, "outcome": "no_global_path",
                "success": False, "collision": False, "fall": False, "time_s": 0.0, "path_length_m": 0.0,
                "route_length_m": None, "route_efficiency": None, "min_clearance_m": float(clear0(scene, obs, cfg)),
                "safety_interventions": 0, "stall_events": 0, "reroutes": 0, "latency_s": [],
                "trajectory": [obs.pose2d.tolist()], "waypoints": [scene.goal.tolist()], "scene": scene.to_dict()}
    route_len = float(np.linalg.norm(np.diff(np.vstack([obs.pos_w[:2], wps]), axis=0), axis=1).sum())
    tracker = WaypointTracker(wps, nc["waypoint_tolerance"])
    controller.reset(scene, obs)
    traj = [obs.pose2d]
    lat, safety_events, clear = [], 0, [scene.clearance(obs.pos_w[:2], cfg["robot_radius"])]
    stall, stall_events, reroutes = 0, 0, 0
    outcome = "timeout"
    t_s = 0.0
    for step in range(nc["max_steps"]):
        xy = obs.pos_w[:2]
        if np.linalg.norm(xy - scene.goal) < nc["goal_tolerance"]:
            outcome = "success"
            break
        wp = tracker.update(xy)
        a, info = controller.act(obs, wp)
        lat.append(info["latency_s"])
        safety_events += bool(info.get("safety"))
        res = adapter.step(VelocityCommand.from_action(a, cfg["control_dt"]))
        t_s += res.info.get("dt_s", cfg["control_dt"])
        obs = res.observation
        traj.append(obs.pose2d)
        clear.append(scene.clearance(obs.pos_w[:2], cfg["robot_radius"]))
        speed = float(np.linalg.norm(obs.lin_vel_b[:2]))
        stall = stall + 1 if speed < nc["stall_speed"] else 0
        if stall == nc["stall_steps"]:
            stall_events += 1
            if reroutes < nc["max_reroutes"]:  # replan from where we are
                tracker = WaypointTracker(gp.plan(scene, obs.pos_w[:2], scene.goal), nc["waypoint_tolerance"])
                reroutes += 1
        if res.terminated:
            outcome = res.termination_reason or "terminated"
            break
    traj = np.array(traj)
    path_len = float(np.linalg.norm(np.diff(traj[:, :2], axis=0), axis=1).sum())
    return {
        "seed": seed,
        "scene_id": scene.scene_id,
        "family": scene.family,
        "outcome": outcome,
        "success": outcome == "success",
        "collision": outcome == "collision",
        "fall": outcome == "fall",
        "time_s": t_s,
        "path_length_m": path_len,
        "route_length_m": route_len,
        "route_efficiency": route_len / path_len if outcome == "success" and path_len > 0 else None,
        "min_clearance_m": float(np.min(clear)),
        "safety_interventions": safety_events,
        "stall_events": stall_events,
        "reroutes": reroutes,
        "latency_s": lat,
        "trajectory": traj.tolist(),
        "waypoints": wps.tolist(),
        "scene": scene.to_dict(),
    }


def aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(results)
    succ = [r for r in results if r["success"]]
    lat = np.concatenate([r["latency_s"] for r in results]) if results else np.array([])
    k_s = len(succ)
    k_c = sum(r["collision"] for r in results)

    def mean(xs):
        xs = [x for x in xs if x is not None]
        return float(np.mean(xs)) if xs else None

    by_family = {}
    for f in sorted({r["family"] for r in results}):
        rf = [r for r in results if r["family"] == f]
        by_family[f] = {"n": len(rf), "success_rate": sum(r["success"] for r in rf) / len(rf),
                        "collision_rate": sum(r["collision"] for r in rf) / len(rf)}
    return {
        "n_episodes": n,
        "success_rate": k_s / n if n else None,
        "success_ci95": wilson_ci(k_s, n),
        "collision_rate": k_c / n if n else None,
        "collision_ci95": wilson_ci(k_c, n),
        "falls": sum(r["fall"] for r in results),
        "no_global_path": sum(r["outcome"] == "no_global_path" for r in results),
        "timeouts": sum(r["outcome"] == "timeout" for r in results),
        "time_to_goal_s_mean_successes": mean([r["time_s"] for r in succ]),
        "path_length_m_mean_successes": mean([r["path_length_m"] for r in succ]),
        "route_efficiency_mean": mean([r["route_efficiency"] for r in succ]),
        "min_clearance_m_mean": mean([r["min_clearance_m"] for r in results]),
        "min_clearance_m_worst": float(min(r["min_clearance_m"] for r in results)) if results else None,
        "safety_interventions_per_episode": mean([r["safety_interventions"] for r in results]),
        "stall_events_per_episode": mean([r["stall_events"] for r in results]),
        "latency_ms_p50": float(np.percentile(lat, 50) * 1e3) if len(lat) else None,
        "latency_ms_p95": float(np.percentile(lat, 95) * 1e3) if len(lat) else None,
        "by_family": by_family,
    }


def scenarios_from_cfg(cfg: dict[str, Any], backend: str) -> list[tuple[int, str | None]]:
    ec = cfg["navigation"]["scenarios"]
    if ec.get("explicit"):
        return [(int(s["seed"]), s.get("scene_id")) for s in ec["explicit"]]
    # explicit scene ids for every backend (adapters would otherwise pick the family from the seed)
    out = []
    for fam in ec["families"]:
        first = int(ec.get("first_index", 0))
        for i in range(first, first + ec["per_family"]):
            seed = ec["base_seed"] + i
            out.append((seed, f"{fam}_{seed:04d}"))
    return out
