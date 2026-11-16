"""Check the inspection posture (crouch + nose-down pitch) on the Isaac Go2: reach it, hold it, stand back up and
walk off without falling. Prints base height / pitch and where the D405's optical axis meets the ground.

  scripts\\isaac_python.bat scripts\\isaac_posture_check.py --drop 0.07 --pitch 0.2
"""

from __future__ import annotations

import argparse
import copy
import math
import sys

import numpy as np

from go2wm.common.config import load_config


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--drop", type=float, default=0.07)
    ap.add_argument("--pitch", type=float, default=0.2)
    ap.add_argument("--config", default="configs/isaac_go2_rough.yaml")
    ap.add_argument("--checkpoint", default=None)
    a = ap.parse_args(argv)
    cfg = copy.deepcopy(load_config(a.config))
    if a.checkpoint:
        cfg["isaac"]["policy_checkpoint"] = a.checkpoint
    cfg["isaac"].update(families=["sidewalk"], terrain_tiles=[{"size": [12.0, 7.0], "width": 3.0}], mission_scene=True,
                        sensor_rig=[], rgb_camera=False, terrain_texture=None, record_height_scan=False,
                        enable_cameras=False)
    from go2wm.common.schema import VelocityCommand
    from go2wm.sim.isaac_go2 import IsaacGo2Adapter
    from go2wm.sim.offline_sensors import quat_to_R
    from go2wm.sim.sensors import SENSOR_SPECS, pitch_matrix

    sim = IsaacGo2Adapter(cfg)
    spec = SENSOR_SPECS["depth_d405"]

    def report(tag, obs):
        R = quat_to_R(obs.quat_w)
        pitch = math.asin(-R[2, 0])
        roll = math.degrees(math.atan2(R[2, 1], R[2, 2]))
        cam = obs.pos_w + R @ np.asarray(spec["mount_pos"])
        ax = R @ pitch_matrix(spec["pitch_deg"]) @ np.array([1.0, 0.0, 0.0])
        g = 0.0  # flat tile
        s = (g - cam[2]) / ax[2]
        hit = cam + s * ax
        print(f"[posture] {tag:<14} base z {obs.pos_w[2]:.3f} m  pitch {math.degrees(pitch):5.1f} roll {roll:5.1f} deg  "
              f"D405 axis hits ground {hit[0] - obs.pos_w[0]:.3f} m ahead of base, range {s:.3f} m, "
              f"view angle {math.degrees(math.asin(-ax[2])):.0f} deg below horizontal", flush=True)

    try:
        obs = sim.reset(0)
        z = VelocityCommand.from_action(np.zeros(3), sim.control_dt)
        for _ in range(10):
            obs = sim.step(z).observation
        # leg model check: forward kinematics vs. the simulated foot positions (base frame)
        d = sim.robot.data
        names = list(sim.robot.joint_names)
        Rb = sim._quat_R(d.root_quat_w[0])
        bodies = list(sim.robot.body_names)
        errs = []
        for leg, (hx, hy, side) in sim.HIPS.items():
            q = [float(d.joint_pos[0, names.index(f"{leg}_{j}_joint")]) for j in ("hip", "thigh", "calf")]
            fk = np.array([hx, hy, 0.0]) + sim._leg_fk(*q, side)
            true = Rb.T @ (d.body_pos_w[0, bodies.index(f"{leg}_foot")].cpu().numpy() - d.root_pos_w[0].cpu().numpy())
            ik = sim._leg_ik3(fk - np.array([hx, hy, 0.0]), side)
            errs.append((leg, np.linalg.norm(fk - true), max(abs(x - y) for x, y in zip(ik, q))))
        print("[posture] FK vs sim foot [mm] / IK round-trip [rad]: " +
              ", ".join(f"{l} {e * 1000:.1f} / {k:.1e}" for l, e, k in errs), flush=True)
        # stop right after a walking turn (uneven stance, as at a real inspection stop)
        turnwalk = VelocityCommand.from_action(np.array([0.5, 0.0, 0.7]), sim.control_dt)
        for _ in range(20):
            obs = sim.step(turnwalk).observation
        for _ in range(3):
            obs = sim.step(z).observation
        report("standing", obs)
        sim.set_posture(a.drop, a.pitch)
        for i in range(30):
            r_ = sim.step(z)
            obs = r_.observation
            if r_.terminated:
                print(f"[posture] TERMINATED during posture: {r_.termination_reason}", flush=True)
                break
        report("inspect pose", obs)
        print(f"[posture] aborts: {getattr(sim, 'posture_aborts', 0)}", flush=True)
        print(f"[posture] settled: {sim.posture_state()}", flush=True)
        sim.set_posture(0.0, 0.0)
        for i in range(30):
            obs = sim.step(z).observation
        report("stood back up", obs)
        walk = VelocityCommand.from_action(np.array([0.5, 0.0, 0.0]), sim.control_dt)
        fell = False
        for i in range(30):
            r = sim.step(walk)
            obs = r.observation
            if r.terminated and not fell:
                print(f"[posture] walk-off terminated: {r.termination_reason} at y={r.observation.pos_w[1]:.2f}", flush=True)
            fell = fell or r.terminated
        report("walked 3 s", obs)
        print(f"[posture] fell/terminated during walk-off: {fell}", flush=True)
    except Exception:
        import traceback

        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
    finally:
        sim.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
