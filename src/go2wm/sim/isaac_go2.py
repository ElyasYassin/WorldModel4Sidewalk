"""Isaac Lab Go2 adapter: high-level body-frame velocity commands -> pretrained RSL-RL locomotion policy.

Must run in a process launched via scripts/isaac_python.bat (Isaac Sim env vars).
The adapter launches the Omniverse app itself, so construct at most one per process.

Design:
  * Task: Isaac Lab's built-in Go2 velocity task (single env). The velocity
    command term is switched to direct mode (no heading control, no standing
    envs, no resampling); we write `vel_command_b` each control period.
  * Isaac's automatic terminations (base contact, timeout) are disabled so the
    env never auto-resets mid-transition; falls/collisions are detected here
    from the robot state, keeping every recorded terminal observation valid.
  * Scene obstacles are kinematic cylinders spawned once in `obstacle_slots`
    and repositioned per scene; unused slots are parked far away.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ..common.schema import Observation, StepResult, VelocityCommand
from .base import SceneSpec
from .mock import make_scene

PARK_XY = (1000.0, 1000.0)


def torch_full_like(t, v):
    out = t.clone()
    out[:] = v
    return out


class IsaacGo2Adapter:
    backend = "isaac"

    def __init__(self, cfg: dict[str, Any]):
        self.cfg = c = cfg["isaac"]
        self.control_dt = float(c["control_dt"])
        self.robot_radius = float(c["robot_radius"])
        self.LEG_L2 = float(c.get("leg_l2", self.LEG_L2))  # calf: knee to foot centre / wheel axle (Go2-W 0.2264)
        self.POSTURE_MIN_FOOT_Y = float(c.get("posture_min_foot_y", self.POSTURE_MIN_FOOT_Y))
        self.scene: SceneSpec | None = None

        from isaaclab.app import AppLauncher

        self._app_launcher = AppLauncher(headless=c.get("headless", True), enable_cameras=c.get("enable_cameras", False) or bool(c.get("rgb_camera")), device=c["device"])
        self.app = self._app_launcher.app

        import gymnasium as gym
        import torch
        import isaaclab.sim as sim_utils
        import isaaclab_tasks  # noqa: F401  (registers tasks)
        if str(c["task"]).startswith("Go2WM-"):  # Go2-W (wheeled legs), our own task (sim/go2w_task.py)
            from . import go2w_task  # noqa: F401
        from isaaclab.assets import RigidObjectCfg
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg
        from rsl_rl.runners import OnPolicyRunner

        self.torch = torch
        env_cfg = parse_env_cfg(c["task"], device=c["device"], num_envs=1)
        # --- direct velocity commands
        bv = env_cfg.commands.base_velocity
        bv.heading_command = False
        bv.rel_heading_envs = 0.0
        bv.rel_standing_envs = 0.0
        bv.resampling_time_range = (1.0e9, 1.0e9)
        bv.debug_vis = False
        self.trained_ranges = {
            "lin_vel_x": tuple(bv.ranges.lin_vel_x),
            "lin_vel_y": tuple(bv.ranges.lin_vel_y),
            "ang_vel_z": tuple(bv.ranges.ang_vel_z),
        }
        # --- no auto-resets: we detect terminal conditions ourselves
        env_cfg.episode_length_s = 1.0e6
        for name in list(vars(env_cfg.terminations)):
            if not name.startswith("_") and name != "time_out":
                setattr(env_cfg.terminations, name, None)
        # deterministic, un-randomized play setup
        for ev in ("push_robot", "base_external_force_torque"):
            if hasattr(env_cfg.events, ev):
                setattr(env_cfg.events, ev, None)
        env_cfg.observations.policy.enable_corruption = False
        env_cfg.seed = int(c.get("seed", 0))
        # --- obstacle slots
        self.n_slots = int(c.get("obstacle_slots", 0))
        for i in range(self.n_slots):
            setattr(
                env_cfg.scene,
                f"obstacle_{i}",
                RigidObjectCfg(
                    prim_path=f"{{ENV_REGEX_NS}}/Obstacle_{i}",
                    spawn=sim_utils.CylinderCfg(
                        radius=float(c["obstacle_radius"]),
                        height=float(c["obstacle_height"]),
                        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                        mass_props=sim_utils.MassPropertiesCfg(mass=10.0),
                        collision_props=sim_utils.CollisionPropertiesCfg(),
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.6, 0.6, 0.6)),
                    ),
                    init_state=RigidObjectCfg.InitialStateCfg(pos=(PARK_XY[0] + 3 * i, PARK_XY[1], c["obstacle_height"] / 2)),
                ),
            )

        # --- optional parametric sidewalk terrain (P1/E4); absent = the task's own ground (flat plane)
        # `terrain_tiles` (list of tile params, merged over `terrain`): one column per tile, episode
        # with seed s runs on tile s % len(tiles).
        self.terrain = c.get("terrain")
        self.tiles: list[dict[str, Any]] | None = None
        self._tile_origin: np.ndarray | None = None
        if c.get("terrain_tiles"):
            self.terrain = {**(self.terrain or {}), "_tiles": True}
        if self.terrain:
            from .terrain import make_isaac_terrain_generator, params_with_defaults

            base = {k: v for k, v in self.terrain.items() if k != "_tiles"}
            if c.get("terrain_tiles"):
                self.tiles = [params_with_defaults({**base, **t}) for t in c["terrain_tiles"]]
                self.terrain = self.tiles[0]
            else:
                self.terrain = params_with_defaults(base)
            t = env_cfg.scene.terrain
            t.terrain_type = "generator"
            t.terrain_generator = make_isaac_terrain_generator(self.tiles or self.terrain)
            t.max_init_terrain_level = None
            mu = float(self.terrain["friction"])
            t.physics_material = sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply", restitution_combine_mode="multiply",
                static_friction=mu, dynamic_friction=mu)
            if hasattr(env_cfg, "curriculum") and hasattr(env_cfg.curriculum, "terrain_levels"):
                env_cfg.curriculum.terrain_levels = None
            if c.get("terrain_texture"):  # procedural concrete/asphalt/grass look (rendering only), made with the mesh
                from .isaac_texture import sidewalk_material_cfg

                tt = c["terrain_texture"] if isinstance(c["terrain_texture"], dict) else {}
                t.visual_material = sidewalk_material_cfg(self.tiles or [self.terrain], tuple(self.terrain["size"]),
                                                          tt.get("cache_dir", "D:/go2wm/cache/textures"),
                                                          float(tt.get("res", 0.02)))
            elif c.get("terrain_visual_color"):  # plain concrete-like surface (videos); physics unchanged
                t.visual_material = sim_utils.PreviewSurfaceCfg(
                    diffuse_color=tuple(float(v) for v in c["terrain_visual_color"]), roughness=0.9)
        # --- optional follow camera for videos (needs enable_cameras: true); render_rgb() returns a frame
        self.video = c.get("video")
        if self.video:
            v = env_cfg.viewer
            v.origin_type, v.asset_name, v.env_index = "asset_root", "robot", 0
            v.eye = tuple(float(x) for x in self.video.get("eye", (-2.2, -2.8, 1.4)))
            v.lookat = tuple(float(x) for x in self.video.get("lookat", (0.8, 0.0, 0.1)))
            v.resolution = tuple(int(x) for x in self.video.get("resolution", (1280, 720)))
        # --- optional E4-C sensor rig (sim/sensors.py): ray-cast range sensors against the terrain mesh,
        # each a RayCaster on the base whose starts/directions are set in the base frame after init
        # (LiDARs get fresh random directions every frame), plus an optional front RGB camera
        if c.get("rgb_camera"):
            from isaaclab.sensors import TiledCameraCfg

            # DLSS (Isaac's default) renders a 224 px camera at ~130 px, below its 300 px minimum, and is
            # temporal; frames came back all black in most runs. FXAA renders at native resolution per frame.
            env_cfg.sim.render.antialiasing_mode = c.get("render_antialiasing", "FXAA")

            from .sensors import RGB_SPEC

            f_mm = 10.0
            th = math.radians(RGB_SPEC["pitch_deg"]) / 2
            setattr(env_cfg.scene, "rig_rgb", TiledCameraCfg(
                prim_path="{ENV_REGEX_NS}/Robot/base/front_cam",
                offset=TiledCameraCfg.OffsetCfg(pos=tuple(RGB_SPEC["mount_pos"]), rot=(math.cos(th), 0.0, math.sin(th), 0.0),
                                                convention="world"),
                data_types=["rgb"], width=RGB_SPEC["width"], height=RGB_SPEC["height"],
                spawn=sim_utils.PinholeCameraCfg(focal_length=f_mm, clipping_range=(0.05, 80.0),
                                                 horizontal_aperture=2 * f_mm * math.tan(math.radians(RGB_SPEC["hfov_deg"]) / 2))))
        self.rig_names = list(c.get("sensor_rig") or [])
        if self.rig_names:
            from isaaclab.sensors import RayCasterCfg
            from isaaclab.sensors.ray_caster.patterns import GridPatternCfg

            from .sensors import SENSOR_SPECS, n_rays

            for name in self.rig_names:
                spec = SENSOR_SPECS[name]
                setattr(env_cfg.scene, f"rig_{name}", RayCasterCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/base", mesh_prim_paths=["/World/ground"], ray_alignment="base",
                    # n placeholder rays on a 1 mm line (overwritten after init)
                    pattern_cfg=GridPatternCfg(resolution=0.001, size=(round((n_rays(spec) - 1) * 0.001, 6), 0.0)),
                    max_distance=float(spec["range_m"][1]) + 1.0, debug_vis=False))
        self.env = gym.make(c["task"], cfg=env_cfg, render_mode="rgb_array" if self.video else None)
        self.uenv = self.env.unwrapped
        self.terrain_texture_png = None
        if self.terrain and c.get("terrain_texture"):  # the texture assumed these tile origins; check them
            from .isaac_texture import generator_tile_origins_xy

            ter = self.uenv.scene.terrain
            got = ter.terrain_origins[0, :, :2].cpu().numpy().astype(np.float64)
            want = generator_tile_origins_xy(len(self.tiles or [self.terrain]), tuple(self.terrain["size"]))
            if not np.allclose(got, want, atol=1e-3):
                raise RuntimeError(f"terrain texture tile origins {want.tolist()} != generator origins {got.tolist()}")
            self.terrain_texture_png = getattr(ter.cfg.visual_material, "png", None)
        self.wrapped = RslRlVecEnvWrapper(self.env)
        agent_cfg = load_cfg_from_registry(c["task"], "rsl_rl_cfg_entry_point")
        self.ckpt_path = self._resolve_checkpoint(c)
        runner = OnPolicyRunner(self.wrapped, agent_cfg.to_dict(), log_dir=None, device=c["device"])
        runner.load(self.ckpt_path)
        self.policy = runner.get_inference_policy(device=self.uenv.device)
        self.robot = self.uenv.scene["robot"]
        self.cmd_term = self.uenv.command_manager.get_term("base_velocity")
        # same signal as Isaac Lab's base_contact termination (net force on base > threshold)
        self.contact = self.uenv.scene.sensors["contact_forces"]
        self.base_body_ids = self.contact.find_bodies("base")[0]
        self.scanner = None
        if self.terrain:  # terrain-only lookups: the flat E0-E3 path is unchanged
            self.foot_ids = self.contact.find_bodies(".*_foot")[0]
            self.robot_foot_ids = self.robot.find_bodies(".*_foot")[0]
            # rough-terrain tasks carry a height scanner (policy input); used for height above ground
            self.scanner = dict(self.uenv.scene.sensors).get("height_scanner")
        self.rig = {n: self.uenv.scene.sensors[f"rig_{n}"] for n in self.rig_names}
        self.rgb_cam = dict(self.uenv.scene.sensors).get("rig_rgb")
        self._rig_rng = np.random.default_rng(0)
        for n in self.rig_names:
            self._set_rig_rays(n)
        self.step_dt = float(self.uenv.step_dt)
        self.n_policy_steps = max(1, int(round(self.control_dt / self.step_dt)))
        if abs(self.n_policy_steps * self.step_dt - self.control_dt) > 1e-6:
            raise ValueError(f"control_dt {self.control_dt} is not a multiple of env step_dt {self.step_dt}")
        self._step_ns = int(round(self.step_dt * 1e9))
        self._env_steps = 0
        self._obs = None
        self._agent_cfg = agent_cfg

    # ------------------------------------------------------------------
    def _resolve_checkpoint(self, c: dict[str, Any]) -> str:
        if c.get("policy_checkpoint", "pretrained") == "pretrained":
            from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

            p = get_published_pretrained_checkpoint("rsl_rl", c["train_task"])
            if not p:
                raise RuntimeError(f"No published pretrained checkpoint for {c['train_task']}")
            return p
        from isaaclab.utils.assets import retrieve_file_path

        return retrieve_file_path(c["policy_checkpoint"])

    def _origin(self) -> np.ndarray:
        if self._tile_origin is not None:
            return self._tile_origin
        return self.uenv.scene.env_origins[0].cpu().numpy().astype(np.float64)

    def _set_command(self, a: np.ndarray) -> None:
        lim = self.cfg["command_limits"]
        cmd = [np.clip(a[0], *lim["vx"]), np.clip(a[1], *lim["vy"]), np.clip(a[2], *lim["yaw_rate"])]
        self.cmd_term.vel_command_b[:] = self.torch.tensor(cmd, device=self.uenv.device, dtype=self.torch.float32)
        self._applied = np.array(cmd, dtype=np.float64)

    # --- inspection posture (crouch + body pitch while standing), joint-level, no retraining -------------
    LEG_L1, LEG_L2, HIP_Y = 0.213, 0.213, 0.0955  # Go2 thigh / calf length, hip-to-thigh lateral offset [m]
    HIPS = {"FL": (0.1934, 0.0465, 1), "FR": (0.1934, -0.0465, -1), "RL": (-0.1934, 0.0465, 1),
            "RR": (-0.1934, -0.0465, -1)}  # hip joint position in the base frame, side (+1 left)
    POSTURE_MAX_ROLL = 0.15  # rad: abort the posture (stand back up) if the roll changes more than this from its start
    POSTURE_RATES = {"drop": 0.04, "pitch": 0.18, "roll": 0.18}  # ramp rates [m/s, rad/s]
    POSTURE_MIN_FOOT_Y = 0.07  # m: refuse to crouch if a foot stands closer than this to the body's centre line
    POSTURE_KP, POSTURE_KD = 60.0, 3.0  # joint PD gains while posing (stand-mode gains of the real Go2 ~ 60 / 5)

    def set_posture(self, drop: float = 0.0, pitch: float = 0.0) -> None:
        """Target body drop [m] (below the height at the start of the posture) and ABSOLUTE nose-down body pitch
        [rad] (0 = level; the walking policy stands ~10 deg nose-up), body roll levelled to 0, while standing.
        drop > 0: the walking policy is suspended and the legs are driven to the matching joint targets (3-joint IK
        per leg with every foot kept where it stands), ramped at 0.08 m/s and 0.35 rad/s; drop = 0: ramps back to
        the start pose and hands control to the policy again. Velocity commands are ignored while a posture is
        active. If the body rolls past POSTURE_MAX_ROLL the posture is released (counted in posture_aborts)."""
        self._posture_target = (float(drop), float(pitch))

    def posture_state(self) -> tuple[float, float, bool]:
        """(drop, commanded absolute pitch, settled at target)."""
        st = getattr(self, "_posture", None)
        tgt = getattr(self, "_posture_target", (0.0, 0.0))
        if st is None:
            return 0.0, 0.0, tgt[0] <= 0.0
        g = self._posture_goal(st, tgt)
        return st["drop"], st["pitch"], all(abs(st[k] - g[k]) < 1e-4 for k in g)

    @staticmethod
    def _posture_goal(st, tgt):
        if tgt[0] > 0.0:
            return {"drop": tgt[0], "pitch": tgt[1], "roll": 0.5 * st["roll0"]}  # level half of the start roll
        return {"drop": 0.0, "pitch": st["pitch0"], "roll": st["roll0"]}

    def _leg_ik(self, x: float, z: float) -> tuple[float, float]:
        l1, l2 = self.LEG_L1, self.LEG_L2
        d2 = x * x + z * z
        c = -math.acos(max(-1.0, min(1.0, (d2 - l1 * l1 - l2 * l2) / (2 * l1 * l2))))
        t = math.atan2(-x, -z) - math.atan2(l2 * math.sin(c), l1 + l2 * math.cos(c))
        return t, c

    def _leg_fk(self, a: float, t: float, c: float, side: int) -> np.ndarray:
        """Foot relative to the hip joint, base frame (hip about x, thigh and calf about y)."""
        x = -self.LEG_L1 * math.sin(t) - self.LEG_L2 * math.sin(t + c)
        z = -self.LEG_L1 * math.cos(t) - self.LEG_L2 * math.cos(t + c)
        y = side * self.HIP_Y
        return np.array([x, y * math.cos(a) - z * math.sin(a), y * math.sin(a) + z * math.cos(a)])

    def _leg_ik3(self, p: np.ndarray, side: int) -> tuple[float, float, float]:
        px, py, pz = (float(v) for v in p)
        Lz = math.sqrt(max(py * py + pz * pz - self.HIP_Y ** 2, 1e-6))
        a = math.atan2(pz, py) - math.atan2(-Lz, side * self.HIP_Y)
        a = (a + math.pi) % (2 * math.pi) - math.pi
        t, c = self._leg_ik(px, -Lz)
        return a, t, c

    @staticmethod
    def _quat_R(q) -> np.ndarray:
        w, x, y, z = (float(v) for v in q)
        return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                         [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                         [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])

    @staticmethod
    def _R_pr(p: float, r: float) -> np.ndarray:
        cp, sp, cr, sr = math.cos(p), math.sin(p), math.cos(r), math.sin(r)
        Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
        Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
        return Ry @ Rx

    def _level_R(self) -> np.ndarray:
        R = self._quat_R(self.robot.data.root_quat_w[0])
        yaw = math.atan2(R[1, 0], R[0, 0])
        c, s = math.cos(-yaw), math.sin(-yaw)
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]) @ R  # attitude with the heading removed

    def _posture_step(self) -> bool:
        """One policy-rate step of the posture controller; False when the walking policy should run."""
        tgt = getattr(self, "_posture_target", (0.0, 0.0))
        st = getattr(self, "_posture", None)
        if st is None:
            if tgt[0] <= 0.0:
                return False
            names = list(self.robot.joint_names)
            q0 = self.robot.data.joint_pos[0].detach().clone()
            Rl = self._level_R()
            p0, r0 = math.asin(max(-1.0, min(1.0, -Rl[2, 0]))), math.atan2(Rl[2, 1], Rl[2, 2])
            legs = {}
            for leg, (hx, hy, side) in self.HIPS.items():
                ii = [names.index(f"{leg}_{j}_joint") for j in ("hip", "thigh", "calf")]
                hip = np.array([hx, hy, 0.0])
                foot_b = hip + self._leg_fk(*(float(q0[i]) for i in ii), side)
                legs[leg] = (ii, side, hip, Rl @ foot_b)  # foot rel. base centre, level heading frame
            if min(v[1] * v[3][1] for v in legs.values()) < self.POSTURE_MIN_FOOT_Y:  # a foot under the body
                self._posture_target = (0.0, 0.0)
                self.posture_aborts = getattr(self, "posture_aborts", 0) + 1
                self.posture_skipped = getattr(self, "posture_skipped", 0) + 1
                return False
            self._set_gains(True)
            st = self._posture = {"ir": 0.0, "ip": 0.0, "drop": 0.0, "pitch": p0, "roll": r0, "pitch0": p0, "roll0": r0, "q0": q0,
                                  "legs": legs, "term": self.uenv.action_manager._terms["joint_pos"],
                                  "fix": self._R_pr(p0, r0).T @ Rl}
        dt = self.step_dt
        Rm = self._level_R()
        roll_m, pitch_m = math.atan2(Rm[2, 1], Rm[2, 2]), math.asin(max(-1.0, min(1.0, -Rm[2, 0])))
        if tgt[0] > 0.0 and abs(roll_m - st["roll0"]) > self.POSTURE_MAX_ROLL:  # safety: stand up if it starts to tip
            self._posture_target = tgt = (0.0, 0.0)
            self.posture_aborts = getattr(self, "posture_aborts", 0) + 1
        goal = self._posture_goal(st, tgt)
        for k, rate in self.POSTURE_RATES.items():
            st[k] += float(np.clip(goal[k] - st[k], -rate * dt, rate * dt))
        # attitude feedback (PI on roll / pitch): the joint PD sags under load, so command past the target
        er, ep = roll_m - st["roll"], pitch_m - st["pitch"]
        st["ir"] = float(np.clip(st["ir"] + er * dt, -0.15, 0.15))
        st["ip"] = float(np.clip(st["ip"] + ep * dt, -0.15, 0.15))
        r_cmd = st["roll"] - 0.3 * er - 1.0 * st["ir"]
        p_cmd = st["pitch"] - 0.2 * ep - 0.8 * st["ip"]
        Rt = self._R_pr(p_cmd, r_cmd) @ st["fix"]  # = the start attitude exactly at the start
        q = st["q0"].clone()
        for leg, (ii, side, hip, foot_l) in st["legs"].items():
            foot_b = Rt.T @ (foot_l - np.array([0.0, 0.0, -st["drop"]]))  # feet stay put, base drops and turns
            a, t, c = self._leg_ik3(foot_b - hip, side)
            q[ii[0]], q[ii[1]], q[ii[2]] = a, t, c
        term = st["term"]
        jid = term._joint_ids
        q_term = q if isinstance(jid, slice) else q[jid]  # the joints of the joint_pos action term
        legs = ((q_term - term._offset[0]) / term._scale).reshape(-1)  # joint_pos action: q = offset + scale * a
        self.wrapped.step(self._full_action(term, legs))
        self._env_steps += 1
        if tgt[0] <= 0.0 and all(abs(st[k] - goal[k]) < 1e-6 for k in goal):
            self._posture = None  # back at the start pose: the walking policy takes over next step
            self._set_gains(False)
        return True

    def _full_action(self, term, leg_action):
        """Action vector for all action terms: the leg joint-position term from the posture controller, zeros for
        the others (Go2-W: wheel velocity 0 = wheels held still). leg_action is in the term's joint order."""
        am = self.uenv.action_manager
        la = leg_action
        parts = []
        for name in am.active_terms:
            t_ = am._terms[name]
            parts.append(la if t_ is term else self.torch.zeros(t_.action_dim, device=la.device))
        return self.torch.cat(parts).unsqueeze(0)

    def _set_gains(self, posing: bool) -> None:
        """Stiffer joint PD while holding a posture; the walking policy's own gains otherwise."""
        from isaaclab.actuators import ImplicitActuator

        if not hasattr(self, "_gains0"):
            self._gains0 = {k: (a.stiffness.clone(), a.damping.clone()) for k, a in self.robot.actuators.items()}
        for k, act in self.robot.actuators.items():
            kp0, kd0 = self._gains0[k]
            kp = torch_full_like(kp0, self.POSTURE_KP) if posing else kp0
            kd = torch_full_like(kd0, self.POSTURE_KD) if posing else kd0
            act.stiffness[:] = kp
            act.damping[:] = kd
            if isinstance(act, ImplicitActuator):
                self.robot.write_joint_stiffness_to_sim(kp, joint_ids=act.joint_indices)
                self.robot.write_joint_damping_to_sim(kd, joint_ids=act.joint_indices)

    def _policy_step(self) -> None:
        # caller holds torch.inference_mode() (as Isaac Lab's play.py): env buffers become
        # inference tensors, so *all* env interaction (reset included) must stay in that mode.
        if self._posture_step():
            return
        obs = self.wrapped.get_observations()  # recompute: includes the current command
        actions = self.policy(obs)
        self.wrapped.step(actions)
        self._env_steps += 1

    def _place_obstacles(self, scene: SceneSpec) -> None:
        torch = self.torch
        origin = self._origin()
        obs_arr = np.asarray(scene.obstacles).reshape(-1, 3)
        if len(obs_arr) > self.n_slots:
            raise ValueError(f"scene {scene.scene_id} has {len(obs_arr)} obstacles > {self.n_slots} slots")
        for i in range(self.n_slots):
            obj = self.uenv.scene[f"obstacle_{i}"]
            z = self.cfg["obstacle_height"] / 2
            if i < len(obs_arr):
                x, y = obs_arr[i, 0] + origin[0], obs_arr[i, 1] + origin[1]
                if self.terrain:  # stand on the surface (world z of the surface = sidewalk_height, see _reset)
                    from .terrain import sidewalk_height

                    z += float(sidewalk_height(obs_arr[i, 0], obs_arr[i, 1], self.terrain))
            else:
                x, y = PARK_XY[0] + 3 * i, PARK_XY[1]
            pose = torch.tensor([[x, y, z, 1, 0, 0, 0]], device=self.uenv.device, dtype=torch.float32)
            obj.write_root_pose_to_sim(pose)
        self._n_static = len(obs_arr)

    def set_movers(self, xy) -> None:
        """Move kinematic cylinders (people in demo missions) to scene xy, in the slots after the static obstacles."""
        from .terrain import sidewalk_height

        torch = self.torch
        origin = self._origin()
        with torch.inference_mode():
            for j, (x, y) in enumerate(np.asarray(xy).reshape(-1, 2)):
                i = self._n_static + j
                if i >= self.n_slots:
                    break
                z = self.cfg["obstacle_height"] / 2 + (float(sidewalk_height(x, y, self.terrain)) if self.terrain else 0.0)
                pose = torch.tensor([[x + origin[0], y + origin[1], z, 1, 0, 0, 0]], device=self.uenv.device,
                                    dtype=torch.float32)
                self.uenv.scene[f"obstacle_{i}"].write_root_pose_to_sim(pose)

    # --- SimulatorAdapter ------------------------------------------------
    def reset(self, seed: int, scene_id: str | None = None) -> Observation:
        with self.torch.inference_mode():
            return self._reset(seed, scene_id)

    def step(self, action: VelocityCommand) -> StepResult:
        with self.torch.inference_mode():
            return self._step(action)

    def _reset(self, seed: int, scene_id: str | None) -> Observation:
        torch = self.torch
        if scene_id is None:
            fams = self.cfg["families"]
            scene_id = f"{fams[seed % len(fams)]}_{seed:04d}"
        self.scene = make_scene(scene_id, self.robot_radius, fixed_radius=float(self.cfg["obstacle_radius"]),
                                max_obstacles=self.n_slots)
        self.scene.params = {"backend": "isaac", "obstacle_radius": self.cfg["obstacle_radius"]}
        if self.tiles:
            i = seed % len(self.tiles)
            self.terrain = self.tiles[i]
            self._tile_origin = self.uenv.scene.terrain.terrain_origins[0, i].cpu().numpy().astype(np.float64)
            self.scene.params["terrain_tile"] = i
            self.scene.params["tile_origin_w"] = self._tile_origin.round(5).tolist()
        if self.terrain:
            self.scene.params["terrain"] = self.terrain
            if not self.cfg.get("scene_obstacles", True):
                self.scene.obstacles = np.zeros((0, 3))
            if self.cfg.get("scene_bounds"):  # e.g. include the road beyond the curb (terrain runs only)
                self.scene.bounds = tuple(float(v) for v in self.cfg["scene_bounds"])
            if self.cfg.get("mission_scene"):  # Phase 1: start/goal/bounds/obstacles from the tile and its defects
                from ..mission.scene import sidewalk_mission_scene

                keep = {k: v for k, v in self.scene.params.items() if k != "terrain"}
                self.scene = sidewalk_mission_scene(self.terrain, scene_id,
                                                    fixed_radius=float(self.cfg["obstacle_radius"]))
                self.scene.params.update(keep)
        if hasattr(self, "_gains0"):
            self._set_gains(False)
        self._posture, self._posture_target = None, (0.0, 0.0)
        torch.manual_seed(seed)
        self._rig_rng = np.random.default_rng(seed + 7_000_000)  # sensor noise / LiDAR directions per episode
        self.wrapped.reset()
        self._place_obstacles(self.scene)
        # place robot at scene start (relative to env origin), zero velocity, default joints
        origin = self._origin()
        x, y, yaw = self.scene.start
        rs = self.robot.data.default_root_state.clone()
        rs[:, 0] = float(x + origin[0])
        rs[:, 1] = float(y + origin[1])
        if self.terrain:
            # default_root_state z is for flat ground at world z = 0. The generator translates the tile
            # only in x/y, so world ground height = sidewalk_height (origin z is just the tile-centre max).
            from .terrain import sidewalk_height

            rs[:, 2] += float(sidewalk_height(x, y, self.terrain))
        rs[:, 3:7] = torch.tensor([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)], device=rs.device)
        rs[:, 7:] = 0.0
        self.robot.write_root_pose_to_sim(rs[:, :7])
        self.robot.write_root_velocity_to_sim(rs[:, 7:])
        self.robot.write_joint_state_to_sim(self.robot.data.default_joint_pos.clone(), self.robot.data.default_joint_vel.clone())
        self.uenv.action_manager.reset()
        # settle with zero command (not recorded)
        self._set_command(np.zeros(3))
        for _ in range(max(1, int(round(self.cfg.get("settle_s", 1.0) / self.step_dt)))):
            self._policy_step()
        self.scene.params["obstacle_placement_max_err_m"] = self._obstacle_placement_error()
        self._env_steps = 0
        return self._observe()

    def _obstacle_placement_error(self) -> float:
        """Max distance between simulated obstacle prims and the map (sanity check, logged in metadata)."""
        origin = self._origin()
        obs_arr = np.asarray(self.scene.obstacles).reshape(-1, 3)
        err = 0.0
        for i in range(len(obs_arr)):
            p = self.uenv.scene[f"obstacle_{i}"].data.root_pos_w[0, :2].cpu().numpy() - origin[:2]
            err = max(err, float(np.linalg.norm(p - obs_arr[i, :2])))
        return err

    def _step(self, action: VelocityCommand) -> StepResult:
        assert self.scene is not None, "call reset() first"
        t0 = self._env_steps * self._step_ns
        self._set_command(action.action)
        n = max(1, int(round(action.duration_s / self.step_dt)))
        collision = fall = False
        for _ in range(n):
            self._policy_step()
            st = self._raw_state()
            clear = self.scene.clearance(st["pos"][:2], self.robot_radius)
            if clear < 0:
                collision = True
            if st["grav"][2] > self.cfg["fall_gravity_z"] or self._height_above_ground(st) < self.cfg["fall_min_height"]:
                fall = True
            elif self._base_contact():
                # base touching something: an obstacle if we're near one, otherwise the ground
                if clear < 0.1:
                    collision = True
                else:
                    fall = True
            if collision or fall:
                break
        obs = self._observe()
        reason = "fall" if fall else ""
        if collision and not fall:
            xmin, xmax, ymin, ymax = self.scene.bounds
            x, y = obs.pos_w[:2]
            r = self.robot_radius
            inside = xmin + r < x < xmax - r and ymin + r < y < ymax - r
            reason = "collision" if inside else "out_of_bounds"
        return StepResult(
            observation=obs,
            action_timestamp_ns=t0,
            collision=collision,
            terminated=bool(reason),
            termination_reason=reason,
            info={"applied_command": self._applied.tolist(), "dt_s": (self._env_steps * self._step_ns - t0) * 1e-9},
        )

    def render_rgb(self) -> np.ndarray:
        """Follow-camera RGB frame [H, W, 3] uint8 (requires isaac.video and enable_cameras)."""
        return np.asarray(self.env.render())

    def close(self) -> None:
        try:
            self.env.close()
        finally:
            self.app.close()

    def metadata(self) -> dict[str, Any]:
        import isaaclab

        return {
            "backend": self.backend,
            "task": self.cfg["task"],
            "policy_checkpoint": str(self.ckpt_path),
            "isaaclab_version": getattr(isaaclab, "__version__", "unknown"),
            "env_step_dt": self.step_dt,
            "physics_dt": float(self.uenv.physics_dt),
            "policy_steps_per_command": self.n_policy_steps,
            "trained_command_ranges": self.trained_ranges,
            "control_dt": self.control_dt,
            "robot_radius": self.robot_radius,
            "collision_detection": "geometric footprint circle vs known obstacle map",
            "terrain": self.terrain or "task default (flat plane)",
            "height_scanner": self.scanner is not None,
            "sensor_rig": self._rig_meta(),
            "height_scan_pattern_xy": (self.scanner.ray_starts[0, :, :2].cpu().numpy().round(4).tolist()
                                       if (self.scanner is not None and self.cfg.get("record_height_scan")) else None),
            "config": self.cfg,
        }

    # --- internals --------------------------------------------------------
    def _ground_z_world(self) -> float | None:
        """Ground height under the base from the height scanner (median of rays within 0.15 m)."""
        if self.scanner is None:
            return None
        hits = self.scanner.data.ray_hits_w[0]  # [R, 3] world
        base = self.robot.data.root_pos_w[0]
        d = (hits[:, :2] - base[:2]).norm(dim=-1)
        sel = hits[d < 0.15, 2]
        sel = sel[self.torch.isfinite(sel)]
        return float(sel.median().item()) if len(sel) else None

    def _height_above_ground(self, st: dict[str, np.ndarray]) -> float:
        if not self.terrain:
            return float(st["pos"][2])  # flat plane at the env origin (E0-E3 behaviour)
        g = self._ground_z_world()
        if g is None:
            from .terrain import sidewalk_height

            g = float(sidewalk_height(st["pos"][0], st["pos"][1], self.terrain))
            return float(st["pos"][2] + self._origin()[2] - g)
        return float(st["pos"][2] + self._origin()[2] - g)

    def _terrain_truth(self, st: dict[str, np.ndarray]) -> dict[str, Any]:
        """Extra ground truth on terrain: height above ground, ground height, foot contacts and slip."""
        torch = self.torch
        out: dict[str, Any] = {"height_above_ground_m": self._height_above_ground(st)}
        g = self._ground_z_world()
        out["ground_z_m"] = float("nan") if g is None else g - float(self._origin()[2])
        f = self.contact.data.net_forces_w[0, self.foot_ids]  # [4, 3]
        contact = (f.norm(dim=-1) > 1.0)
        vel = self.robot.data.body_lin_vel_w[0, self.robot_foot_ids, :2].norm(dim=-1)  # [4]
        out["foot_contact"] = contact.cpu().numpy().astype(np.float64).tolist()
        # slip = horizontal speed of feet that are in contact (0 for a perfect no-slip stance)
        slip = torch.where(contact, vel, torch.zeros_like(vel))
        out["foot_slip_m_s"] = slip.cpu().numpy().astype(np.float64).tolist()
        for n in self.rig_names:
            out.update(self._rig_frame(n, st))
        if self.cfg.get("record_height_scan") and self.scanner is not None:
            z = self.scanner.data.ray_hits_w[0, :, 2] - float(self._origin()[2])
            z = torch.where(torch.isfinite(z), z, torch.full_like(z, float("nan")))
            out["scan_height_m"] = z.cpu().numpy().astype(np.float64).tolist()
        return out

    def _set_rig_rays(self, name: str) -> None:
        from .sensors import SENSOR_SPECS, base_frame_rays

        starts, dirs = base_frame_rays(SENSOR_SPECS[name], self._rig_rng)
        sen = self.rig[name]
        sen.ray_starts[0] = self.torch.as_tensor(starts, dtype=sen.ray_starts.dtype, device=sen.ray_starts.device)
        sen.ray_directions[0] = self.torch.as_tensor(dirs, dtype=sen.ray_directions.dtype, device=sen.ray_directions.device)

    def _rig_frame(self, name: str, st: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """One frame of a range sensor -> elevation grid around the robot (scene-frame heights)."""
        from .sensors import SENSOR_SPECS, bin_elevation, corrupt_hits, pitch_matrix

        spec = SENSOR_SPECS[name]
        sen = self.rig[name]
        if spec["kind"] != "pinhole":  # non-repetitive LiDAR scan: new directions before this frame's cast
            self._set_rig_rays(name)
        hits = sen.data.ray_hits_w[0].detach().cpu().numpy().astype(np.float64)
        starts = sen._ray_starts_w[0].detach().cpu().numpy().astype(np.float64)
        dirs = sen._ray_directions_w[0].detach().cpu().numpy().astype(np.float64)
        w, x, y, z = st["quat"] / np.linalg.norm(st["quat"])
        R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                      [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                      [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])
        axis = R @ pitch_matrix(spec["pitch_deg"]) @ np.array([1.0, 0.0, 0.0])
        pts = corrupt_hits(starts, dirs, hits, spec, self._rig_rng, optical_axis_w=axis)
        o = self._origin()
        if self.terrain and self.terrain.get("defects"):  # standing water is specular: many returns are lost
            from .defects import range_dropout

            ok = np.isfinite(pts[:, 0])
            pd = range_dropout(pts[ok, 0] - o[0], pts[ok, 1] - o[1], self.terrain["defects"], spec["kind"])
            lost = np.flatnonzero(ok)[self._rig_rng.uniform(size=ok.sum()) < pd]
            pts[lost] = np.nan
        yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        # z_ref 0: the generator only shifts tiles in x/y, so world z of the surface = terrain.sidewalk_height
        elev, cnt = bin_elevation(pts, o[:2] + st["pos"][:2], yaw, z_ref=0.0)
        return {f"sensor_{name}_elev_m": elev, f"sensor_{name}_count": cnt}

    def _rig_meta(self) -> dict[str, Any] | None:
        if not self.rig_names and self.rgb_cam is None:
            return None
        from .sensors import ELEV_GRID, GRID_SHAPE, RGB_SPEC, SENSOR_SPECS

        return {"range_sensors": {n: SENSOR_SPECS[n] for n in self.rig_names}, "elev_grid": ELEV_GRID,
                "grid_shape": list(GRID_SHAPE), "rgb": RGB_SPEC if self.rgb_cam is not None else None,
                "frames": "robot-centred yaw-aligned grid; heights = world hit z = terrain frame (terrain.sidewalk_height); "
                          "state z = terrain z - tile origin z (scene.params.tile_origin_w); scan_height_m = hit z - tile origin z"}

    def _rgb_frame(self) -> np.ndarray | None:
        if self.rgb_cam is None:
            return None
        return self.rgb_cam.data.output["rgb"][0, ..., :3].detach().cpu().numpy().astype(np.uint8)

    def _base_contact(self) -> bool:
        f = self.contact.data.net_forces_w_history[0, :, self.base_body_ids]  # [hist, n_base, 3]
        return bool(f.norm(dim=-1).max().item() > float(self.cfg.get("base_contact_threshold_n", 1.0)))

    def _raw_state(self) -> dict[str, np.ndarray]:
        d = self.robot.data
        f = lambda t: t[0].detach().cpu().numpy().astype(np.float64)  # noqa: E731
        pos = f(d.root_pos_w) - self._origin()
        return {"pos": pos, "quat": f(d.root_quat_w), "lin": f(d.root_lin_vel_b), "ang": f(d.root_ang_vel_b),
                "grav": f(d.projected_gravity_b), "jp": f(d.joint_pos), "jv": f(d.joint_vel)}

    def _observe(self) -> Observation:
        s = self._raw_state()
        assert self.scene is not None
        gt = {"clearance_m": float(self.scene.clearance(s["pos"][:2], self.robot_radius))}
        if self.terrain:
            gt.update(self._terrain_truth(s))
        return Observation(
            timestamp_ns=self._env_steps * self._step_ns,
            pos_w=s["pos"],
            quat_w=s["quat"] / np.linalg.norm(s["quat"]),
            lin_vel_b=s["lin"],
            ang_vel_b=s["ang"],
            gravity_b=s["grav"] / max(np.linalg.norm(s["grav"]), 1e-9),
            joint_pos=s["jp"],
            joint_vel=s["jv"],
            ground_truth=gt,
            rgb=self._rgb_frame(),
            valid={"rgb": self.rgb_cam is not None, "depth": False, "joints": True},
        )
