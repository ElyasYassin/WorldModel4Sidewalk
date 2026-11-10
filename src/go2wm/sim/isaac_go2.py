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


class IsaacGo2Adapter:
    backend = "isaac"

    def __init__(self, cfg: dict[str, Any]):
        self.cfg = c = cfg["isaac"]
        self.control_dt = float(c["control_dt"])
        self.robot_radius = float(c["robot_radius"])
        self.scene: SceneSpec | None = None

        from isaaclab.app import AppLauncher

        self._app_launcher = AppLauncher(headless=c.get("headless", True), enable_cameras=c.get("enable_cameras", False), device=c["device"])
        self.app = self._app_launcher.app

        import gymnasium as gym
        import torch
        import isaaclab.sim as sim_utils
        import isaaclab_tasks  # noqa: F401  (registers tasks)
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
        self.env = gym.make(c["task"], cfg=env_cfg)
        self.uenv = self.env.unwrapped
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

    def _policy_step(self) -> None:
        # caller holds torch.inference_mode() (as Isaac Lab's play.py): env buffers become
        # inference tensors, so *all* env interaction (reset included) must stay in that mode.
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
            if i < len(obs_arr):
                x, y = obs_arr[i, 0] + origin[0], obs_arr[i, 1] + origin[1]
            else:
                x, y = PARK_XY[0] + 3 * i, PARK_XY[1]
            pose = torch.tensor([[x, y, self.cfg["obstacle_height"] / 2, 1, 0, 0, 0]], device=self.uenv.device, dtype=torch.float32)
            obj.write_root_pose_to_sim(pose)

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
        if self.terrain:
            self.scene.params["terrain"] = self.terrain
            if not self.cfg.get("scene_obstacles", True):
                self.scene.obstacles = np.zeros((0, 3))
            if self.cfg.get("scene_bounds"):  # e.g. include the road beyond the curb (terrain runs only)
                self.scene.bounds = tuple(float(v) for v in self.cfg["scene_bounds"])
        torch.manual_seed(seed)
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
        return out

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
            valid={"rgb": False, "depth": False, "joints": True},
        )
