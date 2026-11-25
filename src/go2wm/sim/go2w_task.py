"""Unitree Go2-W (wheeled legs) velocity-tracking task for Isaac Lab, built on Isaac Lab's own Go2 rough task.

Import after the Isaac app is running; importing registers

  Go2WM-Velocity-Rough-Go2W-v0   rough terrain (Isaac Lab's terrain curriculum), height-scan policy input

Robot: the Go2-W URDF (go2w_description, from the robot_lab repository, Apache-2.0, used as data only) converted by
Isaac Lab's URDF importer. Legs: implicit PD (kp 25, kd 0.5, as the Go2); wheels: velocity-driven (kp 0, kd 0.5).
Actions: 12 leg joint-position targets (scale 0.125 hips / 0.25 thigh, calf, around the default pose) + 4 wheel
velocity targets (scale 5 rad/s). Rewards: Isaac Lab's locomotion terms re-weighted for a wheeled-legged robot,
following robot_lab's Go2-W choices (velocity tracking 3.0 / 1.5, legs held near the default pose so it rolls
instead of stepping, no feet-air-time reward, leg torque / acceleration / action-rate penalties, undesired contacts
on everything but the wheels). Written here instead of running robot_lab's code.
"""

from __future__ import annotations

import gymnasium as gym
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg
from isaaclab_tasks.manager_based.locomotion.velocity import mdp
from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.rough_env_cfg import UnitreeGo2RoughEnvCfg

GO2W_URDF = "D:/go2wm/ext/robot_lab/source/robot_lab/data/Robots/unitree/go2w_description/urdf/go2w_description.urdf"
LEG_JOINTS = [f"{l}_{j}_joint" for l in ("FR", "FL", "RR", "RL") for j in ("hip", "thigh", "calf")]
WHEEL_JOINTS = [f"{l}_foot_joint" for l in ("FR", "FL", "RR", "RL")]

GO2W_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        asset_path=GO2W_URDF,
        fix_base=False,
        merge_fixed_joints=True,
        replace_cylinders_with_capsules=False,
        activate_contact_sensors=True,
        usd_dir="D:/go2wm/cache/go2w_usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False, retain_accelerations=False, linear_damping=0.0, angular_damping=0.0,
            max_linear_velocity=1000.0, max_angular_velocity=1000.0, max_depenetration_velocity=1.0),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, solver_position_iteration_count=4, solver_velocity_iteration_count=0),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.45),
        joint_pos={".*_hip_joint": 0.0, ".*_thigh_joint": 0.8, ".*_calf_joint": -1.5, ".*_foot_joint": 0.0},
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
                                    effort_limit_sim=23.5, velocity_limit_sim=30.0, stiffness=25.0, damping=0.5),
        "wheels": ImplicitActuatorCfg(joint_names_expr=[".*_foot_joint"], effort_limit_sim=23.5,
                                      velocity_limit_sim=30.0, stiffness=0.0, damping=0.5),
    },
)


@configclass
class Go2WRoughEnvCfg(UnitreeGo2RoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = GO2W_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # actions: legs position around the default pose, wheels velocity
        self.actions.joint_pos = mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=LEG_JOINTS, scale={".*_hip_joint": 0.125, ".*_(thigh|calf)_joint": 0.25},
            use_default_offset=True, preserve_order=True)
        self.actions.joint_vel = mdp.JointVelocityActionCfg(
            asset_name="robot", joint_names=WHEEL_JOINTS, scale=5.0, use_default_offset=True, preserve_order=True)
        # observations: wheel angles are meaningless (continuous joints) -> leg joint positions only
        for grp in (self.observations.policy,):
            grp.joint_pos.params = {"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINTS, preserve_order=True)}
            grp.joint_vel.params = {"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINTS + WHEEL_JOINTS,
                                                                preserve_order=True)}
        # rewards (wheeled-legged)
        r = self.rewards
        r.track_lin_vel_xy_exp.weight = 3.0
        r.track_ang_vel_z_exp.weight = 1.5
        r.lin_vel_z_l2.weight = -2.0
        r.ang_vel_xy_l2.weight = -0.05
        r.dof_torques_l2.weight = -2.5e-5
        r.dof_torques_l2.params = {"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINTS)}
        r.dof_acc_l2.weight = -2.5e-7
        r.dof_acc_l2.params = {"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINTS)}
        r.action_rate_l2.weight = -0.01
        r.feet_air_time = None
        r.dof_pos_limits.weight = -5.0
        r.dof_pos_limits.params = {"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINTS)}
        r.flat_orientation_l2.weight = -1.0
        r.leg_pose = RewTerm(func=mdp.joint_deviation_l1, weight=-0.5,
                             params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINTS)})
        r.wheel_acc = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-9,
                              params={"asset_cfg": SceneEntityCfg("robot", joint_names=WHEEL_JOINTS)})
        r.undesired_contacts = RewTerm(
            func=mdp.undesired_contacts, weight=-1.0,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["base", ".*_hip", ".*_thigh", ".*_calf"]),
                    "threshold": 1.0})


@configclass
class Go2WRoughPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 3000
    save_interval = 100
    experiment_name = "go2w_rough"
    policy = RslRlPpoActorCriticCfg(init_noise_std=1.0, actor_obs_normalization=False, critic_obs_normalization=False,
                                    actor_hidden_dims=[512, 256, 128], critic_hidden_dims=[512, 256, 128],
                                    activation="elu")
    algorithm = RslRlPpoAlgorithmCfg(value_loss_coef=1.0, use_clipped_value_loss=True, clip_param=0.2,
                                     entropy_coef=0.01, num_learning_epochs=5, num_mini_batches=4, learning_rate=1.0e-3,
                                     schedule="adaptive", gamma=0.99, lam=0.95, desired_kl=0.01, max_grad_norm=1.0)


# ------------------------------------------------------------------------------------------------- sim-to-real variant
# Go2WM-Velocity-Rough-Go2W-DR-v0: the same robot / rewards with domain randomization aimed at the real Go2-W and
# sidewalk terrain added to the mix. Randomized: robot friction (static 0.3-1.25, dynamic <= static), base mass -1..+4 kg
# (cameras, bracket, battery), base centre of mass (front-heavy sensor mounts), leg PD gains and wheel damping x0.8-1.2
# (motor strength), action delay 0-2 policy steps (0-40 ms, per env, resampled on reset), pushes every 8-12 s.
# Terrain: Isaac Lab's rough mix plus random 8 m crops of the Trimble X7 sidewalk scan (flipped at random) and
# random sidewalk tiles (slabs, grades, cross-slope, curbs; go2wm.sim.terrain.random_sidewalk_tile).

import numpy as np  # noqa: E402
import torch  # noqa: E402
from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg, JointVelocityActionCfg  # noqa: E402
from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction, JointVelocityAction  # noqa: E402
from isaaclab.managers import EventTermCfg as EventTerm  # noqa: E402

SCAN_HEIGHTMAP = "D:/go2wm/scan/sidewalk_tdx/heightmap.npz"


class _DelayMixin:
    """Applies the policy action from `delay` steps ago (per env, resampled on reset)."""

    def _init_delay(self):
        self._dbuf = torch.zeros(self.cfg.max_delay + 1, self.num_envs, self.action_dim, device=self.device)
        self._delay = torch.randint(self.cfg.min_delay, self.cfg.max_delay + 1, (self.num_envs,), device=self.device)
        self._env_idx = torch.arange(self.num_envs, device=self.device)

    def _delayed(self, actions: torch.Tensor) -> torch.Tensor:
        self._dbuf = torch.roll(self._dbuf, 1, 0)
        self._dbuf[0] = actions
        return self._dbuf[self._delay, self._env_idx]

    def _reset_delay(self, env_ids):
        ids = slice(None) if env_ids is None else env_ids
        n = self.num_envs if env_ids is None else len(env_ids)
        self._dbuf[:, ids] = 0.0
        self._delay[ids] = torch.randint(self.cfg.min_delay, self.cfg.max_delay + 1, (n,), device=self.device)


class DelayedJointPositionAction(_DelayMixin, JointPositionAction):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._init_delay()

    def process_actions(self, actions):
        super().process_actions(self._delayed(actions))

    def reset(self, env_ids=None):
        super().reset(env_ids)
        self._reset_delay(env_ids)


class DelayedJointVelocityAction(_DelayMixin, JointVelocityAction):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._init_delay()

    def process_actions(self, actions):
        super().process_actions(self._delayed(actions))

    def reset(self, env_ids=None):
        super().reset(env_ids)
        self._reset_delay(env_ids)


@configclass
class DelayedJointPositionActionCfg(JointPositionActionCfg):
    class_type: type = DelayedJointPositionAction
    min_delay: int = 0
    max_delay: int = 2


@configclass
class DelayedJointVelocityActionCfg(JointVelocityActionCfg):
    class_type: type = DelayedJointVelocityAction
    min_delay: int = 0
    max_delay: int = 2


def _sidewalk_sub_terrains(scan_file: str = SCAN_HEIGHTMAP, scan_prop: float = 0.15, tile_prop: float = 0.15):
    """Height-field sub-terrains: random crops of the measured scan and random sidewalk tiles (pure numpy)."""
    from isaaclab.terrains.height_field.hf_terrains_cfg import HfTerrainBaseCfg
    from isaaclab.terrains.height_field.utils import height_field_to_mesh

    from .terrain import _load_heightmap, heightmap_height, params_with_defaults, random_sidewalk_tile, sidewalk_height

    def _grid(cfg):
        nx, ny = int(round(cfg.size[0] / cfg.horizontal_scale)), int(round(cfg.size[1] / cfg.horizontal_scale))
        return (np.arange(nx) * cfg.horizontal_scale - 0.5 * cfg.size[0],
                np.arange(ny) * cfg.horizontal_scale - 0.5 * cfg.size[1])

    @height_field_to_mesh
    def _scan_crop(difficulty: float, cfg) -> np.ndarray:
        z, res, x0, _ = _load_heightmap(cfg.scan_file)
        xs, ys = _grid(cfg)
        L = z.shape[0] * res
        cx = np.random.uniform(x0 + 0.5 * cfg.size[0], x0 + L - 0.5 * cfg.size[0])
        sx, sy = np.random.choice([-1.0, 1.0], 2)
        h = heightmap_height(cx + sx * xs[:, None], sy * ys[None, :], cfg.scan_file)
        h = h - np.median(h)
        return np.round(h / cfg.vertical_scale).astype(np.int16)

    @height_field_to_mesh
    def _sidewalk_tile(difficulty: float, cfg) -> np.ndarray:
        p = params_with_defaults(random_sidewalk_tile(np.random.default_rng(np.random.randint(1 << 31))))
        xs, ys = _grid(cfg)
        cx = np.random.uniform(4.0, 9.0)  # features lie at x 2-12 in tile coordinates
        h = sidewalk_height(cx + xs[:, None], ys[None, :], p, grid=cfg.horizontal_scale)
        h = h - np.median(h)
        return np.round(h / cfg.vertical_scale).astype(np.int16)

    hm_path = scan_file

    @configclass
    class ScanCropCfg(HfTerrainBaseCfg):
        function = _scan_crop
        scan_file: str = hm_path

    @configclass
    class SidewalkTileCfg(HfTerrainBaseCfg):
        function = _sidewalk_tile

    return {"scan_crop": ScanCropCfg(proportion=scan_prop, border_width=0.25),
            "sidewalk_tile": SidewalkTileCfg(proportion=tile_prop, border_width=0.25)}


@configclass
class Go2WRoughDREnvCfg(Go2WRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.actions.joint_pos = DelayedJointPositionActionCfg(
            asset_name="robot", joint_names=LEG_JOINTS, scale={".*_hip_joint": 0.125, ".*_(thigh|calf)_joint": 0.25},
            use_default_offset=True, preserve_order=True)
        self.actions.joint_vel = DelayedJointVelocityActionCfg(
            asset_name="robot", joint_names=WHEEL_JOINTS, scale=5.0, use_default_offset=True, preserve_order=True)
        ev = self.events
        ev.physics_material.params.update(static_friction_range=(0.3, 1.25), dynamic_friction_range=(0.25, 1.0),
                                          restitution_range=(0.0, 0.1), make_consistent=True)
        ev.add_base_mass.params["mass_distribution_params"] = (-1.0, 4.0)
        ev.base_com = EventTerm(func=mdp.randomize_rigid_body_com, mode="startup",
                                params={"asset_cfg": SceneEntityCfg("robot", body_names="base"),
                                        "com_range": {"x": (-0.03, 0.06), "y": (-0.03, 0.03), "z": (-0.02, 0.02)}})
        ev.leg_gains = EventTerm(func=mdp.randomize_actuator_gains, mode="reset",
                                 params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINTS),
                                         "stiffness_distribution_params": (0.8, 1.2),
                                         "damping_distribution_params": (0.8, 1.2), "operation": "scale"})
        ev.wheel_gains = EventTerm(func=mdp.randomize_actuator_gains, mode="reset",
                                   params={"asset_cfg": SceneEntityCfg("robot", joint_names=WHEEL_JOINTS),
                                           "damping_distribution_params": (0.8, 1.2), "operation": "scale"})
        ev.push_robot = EventTerm(func=mdp.push_by_setting_velocity, mode="interval", interval_range_s=(8.0, 12.0),
                                  params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}})
        gen = self.scene.terrain.terrain_generator
        if gen is not None:
            gen.sub_terrains = {**gen.sub_terrains, **_sidewalk_sub_terrains()}


@configclass
class Go2WRoughDRPPORunnerCfg(Go2WRoughPPORunnerCfg):
    experiment_name = "go2w_rough_dr"


TASK_ROUGH = "Go2WM-Velocity-Rough-Go2W-v0"
TASK_ROUGH_DR = "Go2WM-Velocity-Rough-Go2W-DR-v0"
for _tid, _env, _run in ((TASK_ROUGH, "Go2WRoughEnvCfg", "Go2WRoughPPORunnerCfg"),
                         (TASK_ROUGH_DR, "Go2WRoughDREnvCfg", "Go2WRoughDRPPORunnerCfg")):
    if _tid not in gym.registry:
        gym.register(id=_tid, entry_point="isaaclab.envs:ManagerBasedRLEnv", disable_env_checker=True,
                     kwargs={"env_cfg_entry_point": f"{__name__}:{_env}",
                             "rsl_rl_cfg_entry_point": f"{__name__}:{_run}"})
