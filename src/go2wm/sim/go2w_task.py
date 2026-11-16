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


TASK_ROUGH = "Go2WM-Velocity-Rough-Go2W-v0"
if TASK_ROUGH not in gym.registry:
    gym.register(id=TASK_ROUGH, entry_point="isaaclab.envs:ManagerBasedRLEnv", disable_env_checker=True,
                 kwargs={"env_cfg_entry_point": f"{__name__}:Go2WRoughEnvCfg",
                         "rsl_rl_cfg_entry_point": f"{__name__}:Go2WRoughPPORunnerCfg"})
