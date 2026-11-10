# Isaac Lab Go2 integration

## How it is wired

- **Launcher:** `scripts\isaac_python.bat` runs the `IsaacLab` conda Python with the env vars
  that `C:\isaacsim\python.bat` sets (`PYTHONPATH+=site`, `CARB_APP_PATH`, `ISAAC_PATH`,
  `EXP_PATH`). Plain `python` cannot `import isaacsim` on this binary install.
- **Task:** `Isaac-Velocity-Flat-Unitree-Go2-Play-v0`, single env. The asset comes from
  `isaaclab_assets.robots.unitree.UNITREE_GO2_CFG` (`.../Robots/Unitree/Go2/go2.usd`).
- **Locomotion policy:** NVIDIA's published RSL-RL checkpoint for
  `Isaac-Velocity-Flat-Unitree-Go2-v0`, fetched by `isaaclab_rl` and loaded through
  `OnPolicyRunner` exactly as in Isaac Lab's `play.py`. It's an MLP actor
  (48 → 128 → 128 → 128 → 12 joint position targets). We don't train any gait.
- **Command interface:** the `base_velocity` command term is set to direct mode
  (`heading_command=False`, no standing envs, no resampling). The adapter writes
  `vel_command_b = (vx, vy, yaw_rate)` and holds it for `control_dt / step_dt` = 0.1 / 0.02
  = 5 policy steps. Physics runs at 0.005 s with decimation 4, inherited unchanged from the task.
- **Terminations:** Isaac's auto-reset terminations are disabled so a terminal
  observation is never replaced by a reset state. The adapter detects:
  - **fall:** tilt (projected gravity z > −0.5), base below 0.08 m, or base contact force
    > 1 N away from obstacles. Base contact uses the same signal as Isaac Lab's `base_contact`
    termination.
  - **collision:** footprint circle intersects the known map, or base contact within 0.1 m
    of an obstacle.

  A first version used a 0.15 m base-height threshold, which fired falsely: this policy walks
  with its base at 0.15–0.19 m at speed (0.25 m standing). That episode was deleted.
- **Inference mode:** all env interaction (reset and step) runs inside `torch.inference_mode()`,
  as in `play.py`. Mixing modes fails with "Inplace update to inference tensor".
- **Scenes:** the same backend-neutral layouts as the mock (`open`, `obstacles`, `sidewalk`).
  Obstacles are kinematic cylinders (r = 0.3 m, h = 0.6 m) in 8 slots, moved per scene, with
  unused slots parked far away. Sidewalk bounds are virtual (no curb geometry yet). After each
  reset, the simulated obstacle positions are read back and compared to the map. The error is
  logged as `scene.params.obstacle_placement_max_err_m` (measured about 3e-7 m).
- **Timestamps:** simulation time = policy steps × 20 ms (deterministic, no wall clock).

## Environment fixes applied (2026-11-03)

- Installed `cloudpickle` and `hydra-core` into the `IsaacLab` env. Installed packages
  (tensordict, gymnasium, isaaclab-rl) declare both as dependencies, but they were
  missing, so Isaac Lab's own `play.py` could not import rsl_rl either. torch and numpy unchanged.
- Kit's `app.close()` ends the process, so scripts write their outputs before closing.

## E1: velocity-tracking smoke test (`scripts\isaac_python.bat scripts\isaac_smoke.py`)

Verified 2026-11-03 on the RTX 3090. Report: `D:\go2wm\outputs\isaac_smoke\report.json`.
Startup took 14 s, then the run went at about 0.6× real time. **No fall.**

Steady-state realized velocity (mean over the second half of each 3 s segment):

| command (vx, vy, wz) | realized (vx, vy, wz) | verdict |
|---|---|---|
| (0.5, 0, 0) | (0.49, 0.00, 0.01) | tracks |
| (1.0, 0, 0) | (1.02, 0.02, −0.02) | tracks |
| (0.5, 0, 0.5) | (0.48, 0.01, 0.46) | tracks |
| (0, 0.3, 0) | (−0.01, 0.00, −0.03) | **does not move** |
| (0, ±0.6, 0) | (0.00, ±0.61, ∓0.04) | tracks |
| (0, 0, −0.8), (0, 0, 0.5), (0, 0, 1.0) | ≈ (0, 0, 0) | **cannot turn in place** |
| (0.3, 0, 0.8) | (0.18, 0.04, 0.41) | under-tracks |

The command term seen by the policy matched the commanded values in every segment, so this
is the policy's behavior, not an adapter bug.

**Finding:** the pretrained flat Go2 policy has a **deadband on small planar commands**.
Below somewhere between 0.3 and 0.6 m/s it stands still, including for pure yaw commands.
A plausible cause (from source, not verified): in training, yaw always came from the
heading controller, and `feet_air_time` is zeroed when the planar command is under 0.1 m/s.

**Consequences:**
1. Command-kinematics MPC will be systematically wrong for turn-in-place and slow
   commands. A learned dynamics model can capture this from data, which makes this a useful
   H2 test case.
2. If the project needs turn-in-place, the options are (a) evaluate Unitree RL Lab's Go2
   policy, or (b) train an Isaac Lab Go2 flat policy with `heading_command=False` and a wider
   standing/turning command distribution. Training a gait is outside the MVP, so this is a
   decision for the project owner.

## Not yet verified

- Rough-terrain task or policy, slopes, curbs, sidewalk geometry (SidewalkAtlas
  `scripts/terrain/` heightfields are the natural source).
- Cameras: `enable_cameras` is false. A body-mounted RGB camera is Phase E.
- Contact-based collision: collisions are geometric only (footprint circle vs map).
- Multi-env parallel collection (currently 1 env, about 0.6× real time).
