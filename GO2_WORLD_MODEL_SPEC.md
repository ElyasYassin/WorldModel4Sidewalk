# Go2 Latent World Model — Engineering Specification

**Status:** MVP implementation brief  
**Robot:** Unitree Go2 EDU  
**Primary simulator:** NVIDIA Isaac Sim / Isaac Lab, using Unitree RL Lab  
**Core research direction:** JEPA-style, action-conditioned latent prediction for local navigation under a conventional long-horizon global planner  
**Primary implementation language:** Python 3.11+ (adapt to the installed Isaac Lab environment)  
**Audience:** Claude Code, implementing engineer, research team

## 0. Your assignment (Claude Code)

Build a reproducible research repository that connects a **Unitree Go2 simulation**, a **timestamped episode dataset**, a **PyTorch latent dynamics world model**, and a **hybrid global/local planner**. The first working system should plan local Go2 velocity commands in simulation, while a conventional waypoint planner handles long-range navigation. Architect the modules so a **pretrained V-JEPA 2 visual encoder** and a later **hierarchical latent planner** can be substituted without redesigning the rest of the system.

**Important:** Deliver *working software*, not only architecture documents. Implement in vertical slices: create a small working end-to-end loop first, then add features. Do not claim Go2 simulation integration or real-robot compatibility is tested unless actually executed. If Isaac Lab is unavailable on the current machine, implement and test a deterministic, lightweight mock backend and provide concrete integration steps; do not pretend mock results validate locomotion.

### Initial priorities, in order
1. Verify environment and official upstream repositories; pin the versions actually used.
2. Get Go2 standing/walking in simulation using a supported existing controller or locomotion policy (do not train gait control from scratch).
3. Export consistent, timestamped observation/action/next-observation transitions and complete episodes.
4. Train and evaluate a **state-only action-conditioned predictive baseline**.
5. Add **frozen V-JEPA 2 visual feature extraction** behind an encoder interface; train a latent predictor when image data is available.
6. Implement a basic map/waypoint global planner plus a receding-horizon local planner; compare against a reactive/geometric controller.
7. Keep hierarchical representation learning, Cosmos integration, and real-robot actuation outside MVP.

## 1. Research problem and success criterion

**Question:** Can an action-conditioned latent world model improve data efficiency and local navigation safety/traversability for a Go2 quadruped in novel sidewalk-like environments, while a conventional global planner handles long-horizon route completion?

**Initial hypotheses:**
- H1: Visual features plus proprioception predict navigational outcomes on variable terrain better than proprioception alone.
- H2: Short-horizon learned model predictive control (MPC) improves selected navigation outcomes relative to a non-learned local baseline under controlled test conditions.
- H3 (later): Hierarchical latent subgoals improve longer-horizon performance, but only if subgoals are reachable by the local controller.

**MVP acceptance:** An executable pipeline can (a) run a Go2 locomotion baseline in Isaac Lab or clearly report why the environment cannot support it, (b) collect episodes with aligned state/action/next state and optional image observations, (c) train a state dynamics model on a dataset, (d) evaluate one-step and rollout error on held-out episodes, and (e) navigate to waypoints using an MPC-style local planner in a *supported simulation backend*, with a nonlearned controller as baseline. A mock backend counts only as a software smoke test, not as proof of Go2 task success.

## 2. Research architecture

```text
   Goal / inspection route + map
             |
             v
   Global graph/waypoint planner                 [long horizon, meters/minutes]
             |
       next reachable waypoint
             |
             v
   Local receding-horizon planner (MPC/CEM)     [short horizon, e.g. 1-3 s]
       |                  ^
       |candidate actions | predicted costs/latents
       v                  |
   Action-conditioned latent world model
       ^                 ^
       |                 |
   Vision encoder     Proprioception encoder
    (V-JEPA 2,        (pose/IMU/velocity/contacts
     optional)         where available)
       ^                 ^
       |                 |
      RGB              Go2 state
             |
             v
     Existing Go2 locomotion policy / high-level velocity interface
             |
             v
        Isaac Lab simulation
             |
       new observations (replan)
```

**Not end-to-end gait learning.** The planner outputs high-level commanded `(vx, vy, yaw_rate)` in the robot body frame (or the supported subset of these values), passed through a known supported locomotion controller. The world model is **non-generative**: it predicts latent future states, physical states, and optionally risk/reachability—not future RGB frames. Keep image reconstruction out of MVP.

### 2.1 Distinguish the models
- **State baseline:** train `f(s_t, a_t) -> delta_s` (optionally history/GRU). No pretrained model needed. Required first.
- **Visual latent model:** freeze official V-JEPA 2 encoder `E(rgb_window) -> z_visual`, fuse `z_visual` with normalized proprioception, and learn action-conditioned predictor `P(z_t, a_t) -> z_(t+1)`; use appropriate stop-gradient/target embeddings in training. This is a custom Go2 predictor, **not** an assertion that standard V-JEPA 2 is already action-conditioned for Go2.
- **Optional later comparison:** a small end-to-end learned encoder with anti-collapse regularization. Do not implement unverified papers or claim equivalence to LeWM without reproducing its exact method.
- **Later hierarchy:** separate abstract global/subgoal latent representation from short-horizon physically grounded prediction. **Not required for MVP**.

## 3. Technology and upstream integration

Preferred:
- Official Unitree RL Lab: https://github.com/unitreerobotics/unitree_rl_lab
- Official Isaac Lab: https://github.com/isaac-sim/IsaacLab
- Official Go2 assets: `unitree_ros` URDF or Unitree model/USD assets; choose based on installed Isaac Sim version and the official Unitree RL Lab README.
- Official V-JEPA 2: https://github.com/facebookresearch/vjepa2
- Official Unitree ROS 2 (read-only telemetry for **future** hardware work): https://github.com/unitreerobotics/unitree_ros2

**Integration rules:**
1. Inspect upstream READMEs and source files **before** assuming environment/task names, checkpoint format, observation structure, ROS topic names, or Go2 asset paths. Unitree example task names in documentation may refer to other robots.
2. Keep external repositories pinned via documented commits/tags. Do not fork or modify upstream repositories unless required; prefer adapters in this repository.
3. Simulator configuration (Hz, camera resolution, environment count, control limits) must be configured, not hard-coded.
4. Provide capability detection: no GPU/Isaac installed -> clear warning and mock smoke tests; missing pretrained checkpoints -> state-only baseline continues to work.
5. No robot IP, network command, hardware execution, or low-level joint actuation in MVP.

## 4. Repository structure

```text
go2-latent-world-model/
├── README.md
├── SPEC.md                       # copy of this specification
├── pyproject.toml
├── configs/
│   ├── mock.yaml
│   ├── isaac_go2.yaml
│   ├── data.yaml
│   ├── world_model_state.yaml
│   ├── world_model_visual.yaml
│   └── planner.yaml
├── src/go2wm/
│   ├── common/
│   │   ├── schema.py             # dataclasses / validation for transitions
│   │   ├── frames.py             # frame conventions + transforms
│   │   ├── config.py
│   │   └── seeds.py
│   ├── sim/
│   │   ├── base.py               # SimulatorAdapter protocol
│   │   ├── mock.py               # deterministic integration-test simulator
│   │   ├── isaac_go2.py          # Unitree RL Lab adapter
│   │   └── policies.py           # existing locomotion policy interface
│   ├── data/
│   │   ├── recorder.py
│   │   ├── episodes.py
│   │   ├── dataset.py
│   │   └── validate.py
│   ├── models/
│   │   ├── proprio_encoder.py
│   │   ├── vjepa_encoder.py
│   │   ├── state_dynamics.py
│   │   ├── latent_dynamics.py
│   │   ├── losses.py
│   │   └── normalization.py
│   ├── planning/
│   │   ├── global_waypoints.py
│   │   ├── local_mpc.py
│   │   ├── costs.py
│   │   ├── baseline.py
│   │   └── safety.py
│   ├── eval/
│   │   ├── prediction.py
│   │   ├── navigation.py
│   │   └── plots.py
│   └── cli.py
├── scripts/
│   ├── verify_environment.py
│   ├── collect.py
│   ├── extract_vjepa.py
│   ├── train.py
│   ├── evaluate.py
│   └── run_navigation.py
├── tests/
│   ├── test_schema.py
│   ├── test_frames.py
│   ├── test_recorder.py
│   ├── test_dataset.py
│   ├── test_dynamics.py
│   ├── test_mpc.py
│   └── test_mock_end_to_end.py
├── data/                         # gitignored
├── outputs/                      # gitignored
└── docs/
    ├── INSTALL.md
    ├── ISAAC_INTEGRATION.md
    ├── DATA_FORMAT.md
    ├── EXPERIMENTS.md
    └── HARDWARE_HANDOFF.md
```

Simplify the structure if needed, but preserve distinct simulator, data, model, planning, and evaluation interfaces.

## 5. Simulator interface and Go2 pipeline

Design a protocol independent of Isaac Lab:

```python
class SimulatorAdapter(Protocol):
    def reset(self, seed: int, scene_id: str | None = None) -> Observation: ...
    def step(self, action: VelocityCommand) -> StepResult: ...
    def close(self) -> None: ...
```

- `Observation`: timestamp, source/environment identifiers, pose/orientation or odometry, linear/angular velocity, IMU/joints if exposed, optional RGB/depth, simulator ground truth for evaluation only, and sensor validity masks.
- `VelocityCommand`: `vx_m_s`, `vy_m_s`, `yaw_rate_rad_s`, `duration_s` (or number of control steps) with explicit body-frame convention.
- `StepResult`: next observation, collision/contact/termination status, task progress, diagnostics.

### 5.1 Simulator tasks
1. Verify official Go2 task exists, inspect its observation and action interfaces, and load the correct robot asset.
2. Load an existing locomotion controller/policy that reliably follows velocity commands; if no ready policy checkpoint exists, document and implement the minimum supported path to produce one or use a compatible verified controller. Do **not** fake a policy.
3. Begin with simple flat ground, then introduce slopes, small obstacles and sidewalk geometry.
4. Build seeded scene variations: lighting, textures, obstacle positions, terrain heights/friction, sensor noise. Log each environment configuration in episode metadata.
5. For visually conditioned training, attach compatible simulated RGB camera(s) and log extrinsics, intrinsics, FOV, resolution, and frame timestamps. Verify the camera is part of the simulated robot and doesn't use privileged tracking.
6. Use asynchronous/batched recording if necessary, and measure actual data-generation throughput. Parallel simulation hours are **aggregate simulated hours**; distinguish them from rendered video hours and wall time.

### 5.2 Recommended starting configuration (tunable, not assumed compatible)
- High-level command/control updates: 10 Hz.
- World-model transition sample interval: 0.1 seconds.
- RGB recordings: start 5–10 FPS at approximately 224–256 pixel scale; adjust after measuring rendering/VRAM.
- Physics timestep: inherit validated Unitree locomotion settings; do not force it equal to observation frequency.
- Initial scenes: flat surface, turns, obstacle bypass, small elevation changes.
- Initial data target: 30–60 minutes varied simulated episodes; scale only after dataset and prediction baselines pass tests.
- Keep every observation timestamp and both actual `dt` and intended action duration.

## 6. Canonical trajectory dataset

Store one file per episode (`.npz` or `.h5` plus metadata `.json`, or another efficient portable format with equivalent documented schema). Avoid saving raw images uncompressed inside giant monolithic NumPy files. The loader must support missing camera modalities with explicit availability masks.

**Mandatory per transition:**

```text
episode_id, scene_id, seed
observation_timestamp_ns, action_timestamp_ns, next_observation_timestamp_ns, dt_s
state_t              # documented numerical state vector, units and frame
command_t            # actual high-level locomotion command; body frame
state_t1             # next observed state
terminated, truncated, termination_reason
collision_flag, progress_to_waypoint_m (where available)
```

**Optional:** `rgb_t`, `rgb_t1`, `depth_t`, `imu_t`, `joint_state_t`, `foot_contacts_t`, `terrain_metadata`, `ground_truth_base_pose_t`, `camera_calibration`, `sensor_valid_mask`.

**Frame policy:** adopt a fixed convention (e.g. world frame ENU, base-frame FLU) and record it explicitly; distinguish commanded velocity, realized velocity, and world-frame pose. Rotation must use quaternions or sin/cos for heading, **never raw wrapped yaw subtraction**. Normalize relative displacement into a local frame for dynamics learning and avoid direct extrapolation of absolute global coordinates. Always handle angle wraparound.

**Temporal alignment:** `a_t` must be the command actually applied between `o_t` and `o_(t+1)`, not a command emitted later. Handle actuator latency and frame drop separately. Reject negative/zero intervals, episode boundary leakage, missing fields, and inconsistent units. Put all transitions from an episode in exactly one data split.

**Train/val/test:** split by episode **and by scene family** for generalization experiments, not by random adjacent frames. Record split manifests and seeds. Keep a final unseen test set untouched until evaluation.

## 7. Baseline world model (must work first)

Start from a compact normalized robot state such as body-frame planar velocity, angular rate, relative pose/heading, IMU, and optional contacts. Avoid privileged variables in deployed inference; privileged data can be evaluation-only or a separate oracle model.

```python
# interface sketch
pred = model.predict(
    state_history,       # [B, T, D_s]
    action_history,      # [B, T, D_a]
    candidate_actions,   # [B, H, D_a]
    dt,                  # actual sampling interval(s)
)
# state predictions: [B, H, D_target]
```

- Implement delta-state dynamics with an MLP first; add a GRU/temporal variant where appropriate.
- Train with MSE/Huber on normalized changes and optional multi-step rollout loss, clipping gradients and checkpointing best validation performance.
- Include a constant-velocity / unicycle kinematics baseline and a geometric collision/obstacle avoidance baseline.
- For multi-step evaluations, explicitly distinguish teacher-forced one-step prediction from **open-loop autoregressive** predictions.
- Log training loss, held-out next-state MAE/RMSE, position error in meters, heading error in radians/degrees, and error at 0.5/1/2/3 seconds (as supported by sample rate).
- Keep train/eval deterministic with fixed seeds. No future-state leakage into inputs.

## 8. V-JEPA 2 visual world model (second implementation)

### 8.1 Scope
- Load official pretrained V-JEPA 2 visual encoder using official repository/weights and documented preprocessing. Do not assume arbitrary single-frame input shapes; implement temporal clip sampling compatible with the chosen checkpoint.
- Freeze pretrained encoder initially, set eval mode, and compute embeddings without gradients.
- Cache embeddings **with source episode IDs, timestamps, window boundaries, checkpoint hash, sampling recipe, and preprocessing version**. Make cache regeneration deterministic.
- Fuse visual embedding with proprioception in a small learned projection/fusion module.
- Train a custom action-conditioned latent predictor (GRU or transformer, starting small). Predict target embeddings for future observed clips, optionally with a supervised physical-state/risk head.
- Prevent trivial latent collapse when training any encoder end-to-end; frozen target embeddings alone simplify the initial experiment. Do not label a simple representation-regression head a complete LeWM reproduction.

### 8.2 Predictive objective

```text
z_t          = fuse(E_video(rgb_context_t), E_proprio(proprio_t))
zhat_(t+k)   = P(z_t, action_sequence_t:t+k-1, dt_sequence)
z_target     = stopgrad(E_target(rgb_context_(t+k)))
L            = lambda_latent * latent_distance(zhat_visual, z_target)
             + lambda_state  * physical_state_loss
             + lambda_risk   * labeled_risk_loss (if valid labels exist)
```

- The predictor must know action durations and the relevant action sequence.
- Train separate models for state-only and state+visual to support ablation.
- Compare latent cosine distance / normalized error but **do not treat low latent loss as proof of planning utility**.
- Test which task-critical variables (distance-to-obstacle, traversability, realized motion) can be decoded or probed from embeddings without using future information.
- Record frozen weights, caching strategy, GPU memory requirements, and throughput. Missing V-JEPA weights must not block state-model development.

## 9. Hybrid long-horizon planning

### 9.1 Global planner
- Implement a simple 2D occupancy-grid A* (or load validated waypoint sequence) for route-level navigation.
- Global route produces spatial waypoints, tracks progress, and triggers reroute when local path is blocked.
- Initially use known maps in simulation; log map assumptions. Do not require learned high-level JEPA in MVP.

### 9.2 Local world-model planner
- Implement receding-horizon optimization (CEM preferred initially, optionally random shooting) over sequences of bounded body-frame velocity commands.
- Evaluate candidates with predicted motion, waypoint progress, collision/terrain-risk output **only if supervised and validated**, smoothness/control cost, and uncertainty penalty if calibrated.
- Use short horizons first (e.g. 1–2 s), select first action, observe, replan. Avoid trusting open-loop rollouts for long durations.
- Include a kinematics/MPC baseline and a simple reactive/geometric local controller for fair comparison.

Illustrative objective:

```text
J = w_goal * predicted_waypoint_distance
  + w_collision * predicted_collision_cost
  + w_stability * predicted_instability_cost
  + w_smooth * control_change_penalty
  + w_uncertainty * calibrated_model_uncertainty
```

**Never include risk terms backed by invented labels or untrained heads**; if a term is unavailable, omit it explicitly and fall back to conservative geometric safeguards.

### 9.3 Safety constraints
- Bound `vx`, `vy`, yaw rate, acceleration, action duration and maximum episode distance per validated simulator/controller capability.
- Treat simulator/robot safety envelope and obstacle proximity rules as hard constraints in addition to learned costs.
- Stop or defer to a safe fallback if no feasible candidate, sensors fail, predictions are invalid, or model uncertainty exceeds calibrated thresholds.
- Do not use learned control for motor torques or joint positions in the initial project.

## 10. Evaluation and experiments

### 10.1 Prediction tests
- One-step vs open-loop 0.5/1/2/3 s (and longer only if meaningful).
- Held-out command patterns, velocities and turns.
- Held-out terrain/texture/obstacle arrangements.
- Performance by scenario (flat, incline, curb-like bump, obstructed sidewalk).
- Simulated perturbations: changes in friction, payload (if supported), camera noise, delays, dropped observations.
- Per-episode error distributions with mean, median, confidence intervals and failure cases; report sample counts.

### 10.2 Navigation comparison
- A: A*/waypoints + conventional local controller.
- B: A*/waypoints + state-only learned MPC.
- C: A*/waypoints + V-JEPA latent learned MPC.
- D (future): H-JEPA-like hierarchical latent planning.

Use **identical test scenarios and random seeds** across baselines. Measure:
- Goal completion rate, collisions, falls, emergency stops.
- Route efficiency/path length, time-to-goal, stall frequency.
- Minimum clearance and terrain hazard exposure.
- Planner latency (p50/p95), hardware/GPU utilization, inference memory.
- World-model rollout error by prediction horizon.

Avoid declaring superiority without sufficient repeated runs; report uncertainty and failure cases. Ensure real-world tests are not conflated with simulation benchmarks.

## 11. CLI and expected developer experience

Implement documented commands with these semantics (exact flags may be adapted consistently):

```bash
# Detect system prerequisites, simulator support, checkpoint paths
python scripts/verify_environment.py

# Fast tests that do not require Isaac or pretrained weights
pytest -q

# Collect a deterministically generated mock episode batch (smoke testing only)
python scripts/collect.py --backend mock --episodes 20 --out data/mock

# Collect Go2 episodes from actual Isaac Lab, only if dependency available
python scripts/collect.py --backend isaac --config configs/isaac_go2.yaml --out data/go2_sim

# Validate timestamps, shapes, split manifests, metadata, and consistency
python -m go2wm.data.validate --data data/go2_sim

# Train and evaluate baseline dynamics
python scripts/train.py --config configs/world_model_state.yaml --data data/go2_sim
python scripts/evaluate.py --checkpoint outputs/state/best.pt --data data/go2_sim --split test

# Optional V-JEPA embedding extraction and latent predictor
python scripts/extract_vjepa.py --data data/go2_sim --config configs/world_model_visual.yaml
python scripts/train.py --config configs/world_model_visual.yaml --data data/go2_sim

# Evaluate a waypoint-navigation scenario, compare baselines
python scripts/run_navigation.py --backend isaac --planner conventional --config configs/planner.yaml
python scripts/run_navigation.py --backend isaac --planner learned_mpc --config configs/planner.yaml
```

Where functionality is not yet implemented, CLI must fail with an actionable message—not silently generate fabricated data.

## 12. Tests and acceptance criteria

**Unit tests:**
- Frames: world/body transformations, heading wraparound, quaternion normalization.
- Data: aligned observation/action transitions, terminal boundaries, unique episode IDs, deterministic splits, missing modalities.
- Model: output shape and gradient flow for batch/time dimensions, checkpoint load/save, reproducible forward pass.
- MPC: respects controls and smoothness limits, collision/geometric fallback, selects progress-making action in simple deterministic case.

**Integration tests:**
- Mock backend -> data collection -> dataset loading -> train -> evaluate -> mock waypoint navigation.
- Isaac adapter smoke test on supported GPU/Isaac installation; capture environment version, task ID, robot model and policy checkpoint provenance.
- Real Go2 locomotion and data logging are separate hardware milestones; not part of today’s acceptance.

**Definition of done for first sprint:**
- [ ] README instructions and pinned dependency versions.
- [ ] Existing Go2 locomotion demonstrably works in Isaac Lab (or blocker recorded precisely).
- [ ] At least one valid recorded Isaac Go2 episode (if environment available).
- [ ] Data schema validator passes and splits are episode/scene-disjoint.
- [ ] State-only baseline trains and outputs multi-step prediction plots.
- [ ] Minimal local MPC can be compared against a conventional controller in a supported backend.
- [ ] V-JEPA integration is implemented as an optional feature or documented as next step, never silently mocked as pretrained predictions.
- [ ] All mock-mode tests pass without Isaac Lab.

## 13. Tomorrow's real Go2 EDU handoff (read-only first)

Prepare `docs/HARDWARE_HANDOFF.md` with:
- Firmware/version, EDU variant, networking and onboard compute information to confirm on site.
- How to discover official state topics using supported Unitree DDS/ROS 2 tooling.
- Read-only logging plan: timestamps, state/IMU/joints and available image/depth/LiDAR interfaces.
- Time synchronization and camera calibration checklist.
- Real/simulation frame and action schema parity checks.
- Human operator, remote stop, clear space, speed limits, and approved high-level test command procedures.

**Absolutely no automatic hardware actuation, remote network discovery, low-level motor control, or autonomous real-world deployment in the implementation sprint.** Tomorrow's first milestone is telemetry validation and a supervised manual safety check, not real-world learned control.

## 14. Execution strategy for Claude Code

Proceed independently with reasonable defaults, but report constraints rather than guess. Follow this work sequence:

**Phase A — Inspect.** Check working tree, available GPU/OS, Python environment, disk; read official upstream docs, find installed Isaac versions/Go2 tasks/checkpoints. Write a short `docs/INSTALL.md` with verified facts and blockers.

**Phase B — Vertical slice.** Create Python package, canonical schema, mock simulator, deterministic collector, state baseline, training CLI, evaluation CLI and tests. Actually run tests and smoke training.

**Phase C — Isaac.** Add a Go2 adapter to official Unitree environment with correct policy/command interface, collect physically meaningful episodes, validate. If no working locomotion policy exists, stop at a clearly reported blocker rather than treating simulated robot positions as actual Go2 physics.

**Phase D — Planning.** Implement waypoint/global path and short-horizon local controller. Compare nonlearned and learned versions on identical seeded scenarios.

**Phase E — Vision.** Add camera logging and official frozen V-JEPA 2 embeddings. Train the action predictor, produce ablations. Do not make this block A–D.

After each phase: run relevant tests, update README, list files changed, report empirical outputs and limitations. Keep source code clean, typed where practical, modular, with minimal dependencies and no inaccessible proprietary services.

## 15. Decisions / non-goals

**Decided:** JEPA-style latent rather than generative frame prediction; Isaac Lab + Go2; existing velocity-following locomotion; global waypoints + local predictive MPC; frozen pretrained visual encoder first; simulation-first; real hardware telemetry later.

**Not decided:** workstation GPU, Isaac version, visual checkpoint size, exact Go2 camera configuration, Go2 policy checkpoint, robot firmware and sensor rate. Probe capabilities and parameterize these rather than hardcoding guesses.

**Not in MVP:** Cosmos video generation, full LeWM or H-JEPA reproduction, training a locomotion policy from zero, joint torque control, autonomous physical deployment, fully learned city-scale map-free planning, or claims of proven sim-to-real generalization.

## 16. Key references

- Unitree RL Lab: https://github.com/unitreerobotics/unitree_rl_lab
- Isaac Lab: https://github.com/isaac-sim/IsaacLab
- V-JEPA 2: https://github.com/facebookresearch/vjepa2
- Unitree ROS 2: https://github.com/unitreerobotics/unitree_ros2
- NVIDIA Isaac Lab documentation: https://isaac-sim.github.io/IsaacLab/

---

**Final instruction to Claude Code:** Begin implementation now. First inspect available hardware/software and report blockers. Then execute Phase B and as much of C/D/E as the environment allows. Prefer tested, reproducible code over ambitious but unexecuted architecture. At the end provide exact run commands, test results, files created, current blockers, and the next 3 actionable tasks.
