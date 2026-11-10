# Episode data format (schema 1.0)

One episode = `<root>/episodes/<episode_id>.npz` (compressed arrays) + `<episode_id>.json` (metadata).
Split manifests live in `<root>/splits/<name>.json`. Validate with:

```
python -m go2wm.data.validate --data <root>
```

## Transitions

An episode with T transitions stores T+1 observations. **Transition i is
`(states[i], commands[i], states[i+1])`**: `commands[i]` is the command actually
applied (after backend clipping) between observation i and i+1, never one issued later.

| array | shape | dtype | meaning |
|---|---|---|---|
| `states` | [T+1, 16] | f64 | canonical state, layout below |
| `obs_timestamp_ns` | [T+1] | i64 | simulator time of each observation (strictly increasing) |
| `commands` | [T, 4] | f64 | `vx_m_s, vy_m_s, yaw_rate_rad_s, duration_s` (body frame FLU) |
| `action_timestamp_ns` | [T] | i64 | when command i was applied; must lie in `[t_obs[i], t_obs[i+1])` |
| `dt_s` | [T] | f64 | actual interval `t_obs[i+1] - t_obs[i]` (not the nominal period) |
| `collision` | [T] | bool | footprint intersected an obstacle/boundary during transition i |
| `terminated` | [T] | bool | true only on the final transition, if the episode ended in a terminal state |
| `truncated` | [T] | bool | true only on the final transition, if the episode hit its step limit |
| `termination_code` | [T] | i8 | 0 none, 1 collision, 2 fall, 3 out_of_bounds, 4 goal, 5 timeout, 6 error |
| `progress_to_waypoint_m` | [T] | f64 | decrease in distance to the scene goal over transition i (NaN if no goal) |
| `distance_to_goal_m` | [T+1] | f64 | distance from base to the scene goal |
| `gt_clearance_m` | [T+1] | f64 | **ground truth, evaluation only**: footprint clearance to the known map |

### State vector (16)

| slice | field | frame / units |
|---|---|---|
| 0:3 | `pos_w` | world, m (relative to the env origin in Isaac) |
| 3:7 | `quat_w` | wxyz, body to world |
| 7:10 | `lin_vel_b` | realized base linear velocity, body frame, m/s |
| 10:13 | `ang_vel_b` | realized base angular velocity, body frame, rad/s |
| 13:16 | `gravity_b` | unit gravity direction in body frame ((0,0,-1) when level) |

### Frames

World is right-handed and z-up. Body is FLU (x forward, y left, z up). The convention
is copied into every episode's `frame_convention` metadata. Commanded velocity
(`commands`) and realized velocity (`lin_vel_b`/`ang_vel_b`) are stored separately.

## Model features (derived, not stored)

`go2wm.models.features`:
- **Inputs:** `lin_vel_b, ang_vel_b, gravity_b` (9). No absolute position or heading.
- **Targets per step (13):** relative motion `dx_b, dy_b` (in the earlier body frame),
  `dz`, `dyaw`, plus deltas of the 9 features. `dyaw` comes from quaternions via atan2 of
  sin/cos; wrapped yaw values are never subtracted directly.

## Metadata (`.json`)

`schema_version, episode_id, scene_id, scene_family, seed, backend, created_utc,
frame_convention, state_fields, command_fields, modalities, scene (full SceneSpec:
bounds, obstacles, start, goal, params), simulator (adapter metadata: task, policy
checkpoint, dt values, config), collector, num_transitions, duration_s, termination_reason`.

## Splits

`make_split_manifest` assigns **whole scenes** to train/val/test with a fixed seed.
`holdout_families` sends entire scene families to test for generalization runs. The
validator checks that no episode and no scene appears in two splits.

## Optional modalities (not yet recorded)

RGB, depth, and camera calibration are reserved for Phase E. `modalities.rgb` is false
on all current episodes. RGB will be stored as per-episode compressed frame files
referenced from metadata, not inside the `.npz`.
