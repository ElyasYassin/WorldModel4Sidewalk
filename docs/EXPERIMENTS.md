# Experiments log

Each entry lists the backend. **Mock-backend numbers are software smoke tests**: the mock is a
first-order-lag planar base, not Go2 physics, and they say nothing about H1/H2.

## E0: mock pipeline smoke test (2026-11-03)

- Data: `python scripts/collect.py --backend mock --episodes 40 --out data/mock`. That gives 40
  episodes and 8,248 transitions in 4.8 s wall, across 4 scene families (open, obstacles, sidewalk,
  slippery). The validator passes.
- Split: scene-disjoint, seed 0. Train 28 / val 6 / test 6 episodes.
- Models: MLP (3×256) and GRU (128) delta dynamics, history 4 (0.4 s), 10-step rollout loss,
  40 epochs, about 40 s each on the RTX 3090.
- Prediction, mock **test split** (751 windows from 6 episodes), mean position error / heading error:

| model | 0.5 s | 1 s | 2 s | 3 s |
|---|---|---|---|---|
| MLP | 0.008 m / 0.7° | 0.018 m / 1.5° | 0.047 m / 3.1° | 0.086 m / 4.6° |
| GRU | 0.008 m / 0.9° | 0.019 m / 1.8° | 0.048 m / 3.2° | 0.079 m / 4.0° |
| command kinematics | 0.032 m / 2.8° | 0.060 m / 4.8° | 0.098 m / 6.4° | 0.133 m / 6.4° |
| constant velocity | 0.039 m / 4.3° | 0.142 m / 15.5° | 0.494 m / 47.7° | 0.928 m / 74.9° |

  The learned models capture the mock's lag, gain and slip terms, which command kinematics
  ignores. That is expected by construction, so it only shows the plumbing works.
- Navigation, mock, 20 identical seeded scenarios (5 per family, seeds 10000–10004, never
  used in training data):

| planner | success [95% CI] | collisions | time to goal (s) | interventions/ep | latency p50 / p95 |
|---|---|---|---|---|---|
| conventional (reactive) | 0.95 [0.76, 0.99] | 0 | 16.8 | 10.8 | 0.2 / 0.3 ms |
| kinematic MPC (CEM) | 0.90 [0.70, 0.97] | 0 | 17.1 | 0.0 | 42 / 56 ms |
| learned MPC (CEM + MLP) | 0.90 [0.70, 0.97] | 0 | 17.6 | 0.0 | 78 / 102 ms |

  All failures are timeouts on narrow `sidewalk` gaps that leave about 0.15 m per side. The
  MPC's 0.3 m clearance margin makes it hover in front of the gap. The reactive controller
  squeezes through but needs about 11 safety-filter interventions per episode. With n=20 the
  CIs overlap completely, so this shows no planner is better than another.

## E1: Isaac Go2 velocity tracking (2026-11-03)

Results are in `docs/ISAAC_INTEGRATION.md`. The pretrained flat policy tracks forward, arc and ≥0.6 m/s
lateral commands. It stands still for small planar commands, including all turn-in-place commands.

## E2: state-only dynamics on Isaac Go2 data (2026-11-03)

- Data: `data/go2_sim`, 60 episodes (seeds 0–59; families open, obstacles, sidewalk), 11,139
  transitions = **18.6 min simulated** at 10 Hz. Collection took 31.7 min wall (single env).
  Endings: 24 collision, 13 goal, 6 out_of_bounds, 17 timeout, **0 falls**. Exploration
  commands are random by design, which is why collisions are frequent. The validator passes.
- Split: scene-disjoint, seed 0. Train 42 / val 9 / test 9 episodes.
- Models: the same configs as E0. About 42 s training each on the 3090.
- Test split (1,157 windows, 9 episodes), mean open-loop position / heading error:

| model | 0.5 s | 1 s | 2 s | 3 s |
|---|---|---|---|---|
| MLP | 0.021 m / 1.1° | 0.044 m / 2.3° | 0.102 m / 4.7° | 0.171 m / 6.7° |
| GRU | 0.023 m / 1.3° | 0.045 m / 2.5° | 0.100 m / 4.6° | 0.166 m / 6.8° |
| command kinematics | 0.042 m / 1.6° | 0.079 m / 3.0° | 0.149 m / 5.5° | 0.221 m / 7.9° |
| constant velocity | 0.063 m / 5.1° | 0.201 m / 14.5° | 0.632 m / 37.6° | 1.143 m / 60.0° |

- 95% bootstrap CIs over the 9 test episodes, position error at 1 s: MLP [0.027, 0.087] m vs
  command kinematics [0.058, 0.121] m. At 3 s: [0.071, 0.435] vs [0.124, 0.475]. The intervals
  overlap. The MLP is better in every family (open/obstacles/sidewalk at 1 s: 0.032/0.055/0.058
  vs 0.069/0.086/0.094 m), but 9 test episodes are too few to claim significance. **Needs
  more test episodes and a paired per-window comparison.**
- **Paired comparison (added 2026-11-04, same windows for every model; `eval_test_paired`)**.
  Difference in per-episode mean position error (other − MLP), with a 95% bootstrap CI over the 9 test
  episodes. Positive means the MLP is better.

| vs MLP | 0.5 s | 1 s | 2 s | 3 s | episodes MLP better |
|---|---|---|---|---|---|
| command kinematics | +0.020 [+0.018, +0.022] | +0.034 [+0.030, +0.037] | +0.046 [+0.036, +0.057] | +0.045 [+0.027, +0.061] | 9/9 at every horizon |
| GRU | +0.002 [+0.001, +0.004] | +0.002 [−0.000, +0.004] | −0.002 [−0.010, +0.005] | −0.006 [−0.019, +0.006] | 6–8/9 |

  Paired, the MLP's improvement over command kinematics is consistent: the CI excludes 0 at every
  horizon. MLP vs GRU is a wash. There are still only 9 test episodes from one exploration generator
  (see caveat).
- Qualitatively (`D:\go2wm\outputs\state_go2_mlp\eval_test\rollout_examples.png`): command
  kinematics overshoots arcs, and it predicts motion for slow and turning commands that the
  policy ignores. The learned models capture both.
- Caveat: the train and test command distributions are the same exploration generator. A
  held-out command-pattern test (spec §10.1) hasn't been run.

## E2b: 3× more Isaac data (2026-11-04)

- Collected 120 more episodes (seeds 60–179, `scripts/collect.py --episodes 120 --seed 60`), adding 20,959 transitions in
  56.6 min wall. `data/go2_sim` now has **180 episodes / 32,098 transitions (≈53.5 min simulated)**, and the validator passes.
- New scene-disjoint split `v2` (seed 0): about 126 train / 27 val / 27 test. Configs: `world_model_state_v2.yaml`, `world_model_gru_v2.yaml`.
- **Leakage caught:** `v2` was split fresh, so 8 of its 27 test episodes were in v1's train/val. Comparing
  v1 and v2 checkpoints on `v2/test` would therefore favor v1. Fair comparison uses manifest `v2_clean_test`: the 19 `v2` test episodes
  in neither v1 train/val nor v2 train/val (`evaluate.py --manifest v2_clean_test`).
- Clean test (19 episodes, 2 training seeds each), mean position / heading error:

| model | train eps | 0.5 s | 1 s | 2 s | 3 s | heading @3 s |
|---|---|---|---|---|---|---|
| MLP v1 seed 0 / 1 | 42 | 0.014 / 0.014 m | 0.025 / 0.026 | 0.050 / 0.051 | 0.079 / 0.080 | 6.2° / 6.3° |
| MLP v2 seed 0 / 1 | ~126 | 0.013 / 0.012 m | 0.027 / 0.025 | 0.057 / 0.051 | 0.092 / 0.083 | 5.2° / 5.2° |
| GRU v1 / v2 (seed 0) | 42 / ~126 | 0.015 / 0.013 m | 0.028 / 0.026 | 0.055 / 0.053 | 0.089 / 0.088 | 6.9° / 5.9° |
| command kinematics | – | 0.032 m | 0.058 | 0.106 | 0.167 | 15.0° |

- **The training seed matters as much as the data size:** v2 seed 0 vs seed 1 differ by +0.012 m [+0.006, +0.019] at 3 s
  (paired). Across seeds, tripling the data gives a small gain at 0.5 s (paired +0.001 m, 14–15/19 episodes) and about 1° better
  heading at 3 s, with **no reliable position gain at 1–3 s**. On flat ground with this command distribution the state-only model
  appears saturated. More of the same data isn't the bottleneck.
- Every learned model still beats command kinematics in 18–19/19 clean episodes at every horizon (v2 seed 0, 1 s: +0.029 m
  [+0.024, +0.034]).
- Implication for H1: any headroom has to come from new information (terrain, vision) or new conditions, not more flat-ground data.

## E3a: Isaac navigation, 15 identical scenes (2026-11-04)

- Planners: conventional (reactive pure-pursuit plus potential field), kinematic MPC (CEM with command
  kinematics), and learned MPC (CEM with the E2 MLP, `outputs/state_go2_mlp/best.pt`). All three use
  the same A* waypoints, the same safety filter and `configs/planner_isaac.yaml`. Scenes are
  `{open,obstacles,sidewalk}_10000–10004`, which were never used in training.
- Commands: conventional `--planner conventional` → `outputs/navigation`; kinematic MPC →
  `outputs/navigation_kin`; learned MPC → `outputs/navigation_learned`. The planners ran
  separately because a combined run was stopped for low system memory. Merge with
  `scripts/merge_navigation.py` → `outputs/nav_merged_e3a`.

| planner | success [95% CI] | collisions | falls | time to goal (s) | min clearance (m) | interventions/ep | latency p50 |
|---|---|---|---|---|---|---|---|
| conventional | 0.87 [0.62, 0.96] | 0 | 0 | 14.8 | 0.76 | 21.5 | 0.2 ms |
| kinematic MPC | 0.87 [0.62, 0.96] | 0 | 0 | 15.3 | 0.77 | 0.0 | 106 ms |
| learned MPC | 0.87 [0.62, 0.96] | 0 | 0 | 14.6 | 0.77 | 0.0 | 103 ms |

- Every planner succeeds on all open and obstacle scenes. Every planner times out on the **same** two
  sidewalk scenes (`sidewalk_10002`, `sidewalk_10004`). The paired McNemar test has 0 discordant
  scenes for every pair, so there's no difference between planners.
- Failure diagnosis: in both scenes the robot **freezes in place** in front of an obstacle (the
  conventional controller has zero displacement over its last 100 steps). The MPCs stop about 0.3 m
  from the obstacle, and the conventional controller stops after 1–2 reroutes. Getting past needs low-speed
  lateral or turn-in-place motion, which the pretrained flat policy ignores (E1 deadband). So the
  failures come from the locomotion policy, not from planner differences. This is the main limit on
  sidewalk performance. Fix options: (A) a policy that turns in place, or a planner-side
  deadband-aware action set that never commands small planar speeds.
- n=15 can't separate the planners. E3b extends this to 51 shared scenes.

## E3b: Isaac navigation, 51 identical scenes (2026-11-04)

- Scenes 10000–10016 per family (E3a + `--first-index 5 --per-family 12`). Learned MPC's extension was
  rerun after a low-memory stop (`outputs/navigation_ext_learned`). Merged: `outputs/nav_merged_e3b`.

| planner | n | success [95% CI] | collisions | falls | timeouts | time to goal (s) | interventions/ep | latency p50 |
|---|---|---|---|---|---|---|---|---|
| conventional | 51 | 0.88 [0.77, 0.94] | 0 | 0 | 6 | 14.8 | 6.3 | 0.2 ms |
| kinematic MPC | 51 | 0.90 [0.79, 0.96] | 0 | 0 | 5 | 15.2 | 0.0 | 102 ms |
| learned MPC | 51 | 0.88 [0.77, 0.94] | 0 | 0 | 6 | 14.5 | 0.0 | 95 ms |

- By family, all three planners: open 1.00, obstacles 0.94. Sidewalk: 0.71 / 0.76 / 0.71.
- Paired (51 shared scenes): conventional vs learned MPC have 0/0 discordant scenes; kinematic MPC is the only
  one to succeed on `sidewalk_10006` (1 discordant scene vs each other planner, exact McNemar p = 1.0).
  **No planner differs from another.** Learned MPC is about 0.3–0.7 s faster to the goal than kinematic MPC, but
  this is a small effect and wasn't tested for significance.
- Every failure is a timeout where the robot freezes before a gap (see the E3a diagnosis). On this
  terrain and with this locomotion policy, better state prediction doesn't turn into better
  navigation: the binding constraint is the policy's deadband, which every planner hits the same way.

## E3c: deadband-aware MPC (2026-11-04)
- Deadband measurement: steady-command transitions in `data/go2_sim` (5,093 samples held ≥0.4 s). The fraction actually moving
  (>0.1 m/s) is 0.01 for planar commands below 0.1 m/s, 0.14 for 0.1–0.2, 0.59 for 0.2–0.3, 0.82 for 0.3–0.4, 0.93 for 0.4–0.5 and 0.98 at 0.5 and above.
  Yaw is barely realized when planar speed is below 0.2 m/s.
- `configs/planner_isaac_db.yaml` (`mpc.deadband`) maps sampled planar commands below 0.1 m/s to a full stop and rescales 0.1–0.4 m/s
  to 0.4 m/s (`project_deadband`, unit-tested). It runs on obstacle + sidewalk scenes 10000–10016 (34 scenes), kinematic and learned (E2 MLP).

| planner (34 obstacle + sidewalk scenes) | success | collisions | mean min clearance |
|---|---|---|---|
| kinematic MPC (E3b) | 29/34 | 0 | 0.78 m |
| learned MPC (E3b) | 28/34 | 0 | 0.78 m |
| kinematic MPC + deadband | 29/34 | 0 | 0.34 m |
| learned MPC + deadband | 29/34 | 0 | 0.33 m |

- Both deadband MPCs fail exactly the same 5 scenes (`obstacles_10014`, `sidewalk_10002/10004/10008/10011`). The only change is that
  learned MPC now also solves `sidewalk_10006`. Paired p = 1.0 everywhere.
- How the failure changes: the robot **no longer freezes**. Over the last 10 s it moves about 0.2 m around a spot, versus under 0.1 m before, and its path is about 5 m longer.
  But it parks at the same place, **0.26–0.33 m from the obstacle, i.e. at the 0.3 m `clearance_margin_m`**. These gaps can't be crossed
  while keeping 0.3 m (e.g. `sidewalk_10002`: 1.04 m wide minus the 0.6 m robot leaves ≤0.22 m per side), and the clearance cost (weight 30) outweighs
  goal progress. So after the deadband fix, the blocker is the **planner's cost tuning**, not locomotion.

## E3d: + 0.15 m clearance margin, the 5 failure scenes only (2026-11-04)
- `configs/planner_isaac_db_m15.yaml` (deadband + `clearance_margin_m: 0.15`), run on the 5 scenes every earlier planner failed.
- **10/10 successes**: kinematic MPC 5/5, learned MPC 5/5. 0 collisions, minimum clearance 0.15–0.20 m, time to goal 14–18 s.
- These scenes were picked *because* they failed, so this isn't an unbiased success rate. The confirmatory run on all 51 scenes,
  with a margin-only ablation (no deadband), is E3e.

## E3e: 0.15 m margin on all 51 E3b scenes (2026-11-05)
- Configs `planner_isaac_db_m15_all.yaml` (deadband + 0.15 m margin) and `planner_isaac_m15_all.yaml` (0.15 m margin only), `--per-family 17`,
  E2 MLP for learned MPC. Outputs: `outputs/navigation_e3e_{dbm15,m15}_{kin,learned}`, merged copies in `outputs/nav_e3e_renamed`.

| planner | success [95% CI] | collisions | falls | worst min clearance | time to goal |
|---|---|---|---|---|---|
| kinematic MPC, deadband + 0.15 m | 51/51 [0.93, 1.00] | 0 | 0 | 0.142 m | 15.7 s |
| learned MPC, deadband + 0.15 m | 51/51 [0.93, 1.00] | 0 | 0 | 0.152 m | 15.1 s |
| kinematic MPC, 0.15 m only | 51/51 [0.93, 1.00] | 0 | 0 | 0.141 m | 15.2 s |
| learned MPC, 0.15 m only | 51/51 [0.93, 1.00] | 0 | 0 | 0.144 m | – |

- **The clearance margin was the entire cause of the remaining failures.** Margin alone works as well as deadband + margin, so the deadband
  projection isn't needed once the margin allows these gaps.
- Paired vs conventional (E3b): tuned kinematic MPC has 6 scenes it alone solves and 0 it alone fails (exact McNemar p = 0.031). **Caveats:** (1) the margin was chosen
  after looking at 5 of these 51 scenes (tuning on the test set), and (2) the conventional controller wasn't given equivalent tuning. A confirmatory run needs fresh
  scene indices (e.g. `--first-index 17`) and a tuned conventional baseline.
- Learned vs kinematic MPC: 0 discordant scenes in every setting. Consistent with E2b, better prediction doesn't change navigation outcomes on flat
  ground with cylinder obstacles.
- Explainer with figures (trajectories, deadband curve, prediction examples): https://claude.ai/artifact/93i1DcD2ecmu8ju8ZFaoA3

## Next experiments (Isaac)
- Confirm E3e on fresh scenes (`--first-index 17`), including a tuned conventional baseline.
- Beyond that, these runs show **the learned model has no navigation advantage on flat ground with cylinders**: every planner difference is
  explained by locomotion deadband and cost tuning. Testing H1/H2 needs conditions where dynamics are hard to model by hand: sidewalk
  terrain (curbs, slopes, friction changes, from SidewalkAtlas `scripts/terrain`), and cameras with V-JEPA 2 for H1.
