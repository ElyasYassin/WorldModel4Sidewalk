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

## P0 integrity pass (2026-11-09/10)

- Baseline commit `e5195d9`. Split audit `outputs/audit/splits_2026-11-09.json` (`scripts/audit_splits.py`): no
  within-manifest overlap by episode, scene id, seed or geometry; one episode per scene; navigation scenes never share id,
  seed or geometry with data. No data episode was untouched, so a fresh final test set was collected. Roles of every
  seed/scene range are frozen in `configs/eval_protocol.yaml`.

## E2c: fresh final prediction test (2026-11-10)

- Data: `data/go2_final_test`, seeds 30000–30059, 60 episodes, same exploration generator as training, collected for
  evaluation only (validator passes; endings 17 collision / 14 goal / 15 out_of_bounds / 14 timeout, 0 falls).
  9,648 windows. `evaluate.py --all-test` → `outputs/p0_pred/final_test/`.

| model | 0.5 s | 1 s | 2 s | 3 s | heading @3 s |
|---|---|---|---|---|---|
| MLP v1 seed 0 / 1 | 0.016 / 0.016 m | 0.030 / 0.030 | 0.060 / 0.060 | 0.098 / 0.098 | 6.9° / 6.8° |
| MLP v2 seed 0 / 1 | 0.015 / 0.014 m | 0.029 / 0.027 | 0.061 / 0.055 | 0.100 / 0.091 | 6.2° / 5.5° |
| GRU v1 / v2 | 0.017 / 0.015 m | 0.033 / 0.029 | 0.063 / 0.060 | 0.101 / 0.097 | 7.5° / 6.0° |
| command kinematics | 0.036 m | 0.065 | 0.116 | 0.178 | 15.7° |

- Paired vs MLP v1: kinematics is worse at every horizon (1 s: +0.032 m [+0.029, +0.035], 59/60 episodes). Confirms E2/E2b
  on untouched data. Learned models differ from each other by ≤0.005 m; more data (v2) helps heading (≈1°), not position.

## E2d: held-out command patterns, spec §10.1 (2026-11-10)

- Data: `data/go2_cmdpattern`, seeds 40000–40079, open family, `HeldOutCommandPatterns` (20 episodes each):
  `sine` (continuous sinusoids), `long_arc` (4–6 s steady arcs), `fast_switch` (unsmoothed, 0.2–0.3 s holds),
  `lateral_ood` (|vy| 0.45–0.7 m/s, outside the training range |vy| ≤ 0.4). 14,787 windows.
  → `outputs/p0_pred/cmdpattern/`.
- Position / heading error by pattern (m / deg):

| pattern | horizon | MLP v1 | GRU v1 | MLP v2 | GRU v2 | kinematics |
|---|---|---|---|---|---|---|
| sine | 3 s | 0.087 / 4.7 | 0.079 / 4.5 | 0.088 / 3.8 | 0.124 / 3.6 | 0.165 / 7.5 |
| long_arc | 3 s | 0.078 / 4.6 | 0.072 / 4.9 | 0.079 / 4.8 | 0.093 / 3.0 | 0.088 / 4.8 |
| fast_switch | 3 s | 0.110 / 7.3 | 0.074 / 5.2 | 0.143 / 4.6 | 0.103 / 4.8 | 0.156 / 4.5 |
| lateral_ood | 1 s | 0.052 / 5.7 | 0.079 / 3.8 | 0.060 / 2.6 | 0.059 / 2.9 | 0.067 / 1.6 |
| lateral_ood | 3 s | 0.263 / 17.7 | 0.255 / 8.4 | 0.161 / 6.5 | 0.204 / 10.1 | **0.147 / 4.5** |

- Paired kinematics − MLP v1 position at 3 s: sine +0.068 [+0.037, +0.098] (17/20 MLP better), fast_switch +0.047
  [+0.036, +0.058] (19/20), long_arc +0.013 [+0.001, +0.026] (14/20), **lateral_ood −0.100 [−0.142, −0.057] (MLP better in 3/20)**.
- **Finding:** new temporal patterns inside the trained command range are handled (learned still best). Commands outside the
  training support are not: every learned model is worse than command kinematics at 2–3 s on `lateral_ood`, and heading error
  is up to 4× worse. The policy tracks |vy| ≥ 0.6 well (E1), so the kinematic assumption holds there and the learned models
  extrapolate wrongly. Pooled over all patterns, the learned advantage at 3 s is gone (kinematics − MLP v1: +0.007 m
  [−0.013, +0.027]).
- Implication for planning: a learned MPC must keep its samples inside the training command support (the current Isaac planner
  limits |vy| ≤ 0.3, so E3 is unaffected), or detect out-of-support inputs and fall back to kinematics. For P1/P2 this means
  terrain data must cover the commands the planner will use.

## Conventional tuning on dev scenes (2026-11-10)

- 7 new configs (`configs/tuning/`) plus the default, 34 obstacle + sidewalk dev scenes each, ranked by the rule fixed
  before any run (success → collisions → interventions → time). Table in `docs/EXPERIMENT_STATUS.md`.
- `min_speed` 0.4 (a forward-speed floor above the policy dead zone) decides it: 32–34/34 with it, 28–31/34 without.
  Chosen and frozen: `min_speed` 0.4, no bound repulsion, influence 0.6 m / k_rep 0.8 → `configs/planner_isaac_conv_tuned.yaml`
  (34/34 on dev, equal to both MPCs).

## E3f: frozen planners on final navigation scenes 17–33 (2026-11-10)

- 51 untouched scenes (`open/obstacles/sidewalk_10017`–`_10033`), run once with `--final-test`; 4 ledger entries in
  `outputs/final_test_ledger.jsonl`. Planners frozen beforehand in `configs/eval_protocol.yaml`.
  `scripts/compare_navigation.py` → `outputs/e3f/compare*.json`.

| planner | success [Wilson 95%] | collisions | falls | freezes | t_goal [s] | interventions/ep | command change/step |
|---|---|---|---|---|---|---|---|
| conventional, default | 45/51 0.88 [0.77, 0.94] | 0 | 0 | 6 | 14.8 | 7.4 | 0.089 |
| conventional, tuned | 50/51 0.98 [0.90, 1.00] | 0 | 0 | 1 | 14.8 | 6.8 | 0.081 |
| kinematic MPC (m 0.15) | 51/51 1.00 [0.93, 1.00] | 0 | 0 | 0 | 15.2 | 0.0 | 0.267 |
| learned MPC (m 0.15) | 51/51 1.00 [0.93, 1.00] | 0 | 0 | 0 | 14.6 | 0.0 | 0.252 |

- Paired success (exact McNemar): default conventional vs either MPC 0 vs 6 discordant, p = 0.031; tuned conventional vs
  either MPC 0 vs 1, p = 1.0; default vs tuned conventional 0 vs 5, p = 0.062; kinematic vs learned MPC 0 vs 0.
- Paired time to goal: learned − kinematic MPC **−0.59 s [−0.67, −0.51]** (dev E3e: −0.65 s), consistent in every family
  (mean differences: open −0.6, obstacles −0.6, sidewalk −0.7 s). Tuned conventional vs MPCs: learned −0.26 s [−0.59, +0.09], kinematic
  +0.33 s [+0.02, +0.67]. On open scenes the conventional controller is fastest (12.7 s vs 13.6/14.2 s).
- Failures: default conventional = 5 sidewalk + 1 obstacle timeouts, all freezes (dead zone, as in E3b). Tuned
  conventional = 1 freeze on `sidewalk_10030` of a different kind: it squeezed between an obstacle and the walkway edge to
  0.02 m clearance, and the safety filter then zeroed its forward command for 347 steps (deadlock, not dead zone).
  Video: `outputs/videos/05_e3f_sidewalk_10030_all4.mp4`.
- Conventional needs the safety filter on sidewalks (≈20 interventions/ep); the MPCs never trigger it. MPC commands change
  ≈3× more per step (0.25–0.27 vs 0.08–0.09, L1 per 0.1 s), i.e. the MPCs are much less smooth.
- **Finding:** E3e replicates on fresh scenes. With matched tuning, the conventional controller is statistically
  indistinguishable from the MPCs in success (50/51 vs 51/51). Learned prediction buys a small, reliable time saving
  (≈0.6 s ≈ 4%) over kinematic prediction and no success difference. Flat ground with cylinders does not separate the
  planners; P1 terrain is where prediction could matter.

## E4 preflight: terrain mesh check with the flat policy (2026-11-10)

- **Not an E4 result.** The rough-terrain checkpoint is unavailable, so the E4-S1 tile (4 % cross-slope, 8 cm slab at
  x = 4.0–5.5 m, 6 % grade, 15 cm curb) was loaded under the *flat* policy only to verify the terrain code in the running app:
  `configs/isaac_go2_flat_e4s1_meshcheck.yaml`, 1 episode at 0.6 m/s → `outputs/e4_meshcheck/report.json`,
  log `D:/go2wm/logs/e4_meshcheck.log`.
- Verified: the generated tile loads, the robot spawns at the scene origin and walks on it; height above ground on the flat
  start section is 0.171 ± 0.001 m (flat-plane runs: median base z 0.18 m), so the tile surface matches the analytic surface
  to within ≈5–10 mm; terrain ground-truth arrays are recorded and the episode validates.
- The blind flat policy stopped at the slab edge (x_max 3.9 m, 40 s timeout, no fall). Expected; says nothing about E4.
- **Open issue:** foot contact is sampled once per 0.1 s control step. Foot 0 shows a 0.3 s swing cycle, the other three
  read "in contact" at every sample, which is either this policy's gait or aliasing. Before relying on contact/slip ground
  truth, log contacts at physics rate (or the contact sensor's air/contact time) and check with the rough policy.
- `ground_z_m` is NaN without a height scanner (by design); the rough policy provides one.

## E1r: velocity tracking, pretrained rough-terrain policy (2026-11-10)

- Checkpoint `.pretrained_checkpoints/rsl_rl/Isaac-Velocity-Rough-Unitree-Go2-v0/checkpoint.pt` (NVIDIA, 6.9 MB), flat
  sidewalk tile, `scripts/isaac_smoke.py --config configs/isaac_go2_rough.yaml` → `outputs/isaac_smoke_rough/report.json`.
  No fall; pass.

| command (vx, vy, wz) | flat policy (E1) realized | rough policy realized |
|---|---|---|
| 0.5, 0, 0 | 0.49, 0.00, 0.01 | 0.46, 0.02, 0.05 |
| 1.0, 0, 0 | 1.02, 0.02, −0.02 | 0.99, −0.07, 0.06 |
| 0.5, 0, 0.5 | 0.48, 0.01, 0.46 | 0.46, −0.01, 0.54 |
| 0, 0.3, 0 | −0.01, 0.00, −0.03 (**dead zone**) | 0.02, 0.27, 0.01 |
| 0, ±0.6, 0 | ±0.61 | 0.57 / −0.55 |
| 0, 0, −0.8 / 0.5 / 1.0 | 0 / 0.03 / 0 (**no turn in place**) | −0.51 / 0.19 / 0.31 (partial) |
| 0.3, 0, 0.8 | 0.18, 0.04, 0.41 | 0.15, −0.01, 0.66 |

- The rough policy has a much smaller dead zone: it tracks 0.3 m/s lateral and turns in place, although only at 40–60 % of
  the commanded rate. Navigation results tuned around the flat policy's dead zone (E3c deadband, conventional `min_speed`)
  need re-checking with this policy.

## E4-S1: traversal smoke test, rough policy (2026-11-10)

- `scripts/isaac_terrain_smoke.py --config configs/isaac_go2_rough_e4s1.yaml`: lane-following along the centreline at
  0.4/0.6/0.8 m/s, 2 episodes each → `data/go2_e4_smoke`, `outputs/e4_smoke/report.json`.
- **6/6 reached the end, 0 falls** (33, 20, 15 s at 0.4/0.6/0.8 m/s). The policy can traverse S1, so S1 failures in later
  experiments are not locomotion limits.
- Mean forward-speed error by section: flat −0.03 to −0.04 m/s; 8 cm slab −0.02 to −0.12 (largest at 0.4 m/s); 6 % grade
  −0.00 to −0.02. Mean foot slip 0.04–0.08 m/s, highest on the slab. Per-foot contact fraction 0.56–0.66, a normal gait, so
  the E4-preflight contact anomaly was the flat policy's gait, not a logging bug (contacts are still sampled at 10 Hz).
- Height scanner works: scanner ground height is finite at every step and equals the analytic surface minus a constant
  0.036 m (std 0.006 m). The constant is the env-origin z (Isaac puts the origin at the highest point within 1 m of the tile
  centre); `gt_ground_z_m` is relative to the env origin, `sidewalk_height` to the start surface. Height above ground
  dips to 0.20 m at the slab edge at 0.4 m/s (scanner rays reach the slab top before the base): a measurement artefact
  near steps, not a fall; use it with care for fall detection near steps.
- Not tested: the curb (lane-following keeps y in 0–0.4 m, the curb is at y = −1.5 m) and cross-slope drift (closed loop
  corrects it).
- **Finding:** S1 is easy for this policy; terrain changes realized speed by at most ≈0.1 m/s. Command kinematics will
  likely be nearly as good as a learned model here, so S1 alone cannot show a benefit from prediction. Next: sweep
  slab height, friction, cross-slope and curb approach to find where the policy's realized motion departs from the command.

## E4 sweep: one terrain feature at a time, rough policy (2026-11-10)

- `configs/e4_sweep.yaml`, `scripts/e4_sweep_queue.ps1` (one Isaac process per variant), analysis
  `python scripts/analyze_terrain_sweep.py` → `outputs/e4_sweep/analysis.json`; data `data/go2_e4_sweep` (seeds 60000+).
  Probes: lane-following at 0.4 and 0.8 m/s, a 20 s open-loop straight command at 0.6 m/s; curb variants cross down to
  the road and back up. **One episode per cell.**
- First pass: 15 of 18 variants (queue stopped for low system memory); `friction_025`, `friction_015`, `curb_25` and
  3-per-speed repeats of the 15 cm slab, 30 % grade and 15 cm curb ran in a second pass. `curb_15_rep` was stopped (low
  memory) after 1 of 9 episodes.
- `kin1s` = command-kinematics 1 s position error on the feature (plain-ground reference ≈ 0.03–0.06 m):

| feature | 0.4 m/s lane | 0.8 m/s lane | 0.6 m/s straight | kin1s on feature (0.4 / 0.8 / straight) |
|---|---|---|---|---|
| flat | goal | goal | – | 0.052 / 0.041 / 0.003 |
| slab 5 cm | goal (hesitates, vx −0.21) | goal | ok | 0.219 / 0.037 / 0.025 |
| slab 10 cm | goal | goal | ok | 0.138 / 0.089 / 0.072 |
| slab 15 cm | **stuck at x = 4.0** | goal (1 of 4 runs, see repeats) | **stuck** | 0.384 / 0.193 / 0.539 |
| slab 20 cm | **stuck** | **stuck** | **stuck** | 0.384 / 0.758 / 0.540 |
| slab 25 cm | **stuck** | **stuck** | **stuck** | 0.377 / 0.759 / 0.544 |
| grade 15 % | goal | goal | ok | 0.030 / 0.017 / 0.033 |
| grade 30 % | **stuck at x = 5.4** | goal | ok | 0.339 / 0.105 / 0.042 |
| cross-slope 6–20 % | goal | goal | drifts downhill | 0.024–0.076 |
| friction 0.6, 0.4 | goal | goal | ok | 0.002–0.050 |
| curb 15 cm, down + up | **stuck on the step up** | down and up | – | 0.249 / 0.062 |

- No falls anywhere: every failure is the robot stopping in front of a step or on the steep grade while still commanded
  forward ("stuck", < 0.1 m in the last 10 s).
- **Repeats (3 per speed) correct the single-run picture:**
  - 30 % grade is speed-dependent and repeatable: 0.4 m/s stuck 4/4, 0.6 m/s pass 3/3, 0.8 m/s pass 4/4.
  - 15 cm slab is **not** speed-dependent; it is a stochastic limit: pass 1/4 at 0.4, 2/3 at 0.6, 1/4 at 0.8 m/s, plus one
    **fall** at 0.8 m/s (the only fall in the sweep). The first-pass "passes at 0.8" was one lucky run.
  - 15 cm curb step-up: stuck in 1 of 2 runs at 0.4 m/s (repeats stopped after one episode). 25 cm curb: 0.4 m/s stalls
    on the step up (x 8.7 m at timeout); 0.8 m/s gets up but slowly (vx −0.34 m/s on the feature, kin1s 0.36 m).
  - ≥ 20 cm slabs fail at every speed, although the policy was trained on steps up to 23 cm (pyramid stairs, not a single
    raised slab).
- Cross-slope: closed-loop traversal unaffected, but the open-loop straight run turns downhill: −13° to −28° heading over
  ≈ 11 m, against +16° on flat ground (the policy itself has a small yaw bias), so a 30–44° downhill effect at 20 %. Slow
  drift barely shows at 1 s (kin1s ≈ flat) but will at 3 s.
- Friction: 0.6 and 0.4 (× foot 0.8/0.6) change nothing measurable. 0.25: still passes, foot slip doubles (0.28 m/s
  at 0.4 m/s), kin1s 0.11 m. 0.15: the 0.4 m/s run only reaches x = 9 m in 40 s (slip 0.36 m/s, kin1s 0.17 m); 0.8 m/s
  passes.
- **Finding:** command kinematics is as good as on flat ground everywhere except near step-ups and steep grades, where it
  is 3–15× worse (up to 0.76 m at 1 s) because it predicts motion that does not happen. That is a planning-relevant error:
  whether a step can be taken decides success. On grades the outcome is a repeatable function of speed, which a
  deterministic terrain-aware model can learn. Near the step limit (15 cm) the outcome is stochastic, so a useful model
  there must predict a *probability* of stalling, not one trajectory. Very low friction (≤ 0.25) is a third regime.
  E4 data collection targets steps of 3–18 cm, curbs and grades up to 25 % at 0.3–0.9 m/s.
- Caveats: 1 episode per cell outside the repeated cells; the height-above-ground signal dips at step edges (E4-S1).

## E4-P2: does terrain input improve prediction? (2026-11-10)

- **Data.** 144 train episodes on 12 random sidewalk tiles (`configs/e4_tiles.yaml`, seeds 70000+, `data/go2_e4_train`,
  manifest `e4`: 100/22/22 train/val/dev-test by episode) and 48 final-test episodes on 6 unseen tiles (seeds 80000+,
  `data/go2_e4_test`, manifest `all_test`). Rough policy, 30 s max, commands: 53 % lane-following at 0.3–0.9 m/s, rest
  exploration. Outcomes (train / test): goal 50/13, timeout 65/21, out of bounds 23/11, fall 6/3. Validator passes.
- **Models.** MLP 256×3 and GRU 128×1 (state + commands only); terrain MLP = the same MLP plus a 16-d height patch
  sampled at every predicted pose from a perfect start-frame elevation map (`src/go2wm/models/terrain_map.py`). 40 epochs,
  seeds 0/1/2 each (`--seed`), no tuning on any test split. Final test run once with all 9 checkpoints.
- **Results, position error (m), mean ± sd over 3 seeds.** (`outputs/e4_p2/{devtest,final_test}/prediction_metrics.json`)

| split | model | 0.5 s | 1 s | 2 s | 3 s |
|---|---|---|---|---|---|
| dev-test (seen tiles) | terrain MLP | 0.022 | **0.044** | **0.086** | **0.133** ± 0.005 |
| | MLP | 0.023 | 0.050 | 0.107 | 0.169 ± 0.003 |
| | GRU | 0.022 | 0.047 | 0.098 | 0.155 ± 0.004 |
| | kinematics | 0.066 | 0.127 | 0.244 | 0.366 |
| final test (unseen tiles) | terrain MLP | 0.025 | 0.051 | 0.108 | 0.181 ± 0.004 |
| | MLP | 0.024 | 0.052 | 0.117 | 0.195 ± 0.005 |
| | GRU | 0.023 | 0.050 | 0.112 | 0.184 ± 0.003 |
| | kinematics | 0.070 | 0.135 | 0.265 | 0.399 |

- Final test by segment, seed-mean error at 2 s (mean / p90): step (2677 windows) terrain 0.158/0.348, MLP 0.184/0.476,
  GRU 0.180/0.478, kinematics 0.209/0.477; grade (712) 0.115 / 0.126 / 0.115 / 0.162; curb (161) 0.131 / 0.131 / 0.109 /
  0.150; plain (6579) 0.086 / 0.088 / 0.084 / 0.302.
- **Findings.**
  - Every learned model beats command kinematics by 2.5–2.7× at 1 s on unseen tiles (paired, 45/48 episodes).
  - The terrain input helps the MLP at 2–3 s (paired vs MLP at 2 s: +0.012 to +0.016 m, CIs exclude 0, 35–38/48
    episodes), and the gain is concentrated on step segments, where it also cuts the p90 error by about a quarter.
  - On unseen tiles the terrain MLP is tied with the state-only GRU overall (0.181 vs 0.184 at 3 s; paired CIs include 0)
    and is no better on curbs and grades. Its seen-tile advantage (0.133 vs 0.155) shrinks on unseen tiles: it overfits
    (train loss 0.126 vs 0.149 for the MLP at equal val loss).
  - So: terrain information is useful, but the effect is modest (1–3 cm at 2–3 s) and limited to steps; the stochastic
    stall-or-climb outcome at 15–18 cm steps is not resolved by a deterministic model.
- **Caveats.** Segment labels use the true future path, so windows where the robot stalls in front of a step are labelled
  plain (this is why kinematics has a p90 of 1.1 m on plain). Perfect elevation map, not the robot's scanner. By chance all 6 test tiles slope
  right-high while 8/12 training tiles slope left-high (independent seeds), so the unseen-tile test is also a
  cross-slope-direction shift. Test-tile slabs are 5–12.5 cm. Prediction quality only; decision utility (navigation with these models) is not tested here.
- **Next.** Terrain GRU (recurrence + terrain), stronger regularisation, a probabilistic head for step outcomes, labels
  from the commanded path; then a navigation test on sidewalk tiles to see if the prediction gain changes decisions.

## E4-P2b: terrain GRU, weight decay, commanded-path labels (2026-11-10)

- **Added.** `terrain_gru` (GRU 128×1, the same 16-d height patch fed to the output head;
  `configs/world_model_e4_terrain_gru.yaml`) and `terrain_wd` (terrain MLP, AdamW weight decay 1e-2 instead of 1e-5).
  3 seeds each, same data and epochs. New segment labels `segment_cmd`: the terrain the *commanded* path would cross
  from the window start, so a robot that stalls in front of a step is labelled `step` (the E4-P2 `segment` labels,
  from the realized path, are kept). All 15 checkpoints evaluated on dev-test, then on the final test. For the 9 E4-P2
  checkpoints this is a second final-test evaluation, only to add the new labels; nothing was retrained or selected.
- **Position error (m), seed mean ± sd.** (`outputs/e4_p2b/{devtest,final_test}/prediction_metrics.json`)

| split | model | 1 s | 2 s | 3 s |
|---|---|---|---|---|
| dev-test | terrain GRU | 0.046 | 0.090 | 0.140 ± 0.006 |
| | terrain MLP, wd 1e-2 | 0.043 | 0.084 | **0.128** ± 0.003 |
| | terrain MLP | 0.044 | 0.086 | 0.133 ± 0.005 |
| | GRU / MLP | 0.047 / 0.050 | 0.098 / 0.107 | 0.155 / 0.169 |
| final test | terrain GRU | 0.050 | **0.103** | **0.165** ± 0.004 |
| | terrain MLP, wd 1e-2 | 0.050 | 0.107 | 0.177 ± 0.008 |
| | terrain MLP | 0.051 | 0.108 | 0.181 ± 0.004 |
| | GRU / MLP | 0.050 / 0.052 | 0.112 / 0.117 | 0.184 / 0.195 |
| | kinematics | 0.135 | 0.265 | 0.399 |

- Final test, 3 s, `segment_cmd` (mean / p90): step (3614 windows) terrain GRU 0.246/0.570, terrain MLP 0.249/0.572,
  GRU 0.288/0.720, MLP 0.293/0.682, kinematics 0.637/1.829; grade (873) 0.160 / 0.229 / 0.165 / 0.179 / 0.437; curb
  (203) 0.287 / 0.318 / 0.272 / 0.282 / 0.423; plain (5439) 0.108 / 0.124 / 0.115 / 0.129 / 0.233.
- **Findings.**
  - On unseen tiles the terrain GRU is the best model: paired vs GRU (seed 0) +0.026 m at 3 s [0.015, 0.038], vs MLP
    +0.037 [0.027, 0.048]; vs the terrain MLPs +0.012 to +0.015, CIs include 0. Combining recurrence and terrain input
    keeps the step gain of the terrain MLP without its loss on unseen grades (terrain MLPs 0.23 m vs GRUs 0.16 m on
    grade windows at 3 s, likely the −24 % ramp on test tile 2).
  - Dev-test and final test disagree on the best model (dev-test would have picked `terrain_wd`, within seed noise of
    the terrain MLP). Weight decay 1e-2 hardly regularised (train loss 0.126, same as wd 1e-5).
  - With commanded-path labels, the stall windows move from plain to step: kinematics' plain p90 at 3 s drops from
    1.56 to 0.50 m (dev-test), and on step windows kinematics is off by 0.64 m on average (final test) while the
    terrain models are at 0.25 m.
- **Caveats.** As E4-P2 (perfect map, cross-slope direction shift, prediction only). The terrain-GRU result was not
  the dev-test winner, so treat it as one more piece of evidence, not a selected model.

## E4-S: terrain GRU from the robot's height scanner vs a perfect map (2026-11-10)

- **Question.** E4-P2b's terrain gain uses a perfect elevation map 3 m ahead. Does it survive when the terrain input
  is the robot's own height scanner (the rough policy's ray caster: 187 rays, 1.6 × 1.0 m grid at 0.1 m, so 0.8 m
  ahead and 0.5 m to the side, no noise)?
- **Data.** E4 re-collected with `--record-scan` (`scripts/e4_collect_queue.ps1 -DataTag e4s`): same tiles and seeds,
  144 train + 48 test episodes (`data/go2_e4s_{train,test}`), 0.01 % of rays missed. Re-collection is not
  bit-identical to E4 (GPU PhysX), so this is a fresh sample; all three models here are trained and tested on it.
  Manifests `e4` (seed 0: 100/22/22) and `all_test`; the split audit is clean for `all_test` (the dev split shares
  tiles with train by design, as in E4).
- **Models.** 3 seeds each, identical except for the terrain input: GRU (none), terrain GRU with the perfect map
  (`terrain_source: analytic`), terrain GRU with the scanner map (`terrain_source: scan`: the scan at the window start,
  interpolated onto the same map grid; beyond the scanner's reach the edge values are held, so it really sees only
  0.8 m ahead). Configs `configs/world_model_e4s_{gru,terrain_gru,scan_gru}.yaml`.
- **Position error (m), seed mean ± sd.** (`outputs/e4s/{devtest,final_test}/prediction_metrics.json`)

| split | model | 1 s | 2 s | 3 s |
|---|---|---|---|---|
| dev-test | GRU | 0.047 | 0.098 | 0.155 ± 0.004 |
| | terrain GRU, perfect map | 0.046 | 0.090 | **0.140** ± 0.008 |
| | terrain GRU, scanner | 0.047 | 0.096 | 0.149 ± 0.002 |
| final test | GRU | 0.050 | 0.112 | 0.184 ± 0.003 |
| | terrain GRU, perfect map | 0.050 | **0.103** | **0.165** ± 0.005 |
| | terrain GRU, scanner | 0.051 | 0.108 | 0.176 ± 0.004 |
| | kinematics | 0.135 | 0.265 | 0.399 |

- **Gain over the GRU on the final test, by `segment_cmd`** (seed-mean error minus seed-mean GRU error, m; [min, max]
  of the paired per-seed differences). Corrected 2026-11-11: recomputed with
  `python scripts/summarize_seeds.py outputs/e4s/final_test/prediction_metrics.json --ref e4s_gru`. The table first
  written here did not match the saved metrics (its source could not be traced); the scanner's step gain at 2 s was
  overstated (−0.015 → −0.011).

| segment (episodes) | map 2 s | scanner 2 s | map 3 s | scanner 3 s |
|---|---|---|---|---|
| step (38) | −0.021 [−0.026, −0.015] | −0.011 [−0.013, −0.009] | −0.043 [−0.049, −0.038] | −0.016 [−0.025, −0.010] |
| grade (18) | −0.006 [−0.013, +0.002] | +0.003 [−0.008, +0.019] | −0.005 [−0.028, +0.023] | +0.004 [−0.021, +0.040] |
| curb (7) | +0.009 [−0.015, +0.030] | +0.008 [−0.016, +0.031] | +0.014 [−0.017, +0.030] | −0.001 [−0.036, +0.022] |
| plain (48) | −0.002 [−0.005, +0.003] | −0.000 [−0.008, +0.005] | −0.007 [−0.015, −0.000] | −0.006 [−0.020, +0.004] |

- **Findings.**
  - The E4-P2b result replicates on fresh data: GRU 0.184 and perfect-map terrain GRU 0.165 m at 3 s, the same
    numbers as before.
  - The scanner keeps part of the gain. On step windows of unseen tiles it recovers about half of the perfect map's
    gain at 2 s (0.011 of 0.021 m) and 37 % at 3 s. At 0.3–0.9 m/s the robot covers 0.9–2.7 m in 3 s, mostly beyond the scanner's
    0.8 m reach, so this is what a short-range sensor should give: it helps for what is about to be stepped on, not
    for terrain further ahead.
  - On dev-test (seen tiles) the scanner's gain is within seed noise overall and slightly negative on steps at 2 s
    (+0.007), so the step gain is not robust across splits. Grades and curbs: differences are within the seed range.
- **Caveats.** The scanner is noise-free and has no occlusion (it ray-casts from 20 m above), so this is an upper bound
  for a real height scan. Its map comes from a single frame; accumulating scans over time (an elevation map as the
  robot walks) would extend the range behind and around, not ahead. The cross-slope direction shift between train
  and test tiles (E4-P2) still applies. Prediction only, no navigation test. Gains are small in absolute terms
  (1–4 cm on a 0.18–0.29 m error).

## Sidewalk appearance (rendering only, 2026-11-10)

- **Problem.** The rough task gives the terrain a white marble-tile material, and every tile was one surface, so
  camera views showed no sidewalk.
- **Change.** `go2wm.sim.sidewalk_texture.sidewalk_albedo` paints the same tile parameters as the geometry:
  - concrete panels with tooled joints every 1.5 m and a centre joint, per-panel tint, wear and hairline cracks;
  - heaved slabs in older concrete;
  - beyond a curb: curb top, gutter pan, asphalt with an edge line; otherwise grass with dirt patches.

  `go2wm.sim.isaac_texture` bakes it into one PNG over the merged terrain mesh (2 cm/px, cached in
  `D:/go2wm/cache/textures`). It is applied as the terrain's visual material while the mesh is created
  (`isaac.terrain_texture`). The 2D replays (`draw_sidewalk`, now in `go2wm.eval.plots`) use the same painter.
  Physics and collision are unchanged.
- **Pitfalls found.**
  - Swapping the material after the scene exists left tiled-camera frames all black in about 2 of 3 launches.
    Creating it with the mesh fixed this (3/3 launches).
  - With the onboard camera on, anti-aliasing is set to FXAA: DLSS renders a 224 px camera at ~130 px, below its
    300 px minimum.
  - The viewport video renderer sometimes returns black frames on a first run, so frames are re-grabbed.
- Videos: `D:/go2wm/outputs/videos/09_e4_cam_textured_{climb_80001,stall_80037,fall_80007}.mp4`.

## P4-a: ground-truth sidewalk scene graph (2026-11-10)

- **Purpose.** One graph for two uses:
  - inspection: an inventory of hazards with ADA checks;
  - mission grounding for the voice / HITL layer: named regions and landmarks, plus keep-out boxes for
    high-risk features.

  Built from the tile parameters, so it is exact ground truth (`go2wm.scene.graph`, `scripts/scene_graph.py`).
- **Nodes.**
  - Regions: walkway, verges, road.
  - Features: slabs, ramps, curb.
  - Panels between joints, two rows on walks ≥ 2 m wide.
  - Cracks: the same polylines the texture paints, so an RGB sensor can be scored on them.
  - Landmarks: start, end.
- **Edges.** `part_of`, `next` (with the joint step in metres and its ADA class), `beside`, `covers`, `spans`,
  `on`, `borders`.
- **Hazards.**
  - Vertical displacement per joint line (ADA 303: ≤ 6.4 mm ok, ≤ 13 mm bevel, above = violation).
  - Running slope (403.3: > 5 % is a ramp, > 8.33 % violation) and cross slope (> 2.08 %).
  - Curb drop-off and cracks.
  - Each hazard carries a robot risk (low / medium / high). It comes from the E4-sweep thresholds (step ≥ 15 cm,
    grade ≥ 28 % high) and is a placeholder for a world-model stall probability.
- **Output.** JSON, a text description (input for an LLM mission compiler) and a figure per tile in
  `D:/go2wm/outputs/scene_graphs/`.
  - Test tiles: 34–43 nodes and 70–105 edges each.
  - Hazards per tile: 7–11, of which 1–6 are ADA violations.
- **Not yet.** A sensor-built graph scored against this one (hazard recall per sensor), and risk from the world model.

## E4-C setup: sensor comparison rig (2026-11-10)

- **Sensors** (`go2wm.sim.sensors`; approximate device models, recorded together in one collection so they see
  the same episodes):

| sensor | model | mount | FOV / rays per 0.1 s | range, noise |
|---|---|---|---|---|
| `depth_d435` | RealSense D435-like | head, 30° down | 87 × 58°, 106 × 60 | 0.2–4 m, σ = 2 mm + 5 mm·z², 3 % dropout |
| `depth_d555` | RealSense D555-like (95 mm baseline) | same as D435 | 87 × 58°, 106 × 60 | 0.52–6 m, σ = 2 mm + 2.5 mm·z², 3 % dropout |
| `l1` | Unitree L1-like (stock Go2) | chin, dome axis forward 15° down | 360 × 90° dome, 2160 random | 0.05–30 m, σ 2 cm |
| `mid360` | Livox Mid-360-like | back, upright | 360° × (−7°, +52°), 20 k random | 0.1–40 m, σ 2 cm |
| `mid360_tilt` | same | back, 30° forward | same | same |
| `scan_height_m` | the policy's height scanner | – | 1.6 × 1.0 m grid, 187 | noise-free |
| RGB | Go2 front camera-like | head, 10° down | 224 × 224, 100° | MP4 per episode |

- **Implementation.**
  - Range sensors are Isaac ray casters whose rays are set in the base frame. The LiDARs get fresh random
    directions every frame, which mimics their non-repetitive scan pattern.
  - Hits are corrupted (range limits, dropout, noise), then binned into a robot-centred, yaw-aligned 5 cm grid
    (x −1 to 3.5 m, y ±1.25 m): mean height in the terrain frame plus hit count.
  - Arrays: `sensor_<name>_elev_m` (float16) and `sensor_<name>_count` (uint8). RGB is `<episode>_rgb.mp4`.
  - Raw point clouds would not fit on disk (25 GB free).
- **Smoke test** (2 test episodes, per frame):
  - Depth camera: 66–71 % of the ground 0.3–2 m ahead, p90 error 0.7–0.9 cm.
  - Tilted Mid-360: 43–50 %, p90 1.0–1.3 cm. L1: 39–40 %, p90 1.5–1.8 cm.
  - Upright Mid-360: 0–1 %. Its lowest beam (−7°) first meets the ground ~3.7 m away, so it is nearly blind to
    terrain near the robot.
  - Median error +0.1 to +0.2 cm for all sensors.
  - Cost: 87 s per 300-step episode (about twice the plain collection) and ~4 MB per episode.
- **Next.** Full collection `data/go2_e4c_{train,test}` (same tiles and seeds), then per sensor:
  1. map accuracy and coverage, single frame and accumulated;
  2. the prediction gain of a terrain GRU fed that sensor's map;
  3. hazard recall against the ground-truth scene graph.

## Next experiments (Isaac)
- P1: sidewalk terrain with the rough-terrain policy (`configs/isaac_go2_rough_e4s1.yaml`, `scripts/isaac_terrain_smoke.py`);
  needs the pretrained rough checkpoint.
- Beyond that, these runs show **the learned model has no navigation advantage on flat ground with cylinders**: every planner difference is
  explained by locomotion deadband and cost tuning. Testing H1/H2 needs conditions where dynamics are hard to model by hand: sidewalk
  terrain (curbs, slopes, friction changes, from SidewalkAtlas `scripts/terrain`), and cameras with V-JEPA 2 for H1.

## E4-C (1): sensor map accuracy (2026-11-11)

- **Data.** Full E4-C collection with the sensor rig: 144 train episodes (12 tiles, seeds 70000+) and 48 final-test
  episodes (6 unseen tiles, seeds 80000+), `data/go2_e4c_{train,test}`. Results below are on the 48 test episodes.
- **Method.** `python scripts/evaluate_sensors.py --data data/go2_e4c_test --out outputs/e4c/test`.
  - Ground truth: the terrain mesh. Cells within 7.5 cm of slab edges or the curb are excluded from the elevation
    metrics. Poses are simulator truth.
  - Fusion: the last K frames with perfect registration, plus a per-frame vertical registration error σ_reg (a
    sensitivity check, not an odometry model).
  - Joint steps: a plane fit each side of every transverse joint within one frame, scored against the ADA classes.
  - Sanity: median signed error is +0.07 to +0.11 cm for every sensor, so frames and ground truth line up.
- **Single frame, p90 |height error| (cm) / coverage, by distance ahead.**

| sensor | p90 < 1 cm range | 0.3–0.8 m | 0.8–1.5 m | 1.5–2.5 m | 2.5–3.5 m |
|---|---|---|---|---|---|
| D555-like | 0.6–2.9 m | 0.48 / 14 % | 0.50 / 70 % | 0.65 / 47 % | 1.04 / 16 % |
| D435-like | 0.6–2.1 m | 0.50 / 15 % | 0.55 / 70 % | 0.93 / 49 % | 1.74 / 17 % |
| L1-like | none (1.0 cm at 1.5–2.5 m) | 1.60 / 50 % | 1.35 / 20 % | 1.01 / 5 % | 1.34 / 1 % |
| Mid-360, tilted 30° | none | 1.41 / 18 % | 1.08 / 60 % | 0.98 / 34 % | 1.03 / 13 % |
| Mid-360, upright | none | (1 % coverage) | 1.42 / 1 % | 1.10 / 1 % | 1.72 / 1 % |
| D555 + L1 | – | 1.42 / 54 % | 0.54 / 74 % | 0.67 / 49 % | 1.03 / 17 % |

- **Fusion over K frames** (p90 cm at 0.8–1.5 m / 1.5–2.5 m; σ_reg = 0 → 5 mm):
  - D555, K = 10: 0.47 / 0.54, rising to 0.58 / 0.76. Coverage 83 % / 84 %.
  - L1 needs K = 10–20 to reach 0.8–0.9 cm, and coverage beyond 1.5 m stays 27–37 %.
  - Tilted Mid-360, K = 20: 0.59 / 0.62, rising to 0.68 / 0.81.
  - Registration error at 5 mm per frame costs the depth camera about 0.1–0.2 cm at p90.
- **Joint step height** (single frame; violation = step > 13 mm):

| sensor | joints at | p90 err | ADA class agree | violation recall | false flags on flush joints |
|---|---|---|---|---|---|
| D555 | 0.5–1.5 m | 2.1 mm | 99 % | 99 % | 0 % |
| D555 | 1.5–2.5 m | 5.7 mm | 98 % | 97 % | 1 % |
| D435 | 1.5–2.5 m | 12.6 mm | 96 % | 96 % | 4 % |
| D435 | 2.5–3.2 m | 35.9 mm | 85 % | 96 % | 21 % |
| L1 | 0.5–1.5 m | 10.9 mm | 89 % | 99 % | 19 % |
| Mid-360 tilted | 0.5–1.5 m | 15.9 mm | 95 % | 99 % | 7 % |

- **Findings.**
  - Only the depth cameras meet the < 1 cm requirement in a single frame, and only from 0.6 m out to 2.1 m (D435)
    or 2.9 m (D555). The D555's wider baseline is what makes the difference beyond 1.5 m: about 3× lower joint-step
    error there, and 1 % vs 4 % false flags.
  - The depth cameras are blind closer than about 0.5 m (14–15 % coverage at 0.3–0.8 m). The L1 covers that zone
    (50 %) but at 1.6 cm p90, so a D555 + L1 pair gives the best coverage near the feet, but not < 1 cm there in
    one frame.
  - The LiDARs (2 cm range noise) reach p90 ≈ 1 cm only by averaging many hits, i.e. fusing 10–20 frames. Even
    then the L1's sparse coverage leaves 19–29 % false flags on flush joints. They are fine for detecting
    violations (recall 97–99 %) but not for measuring steps to ADA precision.
  - An upright Mid-360 on the back is useless for the ground near the robot (1 % coverage).
- **Caveats.** These are approximate device models: Gaussian range noise, random dropout, no multipath, no
  sunlight, no reflectance effects, and no body occlusion beyond what the rays hit. The depth noise model is the
  main assumption; a real D555 outdoors will likely be worse. Registration is perfect apart from the σ_reg
  sensitivity check. The upright Mid-360's 44 cm p90 at 0.3–0.8 m comes from about 1 % of cells, so it is not
  meaningful.

## E4-C (2): does a sensor's map help prediction? Single frame (2026-11-11)

- **Setup.** The E4-S protocol on E4-C data (`data/go2_e4c_train`, manifest `e4`: 100/22/22; final test
  `data/go2_e4c_test`, 48 episodes on 6 unseen tiles, `all_test`). Eight models × 3 seeds, identical except for the
  terrain input: GRU (none), perfect map, height scanner, and the single-frame elevation grid of each E4-C sensor.
  - New `terrain_source: "sensor:<name>"`: the sensor's 5 cm grid at the window start is averaged onto the 10 cm model
    map (`go2wm.models.terrain_map.sensor_maps`).
  - Unobserved points take the mean of observed neighbours, grown outward, as the scanner map holds its edge values.
  - Heights are relative to the ground under the base in the tile frame. This is simulator truth here, standing in
    for the leg-kinematic ground estimate.
  - Bug found and fixed while building this: `gt_ground_z_m` is in a frame shifted by a per-tile constant
    (4–10 cm), so it cannot be used as the reference.
  - Configs `configs/world_model_e4c_*.yaml`; `scripts/e4c_train_queue.ps1`; summary
    `python scripts/summarize_seeds.py outputs/e4c/final_test/prediction_metrics.json --ref e4c_gru`.
- **Position error (m), seed mean ± sd.**

| model | dev-test 3 s | final 1 s | final 2 s | final 3 s |
|---|---|---|---|---|
| GRU, no terrain | 0.201 ± 0.004 | 0.049 | 0.108 | 0.180 ± 0.005 |
| perfect map | **0.163** ± 0.005 | 0.046 | **0.094** | **0.154** ± 0.006 |
| height scanner | 0.175 ± 0.005 | 0.048 | 0.099 | 0.164 ± 0.001 |
| D555 | 0.188 ± 0.005 | 0.049 | 0.108 | 0.178 ± 0.007 |
| D435 | 0.191 ± 0.009 | 0.049 | 0.107 | 0.179 ± 0.009 |
| L1 | 0.190 ± 0.007 | 0.051 | 0.107 | 0.176 ± 0.002 |
| Mid-360 tilted | 0.189 ± 0.008 | 0.048 | 0.104 | 0.174 ± 0.004 |
| Mid-360 upright | 0.203 ± 0.004 | 0.049 | 0.109 | 0.180 ± 0.003 |
| kinematics | 0.427 | 0.161 | 0.316 | 0.478 |

- **Gain over the GRU at 2 s on steps** (seed-mean diff, m; [min, max] over paired seeds):

| model | dev-test (19 eps, seen tiles) | final test (39 eps, unseen tiles) |
|---|---|---|
| perfect map | −0.036 [−0.038, −0.034] | −0.027 [−0.036, −0.017] |
| height scanner | −0.027 [−0.031, −0.025] | −0.010 [−0.020, −0.000] |
| D555 | −0.018 [−0.026, −0.012] | +0.008 [+0.003, +0.017] |
| D435 | −0.015 [−0.026, −0.006] | +0.007 [−0.001, +0.018] |
| L1 | −0.017 [−0.024, −0.012] | +0.004 [−0.002, +0.014] |
| Mid-360 tilted | −0.012 [−0.025, −0.006] | +0.002 [−0.000, +0.004] |

- **Findings.**
  - The perfect-map and scanner gains replicate on a third fresh sample (final 3 s: GRU 0.180, map 0.154, scanner
    0.164 m).
  - **No single-frame sensor map helps on steps of unseen tiles.** The D555 is slightly worse than no terrain
    (+0.008 m at 2 s, all three seeds worse). On seen tiles the same models do help (−0.012 to −0.018 m), so what they
    learn about steps does not transfer.
  - Every terrain input gives a small, consistent gain on plain windows of the final test (−0.012 to −0.019 m at
    3 s for all sensors except the upright Mid-360), so the overall 3 s numbers (0.174–0.179 m) sit just below the GRU.
- **Diagnosis** (final-test step windows, |sensor map − perfect map|, median / p90 cm, |y| ≤ 0.3 m):

| map region | scanner | D555 | L1 | Mid-360 tilted |
|---|---|---|---|---|
| under the robot (x −0.3 to 0.3 m) | 0.1 / 0.3 | 5.1 / 11.8 | 0.6 / 6.5 | 7.4 / 12.7 |
| 0.3–0.8 m | 0.1 / 0.3 | 0.3 / 11.4 | 0.4 / 1.2 | 1.4 / 10.9 |
| 0.8–1.5 m | 0.2 / 11.5 | 0.2 / 0.6 | 0.4 / 2.0 | 0.4 / 2.1 |

  - The depth camera and tilted Mid-360 are blind under and just ahead of the robot. Filling copies whatever they see
    at 0.5 m+ into that zone, so once the robot is at a step, the map shows the far side's level under its feet.
  - Relief in the 1.5 m corridor ahead: 15.6 cm true vs 6.4 cm in the D555 map (L1 16.0, scanner 14.5).
  - The scanner and the perfect map are exact under the robot, which is the part the model uses most at steps.
  - So the comparison was unfair to the cameras: a real robot keeps an elevation map that accumulates frames as it
    walks. That is E4-C (3).

## E4-C (3): fused sensor maps (running, 2026-11-11)

- `terrain_source: "sensor:<a>[+<b>]@K"`: the observed cells of the last K frames, from one or more sensors, moved
  into the current robot frame with the recorded poses and averaged (`fused_sensor_maps`). Registration is perfect,
  so this is an upper bound for an odometry-registered elevation map; the E4-C (1) σ_reg check puts 5 mm/frame at
  about +0.1–0.2 cm p90 on accuracy.
- K = 1 reproduces the single-frame maps exactly (max difference 0), so E4-C (2) checkpoints are unchanged.
- Map check, final-test step windows (subsample of 16 episodes), p90 |error| cm:

| map region | D555 single | D555 @10 | L1 @10 | D555 + L1 @10 | Mid-360 tilted @10 |
|---|---|---|---|---|---|
| under the robot | 12.1 | 6.2 | 1.5 | 1.5 | 10.9 |
| 0.3–0.8 m | 3.0 | 0.6 | 0.6 | 0.5 | 2.3 |
| 0.8–1.5 m | 1.0 | 0.9 | 3.1 | 1.0 | 3.0 |
| 1.5–2.5 m | 2.7 | 1.4 | 4.4 | 1.4 | 2.7 |
| corridor relief (true 14.6 cm) | 12.1 | 14.4 | 14.7 | 14.6 | 13.0 |

- Training: `scripts\e4c_train_queue.ps1 -Models depth_d555_k10_gru,depth_d435_k10_gru,l1_k10_gru,mid360_tilt_k10_gru,depth_d555_l1_k10_gru`
  (5 × 3 seeds).

## E4-C (3) + E5 results: fused sensor maps and pretrained models (2026-11-11, parked after this)

- **Setup.** Same protocol as E4-C (2): `data/go2_e4c_train` (manifest `e4`), 3 seeds per model, final test
  `data/go2_e4c_test` (48 episodes, 6 unseen tiles), evaluated together with the no-terrain GRU and perfect-map
  references. `outputs/e5/final_test/{prediction_metrics,seed_summary}.json`;
  `python scripts/summarize_seeds.py outputs/e5/final_test/prediction_metrics.json --ref e4c_gru`.
- **Position error at 3 s (m, seed mean ± sd)** and gain over the GRU on step windows (39 episodes):

| model | 3 s overall | step, 2 s | step, 3 s |
|---|---|---|---|
| GRU, no terrain | 0.180 ± 0.005 | – | – |
| perfect map | **0.154** ± 0.006 | −0.027 | −0.052 |
| D555 @10 frames | 0.165 ± 0.005 | −0.007 [−0.012, +0.002] | −0.019 [−0.031, −0.003] |
| D435 @10 | 0.168 ± 0.005 | −0.004 | −0.014 |
| L1 @10 | 0.165 ± 0.001 | −0.001 | −0.016 |
| Mid-360 tilted @10 | 0.169 ± 0.002 | +0.004 | −0.001 |
| D555 + L1 @10 | 0.165 ± 0.005 | −0.001 [−0.008, +0.007] | −0.016 [−0.025, −0.000] |
| DINOv2-small CLS + mean (A1) | 0.333 ± 0.010 | +0.192 | +0.341 |
| DINOv2-small 2 × 2 grid (A1) | 0.331 ± 0.002 | +0.209 | +0.348 |
| D555 + L1 @10 + DINOv2 grid | 0.294 ± 0.024 | +0.155 | +0.273 |
| Depth Anything, calibrated @10 (C) | 0.180 ± 0.003 | +0.003 | +0.004 |
| kinematics | 0.478 | | |

- **Findings.**
  - Fusing 1 s of frames turns the sensor maps from no help (single frame, E4-C 2) into a real gain: 3 s error
    0.165–0.169 m vs 0.180 m, about 45–60 % of the perfect map's overall gain. On steps the gain is about a third
    of the perfect map's at 3 s, and within seed noise at 2 s. Adding the L1 to the D555 does not improve prediction,
    although it makes the map more accurate under the robot.
  - **Frozen DINOv2 image features make prediction much worse on unseen tiles** (0.33 m vs 0.18 m at 3 s), most of
    all on steps, and they spoil the fused map as well. The likely cause is overfitting: with 12 training tiles, the
    model can key on tile appearance, which does not transfer. Not checked on dev-test before parking.
  - Calibrated monocular depth (Depth Anything) adds nothing, consistent with its 5–20 cm map error (E5 map check).
- **Status.** Parked on 2026-11-11 when the project was repositioned on the Phase 1 inspection demo. What carries
  over: the fused D555 + L1 map is the sensor map for Phase 1 perception (accuracy: E4-C (1) and (3)); pretrained
  image features need far more tile variety, regularization or a different design before they are worth revisiting.

## Phase 1, step 1: sidewalk defect library (2026-11-11)

- **Purpose.** Realistic things to inspect (Phase 1 repositioning). One parametric description per defect drives the
  geometry, appearance, sensor effects and the ground-truth scene graph (`go2wm.sim.defects`; tile params
  `"defects": [...]`).
- **Core types.**

| type | geometry | appearance | sensors | scene graph (ADA, robot risk) |
|---|---|---|---|---|
| tree-root upheaval | half or full panel tilted up to `lift` (1–8 cm), vertical fault at the high joint | lighter panel, shadowed fault edge; tree, soil and surface roots on the verge | measured as a joint step | level change (303); risk from the E4 step thresholds; tree = `asset` node `causes` it; the joint hazard names it as `cause` |
| open crack | groove, width 3–30 mm, depth 5–30 mm; sub-grid cracks rasterised with the same cross-section area | dark line with halo | mostly visual (too thin for depth beyond close range) | opening > ½ in (302.3) = violation, else monitor |
| hole / pothole | steep-walled elliptical depression, 3–15 cm deep | soil interior, broken rim | geometry | opening = violation; risk high if ≥ 8 cm deep and ≥ 20 cm wide |
| puddle | shallow bowl (0.8–2.5 cm) | water (dark, bluish, sky), wet rim | `range_dropout`: 85 % of depth-camera and 60 % of LiDAR returns lost on water | ponding, ADA n/a, slip risk low |
| obstruction (bin, pole, debris) | none (a SceneSpec obstacle) | – | geometry | clear width < 36 in (403.5.1) = violation |

- **Grid.** Defect tiles use a 2 cm / 1 mm height field (was 5 cm / 5 mm); a 24 × 7 m tile is 1200 × 350 cells.
  `terrain.random_defect_tile` makes tiles with mild grade, cross slope and curb plus 4–8 non-overlapping defects
  in x ∈ [−L/2 + 3, L/2 − 1.5] (route start area clear).
- **Not modelled.** Per-region friction (one physics material per terrain, so puddles are not slippery in physics);
  water's own surface (rays that survive the dropout hit the bowl floor).
- **Offline check.** `python scripts/defect_tiles.py --n 6 --seed 0` → `outputs/defect_tiles/` (params, graph JSON,
  three-view PNG per tile). Seed 0: 5–7 defects and 9–15 hazards per tile, 2–7 ADA violations. Tests:
  `tests/test_defects.py`.
- **Next.** Isaac smoke test on these tiles (mesh build, texture, rough policy over upheavals and holes, obstacles
  from `obstacles_from_defects`, puddle dropout hooked into the sensor rig), then the mission scene and runner.

## Phase 1, step 2: Isaac smoke test on defect tiles (2026-11-11)

- **Setup.** `scripts\isaac_python.bat scripts\isaac_defect_smoke.py --use 3 4 5 --no-rgb` (seed-0 tiles from
  `outputs/defect_tiles/tiles.json`, built side by side on the 2 cm / 1 mm grid). Mission scene
  (`go2wm.mission.scene`): start x = −11, goal x = +10.5, walkway bounds, obstructions as 0.3 m cylinders standing
  on the surface. Rough-terrain policy driven by the existing stack (A* + kinematic CEM-MPC + safety filter).
  Holes, puddles, cracks and upheavals are deliberately not obstacles. D555 + L1 recorded with puddle dropout.
  Data `data/go2_defect_smoke_norgb`, report `outputs/defect_smoke/report_norgb.json`,
  videos `outputs/videos/12_defect_smoke_tile_9000{0,1,2}.mp4`.
- **Result.** 3/3 tiles reached the goal (21.1–21.4 of 21.5 m in 31.2–31.6 s), no falls, no collisions; every
  obstruction passed with ≥ 0.94 m centre distance (0.64 m clearance from the 0.3 m cylinder). The robot walked
  over 6 upheavals (+1 to +8 cm), 2 holes and 2 puddles at 0.59–0.71 m/s without stalling. Foot slip near defects
  is no higher than on plain walkway (p95 0.82–0.96 vs 0.96–1.17 m/s; the slip signal peaks at every touchdown);
  the body dips by up to ~7 cm over holes and upheavals on tile 3 (height p5 0.31 vs 0.38 m). In puddles only
  ~10 % of in-puddle map cells get a return per frame (dropout working). Fused-map error vs truth: median 0.1 cm;
  11/13 scene-graph hazards come into sensor view on tile 5 (the two missed are 3 and 19 mm cracks).
- **Rendering crash.** Two launches with the camera (`--rgb`, rendering experience) crashed at app start-up with an
  access violation in the viewport's Hydra engine (`omni.kit.widget.viewport ... __enable_hydra_engine`), before any
  project code ran; the same configuration worked on 2026-11-10/11 (E4-C). Without the camera the headless app
  starts and runs. Not yet diagnosed (driver / display-state suspects); camera runs are needed for crack and puddle
  detection.
- **Limits.** Range sensors ray-cast only against the ground mesh (`mesh_prim_paths=["/World/ground"]`), so
  obstructions are invisible to the D555/L1 maps; the planner knows them from the scene map. Obstacle cylinders are
  one size (0.3 m). Per-region friction not modelled.

## Phase 1, step 3: crack measurement from the colour cameras (2026-11-11)

`scripts/crack_measurement_study.py --trials 8` (`outputs/crack_study`). Synthetic images (`go2wm.sim.synthetic_camera`:
true-width crack texture, 2x supersampled, 0.8 px blur, noise, 8-bit) of a random ~0.5 m crack on plain concrete,
orthorectified with the known camera geometry and measured without ground truth: Zhang–Suen skeleton, pruned
centre line, length = tile-PCA centre-line length, width = dark area / length. Appearance only (no lighting, glare or
motion blur), so these numbers are optimistic.

Cell: detected % · median width estimate [mm] (|error|) · median length error %

| crack width | D555 RGB 0.80 m (1.09 mm/px) | D555 RGB 1.20 m (2.24 mm/px) | D555 RGB 1.60 m (3.60 mm/px) | D555 RGB 2.00 m (5.14 mm/px) | D405 RGB 0.52 m (0.54 mm/px) |
|---|---|---|---|---|---|
| 1 mm | 100 % · 3.0 (2.0) · 0 % | 0 % | 0 % | 0 % | 100 % · 2.2 (1.2) · 1 % |
| 2 mm | 100 % · 4.8 (2.8) · 0 % | 100 % · 5.3 (3.3) · 5 % | 0 % | 0 % | 100 % · 3.2 (1.2) · 0 % |
| 3 mm | 100 % · 5.9 (2.9) · 1 % | 100 % · 7.5 (4.5) · 1 % | 0 % | 0 % | 100 % · 4.3 (1.3) · 0 % |
| 5 mm | 100 % · 8.0 (3.0) · 1 % | 100 % · 10.6 (5.6) · 2 % | 88 % · 10.0 (5.0) · 26 % | 0 % | 100 % · 6.3 (1.3) · 1 % |
| 8 mm | 100 % · 10.7 (2.7) · 1 % | 100 % · 13.3 (5.3) · 1 % | 100 % · 14.8 (6.8) · 2 % | 38 % · 14.4 (6.4) · 2 % | 100 % · 9.1 (1.1) · 3 % |
| 13 mm | 100 % · 15.7 (2.7) · 2 % | 100 % · 18.5 (5.5) · 2 % | 100 % · 19.4 (6.4) · 2 % | 100 % · 17.6 (4.6) · 3 % | 100 % · 14.2 (1.2) · 2 % |
| 20 mm | 100 % · 22.7 (2.7) · 3 % | 100 % · 24.4 (4.4) · 3 % | 100 % · 25.2 (5.2) · 2 % | 100 % · 23.9 (3.9) · 3 % | 100 % · 20.8 (0.8) · 4 % |

- **Length** is good wherever the crack is detected (≤ 5 % error, except 5 mm cracks at 1.6 m).
- **Detection:** the D555 sees 1 mm cracks only at 0.8 m and 2–3 mm cracks out to 1.2 m. The D405 at its stand-off sees all widths.
- **Width** is over-read by a nearly constant ~2.2 image pixels at ≤ 1.2 m (blur, bilinear orthorectification, dark
  threshold), i.e. +2.7 mm on the D555 at 0.8 m and +1.2 mm on the D405. Because it is constant in pixels it can be
  calibrated: `BIAS_PX = 2.2` (fitted on this run); the 24-trial run uses disjoint seeds (`--seed-offset 100`) and reports
  raw and calibrated width.
- **Crack depth is not measurable** with either sensor for sub-cm cracks. The D405 depth (~0.5 mm footprint, 0.1–0.3 mm
  step precision) resolves the depth of cracks wider than a few mm only.

## Phase 1, step 4: strategy loop, perception fixes and the inspection report (2026-11-11/12)

Mock kinematic missions (`scripts/run_mission.py`, `--workers` for parallel runs) with live D555 + D405 from the offline
rig, detection → spatial memory → measurement, three strategies (walkover / slow / approach to a D405 stand-off).

**Bugs found while building the report, fixed (all earlier mission numbers are void):**
- **Flying pixels wrapped around the image border** (`np.roll`): the nearest image row was averaged with the farthest,
  giving ~7 cm ghost pits at ~1.5 m whenever the robot yawed (tile 0 walkover: 23 false findings). This also affected
  the realistic profile of the sensor error study and the near/far step study, which are rerun (`outputs/*_v2`).
- **The offline rig had no puddle dropout** (only the Isaac rig had it): puddles were invisible wide shallow dips → 0/3
  found. Now applied as in Isaac (85 % of pinhole returns lost on water).
- **Detector:**
  - clusters at the same distance spread ≥ 0.6 m across the walk form one transverse edge;
  - compact raised clusters are the far wall of a hole (a depression), not an edge;
  - strong 2-cell clusters are accepted far away;
  - detection runs every step.
- **Memory:**
  - overlapping findings are consolidated (aliases for policy targets);
  - depressions aligned across the walk are relabelled as an edge;
  - an edge's footprint (its low side and the ground it occludes) and a deep hole's shadow dropout merge into it;
  - puddle dropouts group within 0.9 m.
- **Scoring:**
  - step error is compared by magnitude (the sign depends on the viewing side);
  - depressions on cracks deep/wide enough for depth are counted as crack findings, not false ones.

Dev tiles (the 6 tiles the fixes were developed on, realistic profile): walkover 0 false findings, all upheavals,
holes and puddles found except one hole at the curb edge (y = 1.32 on a 3.1 m walk); approach 2 false findings.
Held-out evaluation: 24 new tiles (`outputs/defect_tiles_eval`, seed 100), realistic and pessimistic profiles
(overnight queue `scripts/phase1_overnight.ps1`).

**Inspection report** (`scripts/inspection_report.py`, `go2wm.mission.report`): HTML + Markdown + JSON per mission,
built only from the robot's memory. It contains:
- per finding: location, measured value (± spread), ADA 303 class (ok / bevel / violation, or unverified for
  dropouts), status, best sensor, detection and measurement counts, recommended action;
- a synthetic camera evidence crop rendered at the recorded pose of the best observation;
- a route map with the findings;
- a separate ground-truth appendix for evaluation.

### Sensor studies rerun after the flying-pixel fix (2026-11-12)

`outputs/sensor_errors_v2`, `outputs/sensor_errors_ablation_v2`, `outputs/near_far_steps_v2`. The wrap-around bug barely
changed these map-level numbers (realistic D555 single-frame p90 at 0.3–0.8 m: 1.45 → 1.35 cm), so the conclusions hold.

Ablation (realistic, D555, p90 at 0.3–0.8 m / step p90):

| errors switched off | absolute height p90 | step p90 |
|---|---|---|
| none | 1.0 cm | 54 mm |
| calibration | 0.44 cm | 52 mm |
| stereo (flying pixels, fixed pattern) | 1.0 cm | 2.9 mm |
| odometry | 1.0 cm | 54 mm |

- **Calibration** dominates absolute height error.
- **Flying pixels at step edges** dominate the D555 step error.
- **Odometry drift** is negligible over 1 s.

Near/far step study, realistic profile:
- **D405 at its 0.53 m stand-off:** 0.1–0.4 mm p90 with 100 % ADA class agreement for every step size, in all profiles.
- **D555:** ≤ 1.1 mm p90 within 1.2 m for steps up to 11 mm; degrades sharply beyond 1.6 m.

## Phase 1, step 5: final strategy evaluation on fresh tiles (2026-11-12, frozen at commit 596cdb4)

24 fresh defect tiles (`outputs/defect_tiles_final`, seed 200), not looked at before the perception was frozen. The
seed-100 tiles were used to diagnose failures, so they count as a second dev set. Mock kinematics, D555 + D405,
realistic profile, `outputs/missions_final`. Defects to find: upheaval faults, holes, puddles (cracks are scored
separately).

| strategy | time [s] | recall | precision | false / mission | upheaval found / D405 / err [mm] / ADA ok | hole found / D405 / depth err [mm] | puddle found |
|---|---|---|---|---|---|---|---|
| walkover | 33.7 | 86 % | 100 % | 0.00 | 22/26 / 9 / 0.5 / 100 % | 22/26 / 1 / 2.8 | 17/19 |
| slow | 37.2 | 83 % | 94 % | 0.17 | 22/26 / 9 / 0.3 / 100 % | 20/26 / 0 / 2.1 | 17/19 |
| approach | 40.0 | 80 % | 86 % | 0.38 | 22/26 / 18 / 0.5 / 100 % | 18/26 / 12 / 0.7 | 17/19 |

Reading:
- **Walkover is the best detector:** no false findings on unseen tiles, and 86 % of defects found.
- **Approach buys close-range evidence:**
  - D405 measurements on 2× the upheavals and 12× the holes;
  - hole depth error 2.8 → 0.7 mm;
  - +6.3 s per 22 m route.
- **Approach's cost in detection quality:** it loses 4 holes and adds 9 false findings. Two mechanisms:
  1. While it dwells, the stereo fixed-pattern error repeats in every frame, so a noise blob ahead gets "confirmed".
  2. Up close, a hole's far wall reads as a short edge, and the memory then absorbed the hole into it.

  Both are addressed after the freeze (multi-view baseline ≥ 15 cm for confirmation; a short edge inside a hole is
  its wall) and evaluated on another fresh set (seed 300), so these numbers stay as the frozen result.
- **All approach missions reached the goal.**
- **Upheaval misses** are mostly small (1.7–2.3 cm) half-width faults.
- **Puddles** are found from depth dropout alone; their depth is not measured (camera needed).

### Crack width calibration on held-out seeds (2026-11-12)

`scripts/crack_measurement_study.py --trials 24 --seed-offset 100` (`outputs/crack_study_24`). These are seeds disjoint
from the 8-trial run the 2.2 px bias was fitted on. Median calibrated width [mm] (|error|):

| crack width | D555 0.80 m | D555 1.20 m | D555 1.60 m | D405 0.52 m |
|---|---|---|---|---|
| 1 mm | 0.6 (0.4) | (4 % detected) | – | 1.0 (0.1) |
| 2 mm | 2.1 (0.1) | 0.2 (1.8) | (8 % detected) | 2.0 (0.1) |
| 3 mm | 3.3 (0.3) | 2.6 (0.4) | (21 % detected) | 3.0 (0.1) |
| 5 mm | 5.3 (0.3) | 5.4 (0.4) | 1.7 (3.3) | 5.0 (0.1) |
| 8 mm | 8.3 (0.3) | 8.3 (0.7) | 6.8 (1.2) | 8.0 (0.1) |
| 13 mm | 13.0 (0.3) | 13.2 (0.3) | 11.8 (1.2) | 12.9 (0.1) |
| 20 mm | 19.7 (0.5) | 18.9 (1.1) | 16.5 (3.5) | 19.2 (0.9) |

- **Calibration generalises within 1.2 m:**
  - D405 at stand-off: ±0.1 mm;
  - D555 at 0.8 m: ±0.5 mm;
  - D555 at 1.2 m: ±0.7 mm for cracks ≥ 3 mm.
- **Beyond 1.6 m** the over-read is no longer constant in pixels (thin cracks fade below the threshold), so the
  correction over-subtracts.
- **The 13 mm ADA opening limit** is decided correctly from ≤ 1.2 m.
- **Caveats:** synthetic, uniformly lit images. Real glare, shadows, wet concrete and motion blur will widen these
  errors. The 2.2 px figure must be re-fitted on the real camera.

## Phase 1, step 5b: final evaluation v2 on fresh tiles (2026-11-12, frozen at commit 408f987)

Perception v2 adds two rules:
- **Multi-view confirmation:** a finding needs detections from viewpoints ≥ 15 cm apart.
- **Hole walls:** a short edge inside a hole is that hole's wall, but only if the hole has at least as many detections
  and no step-shaped votes.

These were developed on the two dev sets only, then evaluated on 24 fresh tiles (`outputs/defect_tiles_final2`,
seed 300), realistic profile (`outputs/missions_final2`):

| strategy | time [s] | recall | precision | false / mission | upheaval found / D405 / err [mm] | hole found / D405 / depth err [mm] | puddle found |
|---|---|---|---|---|---|---|---|
| walkover | 33.5 | 86 % | 91 % | 0.29 | 14/21 / 5 / 2.1 | 30/34 / 5 / 1.2 | 28/29 |
| slow | 37.8 | 83 % | 92 % | 0.25 | 14/21 / 4 / 1.7 | 29/34 / 4 / 1.5 | 27/29 |
| approach | 38.9 | 83 % | 91 % | 0.29 | 15/21 / 11 / 1.1 | 28/34 / 26 / 0.9 | 27/29 |

- **Approach now matches walkover's detection quality** (precision 91 % for both, recall 83 vs 86 %). It adds
  close-range D405 evidence on 26/34 holes (walkover 5) and 11/21 upheavals (5), for +5.4 s per route.
  The approach gap seen in v1 (precision 86 %, 0.38 false findings per mission) is gone.
- **Most "false" findings are type confusions at a real defect:** 5 of 7 walkover and 4 of 7 approach false findings
  are a small upheaval reported as a pit, or a hole reported as an edge. True phantoms are 2 (walkover) and 3 (approach)
  over 24 missions, i.e. ~0.1 per mission. Location-level precision is ~97 %.
- **Weakest point: upheavals.** Recall is 67–71 %; the misses are mostly small (≤ 2.3 cm) half-width faults. The
  upheaval step error (1–2 mm) and ADA class (100 % correct when found) are fine.
- **Puddles** are found from dropout (28/29) but not measured.

Pessimistic sensor errors (`outputs/missions_final2_pess`) and v2 on the seed-200 set (`outputs/missions_final_v2code`,
informative only) follow.

### v2 under pessimistic sensor errors (seed 300, `outputs/missions_final2_pess`)

| strategy | recall | precision | false / mission | upheaval found / D405 / err [mm] / ADA ok | hole found / D405 / depth err [mm] | puddle found |
|---|---|---|---|---|---|---|
| walkover | 81 % | 91 % | 0.29 | 12/21 / 7 / 6.0 / 100 % | 27/34 / 5 / 2.5 | 29/29 |
| slow | 83 % | 91 % | 0.29 | 13/21 / 4 / 7.1 / 100 % | 30/34 / 5 / 2.6 | 27/29 |
| approach | 79 % | 88 % | 0.38 | 13/21 / 11 / 1.3 / 92 % | 25/34 / 22 / 1.9 | 28/29 |

Doubled noise, 10 % dropout, 1° calibration error and 2.5× odometry drift cost only a few points of detection
(recall 86 → 81 % for walkover). Measurement is where the strategies separate:
- **Walkover's upheaval step error triples** (2.1 → 6.0 mm), enough to blur the 6.4 / 13 mm ADA thresholds.
- **Approach stays at 1.3 mm,** because the D405 at its stand-off is barely affected by the long-range error sources.

This is the clearest case for active inspection. Under the realistic profile its benefit is mostly *more defects with
close-range evidence*; under worse sensors it is also *the only strategy that keeps millimetre measurements*.

### v2 on the seed-200 set (informative: seed 200 motivated the v2 rules; `outputs/missions_final_v2code`)

| strategy | recall | precision | false / mission | D405 upheavals / holes |
|---|---|---|---|---|
| walkover | 85 % (v1 86 %) | 98 % (v1 100 %) | 0.04 | 9 / 1 |
| slow | 83 % | 95 % | 0.12 | 10 / 1 |
| approach | 82 % (v1 80 %) | 97 % (v1 86 %) | 0.08 (v1 0.38) | 19 / 15 |

The v2 rules remove most of approach's false findings without hurting walkover. Approach's hole recall on this set
stays at 18/26, against 22 for walkover. The lower walkover precision on seed 300 (91 %) is tile-set variance, not a v2
regression.

## Phase 1, step 6: Isaac physics missions and measured-vs-true (2026-11-12)

New: `PoseErrors` in `mission/runner.py` (odometry drift scaled by distance + camera-IMU latency, default 20 ms), off
by default so the frozen evaluations reproduce exactly (checked on seed-300 tile 8). `scripts/isaac_mission.py` runs the
same mission loop on Isaac Go2 physics (rough-terrain policy, cameras off: works on the 610.88 driver) and records all
link poses; `scripts/render_isaac_mission.py` (separate env `D:/go2wm/render_env`, pyvista) renders a 3D chase view
with measured-vs-true labels. `scripts/measurement_comparison.py` plots every defect, not medians.

### What the real gait exposed (tile 8, seed 300; mock never showed these)
- Approach in the mock assumed the robot arrives facing +x. In Isaac it arrived ~30-60 deg off and wandered until the
  budget expired (approach aborted). Then it stalled 0.17 m short: the policy's dead zone (E1r) ignores the MPC's small
  commands near the waypoint. Fix (Isaac runs only, `Policy(align=True, reach=0.10)`): direct body-frame command
  floored at 0.35 m/s for the last 0.4 m, turn in place (>= 0.6 rad/s) to face the finding.
- After an aborted approach the robot resumed the route offset toward the 7 cm hole and **fell in**. Holes were never
  planner obstacles. Fix (`Policy(avoid_holes=True)`): confirmed holes deeper than 3 cm become A* keep-out circles
  (planner only) and the route is replanned when the set changes.
- Walk past with the real gait: the 65 mm upheaval read as 15 mm (first run) / 41 mm (second run). Stop & look:
  65.6 mm (D405), hole 71.7 vs 70.0 mm. Videos: `outputs/videos/14_isaac_{walkover,approach}_tile8.mp4`.

### Measured vs true, every defect (seed 300, mock kinematics; `outputs/measurement_comparison`)
Medians were 1-2 mm, but the tails are not small: p90 6-25 mm. Systematic failures:
- Deep holes (> 10 cm) under-read by 25-70 mm in ~1 in 8 cases: the camera cannot see the bottom from the stand-off.
- An upheaval on a running-slope ramp (tile 22) read 185-201 mm against 62 mm (realistic); the step measurement
  includes the grade. Not fixed (final set).
- Pessimistic sensors + pose errors: walk past upheaval median 5.7 mm, p90 27.5 mm; stop & look median 0.6 mm,
  p90 25 mm; false findings 0.67 per mission (approach, was 0.29 without pose errors in realistic).
- ADA class agreement 90-100 % is weak evidence: the test tiles have no upheaval under 10 mm or hole under 20 mm.

## Phase 1, step 7: demo mission, body posture, people, Isaac Sim RTX rendering (2026-11-12/13)

### Demo street and mission context (`mission/site.py`)
`demo_tile()`: 56 m street with 5 blocks and 10 defects. The defects include near-threshold ones: an 11 mm upheaval, a 45 mm hole and a 110 mm hole.
- **Work zone:** block 3, with barriers, cones, two workers, a pile and a generator. Static site objects are planner obstacles.
- **People:** a worker pacing inside the zone, a pedestrian crossing from a doorway to the road (triggered by the robot), and an oncoming pedestrian who steps aside.

The runner yields (stops) while a person is in the corridor ahead and caps speed at 0.45 m/s near people. People are known from ground truth, standing in for a person detector; the depth simulation ray-casts only the sidewalk surface. Isaac moves kinematic cylinders for them (`IsaacGo2Adapter.set_movers`).

### Inspection posture (crouch + body attitude), joint level, no retraining
`IsaacGo2Adapter.set_posture(drop, pitch)`: the walking policy is suspended and every leg is driven by 3-joint IK with the feet kept where they stand.
- **Leg model:** forward kinematics matches the simulated foot positions to 0.0 mm.
- **Stiffer joints while posing:** joint gains rise to 60 / 3, close to the real Go2's stand gains of about 60 / 5.
- **Attitude feedback:** roll / pitch PI compensates joint sag.
- **Stance check:** refuses to crouch when a foot stands within 7 cm of the body centre line; the runner then falls back to a standing close-up.
- **Roll abort:** stands back up on a roll change above 8.6 deg.

The first versions fell twice in missions:
- sagittal-only IK slid the body sideways over a skewed stance;
- soft joints let one side buckle under load.

Both are fixed by the full 3D IK and the gains and attitude feedback above. The final gentle pose (5 cm crouch, body level) puts the D405 at 31 cm from the ground at 60 deg, against 53 cm at 48 deg standing. Note that the Go2 rough policy stands about 11 deg nose-up, so the standing D405 looks about 0.68 m ahead, not the 0.52 m the stand-off assumed.

Demo mission results (realistic sensors + drift + latency):

| run | outcome | inspections | notes |
|---|---|---|---|
| stop & look, standing | goal 102 s | 6, 1 yield | 45 mm upheaval 45.3, 110 mm hole 110.2, 28 mm upheaval read 83 mm: one 634 mm D405 outlier wrecked the mean of the best-confidence readings (open issue: no outlier rejection) |
| walk past | goal 86 s | 0 | 28 mm upheaval read 19 mm |
| stop & look + posture | goal 119 s | 1 crouch done, 2 refused by the stance check | the crouched reading of the 28 mm upheaval: 28.0 mm. Other values worse this run (45 mm upheaval 4.6 mm): Isaac run-to-run variance is large, so this is not evidence that posture helps |

Curb false positives: where the work zone squeezes the robot next to the 15 cm curb, the curb edge is logged as 4 "holes".

### Isaac Sim RTX rendering
- **Renderer:** Isaac Sim 5.1's RTX renderer still crashes at start-up on driver 610.88 (in `rtx.scenedb`, before project code). The standalone Isaac Sim 6.0.1 renders on the same driver.
- **Replay script:** `scripts/isaac_rtx_replay.py` (6.0.1) rebuilds the street in USD:
  - the sidewalk height field from the physics terrain, with its concrete albedo;
  - MDL asphalt and grass, and buildings;
  - NVIDIA trees and shrubs, puddles in MDL water, work-zone props;
  - an HDR sky and a sun;
  - NVIDIA's Go2 USD posed from the recorded link poses.
- **Video composition:** `render_demo_mission.py --timeline-out` exports the camera path; `--view-dir` composes the HUD around the RTX frames, with labels, trail and sensor zones projected into the image.
- **People:** the NVIDIA characters use the `RL_BoneRoot` skeleton, while the stock `skelanim` clips use another one (`Root/...`), so the clips never bind; the first render showed T-posed, floating, sliding people. The characters are now animated procedurally on their own skeleton:
  - a walk cycle phased by the distance walked, so there is no foot sliding;
  - arms lowered from the T-pose;
  - idle, look-around and digging poses;
  - unscaled, feet on the ground.
- Video: `outputs/videos/17_demo_mission_isaac_rtx.mp4`.

## Phase 1, step 8: Go2-W port (wheeled legs), curb-aware perception (2026-11-13 to 11-18)

### What is ours and what is NVIDIA's / third-party
- **Robot asset:** the Go2-W URDF from the robot_lab repository (Apache-2.0), used as data only, converted by Isaac Lab's URDF importer (`D:/go2wm/cache/go2w_usd`). Isaac Lab ships no Go2-W.
- **Task and policy: trained by us.** `src/go2wm/sim/go2w_task.py` (`Go2WM-Velocity-Rough-Go2W-v0`) is Isaac Lab's Go2 rough task with:
  - 12 leg position targets plus 4 wheel velocity targets;
  - rewards re-weighted for a wheeled-legged robot, following robot_lab's Go2-W choices: track velocity 3.0 / 1.5, keep the leg pose near default (−0.5) so it rolls instead of stepping, no feet-air-time reward.
  - We rewrote this task ourselves and did not run robot_lab's code.
- **Training:** `scripts/train_go2w.py`, rsl_rl PPO, 4096 envs, 2000 iterations (about 3 h on the RTX 3090, with one resume after a memory-pressure stop at iteration 779). Final episode length 997/1000; linear-velocity tracking reward 2.6 of 3.0. Checkpoint: `D:/go2wm/ext/go2w_runs/rough/model_1999.pt`.
- The Go2 runs still use NVIDIA's pretrained rough Go2 policy. All Go2 defaults are unchanged; Go2-W is its own config (`configs/isaac_go2w_rough.yaml`).

### Locomotion check (`outputs/go2w/smoke_final.json`, flat tile)
Passes with no fall, and **no dead zone**, unlike the pretrained Go2.

| command | realized |
|---|---|
| 0.5 m/s forward | 0.55 m/s |
| 1.0 m/s forward | 1.05 m/s |
| 0.3 m/s lateral | 0.27 m/s |
| 0.6 m/s lateral | 0.58 m/s |
| 0.5 m/s arc + 0.5 rad/s | 0.49 m/s + 0.38 rad/s |

### Sensor geometry finding: the D405 needs the crouch on the Go2-W
Standing, the Go2-W carries the D405 at a slant range of 0.48–0.51 m to the ground, right at the camera's 0.5 m limit (`outputs/go2w_posture.log`). The same 3D-IK inspection pose as the Go2 (`IsaacGo2Adapter.set_posture`, 5 cm drop) brings it to 0.36 m. On the Go2, the crouch was a refinement; on the Go2-W it is required for close-range measurement.

### First demo-street results and their diagnosis (2026-11-13/15)
The first Go2-W demo missions looked weak: stop & look found 4/7 with 17 false findings; with the crouch, 3/7 with 18.
- **Not the cause:** the robot or the crouch. Crouched readings of the 45 mm hole were within 0.1 mm.
- **Cause:** lateral pose drift. Odometry drift reached 0.24 m, and up to 0.8 m once heading drift built up. That exceeds the 12 cm curb margin of the walkway mask, so the curb edge entered the mask and was logged as "holes".
  - Stop & look then approached each of these, looping along the curb (x = 0–11 m).
  - The drifted positions also made real defects score as missed.

### Fixes (all off by default, so the frozen evaluations are unchanged)
- `--curb-aware`:
  - `perception.curb_mask` drops the curb strip found in each depth frame itself, not from the map under the drifting pose;
  - `perception.curb_lateral` measures the curb's offset and angle in the frame, and `PoseErrors.correct` pulls the lateral and heading estimate toward the mapped curb, like localising against a curb;
  - `replan()` falls back to the plain route instead of crashing.
  - Unit tests: `tests/test_curb_aware.py`.
- **Robust memory, partial-view rule** (commit dafab66): while the robot approaches a hole, the D405 first sees only its near part, which can only read shallow. Robust memory now keeps depression readings ≥ 0.85 × the 2nd-largest before the MAD filter and median. Applied to all saved robust runs below.

### Demo street, before vs with `--curb-aware` (Isaac physics, realistic sensors + drift + latency, robust memory)
`D:/go2wm/outputs/curb_aware/comparison.md`. One run per row: Isaac run-to-run variance is large, so read single-defect differences with care.

| run | outcome | time [s] | inspections | found | median / worst error [mm] | false findings |
|---|---|---|---|---|---|---|
| Go2 walk past | goal | 85.9 | 0 | 6/7 | 0.9 / 1.4 | 1 |
| Go2 walk past, curb-aware | goal | 86.0 | 0 | 5/7 | 16.6 / 31.5 | 1 |
| Go2 stop & look | goal | 102.1 | 6 | 6/7 | 0.7 / 9.8 | 4 |
| Go2 stop & look, curb-aware | **fall** at x = 18.9 m | 89.7 | 3 | 4/7 | 0.6 / 1.3 | 0 |
| Go2-W walk past | goal | 80.3 | 0 | 5/7 | 1.0 / 1.1 | 6 |
| Go2-W walk past, curb-aware | goal | 80.3 | 0 | 5/7 | 1.3 / 1.6 | 7 |
| Go2-W stop & look | goal | 115.6 | 18 | 4/7 | 0.2 / 0.4 | 17 |
| **Go2-W stop & look, curb-aware** | goal | 89.0 | 5 | **6/7** | 0.1 / 2.9 | **1** |
| Go2-W stop & look + crouch | goal | 170.7 | 17 | 3/7 | 17.2 / 34.4 | 18 |
| **Go2-W stop & look + crouch, curb-aware** | goal | 101.7 | 4 (4 crouches) | 5/7 | **0.1 / 0.2** | **1** |

- **Go2-W stop & look:** curb-aware removes the curb loop. False findings drop from 17–18 to 1, the mission is 26–69 s shorter, and every measured defect is within 3 mm (crouched: within 0.2 mm).
- **Go2-W walk past:** no change (6 → 7 false findings).
  - 2 are the 45 mm upheaval reported as a depression: a type confusion, not a phantom.
  - 5 lie at x = 22.6–26 m, y ≈ −1.1 to −1.3 m, near the curb at the street's end. The walk-past robot passes there without stopping, so the curb fix has less time to act; still open.
- **Go2 stop & look, curb-aware fell** at the 110 mm hole while turning away after inspecting it. The pose estimate was within 5 cm at the fall, but its lateral error had reached 0.45 m earlier in the run (corrections act only while the heading is within ±18° of the walk). A legged robot near a deep hole stays a risk; one run, so no cause is claimed.
- **Never found, by any robot or strategy: the 11 mm upheaval.** See the next subsection.
- The crouched curb-aware Go2-W run's "missed" 110 mm hole and its 1 false finding are the same finding.
  - The robot crouched over the hole and measured 109.4 mm (−0.6 mm).
  - It placed the hole at (18.70, 0.68), 0.36 m from the true (18.60, 0.32), almost all of it lateral (cause not yet separated: pose error vs. a centroid biased toward the part of the hole in view).
  - That is outside the scoring match radius, so the score counts it once as missed and once as false.
  - By location tolerance alone it is 6/7 found, 0 phantoms. This is a localisation error, not a detection error.

### Why the 11 mm upheaval is always missed
- **Cause, by design:** `detect_candidates` flags cells that deviate from their 35 cm local median by more than 12 mm (more beyond 1 m). That threshold is above the fault itself, and a median filter keeps a clean step edge anyway, so only its corners would deviate.
- **Measured:** in mock D555 frames 0.9–1.6 m before the fault, the largest residual is 6–9 mm; there are no detections at all near x = −18 m.
- **Why it matters:** 6–13 mm is the ADA band where a vertical change needs a bevel, so these faults count for the [< 1 cm requirement].
- **Candidate fix (branch `small-step-detector`, off by default):** `perception.transverse_steps` compares the median of the 3 rows ahead with the 3 rows behind each row boundary, column by column. It is detrended against the same response one window further on, so grades cancel. A step is a run of one sign above 6 mm spanning ≥ 0.6 m across the walk, 0.5–1.8 m ahead.
  - In single frames on the demo street, it finds the 11 mm fault at 10.6–11.2 mm and the other two upheavals.
  - It fires nowhere else: holes, puddles, cracks and the curb give nothing.
  - Dev-set check (seed 100) and a fresh-seed check: see below.

### Video
`outputs/videos/20_demo_mission_go2w_curb_posture_isaac_rtx.mp4`: Go2-W curb-aware stop & look + crouch, Isaac Sim 6.0.1 RTX.
- Video 18 (Nov 13) has the replay bug in which the robot sinks through the slab: render steps simulated the robot's rigid bodies.
- Videos 19 and 20 are rendered with the fix (`isaac_rtx_replay.py` strips the robot's physics).
