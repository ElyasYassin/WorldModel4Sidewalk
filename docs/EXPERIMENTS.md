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

## Next experiments (Isaac)
- P1: sidewalk terrain with the rough-terrain policy (`configs/isaac_go2_rough_e4s1.yaml`, `scripts/isaac_terrain_smoke.py`);
  needs the pretrained rough checkpoint.
- Beyond that, these runs show **the learned model has no navigation advantage on flat ground with cylinders**: every planner difference is
  explained by locomotion deadband and cost tuning. Testing H1/H2 needs conditions where dynamics are hard to model by hand: sidewalk
  terrain (curbs, slopes, friction changes, from SidewalkAtlas `scripts/terrain`), and cameras with V-JEPA 2 for H1.
