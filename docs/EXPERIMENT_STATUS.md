# Experiment status (compact)

Snapshot for planning and review. Full narrative and tables: [EXPERIMENTS.md](EXPERIMENTS.md).
Artifact paths are under the data root `D:/go2wm` (`configs/paths.yaml`). Every result below is
**simulation** (Isaac Lab Go2, pretrained flat-terrain policy, flat ground, cylinder obstacles)
unless marked mock. Nothing has run on hardware.

Updated 2026-11-18 (Phase 1 inspection results: section at the end).

## Evaluation roles (frozen, `configs/eval_protocol.yaml`)

| role | what | status |
|---|---|---|
| development data | `data/go2_sim`, seeds 0–179, 180 eps / 32,098 transitions | all 180 episodes have been in some fit or test split |
| dev prediction test | manifest `v2_clean_test` (19 eps) | used in E2b; a development test, not final |
| final prediction test | `data/go2_final_test`, seeds 30000–30059 | collected 2026-11-09, evaluation only |
| held-out command patterns (spec §10.1) | `data/go2_cmdpattern`, seeds 40000–40079, open family | collected 2026-11-09, evaluation only |
| dev navigation scenes | indices 0–16 (`<family>_10000`–`_10016`) | used in E3a–E3e and for conventional tuning |
| final navigation scenes | indices 17–33 (`_10017`–`_10033`) | **used once, in E3f (2026-11-10)**; must not be used for any further tuning; guarded by `run_navigation.py --final-test`, logged to `outputs/final_test_ledger.jsonl` |

Leakage audit: `python scripts/audit_splits.py --data data/go2_sim --json outputs/audit/splits_2026-11-09.json`.
Result: no within-manifest overlap by episode, scene id, seed or geometry; one episode per scene
(so scene-disjoint ⇒ episode-disjoint); windows never cross episodes (`tests/test_integrity.py`).
Cross-manifest: all 9 `default` test episodes are in `v2` train/val, and 8 `v2` test episodes are
in `default` train/val; `v2_clean_test` excludes both. Navigation scenes share no id, seed or
geometry with any data episode.

## Results so far

| exp | question | result | caveat | artifacts |
|---|---|---|---|---|
| E1 | Does the policy track commands? | Tracks ≥0.5 m/s forward/arcs and ≥0.6 m/s lateral; **dead zone** below ~0.2–0.4 m/s planar, no turn-in-place | policy property, not a model result | `outputs/isaac_smoke/report.json` |
| E2 | Learned vs kinematic prediction | MLP beats command kinematics in 9/9 test eps at every horizon (paired); MLP ≈ GRU | 9 test eps; same command generator as training | `outputs/state_go2_mlp/eval_test_paired/` |
| E2b | Does 3× data help? | No reliable gain at 1–3 s; training seed matters as much as data. Learned still beats kinematics in 18–19/19 clean eps (1 s: 0.025 vs 0.058 m) | flat ground only | `outputs/state_go2_mlp_v2*/eval_clean_test/` |
| E2c | Fresh final prediction test (60 eps) | Learned beats kinematics at every horizon (1 s: 0.030 vs 0.065 m, 59/60 eps); models within 0.005 m of each other | same command generator as training | `outputs/p0_pred/final_test/` |
| E2d | Held-out command patterns (§10.1) | Learned still best on sine/arc/fast-switch; **worse than kinematics on out-of-range lateral commands** (3 s: 0.263 vs 0.147 m, 4× heading error); pooled 3 s advantage gone | flat ground; planner limits keep E3 inside support | `outputs/p0_pred/cmdpattern/` |
| E3a/b | Navigation, 51 dev scenes | conventional 0.88, kinematic MPC 0.90, learned MPC 0.88; no paired differences | all failures = freezing before sidewalk gaps (dead zone) | `outputs/nav_merged_e3b/` |
| E3c | Deadband-aware MPC | No more freezing; robot parks at the 0.3 m clearance margin instead | dev scenes | `outputs/navigation_db_{kin,learned}/` |
| E3d | Margin 0.15 m on the 5 failures | 10/10 | scenes chosen because they failed | `outputs/navigation_db_m15_*` |
| E3e | Margin 0.15 m, all 51 | 51/51 for kinematic and learned MPC, with or without deadband | **margin tuned on these scenes; conventional not tuned** | `outputs/nav_e3e_renamed/` |
| E3f | Frozen planners on 51 **final** scenes | conventional default 45/51, tuned 50/51, kinematic MPC 51/51, learned MPC 51/51; learned − kinematic time −0.59 s [−0.67, −0.51]; tuned conventional vs MPC p = 1.0 | flat ground, cylinders; MPC commands ≈3× less smooth | `outputs/e3f/` |
| E1r | Rough-terrain policy tracking | Passes; much smaller dead zone (tracks 0.3 m/s lateral, partial turn in place) | flat tile | `outputs/isaac_smoke_rough/report.json` |
| E4-S1 | Can the rough policy traverse sidewalk S1? | 6/6 reached end, 0 falls; slab slows it ≤0.12 m/s, grade ≤0.02 | curb and cross-slope drift not exercised | `outputs/e4_smoke/report.json` |
| E4 sweep | Where does the rough policy stop following commands? (18 variants + repeats) | Stalls at steps ≥15 cm (15 cm stochastic: 4/11 pass, 1 fall); 30 % grade stalls at 0.4 m/s only (repeatable); friction ≤0.25 slips and slows. Kinematics 1 s error 0.03–0.06 m elsewhere, up to 0.76 m at steps | mostly 1 episode per cell; curb repeats incomplete | `outputs/e4_sweep/analysis.json` |
| E4-P2 | Does a terrain height map improve prediction on unseen sidewalk tiles? (3 models × 3 seeds) | All learned models beat kinematics 2.6× at 1 s. Terrain MLP beats MLP at 2–3 s (mainly on steps; p90 −25 %) but ties the state-only GRU overall (3 s: 0.181 vs 0.184 m) | perfect elevation map; stall windows labelled plain; prediction only, no navigation test | `outputs/e4_p2/final_test/prediction_metrics.json` |
| E4-P2b | Terrain GRU, weight decay, commanded-path segment labels | Unseen tiles: terrain GRU best (3 s 0.165 vs GRU 0.184, MLP 0.195, terrain MLP 0.181 m); terrain MLPs degrade on unseen grades; stall windows now labelled step (kinematics 0.64 m there) | dev-test picked a different model; perfect map; prediction only | `outputs/e4_p2b/final_test/prediction_metrics.json` |
| P4-a | Ground-truth sidewalk scene graph (inspection + mission grounding) | Per test tile 34–43 nodes, 7–11 hazards (1–6 ADA violations); JSON + text + figure | risk is a heuristic placeholder; no sensor-built graph yet | `outputs/scene_graphs/` |
| E4-C setup | Which sensor sees the sidewalk? (rig smoke test, 2 episodes) | Ground coverage 0.3–2 m ahead per frame: depth 66–71 %, Mid-360 tilted 43–50 %, L1 39–40 %, Mid-360 upright 0–1 %; errors ≤ 2 cm p90 | 2 episodes; full collection running | `data/go2_e4c_smoke` |
| E4-C (1) | Sensor map accuracy, 48 test eps on unseen tiles | Only depth cameras meet < 1 cm in one frame (D555 p90 ≤ 1 cm at 0.6–2.9 m, D435 0.6–2.1 m); D555 joint steps p90 2–6 mm to 2.5 m, ADA class 98–99 %. LiDARs ≈ 1–1.6 cm single frame, need 10–20-frame fusion; upright Mid-360 sees 1 % of near ground | approximate noise models, no sunlight/multipath; perfect registration except σ_reg check | `outputs/e4c/test/` |
| E4-C (2) | Does a single-frame sensor map help prediction? (8 models × 3 seeds) | Map/scanner replicate (final 3 s: GRU 0.180, map 0.154, scanner 0.164 m). **No single-frame sensor map helps on unseen-tile steps** (D555 +0.008 m at 2 s); they help on seen tiles only. Cause: cameras blind under the robot (p90 12 cm there) | perfect ground reference; single frame | `outputs/e4c/final_test/seed_summary.json` |
| E4-C (3) | Same with maps fused over 10 frames | Fused maps help: 3 s 0.165–0.169 vs GRU 0.180 m (perfect map 0.154); on steps about a third of the map's gain at 3 s. D555 + L1 = D555 alone for prediction | perfect registration | `outputs/e5/final_test/seed_summary.json` |
| E5 | Pretrained models (frozen DINOv2 features; Depth Anything maps) | DINOv2 makes unseen-tile prediction much worse (3 s 0.33 m; steps +0.34 m), likely tile-appearance overfitting; monocular depth adds nothing. **Parked** (Phase 1 repositioning) | 12 training tiles; not checked on dev-test | `outputs/e5/final_test/` |
| E4-S | Does the terrain gain survive on the robot's own height scanner (0.8 m reach) instead of a perfect 3 m map? (fresh E4 re-collection, 3 models × 3 seeds) | E4-P2b replicates (3 s: GRU 0.184, map 0.165 m). Scanner 0.176 m: on unseen-tile steps it keeps ~50 % of the map's gain at 2 s, ~37 % at 3 s (segment table corrected 2026-11-11) | dev-test scanner gain within noise; noise-free, unoccluded scanner; single-frame scan; prediction only | `outputs/e4s/final_test/prediction_metrics.json` |

Main findings so far: on flat ground, better prediction does not change navigation outcomes;
every planner difference is explained by the locomotion dead zone and the clearance margin. Confirmed
on untouched scenes (E3f): a tuned conventional controller is statistically tied with both MPCs;
learned prediction only saves ≈0.6 s per run over kinematics.
Learned models generalize to new command timing but not to commands outside their training support.

## P0 (done, 2026-11-10)

| step | status | how to reproduce |
|---|---|---|
| Baseline commit | done, `e5195d9` | – |
| Split audit | done (above) | `scripts/audit_splits.py` |
| Frozen protocol + final-test guard | done | `configs/eval_protocol.yaml`, `run_navigation.py --final-test` |
| Fresh final prediction test set | done, E2c | `scripts\p0_queue.ps1 -Phase data`; `evaluate.py --data data/go2_final_test --all-test` |
| §10.1 held-out command patterns | done, E2d | `evaluate.py --data data/go2_cmdpattern --all-test` |
| Conventional tuning, dev scenes only | done 2026-11-10; frozen `configs/planner_isaac_conv_tuned.yaml` | `scripts\p0_queue.ps1 -Phase tune` → `outputs/p0_tuning/` |
| E3f: E3e confirmed on final scenes 17–33 | done 2026-11-10 | `scripts\p0_queue.ps1 -Phase final` → `outputs/e3f/` |

Conventional tuning grid (`configs/tuning/`): `min_speed` {0, 0.4} (deadband-aware forward-speed
floor, the counterpart of the MPC's deadband projection) × `bound_repulsion` {off, on} ×
(`influence_m`, `k_rep`) {(1.0, 1.2), (0.6, 0.8)}. The default (0, off, 1.0/1.2) reuses E3b.
Selection rule, fixed before any run: highest dev success; ties → fewer collisions → fewer safety
interventions → shorter mean time to goal. The MPC's comparable budget was 4 configs
(margin 0.30/0.15 × deadband on/off).

Tuning result (34 obstacle + sidewalk dev scenes, `python scripts/select_conventional.py`):

| config (min_speed, bound_rep, influence/k_rep) | success | collisions | interventions/ep | mean t_goal [s] |
|---|---|---|---|---|
| **0.4, off, 0.6/0.8 (chosen)** | **34/34** | 0 | 0.0 | 15.9 |
| 0.4, on, 0.6/0.8 | 34/34 | 0 | 0.0 | 16.0 |
| 0.4, on, 1.0/1.2 | 34/34 | 0 | 0.0 | 16.0 |
| 0.4, off, 1.0/1.2 | 32/34 | 0 | 16.5 | 15.7 |
| 0, off, 0.6/0.8 | 31/34 | 0 | 0.0 | 17.4 |
| 0, on, 1.0/1.2 | 28/34 | 0 | 0.0 | 16.5 |
| 0, on, 0.6/0.8 | 28/34 | 0 | 0.0 | 16.9 |
| 0, off, 1.0/1.2 (default, E3b) | 28/34 | 0 | 9.5 | 16.1 |

The deadband-aware speed floor (`min_speed` 0.4) is what fixes the conventional controller: every
config with it reaches 32–34/34, every config without it 28–31/34. With a comparable tuning budget
the conventional controller matches the MPCs (34/34) on the dev scenes. Whether any planner
difference survives on fresh scenes is the question for E3f.

## Phase 1: sidewalk inspection missions (2026-11-11 to 11-18)

Defect tiles (upheaval faults, holes, puddles, cracks), D555 detection + D405 close-up, scene-graph memory, inspection
strategies walk past / slow / stop & look, report generator. Mock = kinematic robot with ray-cast sensors; Isaac = Isaac Lab
physics with ray-cast sensors (cameras off). Dev tile sets: seeds 0 and 100. Fresh final sets, each used once: 200 (v1),
300 (v2), 400 (curb-aware + robust), 500 (step detector).

| exp | question | result | caveat | artifacts |
|---|---|---|---|---|
| P1-5 | Strategies on fresh tiles (seed 200, frozen 596cdb4) | walk past 86 % recall, 0 false findings; stop & look adds D405 evidence (hole err 2.8 → 0.7 mm) but 0.38 false / mission | mock, no pose errors | `outputs/missions_final` |
| P1-5b | v2 rules (multi-view, hole walls) on fresh seed 300 | approach precision 86 → 91 %, recall 83 % = walk past; upheavals weakest (67–71 %) | mock, no pose errors | `outputs/missions_final2` |
| P1-6 | Isaac physics missions | real gait exposed approach heading/dead-zone stalls and a fall into a hole; fixed with align + hole keep-out | Isaac Go2 rough policy, single tiles | `outputs/isaac_missions` |
| P1-7 | Demo street (people, work zone, crouch) | missions complete with yields; crouch gives D405 at 31 cm | one run per config | `outputs/demo_mission` |
| P1-8a | Go2-W port | our own Go2-W task + PPO policy (2000 it): no dead zone; D405 at its 0.5 m limit standing → crouch needed | URDF from robot_lab (data only) | `D:/go2wm/ext/go2w_runs/rough` |
| P1-8b | Curb-aware perception, Isaac demo street | Go2-W stop & look false findings 17–18 → 1, errors ≤ 3 mm (crouched ≤ 0.2 mm); Go2 curb-aware stop & look fell once at the 110 mm hole | one run per row | `outputs/curb_aware/comparison.md` |
| P1-8c | Curb-aware + robust memory on fresh seed 400 (pose errors on) | curb-aware: false / mission 1.42 → 0.83 (walk past), 1.38 → 0.62 (stop & look), worse on 0/48 tiles, +2 goals; robust: p90 error 6.8 → 3.7 mm but one 79 mm hole read 371 mm (partial-view rule) | effect comes from 2–3 drift blow-up tiles | `outputs/curb_aware_final3.md` |
| P1-8d | 11 mm upheaval never found | the 12 mm detection threshold sits above the fault height; a transverse step detector (branch `small-step-detector`) finds every dev upheaval (21/21, was 15–16) with no extra false findings, and the 11 mm fault in Isaac (11.0 mm, D405) | the extra inspection stop shifts the robot sideways and puddles were missed (coverage) | `outputs/dev_step`, `outputs/small_step` |
| P1-8e | Step detector v2 (cap 20 mm, skip near residual detections) | dev: 21/21 upheavals, false 0.25 → 0.08 / 0.12, no error tail; fresh seed 600: upheavals 13–14/22 → 21/22, false 0.25 → 0.17 / 0.42 → 0.25 (n.s.), no error regression; Isaac: 0 false, 11 mm read 11.0 mm | 3 puddles lost on seed 600 (1 explained by coverage); branch not merged | `outputs/small_step_final5.md` |

## Not implemented / not verified

Terrain sweep beyond S1, terrain-aware models, cameras, terrain-aware models, V-JEPA 2, scene graph,
voice/mission manager, any real-robot control, paper results sections.
