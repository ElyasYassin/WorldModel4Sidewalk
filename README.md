# go2wm: Go2 latent world model + hybrid planner

This repo implements [GO2_WORLD_MODEL_SPEC.md](GO2_WORLD_MODEL_SPEC.md): an action-conditioned
world model for local navigation with a Unitree Go2, simulation first. A conventional A*
waypoint planner handles the long route. A receding-horizon CEM planner picks body-frame
velocity commands `(vx, vy, yaw_rate)`, using either kinematics or a learned dynamics model,
and an existing pretrained locomotion policy turns those commands into gait.

## Status

| piece | state |
|---|---|
| Mock backend → collect → validate → train → evaluate → navigate | working, 25 tests pass |
| State-only dynamics (MLP, GRU) + kinematic baselines | working (mock-validated only) |
| A* global planner, CEM local MPC, reactive baseline, safety filter | working (mock-validated only) |
| Isaac Lab Go2 adapter (pretrained flat-terrain policy, obstacle cylinders) | see [docs/ISAAC_INTEGRATION.md](docs/ISAAC_INTEGRATION.md) |
| V-JEPA 2 visual latent model | not implemented (Phase E); CLI fails with next steps |
| Real Go2 | read-only telemetry plan only: [docs/HARDWARE_HANDOFF.md](docs/HARDWARE_HANDOFF.md) |

The mock backend is a software test harness, not Go2 physics.

## Setup

Uses the existing `IsaacLab` conda env (verified versions in [docs/INSTALL.md](docs/INSTALL.md)):

```powershell
conda activate IsaacLab
pip install --no-deps -e .
python scripts/verify_environment.py
pytest -q
```

Datasets and outputs go under `D:/go2wm` (`configs/paths.yaml`, or set `GO2WM_ROOT`).
Relative `--data`/`--out` paths resolve there, so nothing large lands in OneDrive.

## Run (mock: no Isaac needed)

```powershell
python scripts/collect.py --backend mock --episodes 40 --out data/mock
python -m go2wm.data.validate --data data/mock
python scripts/train.py --config configs/world_model_state.yaml --data data/mock --out outputs/state_mock_mlp
python scripts/train.py --config configs/world_model_gru.yaml   --data data/mock --out outputs/state_mock_gru
python scripts/evaluate.py --checkpoint outputs/state_mock_mlp/best.pt outputs/state_mock_gru/best.pt --data data/mock --split test
python scripts/run_navigation.py --backend mock --planner all --checkpoint outputs/state_mock_mlp/best.pt
```

## Run (Isaac Go2)

Every Isaac process must go through `scripts\isaac_python.bat`, which sets the Isaac Sim
binary-install environment variables around the IsaacLab conda Python:

```powershell
scripts\isaac_python.bat scripts\isaac_smoke.py
scripts\isaac_python.bat scripts\collect.py --backend isaac --out data/go2_sim
python -m go2wm.data.validate --data data/go2_sim
python scripts/train.py --config configs/world_model_state.yaml --data data/go2_sim --out outputs/state_go2
python scripts/evaluate.py --checkpoint outputs/state_go2/best.pt --data data/go2_sim --split test
scripts\isaac_python.bat scripts\run_navigation.py --backend isaac --planner all --checkpoint outputs/state_go2/best.pt --config configs/planner_isaac.yaml
```

Isaac Sim needs about 16 GB of free commit memory. Close browsers and other heavy apps first.

## Layout

```
configs/            mock, isaac_go2, world_model_{state,gru,visual}, planner, paths
src/go2wm/common    schema (state/command layout), frames, config, seeds
src/go2wm/sim       SimulatorAdapter protocol, mock backend, Isaac Go2 adapter
src/go2wm/data      recorder + exploration commands, episode storage, splits/windows, validator
src/go2wm/models    features, normalization, MLP/GRU/kinematic dynamics, losses, training
src/go2wm/planning  A* global waypoints, CEM local MPC, reactive baseline, costs, safety filter
src/go2wm/eval      prediction metrics, navigation benchmark, plots
scripts/            verify_environment, collect, train, evaluate, run_navigation, isaac_smoke, extract_vjepa
docs/               INSTALL, DATA_FORMAT, ISAAC_INTEGRATION, EXPERIMENTS, HARDWARE_HANDOFF
```
