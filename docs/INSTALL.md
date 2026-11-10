# INSTALL — verified environment facts

Probed 2026-11-03 on the development workstation. Everything below was observed,
not assumed; items marked **untested** have not been executed yet.

## Hardware / OS

| Item | Value |
|---|---|
| OS | Windows 11 Home 10.0.26200 |
| GPU | NVIDIA GeForce RTX 3090, 24 GB VRAM |
| Driver / CUDA | 580.88 / CUDA 13.0 (driver-reported) |
| Disk C: | **~0.3 GB free (BLOCKER, see below)** |
| Disk D: | ~20 GB free |

## Simulator stack (already installed)

| Component | Location | Version |
|---|---|---|
| Isaac Sim | `C:\isaacsim` (binary install) | `5.1.0-rc.19+release.26219.9c81211b` |
| Isaac Lab | `C:\IsaacLab` (`_isaac_sim` -> `C:\isaacsim` symlink) | `2.3.0`, commit `cbf51abb5e98d1b3d497c8c73dc989e9f3628b89` (`v2.2.1-232`) |
| Python env | conda `IsaacLab` (`C:\Users\Elyas\anaconda3\envs\IsaacLab`) | Python 3.11.14 |
| torch | in `IsaacLab` env | 2.7.0+cu128, CUDA available = True |
| rsl-rl-lib | in `IsaacLab` env | 3.1.2 |
| other | in `IsaacLab` env | numpy 1.26.4, h5py 3.15.1, gymnasium 1.2.1, PyYAML 6.0.3, matplotlib 3.10.8, pytest 9.0.2, transformers 5.0.0 |

Isaac Lab packages (editable): `isaaclab 0.54.0`, `isaaclab_assets 0.2.4`,
`isaaclab_tasks 0.11.12`, `isaaclab_rl 0.4.7`.

Also present but unused here: `C:\isaacsim51clean` (second Isaac Sim 5.1 copy).

## Go2 locomotion path

**Decision:** use Isaac Lab's built-in Go2 velocity tasks plus NVIDIA's
*published pretrained* RSL-RL policy. This is an official, already-trained
velocity-following controller, so no gait training is needed (spec §5.1.2).

- Tasks registered in `isaaclab_tasks/manager_based/locomotion/velocity/config/go2/__init__.py`:
  `Isaac-Velocity-Flat-Unitree-Go2-v0`, `Isaac-Velocity-Rough-Unitree-Go2-v0`
  and their `-Play-v0` variants.
- Robot asset: `UNITREE_GO2_CFG` in `isaaclab_assets/robots/unitree.py`,
  USD `{ISAACLAB_NUCLEUS_DIR}/Robots/Unitree/Go2/go2.usd` (HTTP 200 from the 5.1 S3 asset root).
- Pretrained checkpoints (HTTP 200, read-only HEAD check):
  `.../Assets/Isaac/5.1/Isaac/IsaacLab/PretrainedCheckpoints/rsl_rl/Isaac-Velocity-{Flat,Rough}-Unitree-Go2-v0/checkpoint.pt`.
  Fetched by `isaaclab_rl.utils.pretrained_checkpoint.get_published_pretrained_checkpoint`
  or `scripts/reinforcement_learning/rsl_rl/play.py --use_pretrained_checkpoint`.
- Local Go2 training runs: none (`C:\IsaacLab\logs\rsl_rl` only has `ant`, `anymal_c_rough`).
- **Unitree RL Lab** (spec's preferred upstream): **not installed**. Being
  deferred; the Isaac Lab built-in task covers MVP needs. Revisit if we need
  Unitree's sim2real deploy configs.
- **Untested:** launching the Go2 Play task and the policy actually walking on
  this machine. That is Phase C's first step.

## V-JEPA 2

- No V-JEPA 2 weights in the HF cache (`%USERPROFILE%\.cache\huggingface\hub`)
  or `D:\hf_cache`.
- `transformers 5.0.0` is present. Whether its V-JEPA 2 classes load the
  official `facebook/vjepa2-*` checkpoints is **untested**.
- The official repo `facebookresearch/vjepa2` is not cloned. Not needed until Phase E.

## Sibling project

`..\SidewalkAtlas` (Spot-based) has reusable pure-numpy sidewalk terrain
generation (`scripts/terrain/`) and Isaac capture tooling. We reference it but
don't modify it.

## Blockers and constraints

1. **C: drive is full (~0.3 GB free).** Isaac Sim writes kit/shader/extension
   caches and logs to `%LOCALAPPDATA%` / `C:\IsaacLab\logs` and will likely
   fail or crash at startup. Free at least ~20–30 GB on C: before Phase C.
   (Or move the Omniverse caches to D:, though that hasn't been tried.)
2. The repo lives in OneDrive. Datasets and checkpoints must **not** go under
   it. The default data and output root will be `D:\go2wm\` (configurable).
3. D: has ~20 GB free, which is enough for the initial 30–60 min state+RGB dataset
   at 224 px / 5–10 FPS but not for large-scale collection.

## Fixes applied later on 2026-11-03

- C: was freed (95 GB free afterwards).
- Plain conda `python` cannot `import isaacsim`. Run Isaac processes through
  `scripts\isaac_python.bat`, which sets the same env vars as `C:\isaacsim\python.bat`.
- `pip install cloudpickle hydra-core` in the `IsaacLab` env. Both are declared dependencies of
  installed packages (tensordict, gymnasium, isaaclab-rl) but were missing, which broke
  `import rsl_rl`. torch and numpy unchanged. (There is a pre-existing rl-games/psutil version
  warning, unrelated.)
- `pip install --no-deps -e .` for this repo (no other packages changed).
- **RAM:** 32 GB total. An Isaac start with about 6 GB of free commit memory crashed with
  `LLVM ERROR: out of memory`. Later runs with ~8 GB free (no cameras) worked. Close browsers
  and other heavy apps before Isaac runs, especially once cameras are enabled.

## Python environment for this repo

The plan is to use the existing conda env `IsaacLab` for everything (it already has
torch, h5py, pytest, and Isaac Lab), installing this package editable:

```powershell
conda activate IsaacLab
pip install -e .
pytest -q
```

Mock-backend code and tests import no Isaac modules, so they also run in any
Python ≥3.10 env with torch and numpy.
