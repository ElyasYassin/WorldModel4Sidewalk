"""Train the Unitree Go2-W (wheeled legs) velocity-tracking policy with rsl_rl PPO in Isaac Lab.

  scripts\\isaac_python.bat scripts\\train_go2w.py --task Go2WM-Velocity-Rough-Go2W-v0 \
      --num-envs 4096 --iterations 3000 --out D:/go2wm/ext/go2w_runs/rough [--resume path/model_N.pt]

Task: go2wm.sim.go2w_task (Isaac Lab's Go2 rough task with the Go2-W URDF, wheel velocity actions and wheeled-legged
rewards modelled on robot_lab's Go2-W settings). Mirrors Isaac Lab's scripts/reinforcement_learning/rsl_rl/train.py;
saves model_<it>.pt every save_interval iterations in --out and copies the last one to model_latest.pt.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path


ap = argparse.ArgumentParser()
ap.add_argument("--task", default="Go2WM-Velocity-Rough-Go2W-v0")
ap.add_argument("--num-envs", type=int, default=4096)
ap.add_argument("--iterations", type=int, default=3000)
ap.add_argument("--out", default="D:/go2wm/ext/go2w_runs/rough")
ap.add_argument("--resume", default=None)
ap.add_argument("--seed", type=int, default=42)
a = ap.parse_args()

from isaaclab.app import AppLauncher  # noqa: E402

app = AppLauncher(headless=True).app

import gymnasium as gym  # noqa: E402
import isaaclab_tasks  # noqa: E402,F401
import go2wm.sim.go2w_task  # noqa: E402,F401  (registers the Go2-W task)
from isaaclab.utils.io import dump_yaml  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

env_cfg = parse_env_cfg(a.task, device="cuda:0", num_envs=a.num_envs)
agent_cfg = load_cfg_from_registry(a.task, "rsl_rl_cfg_entry_point")
agent_cfg.max_iterations = a.iterations
agent_cfg.seed = a.seed
env_cfg.seed = a.seed
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
env_cfg.log_dir = str(out)
env = gym.make(a.task, cfg=env_cfg)
env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)


class CsvWriter:
    """Minimal stand-in for TensorBoard's SummaryWriter (tensorboard's absl dependency is not installed in the
    Isaac Lab env): scalars go to <out>/progress.csv, one row per (iteration, tag)."""

    def __init__(self, path):
        self.f = open(path, "a", buffering=1)

    def add_scalar(self, tag, value, step):
        self.f.write(f"{step},{tag},{float(value):.6g}\n")

    def flush(self):
        self.f.flush()

    def close(self):
        self.f.close()

    def save_file(self, *a_, **k_):
        pass


class Runner(OnPolicyRunner):
    def _prepare_logging_writer(self) -> None:
        if self.log_dir is not None and self.writer is None and not self.disable_logs:
            self.logger_type = "tensorboard"
            self.writer = CsvWriter(Path(self.log_dir) / "progress.csv")


runner = Runner(env, agent_cfg.to_dict(), log_dir=str(out), device=agent_cfg.device)
if a.resume:
    print(f"[train_go2w] resuming from {a.resume}", flush=True)
    runner.load(a.resume)
dump_yaml(str(out / "params" / "env.yaml"), env_cfg)
dump_yaml(str(out / "params" / "agent.yaml"), agent_cfg)
t0 = time.time()
runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
print(f"[train_go2w] training time {time.time() - t0:.0f} s", flush=True)
models = sorted(out.glob("model_*.pt"), key=lambda p: int(p.stem.split("_")[1]) if p.stem.split("_")[1].isdigit() else -1)
models = [m for m in models if m.stem.split("_")[1].isdigit()]
if models:
    shutil.copy(models[-1], out / "model_latest.pt")
    print(f"[train_go2w] latest {models[-1]} -> model_latest.pt", flush=True)
env.close()
app.close()
