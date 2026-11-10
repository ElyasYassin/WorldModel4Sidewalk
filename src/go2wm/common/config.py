"""YAML config loading and data-root path resolution."""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = REPO_ROOT / "configs"


def load_config(path: str | Path, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    path = Path(path)
    if not path.is_absolute() and not path.exists():
        path = CONFIG_DIR / path.name
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    base = cfg.pop("_base_", None)
    if base:
        cfg = deep_merge(load_config(path.parent / base), cfg)
    if overrides:
        cfg = deep_merge(cfg, overrides)
    return cfg


def deep_merge(a: dict, b: dict) -> dict:
    out = copy.deepcopy(a)
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def data_root() -> Path:
    """Root for datasets/outputs. GO2WM_ROOT env var > configs/paths.yaml > repo dir.

    The repo lives in OneDrive, so large artifacts default to a local disk.
    """
    env = os.environ.get("GO2WM_ROOT")
    if env:
        return Path(env)
    paths_cfg = CONFIG_DIR / "paths.yaml"
    if paths_cfg.exists():
        with open(paths_cfg, "r", encoding="utf-8") as f:
            root = (yaml.safe_load(f) or {}).get("root")
        if root and Path(root).anchor and Path(Path(root).anchor).exists():
            return Path(root)
    return REPO_ROOT


def resolve_path(p: str | Path) -> Path:
    """Absolute paths are kept; relative ones (e.g. data/mock) go under data_root()."""
    p = Path(p)
    return p if p.is_absolute() else data_root() / p
