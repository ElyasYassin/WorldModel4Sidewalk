"""Training loop for state-only dynamics models."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from ..common.seeds import seed_everything
from ..data.dataset import WindowArrays, build_windows, load_split
from .losses import rollout_loss
from .state_dynamics import DynamicsBase, build_model, save_checkpoint

BATCH_KEYS = ("hist_feat", "hist_act", "hist_dt", "fut_act", "fut_dt", "targets", "fut_pose")


def pick_device(name: str = "auto") -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def to_tensors(w: WindowArrays, device: torch.device) -> dict[str, torch.Tensor]:
    out = {k: torch.as_tensor(getattr(w, k), device=device) for k in BATCH_KEYS}
    if w.terrain_map is not None:
        out["terrain_map"] = torch.as_tensor(w.terrain_map, device=device)
    return out


def batch_loss(model: DynamicsBase, b: dict[str, torch.Tensor], rollout_weight: float, huber_delta: float) -> torch.Tensor:
    out = model.rollout(b["hist_feat"], b["hist_act"], b["hist_dt"], b["fut_act"], b["fut_dt"], return_normalized=True,
                        terrain_map=b.get("terrain_map"))
    return rollout_loss(out["deltas_norm"], model.target_norm(b["targets"]), rollout_weight, huber_delta)


@torch.no_grad()
def evaluate_loss(model, data: dict[str, torch.Tensor], tc: dict[str, Any], bs: int = 4096) -> dict[str, float]:
    model.eval()
    n = len(data["hist_feat"])
    tot, pos_err = 0.0, 0.0
    for i in range(0, n, bs):
        b = {k: v[i : i + bs] for k, v in data.items()}
        m = len(b["hist_feat"])
        tot += batch_loss(model, b, tc["rollout_weight"], tc["huber_delta"]).item() * m
        out = model.rollout(b["hist_feat"], b["hist_act"], b["hist_dt"], b["fut_act"], b["fut_dt"], terrain_map=b.get("terrain_map"))
        pos_err += (out["poses"][:, -1, :2] - b["fut_pose"][:, -1, :2]).norm(dim=-1).sum().item()
    return {"loss": tot / n, "final_pos_err_m": pos_err / n}


def train(cfg: dict[str, Any], data_root: str | Path, out_dir: str | Path, log: Callable[[str], None] = print) -> dict[str, Any]:
    dc, tc = cfg["data"], cfg["train"]
    seed_everything(tc["seed"])
    device = pick_device(tc.get("device", "auto"))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mkw = dict(seed=dc.get("split_seed", 0), val_frac=dc.get("val_frac", 0.15), test_frac=dc.get("test_frac", 0.15),
               holdout_families=dc.get("holdout_families"))
    train_eps = load_split(data_root, "train", dc.get("manifest", "default"), **mkw)
    val_eps = load_split(data_root, "val", dc.get("manifest", "default"), **mkw)
    K, H = dc["history"], dc["horizon_train"]
    from .state_dynamics import MODEL_TYPES

    uses_t = getattr(MODEL_TYPES.get(cfg["model"]["type"]), "uses_terrain", False)
    src = cfg["model"].get("terrain_source", "analytic") if uses_t else None
    scan, sensor = src == "scan", bool(src and src.startswith("sensor:"))
    wkw = dict(terrain_map=src == "analytic", terrain_scan=scan, terrain_sensors=(src,) if sensor else ())
    w_tr = build_windows(train_eps, K, H, dc.get("stride", 1), **wkw)
    w_va = build_windows(val_eps, K, H, dc.get("stride", 1), **wkw)
    if scan:  # the model reads `terrain_map`; feed it the scanner-built map
        w_tr.terrain_map, w_va.terrain_map = w_tr.terrain_map_scan, w_va.terrain_map_scan
    if sensor:
        w_tr.terrain_map, w_va.terrain_map = w_tr.terrain_map_sensor[src], w_va.terrain_map_sensor[src]
    log(f"[train] device={device} train: {len(train_eps)} eps / {len(w_tr)} windows, val: {len(val_eps)} eps / {len(w_va)} windows")

    model = build_model(cfg)
    model.fit_normalizers(w_tr)  # train split statistics only
    model.to(device)
    tr, va = to_tensors(w_tr, device), to_tensors(w_va, device)
    history: list[dict[str, float]] = []
    best = float("inf")
    extra = {"config": cfg, "data_root": str(data_root), "train_episodes": [e.episode_id for e in train_eps]}
    if not model.learned:
        metrics = evaluate_loss(model, va, tc)
        save_checkpoint(out_dir / "best.pt", model, extra)
        log(f"[train] non-learned baseline; val {metrics}")
        return {"best_val_loss": metrics["loss"], "history": [metrics]}

    opt = torch.optim.AdamW(model.parameters(), lr=tc["lr"], weight_decay=tc.get("weight_decay", 0.0))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=tc["epochs"])
    g = torch.Generator(device="cpu").manual_seed(tc["seed"])
    n = len(w_tr)
    t0 = time.perf_counter()
    for epoch in range(tc["epochs"]):
        model.train()
        perm = torch.randperm(n, generator=g).to(device)
        run, cnt = 0.0, 0
        for i in range(0, n, tc["batch_size"]):
            idx = perm[i : i + tc["batch_size"]]
            b = {k: v[idx] for k, v in tr.items()}
            loss = batch_loss(model, b, tc["rollout_weight"], tc["huber_delta"])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), tc["grad_clip"])
            opt.step()
            run += loss.item() * len(idx)
            cnt += len(idx)
        sched.step()
        vm = evaluate_loss(model, va, tc)
        rec = {"epoch": epoch, "train_loss": run / cnt, "val_loss": vm["loss"], "val_final_pos_err_m": vm["final_pos_err_m"]}
        history.append(rec)
        if vm["loss"] < best:
            best = vm["loss"]
            save_checkpoint(out_dir / "best.pt", model, {**extra, "epoch": epoch, "val": vm})
        if epoch % max(1, tc["epochs"] // 10) == 0 or epoch == tc["epochs"] - 1:
            log(f"[train] ep {epoch:3d} train {rec['train_loss']:.4f} val {vm['loss']:.4f} "
                f"val pos err @step{H} {vm['final_pos_err_m']:.3f} m")
    save_checkpoint(out_dir / "last.pt", model, {**extra, "epoch": tc["epochs"] - 1})
    result = {"best_val_loss": best, "history": history, "wall_s": time.perf_counter() - t0,
              "n_train_windows": len(w_tr), "n_val_windows": len(w_va)}
    with open(out_dir / "train_metrics.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    log(f"[train] done in {result['wall_s']:.1f}s, best val loss {best:.4f} -> {out_dir / 'best.pt'}")
    return result
