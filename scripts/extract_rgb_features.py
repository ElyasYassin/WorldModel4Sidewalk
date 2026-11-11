"""Frozen pretrained image features for every RGB frame of an episode set (E5, option A1).

Per episode writes <data>/derived/<tag>/<episode_id>.npz with, for frames 0..T (aligned with states):
  feat_<tag>_cls   [T+1, D]        CLS token
  feat_<tag>_mean  [T+1, D]        mean of patch tokens
  feat_<tag>_grid  [T+1, 2, 2, D]  patch tokens average-pooled to 2 x 2 (top/bottom x left/right of the image)
all float16. Episodes already done are skipped.

  python scripts/extract_rgb_features.py --data data/go2_e4c_train --model facebook/dinov2-small --tag dinov2s
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from go2wm.common.config import resolve_path  # noqa: E402
from go2wm.data.episodes import list_episodes, load_episode, load_rgb  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", default="facebook/dinov2-small")
    ap.add_argument("--tag", default="dinov2s")
    ap.add_argument("--batch", type=int, default=128)
    a = ap.parse_args()
    from transformers import AutoImageProcessor, AutoModel

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    proc = AutoImageProcessor.from_pretrained(a.model)
    model = AutoModel.from_pretrained(a.model).to(dev).eval().half()
    root = resolve_path(a.data)
    out_dir = root / "derived" / a.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = list_episodes(root)
    t0 = time.time()
    for n, p in enumerate(paths):
        ep = load_episode(p, load_arrays=False)
        f = out_dir / f"{ep.episode_id}.npz"
        if f.exists():
            continue
        frames = load_rgb(p)
        if frames is None:
            print(f"[skip] {ep.episode_id}: no RGB")
            continue
        cls, mean, grid = [], [], []
        for i in range(0, len(frames), a.batch):
            # 224 x 224 frames go in at native size (no crop): patch 14 -> 16 x 16 tokens
            x = proc(images=list(frames[i : i + a.batch]), return_tensors="pt", do_resize=False, do_center_crop=False)
            with torch.no_grad():
                h = model(pixel_values=x["pixel_values"].to(dev).half()).last_hidden_state.float()
            tok = h[:, 1:]
            g = int(round(tok.shape[1] ** 0.5))
            tok = tok.reshape(len(tok), g, g, -1)
            cls.append(h[:, 0].cpu())
            mean.append(tok.mean((1, 2)).cpu())
            grid.append(tok.reshape(len(tok), 2, g // 2, 2, g // 2, -1).mean((2, 4)).cpu())
        np.savez(f, **{f"feat_{a.tag}_cls": torch.cat(cls).numpy().astype(np.float16),
                       f"feat_{a.tag}_mean": torch.cat(mean).numpy().astype(np.float16),
                       f"feat_{a.tag}_grid": torch.cat(grid).numpy().astype(np.float16)})
        if (n + 1) % 16 == 0:
            print(f"  {n + 1}/{len(paths)} ({time.time() - t0:.0f}s)", flush=True)
    print(f"done: {len(paths)} episodes -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
