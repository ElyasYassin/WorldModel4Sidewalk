"""Frozen V-JEPA 2 embedding extraction (Phase E, not implemented yet).

  python scripts/extract_vjepa.py --data data/go2_sim --config configs/world_model_visual.yaml

Fails with an actionable message instead of producing placeholder embeddings.
"""

from __future__ import annotations

import argparse
import sys

from go2wm.common.config import resolve_path
from go2wm.data.episodes import list_episodes, load_episode


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--config", default="configs/world_model_visual.yaml")
    args = ap.parse_args(argv)
    root = resolve_path(args.data)
    eps = list_episodes(root)
    if not eps:
        raise SystemExit(f"[ERROR] no episodes in {root}")
    with_rgb = sum(bool(load_episode(p, load_arrays=False).meta.get("modalities", {}).get("rgb")) for p in eps)
    if with_rgb == 0:
        raise SystemExit(
            f"[ERROR] none of the {len(eps)} episodes in {root} contain RGB. Next steps (Phase E):\n"
            "  1. set isaac.enable_cameras: true and add a body-mounted camera to the Isaac adapter;\n"
            "  2. re-collect with scripts/collect.py --backend isaac;\n"
            "  3. download the official V-JEPA 2 weights and implement go2wm/models/vjepa_encoder.py.\n"
            "See docs/EXPERIMENTS.md."
        )
    raise SystemExit("[ERROR] V-JEPA 2 extraction is not implemented yet (Phase E). See docs/EXPERIMENTS.md.")


if __name__ == "__main__":
    sys.exit(main())
