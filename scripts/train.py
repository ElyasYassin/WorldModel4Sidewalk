"""Train a dynamics model.

  python scripts/train.py --config configs/world_model_state.yaml --data data/mock --out outputs/state_mock
"""

from __future__ import annotations

import argparse
import sys

from go2wm.common.config import load_config, resolve_path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default=None, help="default: outputs/<config stem>")
    ap.add_argument("--epochs", type=int, default=None)
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    if "visual" in cfg:
        raise SystemExit(
            "[ERROR] Visual latent model training is not implemented yet (Phase E). "
            "Run scripts/extract_vjepa.py once RGB episodes exist; see docs/EXPERIMENTS.md."
        )
    if args.epochs is not None:
        cfg["train"]["epochs"] = args.epochs
    from pathlib import Path

    from go2wm.models.training import train

    out = resolve_path(args.out or f"outputs/{Path(args.config).stem}")
    train(cfg, resolve_path(args.data), out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
