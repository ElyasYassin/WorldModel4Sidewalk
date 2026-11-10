"""Evaluate dynamics checkpoints (plus kinematic baselines) on a held-out split.

  python scripts/evaluate.py --checkpoint outputs/state_mock/best.pt --data data/mock --split test
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from go2wm.cli import write_json
from go2wm.common.config import resolve_path
from go2wm.data.dataset import load_split
from go2wm.eval.plots import plot_prediction_report
from go2wm.eval.prediction import default_baselines, evaluate_models
from go2wm.models.state_dynamics import load_checkpoint
from go2wm.models.training import pick_device


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", nargs="+", required=True, help="one or more .pt files")
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--out", default=None, help="default: <first checkpoint dir>/eval_<split>")
    ap.add_argument("--no-baselines", action="store_true")
    ap.add_argument("--manifest", default=None,
                    help="split manifest name (default: the first checkpoint's). Comparing checkpoints trained on "
                         "different manifests needs a test split none of them trained on.")
    args = ap.parse_args(argv)

    models, cfg = {}, None
    for c in args.checkpoint:
        p = resolve_path(c)
        if not p.exists():
            raise SystemExit(f"[ERROR] checkpoint not found: {p} (train first with scripts/train.py)")
        m, ck = load_checkpoint(p)
        cfg = cfg or ck["config"]
        models[f"{m.model_type}:{p.parent.name}"] = m
    dc = cfg["data"]
    if not args.no_baselines:
        models.update(default_baselines(dc["history"]))
    root = resolve_path(args.data)
    manifest = args.manifest or dc.get("manifest", "default")
    eps = load_split(root, args.split, manifest)
    res = evaluate_models(models, eps, dc["history"], dc["eval_horizons_s"], pick_device("auto"))
    res["split"] = args.split
    res["manifest"] = manifest
    res["data_root"] = str(root)
    backends = sorted({e.meta.get("backend") for e in eps})
    res["backends"] = backends
    out = resolve_path(args.out) if args.out else resolve_path(args.checkpoint[0]).parent / f"eval_{args.split}"
    write_json(out / "prediction_metrics.json", res)
    plot_prediction_report(res, out, title_suffix=" [MOCK backend]" if backends == ["mock"] else "")

    print(f"[eval] split={args.split} backends={backends} windows={res['n_windows']} episodes={res['n_episodes']} dt~{res['nominal_dt_s']:.3f}s")
    hdr = "model".ljust(28) + "".join(f"pos@{h}s".rjust(11) for h in res["horizon_steps"]) + "".join(f"yaw@{h}s".rjust(11) for h in res["horizon_steps"])
    print(hdr)
    for name, r in res["models"].items():
        row = name.ljust(28)
        row += "".join(f"{r['open_loop'][h]['pos_err_m_mean']:10.3f}m" for h in res["horizon_steps"])
        row += "".join(f"{r['open_loop'][h]['heading_err_deg_mean']:9.2f}deg" for h in res["horizon_steps"])
        print(row)
    pc = res.get("paired") or {}
    for other, cmp in pc.get("comparisons", {}).items():
        parts = [f"{h}s {c['mean_diff_m']:+.3f} [{c['diff_ci95_over_episodes'][0]:+.3f},{c['diff_ci95_over_episodes'][1]:+.3f}]"
                 f" ep {c['episodes_ref_better']}/{c['n_episodes']}" for h, c in cmp.items()]
        print(f"[paired] {other} - {pc['reference']} pos err: " + "; ".join(parts))
    print(f"[eval] wrote {out / 'prediction_metrics.json'} and plots")
    return 0


if __name__ == "__main__":
    sys.exit(main())
