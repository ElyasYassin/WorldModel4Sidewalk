"""Dataset validator: python -m go2wm.data.validate --data <root>"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from ..common.config import resolve_path
from ..common.schema import COMMAND_DIM, SCHEMA_VERSION, STATE_DIM, STATE_FIELDS
from .episodes import ARRAY_SPEC, OPTIONAL_ARRAYS, Episode, list_episodes, load_episode

REQUIRED_META = ["schema_version", "episode_id", "scene_id", "seed", "backend", "frame_convention", "num_transitions"]


def validate_episode(ep: Episode, max_dt_s: float = 1.0) -> list[str]:
    errs: list[str] = []
    m, a = ep.meta, ep.arrays
    for k in REQUIRED_META:
        if k not in m:
            errs.append(f"missing meta '{k}'")
    if m.get("schema_version") != SCHEMA_VERSION:
        errs.append(f"schema_version {m.get('schema_version')} != {SCHEMA_VERSION}")
    for name, (dtype, kind) in ARRAY_SPEC.items():
        if name not in a:
            if name not in OPTIONAL_ARRAYS:
                errs.append(f"missing array '{name}'")
            continue
        if np.dtype(a[name].dtype) != np.dtype(dtype):
            errs.append(f"{name}: dtype {a[name].dtype} != {dtype}")
    if errs:
        return errs
    T = len(a["commands"])
    if T == 0:
        return ["zero transitions"]
    if m.get("num_transitions") != T:
        errs.append(f"meta num_transitions {m.get('num_transitions')} != {T}")
    for name, (_, kind) in ARRAY_SPEC.items():
        if name in a and len(a[name]) != (T + 1 if kind == "obs" else T):
            errs.append(f"{name}: length {len(a[name])} (expected {'T+1' if kind == 'obs' else 'T'}={T + (kind == 'obs')})")
    if a["states"].shape[1:] != (STATE_DIM,):
        errs.append(f"states shape {a['states'].shape}")
    if a["commands"].shape[1:] != (COMMAND_DIM,):
        errs.append(f"commands shape {a['commands'].shape}")
    if errs:
        return errs
    for name in ("states", "commands", "dt_s"):
        if not np.all(np.isfinite(a[name])):
            errs.append(f"{name}: non-finite values")
    ts = a["obs_timestamp_ns"]
    d = np.diff(ts)
    if np.any(d <= 0):
        errs.append(f"non-increasing observation timestamps at {np.nonzero(d <= 0)[0][:5].tolist()}")
    if not np.allclose(a["dt_s"], d * 1e-9, atol=1e-6):
        errs.append("dt_s inconsistent with observation timestamps")
    if np.any(a["dt_s"] > max_dt_s):
        errs.append(f"dt_s exceeds {max_dt_s}s (dropped frames?) at {np.nonzero(a['dt_s'] > max_dt_s)[0][:5].tolist()}")
    at = a["action_timestamp_ns"]
    if np.any(at < ts[:-1]) or np.any(at >= ts[1:]):
        errs.append("action timestamp outside [t_obs, t_next_obs) -> misaligned action")
    qn = np.linalg.norm(a["states"][:, STATE_FIELDS["quat_w"]], axis=1)
    if np.any(np.abs(qn - 1.0) > 1e-3):
        errs.append("quaternion not unit norm")
    gn = np.linalg.norm(a["states"][:, STATE_FIELDS["gravity_b"]], axis=1)
    if np.any(np.abs(gn - 1.0) > 0.05):
        errs.append("projected gravity not unit norm (units?)")
    if np.any(a["commands"][:, 3] <= 0):
        errs.append("non-positive command duration")
    end = a["terminated"] | a["truncated"]
    if np.any(end[:-1]):
        errs.append(f"terminal flag before last transition at {np.nonzero(end[:-1])[0][:5].tolist()} (boundary leakage)")
    if not end[-1]:
        errs.append("last transition neither terminated nor truncated")
    return errs


def validate_dataset(root: str | Path, manifest_name: str | None = "default") -> tuple[dict[str, list[str]], list[str]]:
    root = Path(root)
    per_ep: dict[str, list[str]] = {}
    global_errs: list[str] = []
    paths = list_episodes(root)
    if not paths:
        return {}, [f"no episodes found under {root / 'episodes'}"]
    seen: dict[str, Path] = {}
    scene_of: dict[str, str] = {}
    for p in paths:
        try:
            ep = load_episode(p)
        except Exception as e:  # noqa: BLE001
            per_ep[p.stem] = [f"failed to load: {e}"]
            continue
        eid = ep.meta.get("episode_id", p.stem)
        if eid != p.stem:
            global_errs.append(f"{p.name}: episode_id '{eid}' != filename")
        if eid in seen:
            global_errs.append(f"duplicate episode_id {eid}")
        seen[eid] = p
        scene_of[eid] = ep.meta.get("scene_id")
        errs = validate_episode(ep)
        if errs:
            per_ep[eid] = errs
    split_file = root / "splits" / f"{manifest_name}.json"
    if manifest_name and split_file.exists():
        from .dataset import load_manifest

        m = load_manifest(root, manifest_name)
        sets = {k: set(v) for k, v in m["episodes"].items()}
        for a_ in sets:
            for b_ in sets:
                if a_ < b_ and sets[a_] & sets[b_]:
                    global_errs.append(f"split overlap {a_}/{b_}: {sorted(sets[a_] & sets[b_])[:3]}")
        scenes = {k: {scene_of.get(e) for e in v} for k, v in sets.items()}
        for a_ in scenes:
            for b_ in scenes:
                if a_ < b_ and scenes[a_] & scenes[b_]:
                    global_errs.append(f"scene leakage between {a_}/{b_}: {sorted(map(str, scenes[a_] & scenes[b_]))[:3]}")
        unknown = set().union(*sets.values()) - set(seen)
        if unknown:
            global_errs.append(f"manifest references missing episodes: {sorted(unknown)[:3]}")
    elif manifest_name:
        global_errs.append(f"(info) no split manifest at {split_file}; splits not checked")
    return per_ep, global_errs


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Validate a go2wm episode dataset")
    ap.add_argument("--data", required=True)
    ap.add_argument("--manifest", default="default")
    args = ap.parse_args(argv)
    root = resolve_path(args.data)
    per_ep, glob = validate_dataset(root, args.manifest)
    n = len(list_episodes(root))
    for eid, errs in per_ep.items():
        for e in errs:
            print(f"[FAIL] {eid}: {e}")
    hard = [g for g in glob if not g.startswith("(info)")]
    for g in glob:
        print(("[INFO] " if g.startswith("(info)") else "[FAIL] ") + g)
    ok = not per_ep and not hard
    print(f"{'PASS' if ok else 'FAIL'}: {n} episodes checked in {root}, {len(per_ep)} with errors")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
