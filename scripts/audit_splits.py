"""Audit dataset splits and navigation scenes for leakage.

  python scripts/audit_splits.py --data data/go2_sim --outputs outputs [--json outputs/audit/splits.json]

Checks, per manifest: every episode is in exactly one split; splits are disjoint by episode,
scene id, seed and scene geometry; no episode's frames are split (episodes are the unit, and
windows never cross episodes, see dataset.build_windows). Across manifests: which test episodes
were ever in another manifest's train/val. For navigation: every scene id any saved run used,
and whether a navigation scene shares an id, seed or geometry with any training episode.
Exit code 1 if a within-manifest violation is found.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from go2wm.common.config import resolve_path
from go2wm.data.episodes import list_episodes, load_episode


def geom_key(scene: dict) -> str:
    """Hash of rounded obstacle layout + start + goal + bounds (identifies identical scenes)."""
    obs = np.round(np.asarray(scene["obstacles"], dtype=float).reshape(-1, 3), 2)
    obs = obs[np.lexsort(obs.T[::-1])] if len(obs) else obs
    parts = [obs, np.round(np.asarray(scene["start"][:2], float), 2), np.round(np.asarray(scene["goal"], float), 2),
             np.round(np.asarray(scene["bounds"], float), 2)]
    return hashlib.sha1(b"|".join(p.tobytes() for p in parts)).hexdigest()[:16]


def audit_manifest(name: str, m: dict, meta: dict[str, dict]) -> dict:
    eps = m["episodes"]
    problems = []
    all_listed = [e for s in eps.values() for e in s]
    dup = sorted({e for e in all_listed if all_listed.count(e) > 1})
    if dup:
        problems.append(f"episodes in more than one split: {dup}")
    missing = sorted(set(all_listed) - set(meta))
    if missing:
        problems.append(f"manifest lists episodes not on disk: {missing[:5]}...")
    keyed = {"scene_id": lambda e: meta[e]["scene_id"], "seed": lambda e: meta[e]["seed"],
             "geometry": lambda e: geom_key(meta[e]["scene"])}
    overlaps = {}
    for kname, f in keyed.items():
        sets = {s: {f(e) for e in ids if e in meta} for s, ids in eps.items()}
        for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
            inter = sets.get(a, set()) & sets.get(b, set())
            if inter:
                overlaps[f"{kname}:{a}&{b}"] = sorted(map(str, inter))[:10]
                problems.append(f"{kname} shared between {a} and {b}: {len(inter)}")
    unassigned = sorted(set(meta) - set(all_listed)) if m["episodes"].get("train") else []
    return {"counts": {s: len(v) for s, v in eps.items()}, "unassigned_on_disk": len(unassigned),
            "overlaps": overlaps, "problems": problems}


def nav_scenes(out_root: Path) -> dict[str, dict]:
    used: dict[str, dict] = {}
    for f in out_root.rglob("*_episodes.json"):
        if "mock" in f.parts:
            continue
        try:
            res = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        for r in res:
            d = used.setdefault(r["scene_id"], {"seed": r["seed"], "geometry": geom_key(r["scene"]), "runs": set()})
            d["runs"].add(str(f.parent.parent.relative_to(out_root)))
    return used


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--outputs", default="outputs")
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)
    root = resolve_path(args.data)
    meta = {}
    for p in list_episodes(root):
        mm = load_episode(p, load_arrays=False).meta
        meta[mm["episode_id"]] = mm
    report: dict = {"data": str(root), "n_episodes_on_disk": len(meta), "manifests": {}}

    # one episode per scene id? (if not, scene-disjoint is stronger than episode-disjoint)
    per_scene = defaultdict(list)
    for e, mm in meta.items():
        per_scene[mm["scene_id"]].append(e)
    report["episodes_per_scene_max"] = max(len(v) for v in per_scene.values())
    geoms = defaultdict(list)
    for e, mm in meta.items():
        geoms[geom_key(mm["scene"])].append(e)
    report["duplicate_geometries_on_disk"] = {k: v for k, v in geoms.items() if len(v) > 1}

    manifests = {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in sorted((root / "splits").glob("*.json"))}
    bad = False
    for name, m in manifests.items():
        r = audit_manifest(name, m, meta)
        report["manifests"][name] = r
        bad |= bool(r["problems"])

    # cross-manifest: test episodes that some other manifest trained/validated on
    seen_fit = {n: set(m["episodes"].get("train", [])) | set(m["episodes"].get("val", [])) for n, m in manifests.items()}
    cross = {}
    for n, m in manifests.items():
        test = set(m["episodes"].get("test", []))
        cross[n] = {o: len(test & s) for o, s in seen_fit.items() if o != n and test & s}
    report["test_episodes_fit_by_other_manifests"] = cross
    ever_fit = set().union(*seen_fit.values()) if seen_fit else set()
    ever_test = set().union(*(set(m["episodes"].get("test", [])) for m in manifests.values())) if manifests else set()
    report["episodes_never_used_for_fit_or_test"] = sorted(set(meta) - ever_fit - ever_test)

    # navigation scenes
    used = nav_scenes(resolve_path(args.outputs))
    train_ids = {mm["scene_id"] for mm in meta.values()}
    train_seeds = {mm["seed"] for mm in meta.values()}
    train_geoms = set(geoms)
    nav_rep = {}
    for sid, d in sorted(used.items()):
        nav_rep[sid] = {"seed": d["seed"], "runs": sorted(d["runs"]), "id_in_data": sid in train_ids,
                        "seed_in_data": d["seed"] in train_seeds, "geometry_in_data": d["geometry"] in train_geoms}
    report["navigation_scenes_used"] = nav_rep
    idx = sorted({int(s.rpartition("_")[2]) for s in used})
    report["navigation_index_range_used"] = [idx[0], idx[-1]] if idx else None
    report["navigation_scenes_overlapping_data"] = [s for s, v in nav_rep.items()
                                                    if v["id_in_data"] or v["seed_in_data"] or v["geometry_in_data"]]

    print(f"[audit] {len(meta)} episodes on disk, max episodes per scene = {report['episodes_per_scene_max']}, "
          f"duplicate geometries = {len(report['duplicate_geometries_on_disk'])}")
    for n, r in report["manifests"].items():
        print(f"[audit] manifest {n}: {r['counts']} problems={r['problems'] or 'none'}")
    print(f"[audit] test episodes fit by other manifests: {cross}")
    print(f"[audit] episodes never used for fit or test: {len(report['episodes_never_used_for_fit_or_test'])}")
    print(f"[audit] navigation scenes used: {len(used)}, index range {report['navigation_index_range_used']}, "
          f"overlapping training data: {report['navigation_scenes_overlapping_data'] or 'none'}")
    if args.json:
        p = resolve_path(args.json)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(report, indent=2, default=list), encoding="utf-8")
        print(f"[audit] wrote {p}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
