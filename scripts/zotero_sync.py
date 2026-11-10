"""Mirror the project's literature into a Zotero collection via pyzotero.

Every item is resolved from an authoritative record (arXiv API or Crossref)
before upload; unresolved titles are reported, never invented.

Credentials (never commit them): set env vars ZOTERO_LIBRARY_ID and
ZOTERO_API_KEY (optional ZOTERO_LIBRARY_TYPE=user|group, default user), or put
the same KEY=VALUE lines in ~/.zotero.env.

Usage:
  python scripts/zotero_sync.py resolve           # build/refresh paper/zotero_manifest.json
  python scripts/zotero_sync.py push              # upload unsynced manifest items
  python scripts/zotero_sync.py add --arxiv 2409.13393 [--doi 10.x/...] [--title "..."]
"""
from __future__ import annotations

import argparse
import difflib
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BIB = ROOT / "paper" / "references.bib"
MANIFEST = ROOT / "paper" / "zotero_manifest.json"
COLLECTION = "World Model Sidewalk"
UA = {"User-Agent": "WorldModel4Sidewalk-zotero-sync/0.1"}
ATOM = {"a": "http://www.w3.org/2005/Atom"}

# Papers in the NotebookLM notebook that are not cited in references.bib.
# Resolved by title; kept only when the authoritative title matches.
EXTRA_TITLES = [
    "Learning Forward Dynamics Model and Informed Trajectory Sampler for Safe Quadruped Navigation",
    "Benchmarking World Models for Continual Learning on Compositional Tasks",
    "Behavioral Conflict Avoidance Between Humans and Quadruped Robots in Shared Environments",
    "Control Barrier Function for Aligning Large Language Models",
    "Control Barrier Function-Based Quadratic Programming for SafeOperation of Tethered UAVs",  # arXiv title has no space
    "Control Barrier Functions: Theory and Applications",
    "Cybersecurity of Teleoperated Quadruped Robots: A Systematic Survey of Vulnerabilities, Threats, and Open Defense Gaps",
    "DR-VLA: Learning Human-Like Quadruped Navigation from Egocentric Web Videos via Domain-Invariant Scene Representations",
    "Dream to Fly: Model-Based Reinforcement Learning for Vision-Based Drone Flight",
    "Dynamic Object Detection and Tracking in Construction: A Fisheye Camera and LiDAR Sensor Fusion Model",
    "EfficientTDMPC: Improved MPC Objectives for Sample-Efficient Continuous Control",
    "Explicit Control Barrier Function-based Safety Filters and their Resource-Aware Computation",
    "Explicit Solutions for Safety Problems Using Control Barrier Functions",
    "Fast Traversability Estimation for Wild Visual Navigation",
    "Geometry-Aware Predictive Safety Filters on Humanoids: From Poisson Safety Functions to CBF Constrained MPC",
    "STONE Dataset: A Scalable Multi-Modal Surround-View 3D Traversability Dataset for Off-Road Robot Navigation",
    "Hierarchical Open-Vocabulary 3D Scene Graphs for Language-Grounded Robot Navigation",
    "HuGo: LLMs as Whole-Body Policy Code Designers for Humanoid Loco-Manipulation",
    "Hydra++: Real-Time Hierarchical 3D Scene Graph Construction With Object-Level Shape Estimation",
    "Learning Massively Multitask World Models for Continuous Control",
    "Learning Task-Invariant Properties via Dreamer: Enabling Efficient Policy Transfer for Quadruped Robots",
    "Quadruped Robot Aided Construction Inspection and Documentation",
    "RT-Trajectory: Robotic Task Generalization via Hindsight Trajectory Sketches",
    "Robotic Replanning with Perception and Language Models",
    "Scene-Agnostic Traversability Labeling and Estimation via a Multimodal Self-supervised Framework",
    "Security of World-Model-Based Embodied AI: A Lifecycle of Threats, Defenses, and Evaluation",
    "SemantI-Nav: Human-Guided Imitation Learning for Safe Quadruped Navigation on Construction Sites",
    "Semantic Navigation Using Building Information on Construction Sites",
    "TD-GRPC: Temporal Difference Learning with Group Relative Policy Constraint for Humanoid Locomotion",
    "Temporal Difference Learning for Model Predictive Control",
    "VLMPC: Vision-Language Model Predictive Control for Robotic Manipulation",
    "Foundations of Spatial Perception for Robotics: Hierarchical Representations and Real-time Systems",
    "Quadrupped-Legged Robot Movement Plan Generation using Large Language Model",
]

# Items with no arXiv ID or DOI, entered from their canonical record.
MANUAL = [
    {
        "key": "lecun2022path", "itemType": "report",
        "title": "A Path Towards Autonomous Machine Intelligence",
        "creators": [{"creatorType": "author", "firstName": "Yann", "lastName": "LeCun"}],
        "date": "2022", "url": "https://openreview.net/forum?id=BZ5a1r-kVsf",
        "extra": "OpenReview preprint, version 0.9.2", "source": "manual",
    }
]


# ----------------------------------------------------------------- helpers
def _get(url: str) -> bytes:
    req = urllib.request.Request(url, headers=UA)
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read()
        except Exception:
            if attempt == 3:
                raise
            time.sleep(3 * (attempt + 1))
    raise RuntimeError("unreachable")


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def _similar(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, _norm(a), _norm(b)).ratio()


def _split_name(full: str) -> dict:
    full = " ".join(full.split())
    if " " not in full:
        return {"creatorType": "author", "name": full}
    first, last = full.rsplit(" ", 1)
    return {"creatorType": "author", "firstName": first, "lastName": last}


def _arxiv_entries(xml: bytes) -> list[dict]:
    out = []
    for e in ET.fromstring(xml).findall("a:entry", ATOM):
        idurl = e.find("a:id", ATOM).text
        if "/abs/" not in idurl:
            continue
        aid = re.sub(r"v\d+$", "", idurl.split("/abs/")[-1])
        doi_el = e.find("{http://arxiv.org/schemas/atom}doi")
        out.append({
            "itemType": "preprint",
            "title": " ".join(e.find("a:title", ATOM).text.split()),
            "creators": [_split_name(a.find("a:name", ATOM).text)
                         for a in e.findall("a:author", ATOM)],
            "date": e.find("a:published", ATOM).text[:10],
            "abstractNote": " ".join(e.find("a:summary", ATOM).text.split()),
            "repository": "arXiv", "archiveID": f"arXiv:{aid}",
            "url": f"https://arxiv.org/abs/{aid}",
            "DOI": doi_el.text if doi_el is not None else "",
            "arxiv": aid, "source": "arxiv",
        })
    return out


def resolve_arxiv_ids(ids: list[str]) -> dict[str, dict]:
    res = {}
    for i in range(0, len(ids), 40):
        chunk = ids[i:i + 40]
        xml = _get("https://export.arxiv.org/api/query?max_results=50&id_list=" + ",".join(chunk))
        for it in _arxiv_entries(xml):
            res[it["arxiv"]] = it
        time.sleep(3)
    return res


def resolve_arxiv_title(title: str) -> dict | None:
    # Exact-phrase search on the title (up to the first colon-free 8 words);
    # a bag-of-words AND query silently returns nothing for many titles.
    phrase = " ".join(re.sub(r"[:;,?!()\"]", " ", title).split()[:8])
    q = urllib.parse.quote(f'ti:"{phrase}"')
    cands: list[dict] = []
    for attempt in range(2):
        xml = _get(f"https://export.arxiv.org/api/query?search_query={q}&max_results=5")
        time.sleep(3)
        cands = _arxiv_entries(xml)
        if cands:
            break
        time.sleep(5)
    best = max(cands, key=lambda it: _similar(it["title"], title), default=None)
    return best if best and _similar(best["title"], title) >= 0.88 else None


CROSSREF_TYPES = {"journal-article": "journalArticle", "proceedings-article": "conferencePaper",
                  "posted-content": "preprint", "dissertation": "thesis",
                  "book-chapter": "bookSection", "report": "report"}


def _crossref_item(m: dict) -> dict:
    itype = CROSSREF_TYPES.get(m.get("type", ""), "journalArticle")
    title = " ".join((m.get("title") or [""])[0].split())
    container = (m.get("container-title") or [""])[0]
    parts = (m.get("issued") or {}).get("date-parts") or [[None]]
    date = "-".join(str(p) for p in parts[0] if p)
    creators = []
    for a in m.get("author", []):
        if "family" in a:
            creators.append({"creatorType": "author", "firstName": a.get("given", ""),
                             "lastName": a["family"]})
        elif "name" in a:
            creators.append({"creatorType": "author", "name": a["name"]})
    it = {"itemType": itype, "title": title, "creators": creators, "date": date,
          "DOI": m.get("DOI", ""), "url": m.get("URL", ""), "source": "crossref"}
    if itype == "journalArticle":
        it.update(publicationTitle=container, volume=m.get("volume", ""),
                  issue=m.get("issue", ""), pages=m.get("page", ""))
    elif itype == "conferencePaper":
        it.update(proceedingsTitle=container, pages=m.get("page", ""))
    elif itype == "bookSection":
        it.update(bookTitle=container, pages=m.get("page", ""))
    return it


def resolve_doi(doi: str) -> dict | None:
    try:
        m = json.loads(_get(f"https://api.crossref.org/works/{urllib.parse.quote(doi)}"))["message"]
    except Exception:
        return None
    return _crossref_item(m)


def resolve_crossref_title(title: str) -> dict | None:
    url = ("https://api.crossref.org/works?rows=5&query.bibliographic="
           + urllib.parse.quote(title))
    items = json.loads(_get(url))["message"]["items"]
    best = max(items, key=lambda m: _similar((m.get("title") or [""])[0], title), default=None)
    if best and _similar((best.get("title") or [""])[0], title) >= 0.88:
        return _crossref_item(best)
    return None


def bib_ids() -> list[tuple[str, str, str]]:
    """(citekey, kind, id) for every entry in references.bib with an eprint or DOI."""
    text = BIB.read_text(encoding="utf-8")
    out = []
    for m in re.finditer(r"@\w+\{([^,]+),(.*?)\n\}", text, re.S):
        key, body = m.group(1).strip(), m.group(2)
        ep = re.search(r"eprint\s*=\s*\{([^}]+)\}", body)
        doi = re.search(r"doi\s*=\s*\{([^}]+)\}", body)
        if ep:
            out.append((key, "arxiv", ep.group(1)))
        elif doi:
            out.append((key, "doi", doi.group(1)))
    return out


# ---------------------------------------------------------------- manifest
def load_manifest() -> list[dict]:
    return json.loads(MANIFEST.read_text(encoding="utf-8")) if MANIFEST.exists() else []


def save_manifest(items: list[dict]) -> None:
    MANIFEST.write_text(json.dumps(items, indent=2, ensure_ascii=False), encoding="utf-8")


def ident(it: dict) -> str:
    return (it.get("arxiv") or it.get("DOI", "").lower() or _norm(it["title"]))


def merge(manifest: list[dict], new: list[dict]) -> list[dict]:
    seen = {ident(it): it for it in manifest}
    for it in new:
        k = ident(it)
        if k in seen:
            for f in ("key", "tags"):
                if it.get(f) and not seen[k].get(f):
                    seen[k][f] = it[f]
        else:
            manifest.append(it)
            seen[k] = it
    return manifest


def cmd_resolve(_args) -> None:
    new: list[dict] = []
    entries = bib_ids()
    arx = resolve_arxiv_ids([i for _, k, i in entries if k == "arxiv"])
    for key, kind, i in entries:
        it = arx.get(i) if kind == "arxiv" else resolve_doi(i)
        if it:
            new.append({**it, "key": key, "tags": ["cited-in-paper"]})
        else:
            print(f"UNRESOLVED bib entry {key} ({kind} {i})")
    for m in MANUAL:
        new.append({**m, "tags": ["cited-in-paper"]})
    unresolved = []
    known = [x["title"] for x in load_manifest() + new]
    for t in EXTRA_TITLES:
        if any(_similar(t, k) >= 0.88 for k in known):
            continue  # already resolved on an earlier run
        it = resolve_arxiv_title(t) or resolve_crossref_title(t)
        if it:
            new.append({**it, "tags": ["notebooklm"]})
            print(f"ok    {it['source']:8s} {it['title'][:80]}")
        else:
            unresolved.append(t)
            print(f"MISS  {t[:80]}")
    manifest = merge(load_manifest(), new)
    save_manifest(manifest)
    print(f"\nmanifest: {len(manifest)} items -> {MANIFEST}")
    if unresolved:
        print(f"{len(unresolved)} titles not resolved (not uploaded):")
        for t in unresolved:
            print("  -", t)


# ------------------------------------------------------------------ zotero
def zotero_client():
    env = dict(os.environ)
    cfg = Path.home() / ".zotero.env"
    if cfg.exists():
        for line in cfg.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.strip().startswith("#"):
                k, v = line.split("=", 1)
                env.setdefault(k.strip(), v.strip().strip('"'))
    lib, key = env.get("ZOTERO_LIBRARY_ID"), env.get("ZOTERO_API_KEY")
    if not lib or not key:
        sys.exit("Missing ZOTERO_LIBRARY_ID / ZOTERO_API_KEY (env vars or ~/.zotero.env).")
    from pyzotero import zotero
    return zotero.Zotero(lib, env.get("ZOTERO_LIBRARY_TYPE", "user"), key)


def get_collection(zot) -> str:
    for c in zot.everything(zot.collections()):
        if c["data"]["name"] == COLLECTION:
            return c["key"]
    resp = zot.create_collections([{"name": COLLECTION}])
    return resp["successful"]["0"]["key"]


def to_zotero(zot, it: dict, coll: str, templates: dict) -> dict:
    itype = it["itemType"]
    if itype not in templates:
        templates[itype] = zot.item_template(itype)
    tmpl = json.loads(json.dumps(templates[itype]))
    for f, v in it.items():
        if f in tmpl and f not in ("itemType", "collections", "tags") and v:
            tmpl[f] = v
    if it.get("arxiv") and "archiveID" not in tmpl:
        tmpl["extra"] = (tmpl.get("extra", "") + f"\narXiv:{it['arxiv']}").strip()
    tmpl["collections"] = [coll]
    tmpl["tags"] = [{"tag": t} for t in it.get("tags", [])] + [{"tag": "world-model-sidewalk"}]
    return tmpl


def existing_idents(zot, coll: str) -> set[str]:
    have = set()
    for z in zot.everything(zot.collection_items_top(coll)):
        d = z["data"]
        arx = re.search(r"(\d{4}\.\d{4,5})", (d.get("archiveID", "") + " " + d.get("extra", "")
                                               + " " + d.get("url", "")))
        if arx:
            have.add(arx.group(1))
        if d.get("DOI"):
            have.add(d["DOI"].lower())
        if d.get("title"):
            have.add(_norm(d["title"]))
    return have


def cmd_push(_args) -> None:
    manifest = load_manifest()
    if not manifest:
        sys.exit("Empty manifest; run `resolve` first.")
    zot = zotero_client()
    coll = get_collection(zot)
    have = existing_idents(zot, coll)
    todo = [it for it in manifest
            if not it.get("zotero_key") and ident(it) not in have and _norm(it["title"]) not in have]
    print(f"{len(manifest)} in manifest, {len(todo)} to upload to '{COLLECTION}'")
    templates: dict = {}
    for i in range(0, len(todo), 50):
        batch = todo[i:i + 50]
        resp = zot.create_items([to_zotero(zot, it, coll, templates) for it in batch])
        for idx, obj in resp.get("successful", {}).items():
            batch[int(idx)]["zotero_key"] = obj["key"]
        for idx, err in resp.get("failed", {}).items():
            print(f"FAILED {batch[int(idx)]['title'][:70]}: {err}")
    save_manifest(manifest)
    print(f"uploaded {sum(1 for it in todo if it.get('zotero_key'))}")


def cmd_add(args) -> None:
    it = None
    if args.arxiv:
        it = resolve_arxiv_ids([args.arxiv]).get(args.arxiv)
    elif args.doi:
        it = resolve_doi(args.doi)
    elif args.title:
        it = resolve_arxiv_title(args.title) or resolve_crossref_title(args.title)
    if not it:
        sys.exit("Could not resolve that paper from arXiv/Crossref; nothing added.")
    it["tags"] = args.tag or ["notebooklm"]
    save_manifest(merge(load_manifest(), [it]))
    print("resolved:", it["title"])
    if not args.no_push:
        cmd_push(args)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("resolve")
    sub.add_parser("push")
    a = sub.add_parser("add")
    a.add_argument("--arxiv")
    a.add_argument("--doi")
    a.add_argument("--title")
    a.add_argument("--tag", action="append")
    a.add_argument("--no-push", action="store_true")
    args = p.parse_args()
    {"resolve": cmd_resolve, "push": cmd_push, "add": cmd_add}[args.cmd](args)


if __name__ == "__main__":
    main()
