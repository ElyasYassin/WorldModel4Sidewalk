"""Write inspection reports (HTML + Markdown + JSON) for finished missions.

  python scripts/inspection_report.py --missions outputs/missions --tiles outputs/defect_tiles/tiles.json
  python scripts/inspection_report.py --missions outputs/missions --only approach_tile2

Reports go to <missions>/reports/<policy>_tile<i>.{html,md,json}.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from go2wm.common.config import resolve_path  # noqa: E402
from go2wm.mission.report import build_report, save_report  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--missions", default="outputs/missions")
    ap.add_argument("--tiles", default="outputs/defect_tiles/tiles.json")
    ap.add_argument("--only", nargs="*", default=None, help="mission file stems, e.g. approach_tile2")
    ap.add_argument("--no-evidence", action="store_true", help="skip the camera evidence renders (fast)")
    ap.add_argument("--no-gt", action="store_true", help="omit the ground-truth evaluation appendix")
    a = ap.parse_args()
    md = resolve_path(a.missions)
    tiles = json.loads(resolve_path(a.tiles).read_text())
    files = sorted(f for f in md.glob("*_tile*.json") if a.only is None or f.stem in a.only)
    for f in files:
        t0 = time.time()
        res = json.loads(f.read_text())
        rep = build_report(res, tiles[res["tile"]], title=f"Sidewalk inspection: tile {res['tile']} ({res['policy']})",
                           evidence=not a.no_evidence, gt_appendix=not a.no_gt)
        p = save_report(rep, md / "reports", f.stem)
        print(f"{f.stem}: {len(rep['rows'])} findings {rep['counts']} -> {p} ({time.time() - t0:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
