"""Score a mission on a scanned-sidewalk tile against defects measured in the scan (the tile itself has none).

  PYTHONPATH=src python scripts/rescore_scan_mission.py --mission <mission.json> \
      --upheaval 10.68 0.0 1.1 0.0134 --reference "Trimble X7 scan heightmap: ..."

Writes <mission>_rescored.json with a new "score" and "score_reference" (shown in the render_demo_mission.py report).
The reference defects are used for scoring only, never added to the simulated terrain.

Default = the faulted slab of the 2026-11-18 scan (scan/sidewalk_tdx): a step down walking +x at scene x 10.68 on the
+y half, 12.3-14.4 mm over y 0.3-0.9 m (median 13.4 mm), ~7 mm near the centreline, ~17 mm at the outer corner.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from go2wm.mission.runner import score_mission

REF = "Trimble X7 scan heightmap: faulted slab at scene x 10.68, +y half, 13.4 mm (median over y 0.3-0.9 m)"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mission", required=True)
    ap.add_argument("--upheaval", type=float, nargs=4, action="append", metavar=("X_EDGE", "Y0", "Y1", "LIFT"),
                    help="step down walking +x at X_EDGE over y in [Y0, Y1], LIFT [m] (repeatable)")
    ap.add_argument("--reference", default=REF)
    a = ap.parse_args(argv)
    ups = a.upheaval or [[10.68, 0.0, 1.1, 0.0134]]
    src = Path(a.mission)
    r = json.loads(src.read_text())
    tile = copy.deepcopy(json.loads(Path(r["tiles_file"]).read_text())[r["tile"]])
    tile["defects"] = [{"type": "upheaval", "x0": x - 1.5, "x1": x, "y0": y0, "y1": y1, "lift": lift, "high_end": "x1"}
                       for x, y0, y1, lift in ups]
    r["score"] = score_mission(r, tile)
    r["score_reference"] = a.reference
    out = src.with_name(src.stem + "_rescored.json")
    out.write_text(json.dumps(r))
    print(json.dumps(r["score"]["per_defect"], indent=1))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
