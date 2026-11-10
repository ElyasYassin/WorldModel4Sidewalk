"""Write the frozen E4 tile sets (configs/e4_tiles.yaml) from fixed seeds.

  python scripts/make_terrain_tiles.py [--train 12 --test 6]

train: random tiles for the training/dev pool (seed 1). test: random tiles from a different seed
(2), never used for fitting. Slabs reach 18 cm and grades 25 %: around the speed-dependent limit
found in the E4 sweep (15 cm / 30 % stall at 0.4 m/s), so the data contains both clean traversals
and stalls, which is exactly what command kinematics cannot predict.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import yaml

from go2wm.sim.terrain import random_sidewalk_tile


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", type=int, default=12)
    ap.add_argument("--test", type=int, default=6)
    ap.add_argument("--out", default="configs/e4_tiles.yaml")
    args = ap.parse_args(argv)
    sets = {"train": [random_sidewalk_tile(np.random.default_rng([1, i])) for i in range(args.train)],
            "test": [random_sidewalk_tile(np.random.default_rng([2, i])) for i in range(args.test)]}
    head = ("# FROZEN E4 tile sets, written by scripts/make_terrain_tiles.py (train seed 1, test seed 2).\n"
            "# Tile params are sidewalk_height() inputs (src/go2wm/sim/terrain.py); seed s runs on tile s % n.\n")
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(head)
        yaml.safe_dump(sets, f, default_flow_style=None, sort_keys=False)
    for k, v in sets.items():
        print(k, len(v), "max slab", max((s[2] for t in v for s in t["slabs"]), default=0),
              "curbs", sum(t["curb_drop"] > 0 for t in v), "grades", sum(bool(t["grade"]) for t in v))
    return 0


if __name__ == "__main__":
    sys.exit(main())
