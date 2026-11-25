"""Preliminary camera mount study (mock kinematics): stop & look missions per D555 / D405 mount, on dev tiles + the scan.

  python scripts/mount_sweep.py --out D:/go2wm/outputs/mount_sweep [--workers 4] [--summary-only]

One-at-a-time sweeps around the current mounts (src/go2wm/sim/sensors.py): D555 pitch with the D405 fixed, then D405
pitch x height with the D555 fixed. Every config runs approach (stop & look) with the frozen perception (robust,
curb-aware, step detector v2, pose errors) and --dyn-d405, so the stand-off follows where the D405 actually looks.
Tiles: the dev set outputs/defect_tiles_eval (NOT the held-out final sets) and the scanned sidewalk (rescored against
the scan-measured fault). Writes <out>/<config>/ per config and <out>/summary.md.

Not modelled yet: legs / wheels occluding the cameras, the Go2-W's own body height (mock = Go2 kinematics), the
inspection crouch (Isaac only). Treat the ranking as preliminary until those are in.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEV_TILES = "outputs/defect_tiles_eval/tiles.json"
SCAN_TILES = "D:/go2wm/scan/sidewalk_tdx/tiles.json"
PERCEPTION = ["--profile", "realistic", "--pose-errors", "--robust", "--curb-aware", "--step-thresh", "0.006",
              "--step-max", "0.02", "--step-clear", "0.6", "--dyn-d405", "--policies", "approach"]
D555_POS, D405_POS = [0.30, 0.0, 0.08], [0.33, 0.0, -0.03]


def configs():
    out = {"base": {}}
    for p in (20, 25, 35, 40):
        out[f"d555_p{p}"] = {"depth_d555": {"pitch_deg": float(p)}}
    for z, zn in ((-0.03, "chin"), (0.08, "top")):
        for p in (45, 50, 60, 70, 80):
            if zn == "chin" and p == 60:
                continue  # = base
            out[f"d405_{zn}_p{p}"] = {"depth_d405": {"mount_pos": [D405_POS[0], 0.0, z], "pitch_deg": float(p)}}
    return out


def run(name, mounts, out, workers, n_tiles):
    py = sys.executable
    common = PERCEPTION + ["--workers", str(workers), "--mounts", json.dumps(mounts)]
    d = out / name
    if not (d / "comparison.md").exists():
        subprocess.run([py, "scripts/run_mission.py", "--tiles", DEV_TILES, "--use", *map(str, range(n_tiles)),
                        "--out", str(d), *common], cwd=ROOT, check=False)
    s = out / name / "scan"
    if not (s / "comparison.md").exists():
        subprocess.run([py, "scripts/run_mission.py", "--tiles", SCAN_TILES, "--use", "0", "--out", str(s),
                        *PERCEPTION, "--workers", "1", "--mounts", json.dumps(mounts)], cwd=ROOT, check=False)
        f = s / "approach_tile0.json"
        if f.exists():
            subprocess.run([py, "scripts/rescore_scan_mission.py", "--mission", str(f)], cwd=ROOT, check=False)


def summarize(out):
    rows = ["| config | mount change | goal | time [s] | upheaval found / D405 / err [mm] | hole found / D405 / err [mm] | "
            "puddle found / D405 | false | scan fault: found / err [mm] |", "|" + "---|" * 9]
    for name, mounts in configs().items():
        runs = [json.loads(f.read_text()) for f in sorted((out / name).glob("approach_tile*.json"))]
        if not runs:
            continue
        cells = []
        for typ in ("upheaval", "hole", "puddle"):
            d = [x for r in runs for x in r["score"]["per_defect"] if x["type"] == typ]
            errs = [x["abs_err_m"] * 1000 for x in d if "abs_err_m" in x]
            c = f"{sum(x['found'] for x in d)}/{len(d)} / {sum(x.get('best_sensor') == 'depth_d405' for x in d)}"
            if typ != "puddle":
                c += f" / {np.median(errs):.1f}" if errs else " / -"
            cells.append(c)
        sf = out / name / "scan" / "approach_tile0_rescored.json"
        scan = "-"
        if sf.exists():
            pd = json.loads(sf.read_text())["score"]["per_defect"]
            scan = " / ".join([f"{sum(x['found'] for x in pd)}/{len(pd)}"] +
                              [f"{x['abs_err_m'] * 1000:.1f}" for x in pd if "abs_err_m" in x])
        rows.append(f"| {name} | {json.dumps(mounts) if mounts else 'current mounts'} | "
                    f"{sum(r['outcome'] == 'goal' for r in runs)}/{len(runs)} | {np.mean([r['time_s'] for r in runs]):.0f} | "
                    + " | ".join(cells) + f" | {sum(r['score']['false_findings'] for r in runs)} | {scan} |")
    head = ["# Preliminary mount sweep (mock kinematics, stop & look, frozen perception + step v2, dev tiles "
            "defect_tiles_eval + scan)", "",
            "Not modelled: leg / wheel occlusion, Go2-W body height, the crouch. Preliminary ranking only.", ""]
    (out / "summary.md").write_text("\n".join(head + rows) + "\n", encoding="utf-8")
    print((out / "summary.md").read_text(encoding="utf-8"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="D:/go2wm/outputs/mount_sweep")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--tiles", type=int, default=12, help="first N dev tiles (of 24)")
    ap.add_argument("--only", nargs="*", default=None, help="run only these configs")
    ap.add_argument("--summary-only", action="store_true")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if not a.summary_only:
        for name, mounts in configs().items():
            if a.only is None or name in a.only:
                print(f"[mount-sweep] {name}: {mounts or 'current mounts'}", flush=True)
                run(name, mounts, out, a.workers, a.tiles)
                summarize(out)
    summarize(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
