"""Phase 1: run inspection missions on defect tiles and compare inspection strategies (mock backend).

  python scripts/run_mission.py --tiles outputs/defect_tiles/tiles.json --use 0 1 2 3 4 5 \
      --policies walkover slow approach --profile realistic --out outputs/missions

Writes per mission <out>/<policy>_tile<i>.json (trajectory, events, memory, score) and a comparison table
<out>/comparison.md: per strategy, mission time, defects found / measured with the D405, median measurement error
per type, ADA class accuracy for upheavals, false findings. The mock backend is kinematic (not Go2 physics);
the D555 / D405 are ray-cast with the chosen error profile.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from go2wm.common.config import load_config, resolve_path  # noqa: E402
from go2wm.mission.runner import MockMissionSim, Policy, run_mission, score_mission  # noqa: E402
from go2wm.models.state_dynamics import KinematicBaseline  # noqa: E402
from go2wm.planning.local_mpc import ModelPredictor, MPCController  # noqa: E402
from go2wm.planning.safety import SafetyFilter  # noqa: E402


def one_mission(job):
    """Run (and save) one mission; returns its result. Top-level so worker processes can run it."""
    pol, ti, tile, a = job
    pc = load_config(a["planner"])
    sim = MockMissionSim(load_config("configs/mock.yaml"), tile)
    safety = SafetyFilter({**pc["safety"], "limits": pc["limits"], "robot_radius": pc["robot_radius"]})
    ctrl = MPCController(ModelPredictor(KinematicBaseline(1, "command")), pc, safety, a["seed"], "kinematic_mpc")
    res = run_mission(sim, ctrl, tile, Policy(pol), a["profile"], seed=a["seed"] + ti, log=lambda *x: None,
                      pose_errors=a.get("pose_errors", False))
    res["score"] = score_mission(res, tile)
    res["tile"] = ti
    (Path(a["out"]) / f"{pol}_tile{ti}.json").write_text(json.dumps(res, default=float))
    sc = res["score"]
    print(f"[{pol}] tile {ti}: {res['outcome']} in {res['time_s']:.1f} s, findings {sc['n_findings']} "
          f"(false {sc['false_findings']}), " + ", ".join(f"{k} {v['found']}/{v['n']} found, {v['measured_d405']} by D405"
                                                           for k, v in sc["by_type"].items()), flush=True)
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiles", default="outputs/defect_tiles/tiles.json")
    ap.add_argument("--use", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5])
    ap.add_argument("--policies", nargs="+", default=["walkover", "slow", "approach"])
    ap.add_argument("--profile", default="realistic")
    ap.add_argument("--planner", default="configs/planner.yaml")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="outputs/missions")
    ap.add_argument("--pose-errors", action="store_true", help="odometry drift + camera latency in perception")
    ap.add_argument("--overwrite", action="store_true", help="rerun missions that already have a result file")
    ap.add_argument("--workers", type=int, default=1, help="missions run in parallel processes")
    a = ap.parse_args()
    tiles = json.loads(resolve_path(a.tiles).read_text())
    out = resolve_path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = {"planner": a.planner, "seed": a.seed, "profile": a.profile, "out": str(out), "pose_errors": a.pose_errors}
    done, jobs = {}, []
    for pol in a.policies:
        for ti in a.use:
            f = out / f"{pol}_tile{ti}.json"
            if f.exists() and not a.overwrite:  # resume: keep finished missions
                done[(pol, ti)] = json.loads(f.read_text())
            else:
                jobs.append((pol, ti, tiles[ti], cfg))
    if a.workers > 1 and len(jobs) > 1:
        import os
        from concurrent.futures import ProcessPoolExecutor

        os.environ.setdefault("OMP_NUM_THREADS", "1")  # one core per mission
        os.environ.setdefault("MKL_NUM_THREADS", "1")
        with ProcessPoolExecutor(min(a.workers, len(jobs))) as ex:
            for job, res in zip(jobs, ex.map(one_mission, jobs)):
                done[(job[0], job[1])] = res
    else:
        for job in jobs:
            done[(job[0], job[1])] = one_mission(job)
    summary = [done[(pol, ti)] for pol in a.policies for ti in a.use]
    write_table(summary, out / "comparison.md", a.profile)
    print((out / "comparison.md").read_text(encoding="utf-8"))
    return 0


def write_table(runs, path, profile):
    L = [f"# Inspection strategies on defect tiles (mock kinematics, sensors: D555 + D405, profile {profile})", "",
         "| strategy | missions | goal | mean time [s] | upheaval found / D405 / err [mm] / ADA ok | hole found / D405 / depth err [mm] | "
         "puddle found / D405 / depth err [mm] | false findings |", "|" + "---|" * 8]
    for pol in dict.fromkeys(r["policy"] for r in runs):
        rr = [r for r in runs if r["policy"] == pol]
        cells = []
        for typ in ("upheaval", "hole", "puddle"):
            d = [x for r in rr for x in r["score"]["per_defect"] if x["type"] == typ]
            if not d:
                cells.append("–")
                continue
            errs = [x["abs_err_m"] * 1000 for x in d if "abs_err_m" in x]
            ada = [x["ada_ok"] for x in d if "ada_ok" in x]
            s = f"{sum(x['found'] for x in d)}/{len(d)} / {sum(x.get('best_sensor') == 'depth_d405' for x in d)} / " + \
                (f"{np.median(errs):.1f}" if errs else "–")
            if typ == "upheaval":
                s += f" / {np.mean(ada) * 100:.0f} %" if ada else " / –"
            cells.append(s)
        L.append(f"| {pol} | {len(rr)} | {sum(r['outcome'] == 'goal' for r in rr)} | {np.mean([r['time_s'] for r in rr]):.1f} | "
                 + " | ".join(cells) + f" | {sum(r['score']['false_findings'] for r in rr)} |")
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
