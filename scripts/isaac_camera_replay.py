"""Isaac camera video of a recorded terrain episode: rebuild the same tile set, reset with the
episode's seed (same tile), and replay its logged command sequence open-loop with a follow camera.

  scripts\isaac_python.bat scripts\isaac_camera_replay.py --data data/go2_e4_test --split test \
      --episode isaac_sidewalk_80013_s080013 --out outputs/videos/08_e4_cam_stall_80013.mp4

The replay is not guaranteed to be bit-identical to the recording (PhysX on GPU is not fully
deterministic); the script reports the final-position difference between replay and log.
"""

from __future__ import annotations

import argparse
import copy
import sys

import numpy as np
import yaml

from go2wm.common.config import load_config, resolve_path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--episode", required=True)
    ap.add_argument("--split", choices=["train", "test"], required=True, help="tile set the episode was recorded on")
    ap.add_argument("--tiles", default="configs/e4_tiles.yaml")
    ap.add_argument("--config", default="configs/isaac_go2_rough.yaml")
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=int, default=10, help="10 = real time at the 0.1 s control step")
    args = ap.parse_args(argv)

    from go2wm.common.schema import VelocityCommand, state_pose2d
    from go2wm.data.episodes import list_episodes, load_episode

    root = resolve_path(args.data)
    path = next(p for p in list_episodes(root) if load_episode(p, load_arrays=False).meta["episode_id"] == args.episode)
    ep = load_episode(path)
    with open(args.tiles, encoding="utf-8") as f:
        tiles = yaml.safe_load(f)[args.split]
    cfg = copy.deepcopy(load_config(args.config))
    c = cfg["isaac"]
    c.update(families=["sidewalk"], scene_obstacles=False, scene_bounds=[-1.0, 14.0, -3.5, 3.5], terrain_tiles=tiles,
             enable_cameras=True, video={"resolution": [1280, 720], "eye": [-2.6, -1.1, 1.1], "lookat": [1.6, 0.0, 0.0]}, terrain_visual_color=[0.42, 0.41, 0.39])

    from go2wm.sim.isaac_go2 import IsaacGo2Adapter

    sim = IsaacGo2Adapter(cfg)
    seed = int(ep.meta["seed"])
    frames, poses = [], []
    try:
        obs = sim.reset(seed, ep.meta["scene_id"])
        for _ in range(3):  # let the renderer warm up
            frames.append(sim.render_rgb())
        frames = [frames[-1]]
        poses.append(state_pose2d(obs.state_vector()))
        for k, cmd in enumerate(ep["commands"][:, :3]):
            r = sim.step(VelocityCommand(float(cmd[0]), float(cmd[1]), float(cmd[2])))
            frames.append(sim.render_rgb())
            poses.append(state_pose2d(r.observation.state_vector()))
            if r.terminated:
                break
    finally:
        log_pose = state_pose2d(ep["states"])
        n = min(len(poses), len(log_pose))
        if n:
            d = np.linalg.norm(np.asarray(poses)[:n, :2] - log_pose[:n, :2], axis=1)
            print(f"[replay] {len(poses) - 1} steps (log {len(log_pose) - 1}); replay vs log position diff: "
                  f"mean {d.mean():.2f} m, final {d[-1]:.2f} m", flush=True)
        if frames:
            write_video(frames, ep, resolve_path(args.out), args.fps)
        sim.close()
    return 0


def write_video(frames, ep, out, fps):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter

    out.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames[0].shape[:2]
    fig = plt.figure(figsize=(w / 100, h / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    im = ax.imshow(frames[0][..., :3])
    txt = ax.text(0.01, 0.97, "", transform=ax.transAxes, va="top", color="white", fontsize=13, family="monospace",
                  bbox=dict(fc="black", alpha=0.5, ec="none"))
    m = ep.meta
    cmds = ep["commands"]
    wr = FFMpegWriter(fps=fps, codec="libx264", extra_args=["-pix_fmt", "yuv420p"])
    with wr.saving(fig, str(out), dpi=100):
        for k, f in enumerate(frames):
            im.set_data(f[..., :3])
            c = cmds[min(max(k - 1, 0), len(cmds) - 1)]
            txt.set_text(f"{m['episode_id']}  tile {m.get('e4_tile')}  (recorded outcome: {m.get('termination_reason')})\n"
                         f"t={k * 0.1:5.1f}s  cmd vx {c[0]:+.2f} vy {c[1]:+.2f} wz {c[2]:+.2f}   [replay of logged commands]")
            wr.grab_frame()
    plt.close(fig)
    print(f"[replay] wrote {out} ({len(frames)} frames)", flush=True)


if __name__ == "__main__":
    sys.exit(main())
