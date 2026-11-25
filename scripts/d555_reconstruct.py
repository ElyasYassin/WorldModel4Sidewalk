"""Reconstruct a D555 sidewalk recording (scripts/d555_record.py) into a levelled point cloud / mesh for comparison with
the Trimble X7 scan. RGB-D odometry (Open3D, depth + color, raw frame vs keyframe) + TSDF voxel-block fusion, levelled with the
IMU gravity from the standing-still start.

  pip install pyrealsense2 open3d numpy
  python scripts/d555_reconstruct.py recordings/pass1_20261126_101500.db3 --voxel 0.01 [--device CUDA:0]

Writes next to the bag, in <stem>_recon/: cloud.ply (levelled: z up, origin = first camera position on the ground
projection, x = initial walking direction), mesh.ply, poses.npy (N x 4 x 4, camera -> levelled world), poses_tum.txt,
recon.json (settings, frames used / tracking failures, gravity, trajectory length).

Accuracy caveats: frame-to-model visual odometry drifts (typically ~1 % of the distance walked); compare local
geometry (step heights, slab tilt, crack widths) after a rigid ICP alignment per few-metre patch, not absolute
positions over the whole walk. Concrete with little texture plus sun can make tracking fail: the script reports it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

try:
    import open3d as o3d
    import pyrealsense2 as rs
except ImportError as e:  # pragma: no cover
    raise SystemExit(f"{e.name} is missing: pip install open3d pyrealsense2")


def read_bag(path, stride, max_frames, still_s, info):
    """Generator of aligned (depth uint16, color uint8 RGB, t_s) from a .bag. Fills `info` with the intrinsics (K),
    depth scale and the accelerometer samples of the first still_s seconds (camera frame)."""
    pipe, cfg = rs.pipeline(), rs.config()
    cfg.enable_device_from_file(str(path), repeat_playback=False)
    prof = pipe.start(cfg)
    prof.get_device().as_playback().set_real_time(False)
    has_color = rs.stream.color in {sp.stream_type() for sp in prof.get_streams()}
    align = rs.align(rs.stream.color) if has_color else None
    info.update(depth_scale=prof.get_device().first_depth_sensor().get_depth_scale(), accel=[], has_color=has_color)
    i, n, t0 = 0, 0, None
    try:
        while True:
            try:
                fs = pipe.wait_for_frames(5000)
            except RuntimeError:
                return  # end of file
            t = fs.get_timestamp() / 1000.0
            t0 = t if t0 is None else t0
            for f in fs:
                if f.is_motion_frame() and f.get_profile().stream_type() == rs.stream.accel and t - t0 < still_s:
                    m = f.as_motion_frame().get_motion_data()
                    info["accel"].append([m.x, m.y, m.z])
            if align is not None:
                fs = align.process(fs)
            d = fs.get_depth_frame()
            c = fs.get_color_frame() if has_color else None
            if not d or (has_color and not c):
                continue
            i += 1
            if (i - 1) % stride:
                continue
            ii = (c if c else d).get_profile().as_video_stream_profile().get_intrinsics()
            info["K"] = np.array([[ii.fx, 0, ii.ppx], [0, ii.fy, ii.ppy], [0, 0, 1]], dtype=np.float64)
            dep = np.asanyarray(d.get_data()).copy()
            col = np.asanyarray(c.get_data()).copy() if c else np.repeat(
                (np.clip(dep / max(int(dep.max()), 1), 0, 1) * 255).astype(np.uint8)[..., None], 3, axis=2)
            yield dep, col, t - t0
            n += 1
            if max_frames and n >= max_frames:
                return
    finally:
        pipe.stop()


def ground_up(dep, K, depth_scale):
    """Up vector in the camera frame from a RANSAC ground plane of one depth frame (no IMU)."""
    z = dep.astype(np.float64) * depth_scale
    v, u = np.nonzero((z > 0.3) & (z < 3.0))
    sel = np.random.default_rng(0).choice(len(z[v, u]), min(20000, len(v)), replace=False)
    v, u, zz = v[sel], u[sel], z[v[sel], u[sel]]
    pts = np.stack([(u - K[0, 2]) * zz / K[0, 0], (v - K[1, 2]) * zz / K[1, 1], zz], 1)
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    (a, b, c, _), _ = pc.segment_plane(0.01, 3, 500)
    n = np.array([a, b, c]) / np.linalg.norm([a, b, c])
    return n if n[1] < 0 else -n  # camera y points down: up has negative y


def level_rotation(accel, first_pose, poses, plane_up=None):
    """Rotation world_slam -> levelled (z up, x along the initial walking direction). accel: camera-frame
    accelerometer at rest (RealSense: reads the reaction to gravity, i.e. points up)."""
    if len(accel):
        up_cam = np.mean(accel, 0)
        up_cam /= np.linalg.norm(up_cam)
    elif plane_up is not None:  # no IMU: ground plane of the first frame
        up_cam = plane_up
    else:
        up_cam = np.array([0.0, -1.0, 0.0])
    up = first_pose[:3, :3] @ up_cam  # SLAM world frame = first camera frame
    travel = poses[-1][:3, 3] - poses[0][:3, 3]
    fwd = travel - up * (travel @ up)
    if np.linalg.norm(fwd) < 0.2:  # barely moved: use the camera's viewing direction
        fwd = first_pose[:3, 2] - up * (first_pose[:3, 2] @ up)
    fwd /= np.linalg.norm(fwd)
    left = np.cross(up, fwd)
    return np.stack([fwd, left, up]), up_cam


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bag")
    ap.add_argument("--out", default=None)
    ap.add_argument("--voxel", type=float, default=0.01, help="TSDF voxel size [m] (0.005 for close-range detail)")
    ap.add_argument("--depth-min", type=float, default=0.5)
    ap.add_argument("--depth-max", type=float, default=3.0, help="farther depth is noisier (error ~ z^2)")
    ap.add_argument("--stride", type=int, default=1, help="use every k-th frame")
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--still", type=float, default=2.0, help="seconds of standing still at the start (gravity)")
    ap.add_argument("--blocks", type=int, default=60000, help="voxel blocks to allocate (raise for long walks)")
    ap.add_argument("--track", choices=["hybrid", "depth"], default="hybrid", help="hybrid = depth + color (needed on "
                    "a flat sidewalk: depth-only ICP slides along a plane)")
    ap.add_argument("--odom", choices=["keyframe", "model"], default="keyframe", help="keyframe = raw RGB-D frame vs "
                    "the last keyframe; model = frame-to-model (biased on a grazing-angle ground plane in tests)")
    ap.add_argument("--kf-dist", type=float, default=0.3, help="new keyframe after this much motion [m]")
    ap.add_argument("--depth-diff", type=float, default=0.07, help="max depth residual for tracking correspondences [m]")
    ap.add_argument("--device", default="CPU:0", help="CPU:0 or CUDA:0 (needs a CUDA build of open3d)")
    a = ap.parse_args(argv)

    bag = Path(a.bag)
    out = Path(a.out) if a.out else bag.with_name(bag.stem + "_recon")
    out.mkdir(parents=True, exist_ok=True)
    print(f"[recon] reading {bag} ...")
    info, poses, times, fails = {}, [], [], 0
    model = inp = ray = T = dev = None
    for i, (dep, col, t) in enumerate(read_bag(bag, a.stride, a.max_frames, a.still, info)):
        if model is None:  # first frame: intrinsics known now
            dev = o3d.core.Device(a.device)
            K = o3d.core.Tensor(info["K"])
            scale = 1.0 / info["depth_scale"]
            H, W = dep.shape
            T = o3d.core.Tensor(np.eye(4))
            model = o3d.t.pipelines.slam.Model(a.voxel, 16, a.blocks, T, dev)
            inp = o3d.t.pipelines.slam.Frame(H, W, K, dev)
            ray = o3d.t.pipelines.slam.Frame(H, W, K, dev)
            use_color = info["has_color"] and a.track != "depth"
            method = (o3d.t.pipelines.odometry.Method.Hybrid if use_color
                      else o3d.t.pipelines.odometry.Method.PointToPlane)
            print(f"[recon] {W}x{H}, color {info['has_color']}, depth scale {info['depth_scale']}, "
                  f"tracking {'depth + color' if use_color else 'depth only'}")
            first = (dep, info["K"], info["depth_scale"])
        times.append(t)
        inp.set_data_from_image("depth", o3d.t.geometry.Image(o3d.core.Tensor(dep)).to(dev))
        inp.set_data_from_image("color", o3d.t.geometry.Image(o3d.core.Tensor(col)).to(dev))
        rgbd = o3d.t.geometry.RGBDImage(o3d.t.geometry.Image(o3d.core.Tensor(col)).to(dev),
                                        o3d.t.geometry.Image(o3d.core.Tensor(dep)).to(dev)) if a.odom == "keyframe" else None
        if i == 0 and a.odom == "keyframe":
            kf, T_kf, T_rel = rgbd, T, o3d.core.Tensor(np.eye(4))
        if i > 0:
            try:
                if a.odom == "keyframe":  # raw frame vs raw keyframe: no bias from the fused model's raycast
                    res = o3d.t.pipelines.odometry.rgbd_odometry_multi_scale(
                        rgbd, kf, K, T_rel, scale, a.depth_max, method=method)
                    if res.fitness < 0.1:
                        raise RuntimeError(f"low fitness {res.fitness:.2f}")
                    T_rel = res.transformation  # current -> keyframe
                    T = T_kf @ T_rel
                    if np.linalg.norm(T_rel.cpu().numpy()[:3, 3]) > a.kf_dist:
                        kf, T_kf, T_rel = rgbd, T, o3d.core.Tensor(np.eye(4))
                else:
                    res = model.track_frame_to_model(inp, ray, scale, a.depth_max, a.depth_diff, method)
                    if res.fitness < 0.1:
                        raise RuntimeError(f"low fitness {res.fitness:.2f}")
                    T = T @ res.transformation
            except RuntimeError as e:  # keep the previous pose, do not fuse this frame
                fails += 1
                if fails <= 10 or fails % 50 == 0:
                    print(f"[recon] frame {i}: tracking failed ({e})")
                poses.append(T.cpu().numpy())
                continue
        poses.append(T.cpu().numpy())
        model.update_frame_pose(i, T)
        model.integrate(inp, scale, a.depth_max)
        if a.odom == "model":
            model.synthesize_model_frame(ray, scale, a.depth_min, a.depth_max, 8.0, use_color)
        if i % 100 == 0:
            print(f"[recon] frame {i}  t {t:.1f} s  pos {poses[-1][:3, 3].round(2)}  failures {fails}")
    if len(poses) < 2:
        raise SystemExit("fewer than 2 frames in the recording")
    print(f"[recon] {len(poses)} frames, {times[-1]:.1f} s, {len(info['accel'])} accel samples at rest")

    poses = np.array(poses)
    plane_up = None if info["accel"] else ground_up(*first)
    if plane_up is not None:
        print("[recon] no IMU samples: levelling with the ground plane of the first frame")
    R, up_cam = level_rotation(info["accel"], poses[0], poses, plane_up)
    Lw = np.eye(4)
    Lw[:3, :3] = R
    pcd = model.voxel_grid.extract_point_cloud().to_legacy()
    pcd.transform(Lw)
    P = np.asarray(pcd.points)
    near = P[(P[:, 0] > 0.5) & (P[:, 0] < 2.0) & (np.abs(P[:, 1]) < 0.4)] if len(P) else P
    z_ground = float(np.median(near[:, 2])) if len(near) else (float(np.percentile(P[:, 2], 5)) if len(P) else 0.0)
    shift = np.eye(4)
    shift[:3, 3] = -(R @ poses[0][:3, 3])
    shift[2, 3] = -z_ground  # ground ~ z = 0
    pcd.transform(shift)
    W2L = shift @ Lw
    poses_l = np.einsum("ij,njk->nik", W2L, poses)
    o3d.io.write_point_cloud(str(out / "cloud.ply"), pcd)
    try:
        mesh = model.voxel_grid.extract_triangle_mesh().to_legacy()
        mesh.transform(W2L)
        o3d.io.write_triangle_mesh(str(out / "mesh.ply"), mesh)
    except RuntimeError as e:
        print(f"[recon] mesh extraction failed: {e}")
    np.save(out / "poses.npy", poses_l)
    try:  # TUM trajectory (t x y z qx qy qz qw) for evo etc.; needs scipy
        from scipy.spatial.transform import Rotation

        with open(out / "poses_tum.txt", "w") as f:
            for t, Pc in zip(times, poses_l):
                q = Rotation.from_matrix(Pc[:3, :3]).as_quat()
                f.write(f"{t:.4f} " + " ".join(f"{v:.5f}" for v in (*Pc[:3, 3], *q)) + "\n")
    except ImportError:
        print("[recon] scipy missing: skipped poses_tum.txt")
    length = float(np.linalg.norm(np.diff(poses_l[:, :3, 3], axis=0), axis=1).sum())
    rep = {"bag": str(bag), "frames": len(poses), "tracking_failures": fails, "voxel_m": a.voxel,
           "depth_range_m": [a.depth_min, a.depth_max], "stride": a.stride, "trajectory_m": round(length, 2),
           "levelling": "imu" if info["accel"] else "ground plane", "up_in_camera": up_cam.round(4).tolist(),
           "camera_height_m_start": round(float(poses_l[0][2, 3]), 3), "points": len(pcd.points)}
    (out / "recon.json").write_text(json.dumps(rep, indent=2))
    print(f"[recon] done: {length:.1f} m walked, {fails} tracking failures, {len(pcd.points)} points -> {out}")
    if fails > 0.05 * len(poses):
        print("[recon] WARNING: >5 % tracking failures: walk slower, keep more texture in view, or try --stride 1")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
