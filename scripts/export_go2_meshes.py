"""Export the Go2's visual meshes (per link, in link frame) from the Isaac stage, for the offline 3D renderer.

  scripts\\isaac_python.bat scripts\\export_go2_meshes.py --out outputs/go2_meshes.npz

Loads the same Isaac Go2 scene as isaac_mission.py (no cameras), walks the USD under the robot prim (including
instance proxies), and saves every visual mesh's vertices / triangles / display colour expressed in the frame of the
robot link it belongs to. scripts/render_isaac_mission.py --go2-meshes then poses them with the recorded link poses.
"""

from __future__ import annotations

import argparse
import copy
import sys

import numpy as np

from go2wm.common.config import load_config, resolve_path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/isaac_go2_rough.yaml")
    ap.add_argument("--out", default="outputs/go2_meshes.npz")
    ap.add_argument("--checkpoint", default=None, help="override policy_checkpoint (any compatible model)")
    a = ap.parse_args(argv)
    cfg = copy.deepcopy(load_config(a.config))
    cfg["isaac"].update(families=["sidewalk"], sensor_rig=[], rgb_camera=False, terrain_texture=None,
                        record_height_scan=False, enable_cameras=False)
    if a.checkpoint:
        cfg["isaac"]["policy_checkpoint"] = a.checkpoint
    from go2wm.sim.isaac_go2 import IsaacGo2Adapter

    sim = IsaacGo2Adapter(cfg)
    try:
        import omni.usd
        from pxr import Usd, UsdGeom

        stage = omni.usd.get_context().get_stage()
        names = set(sim.robot.body_names)
        root = stage.GetPrimAtPath("/World/envs/env_0/Robot")
        cache = UsdGeom.XformCache(Usd.TimeCode.Default())
        out = {}
        n = 0
        for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
            if not prim.IsA(UsdGeom.Mesh):
                continue
            path = str(prim.GetPath())
            if "collision" in path.lower():
                continue
            link = None
            q = prim
            while q and q.GetPath() != root.GetPath():
                if q.GetName() in names:
                    link = q
                    break
                q = q.GetParent()
            if link is None:
                continue
            mesh = UsdGeom.Mesh(prim)
            pts = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float64)
            counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get(), dtype=np.int64)
            idx = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int64)
            if len(pts) == 0 or len(counts) == 0:
                continue
            tris, o = [], 0
            for c in counts:  # fan-triangulate
                for j in range(1, c - 1):
                    tris.append((idx[o], idx[o + j], idx[o + j + 1]))
                o += c
            Mw = np.array(cache.GetLocalToWorldTransform(prim), dtype=np.float64)  # row-vector convention
            Ml = np.array(cache.GetLocalToWorldTransform(link), dtype=np.float64)
            M = Mw @ np.linalg.inv(Ml)  # mesh -> link
            ph = np.c_[pts, np.ones(len(pts))] @ M
            col = mesh.GetDisplayColorAttr().Get()
            col = np.asarray(col[0], dtype=np.float64) if col else np.array([0.15, 0.15, 0.15])
            key = f"m{n}"
            out[key + "_link"] = np.array(link.GetName())
            out[key + "_v"] = ph[:, :3].astype(np.float32)
            out[key + "_f"] = np.asarray(tris, dtype=np.int32)
            out[key + "_c"] = col.astype(np.float32)
            n += 1
            print(f"[export] {path} -> {link.GetName()} ({len(pts)} verts)", flush=True)
        f = resolve_path(a.out)
        f.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(f, n=n, **out)
        print(f"[export] {n} meshes -> {f}", flush=True)
    finally:
        sim.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
