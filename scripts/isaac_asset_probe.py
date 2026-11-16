"""Probe Isaac Sim asset access (Go2 robot, sky HDRIs, people, props) for the RTX mission replay."""

from isaacsim import SimulationApp

app = SimulationApp({"headless": True})
import omni.client  # noqa: E402
import omni.usd  # noqa: E402
from isaacsim.storage.native import get_assets_root_path  # noqa: E402
from pxr import Usd, UsdGeom  # noqa: E402

root = get_assets_root_path()
print("[probe] assets root:", root, flush=True)


def ls(path, depth=0, maxd=1):
    res, entries = omni.client.list(path)
    if res != omni.client.Result.OK:
        print(f"[probe] {path}: {res}", flush=True)
        return
    for e in entries[:60]:
        print("[probe] " + "  " * depth + f"{path.rstrip('/')}/{e.relative_path}", flush=True)
        if depth + 1 < maxd and not e.relative_path.endswith((".usd", ".usda", ".hdr", ".exr", ".png", ".jpg")):
            ls(f"{path.rstrip('/')}/{e.relative_path}", depth + 1, maxd)


for sub in ("Isaac/Robots/Unitree", "Isaac/People/Characters", "NVIDIA/Assets/Skies", "Isaac/Environments",
            "Isaac/Props", "NVIDIA/Assets/ArchVis/Commercial", "NVIDIA/Materials/Base"):
    ls(f"{root}/{sub}", 0, 2)

go2 = f"{root}/Isaac/Robots/Unitree/Go2/go2.usd"
stage = omni.usd.get_context().get_stage()
p = stage.DefinePrim("/World/Go2", "Xform")
p.GetReferences().AddReference(go2)
for _ in range(5):
    app.update()
n = 0
for q in Usd.PrimRange(p):
    if q.GetPath().pathElementCount <= 4:
        print("[probe] go2 prim", q.GetPath(), q.GetTypeName(), flush=True)
        n += 1
print(f"[probe] go2 prims listed {n}", flush=True)
app.close()
