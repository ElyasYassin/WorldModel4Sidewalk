"""Probe Isaac people animations, vegetation and street props."""

from isaacsim import SimulationApp

app = SimulationApp({"headless": True})
import omni.client  # noqa: E402
from isaacsim.storage.native import get_assets_root_path  # noqa: E402

root = get_assets_root_path()


def ls(path, depth=0, maxd=2):
    res, entries = omni.client.list(path)
    if res != omni.client.Result.OK:
        print(f"[probe] MISSING {path}", flush=True)
        return
    for e in entries[:80]:
        name = e.relative_path
        if name.startswith("."):
            continue
        print("[probe] " + "  " * depth + f"{path.replace(root + '/', '')}/{name}", flush=True)
        if depth + 1 < maxd and "." not in name[-6:]:
            ls(f"{path}/{name}", depth + 1, maxd)


for sub in ("Isaac/People", "Isaac/People/Animations", "NVIDIA/Assets/Vegetation", "NVIDIA/Assets/DigitalTwin/Assets",
            "NVIDIA/Assets/ArchVis", "Isaac/Environments/Outdoor/Rivermark", "NVIDIA/Assets/Vehicles",
            "NVIDIA/Assets/Isaac/6.0/Isaac/Props/Street"):
    ls(f"{root}/{sub}", 0, 2)
app.close()
