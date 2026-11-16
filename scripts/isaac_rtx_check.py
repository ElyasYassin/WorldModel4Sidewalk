"""Minimal Isaac Sim RTX rendering check: a lit cube on a ground plane, one camera frame saved to PNG.

  scripts\\isaac_python.bat scripts\\isaac_rtx_check.py --out outputs/rtx_check.png
  (or the standalone Isaac Sim's python.bat for another version)

Tells whether the RTX renderer starts on this machine / driver at all, independent of Isaac Lab.
"""

import argparse
import sys

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="D:/go2wm/outputs/rtx_check.png")
a = ap.parse_args()

from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": True, "renderer": "RayTracedLighting", "width": 640, "height": 360})
print("[rtx] app started", flush=True)
import numpy as np  # noqa: E402
import omni.replicator.core as rep  # noqa: E402
from PIL import Image  # noqa: E402

with rep.new_layer():
    rep.create.plane(scale=10, position=(0, 0, 0))
    rep.create.cube(position=(0, 0, 0.5), scale=1.0)
    rep.create.light(light_type="distant", intensity=3000, rotation=(-45, 0, 0))
    cam = rep.create.camera(position=(4, 4, 3), look_at=(0, 0, 0.5))
    rp = rep.create.render_product(cam, (640, 360))
    ann = rep.AnnotatorRegistry.get_annotator("rgb")
    ann.attach([rp])
    for _ in range(10):
        rep.orchestrator.step()
    img = np.asarray(ann.get_data())
print(f"[rtx] frame {img.shape} mean {img[..., :3].mean():.1f}", flush=True)
Image.fromarray(img[..., :3].astype(np.uint8)).save(a.out)
print(f"[rtx] saved {a.out}", flush=True)
app.close()
sys.exit(0)
