"""Detect prerequisites without launching the simulator.

  python scripts/verify_environment.py
Reports: Python/torch/CUDA, disk + memory headroom, Isaac Sim/Lab installs, Go2 task
registration (source scan), published Go2 checkpoint reachability, V-JEPA 2 availability,
and the data root. Exit code 0 even when optional pieces are missing (mock mode still works).
"""

from __future__ import annotations

import importlib.util
import os
import platform
import shutil
import sys
import urllib.request
from pathlib import Path

from go2wm.common.config import data_root

ISAACSIM_ROOT = Path(os.environ.get("ISAACSIM_ROOT", r"C:\isaacsim"))
ISAACLAB_ROOT = Path(os.environ.get("ISAACLAB_ROOT", r"C:\IsaacLab"))
S3 = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/IsaacLab"
GO2_TASKS = ["Isaac-Velocity-Flat-Unitree-Go2-v0", "Isaac-Velocity-Rough-Unitree-Go2-v0"]


def row(ok, name, detail=""):
    tag = {True: " OK ", False: "MISS", None: "WARN"}[ok]
    print(f"[{tag}] {name:34s} {detail}")


def head(url: str) -> bool:
    try:
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status == 200
    except Exception:  # noqa: BLE001
        return False


def main() -> int:
    print(f"go2wm environment check ({platform.platform()})")
    row(True, "python", sys.version.split()[0] + f" ({sys.executable})")
    try:
        import torch

        cuda = torch.cuda.is_available()
        dev = torch.cuda.get_device_name(0) if cuda else "-"
        vram = f"{torch.cuda.get_device_properties(0).total_memory / 2**30:.0f} GB" if cuda else ""
        row(cuda or None, "torch / CUDA", f"{torch.__version__}, cuda={cuda} {dev} {vram}")
    except ImportError:
        row(False, "torch", "not installed")

    for drive in ("C:\\", "D:\\") if os.name == "nt" else ("/",):
        if os.path.exists(drive):
            free = shutil.disk_usage(drive).free / 2**30
            row(free > 20 or None, f"disk free {drive}", f"{free:.0f} GB" + ("" if free > 20 else "  (Isaac caches need ~20 GB)"))
    if os.name == "nt":
        try:
            import ctypes

            class MS(ctypes.Structure):
                _fields_ = [(n, ctypes.c_ulonglong if n != "dwLength" and n != "dwMemoryLoad" else ctypes.c_ulong)
                            for n in ("dwLength", "dwMemoryLoad", "ullTotalPhys", "ullAvailPhys", "ullTotalPageFile",
                                      "ullAvailPageFile", "ullTotalVirtual", "ullAvailVirtual", "ullAvailExtendedVirtual")]

            m = MS()
            m.dwLength = ctypes.sizeof(MS)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
            commit = m.ullAvailPageFile / 2**30
            row(commit > 16 or None, "free RAM / commit", f"{m.ullAvailPhys / 2**30:.1f} GB / {commit:.1f} GB"
                + ("" if commit > 16 else "  (Isaac Sim needs ~16 GB commit headroom; close apps)"))
        except Exception:  # noqa: BLE001
            pass

    ver = ISAACSIM_ROOT / "VERSION"
    row(ver.exists(), "Isaac Sim", f"{ISAACSIM_ROOT} {ver.read_text().strip() if ver.exists() else ''}")
    lab_ver = ISAACLAB_ROOT / "VERSION"
    row(lab_ver.exists(), "Isaac Lab", f"{ISAACLAB_ROOT} {lab_ver.read_text().strip() if lab_ver.exists() else ''}")
    for mod in ("isaaclab", "isaaclab_tasks", "isaaclab_rl", "rsl_rl"):
        row(importlib.util.find_spec(mod) is not None or None, f"python module {mod}", "(Isaac runs need scripts/isaac_python.bat)")
    go2_init = ISAACLAB_ROOT / "source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/go2/__init__.py"
    src = go2_init.read_text() if go2_init.exists() else ""
    for t in GO2_TASKS:
        row(t in src, f"task {t}", "registered" if t in src else "not found in Isaac Lab source")
    for t in GO2_TASKS:
        ok = head(f"{S3}/PretrainedCheckpoints/rsl_rl/{t}/checkpoint.pt")
        row(ok or None, "published policy", f"{t}: {'reachable' if ok else 'unreachable (offline?)'}")
    local_ckpt = ISAACLAB_ROOT / ".pretrained_checkpoints"
    row(local_ckpt.exists() or None, "cached pretrained checkpoints", str(local_ckpt))

    hf = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")) / "hub"
    vj = sorted(p.name for p in hf.glob("models--facebook--vjepa2*")) if hf.exists() else []
    row(bool(vj) or None, "V-JEPA 2 weights", ", ".join(vj) if vj else "not downloaded (optional; Phase E)")
    row(importlib.util.find_spec("transformers") is not None or None, "transformers", "(V-JEPA 2 loader)")

    root = data_root()
    row(True, "data root", f"{root}  (configs/paths.yaml or GO2WM_ROOT)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
