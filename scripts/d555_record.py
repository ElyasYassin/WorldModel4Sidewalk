"""Record a sidewalk walk with a RealSense D555 on a laptop (depth + color + IMU) for offline reconstruction and comparison
with the Trimble X7 scan. Standalone: needs only pyrealsense2 + numpy (+ opencv-python for the live preview).

  pip install pyrealsense2 numpy opencv-python
  python d555_record.py --out recordings --name pass1 --height 0.40 --pitch 30 --notes "east -> west, cloudy"

Writes <out>/<name>_<timestamp>.db3 (.bag with older SDKs; lossless RealSense recording: every depth / color / IMU frame with intrinsics and
extrinsics) and a .json sidecar (device, firmware, streams, intrinsics, depth scale, mount, notes, markers, frame
counts, dropped frames). Reconstruct later with scripts/d555_reconstruct.py.

Preview keys (window focused): q / Esc = stop, space = drop a marker (e.g. "passing the fault now"; logged with its
time in the sidecar). Without a display use --no-preview and stop with Ctrl+C or --duration.

The D555 is a PoE camera (Ethernet + PoE injector) that talks DDS: use a recent pyrealsense2 (2.57 or newer) and check
first that the RealSense Viewer sees it. The script uses the first device found (or --serial).

Field protocol (see the --help epilog): hold / mount it at the robot's camera height and pitch, walk slowly
(~0.3-0.5 m/s, no fast turns), start and end on a few seconds standing still, cover the scanned stretch incl. the
faulted slab, and do 2-3 passes. Data rate is roughly 40-110 MB/s depending on streams: check free disk space.
"""

from __future__ import annotations

import argparse
import json
import shutil
import signal
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

try:
    import pyrealsense2 as rs
except ImportError:  # pragma: no cover
    sys.exit("pyrealsense2 is missing: pip install pyrealsense2 (2.57+ for the D555)")

EPILOG = """field protocol:
  1. RealSense Viewer first: camera found, depth looks clean on concrete, firmware noted.
  2. Mount: same height / pitch as planned on the Go2-W (default 0.40 m, 30 deg down); a pole or a board on a
     trolley works. Measure and pass --height / --pitch.
  3. Stand still 3 s at the start (IMU gravity estimate for levelling), walk ~0.3-0.5 m/s along the sidewalk,
     stand still 3 s at the end.
  4. Press space when passing the faulted slab (and other landmarks) so we can find them in the recording.
  5. 2-3 passes (both directions); one pass along each edge if the sidewalk is wider than the view.
  6. Optional: a few flat targets (A4 checkerboards / cones) left in place during both the X7 scan and the walk make
     the alignment and the accuracy check much easier.
"""


def pick_device(ctx, serial):
    devs = list(ctx.query_devices())
    if not devs:
        sys.exit("no RealSense device found (PoE link up? same subnet? does the RealSense Viewer see it?)")
    for d in devs:
        info = {k: (d.get_info(getattr(rs.camera_info, k)) if d.supports(getattr(rs.camera_info, k)) else None)
                for k in ("name", "serial_number", "firmware_version", "product_line")}
        print(f"[d555] found {info['name']}  serial {info['serial_number']}  fw {info['firmware_version']}")
        if serial is None or info["serial_number"] == serial:
            return d, info
    sys.exit(f"device {serial} not found")


def try_start(serial, path, depth_wh, color_wh, fps, imu, color):
    """Start the pipeline with the requested streams, falling back to device defaults if a mode is unsupported."""
    attempts = []
    for exact in (True, False):
        for with_imu in ((True, False) if imu else (False,)):
            attempts.append((exact, with_imu))
    last = None
    for exact, with_imu in attempts:
        cfg = rs.config()
        cfg.enable_device(serial)
        if exact:
            cfg.enable_stream(rs.stream.depth, depth_wh[0], depth_wh[1], rs.format.z16, fps)
            if color:
                cfg.enable_stream(rs.stream.color, color_wh[0], color_wh[1], rs.format.rgb8, fps)
        else:
            cfg.enable_stream(rs.stream.depth, rs.format.z16, fps)
            if color:
                cfg.enable_stream(rs.stream.color, rs.format.rgb8, fps)
        if with_imu:
            cfg.enable_stream(rs.stream.accel)
            cfg.enable_stream(rs.stream.gyro)
        cfg.enable_record_to_file(str(path))
        pipe = rs.pipeline()
        try:
            prof = pipe.start(cfg)
            if not exact:
                print("[d555] requested modes not supported -> using device default resolutions")
            if imu and not with_imu:
                print("[d555] IMU streams unavailable -> recording without IMU")
            return pipe, prof, with_imu
        except RuntimeError as e:
            last = e
    raise last


def set_options(dev, preset, exposure, emitter):
    out = {}
    for s in dev.query_sensors():
        if not s.is_depth_sensor():
            continue
        if preset and s.supports(rs.option.visual_preset):
            try:
                rng = s.get_option_range(rs.option.visual_preset)
                names = {s.get_option_value_description(rs.option.visual_preset, v).lower(): v
                         for v in np.arange(rng.min, rng.max + 1)}
                hit = next((v for n, v in names.items() if preset.lower() in n), None)
                if hit is not None:
                    s.set_option(rs.option.visual_preset, float(hit))
                    out["visual_preset"] = s.get_option_value_description(rs.option.visual_preset, hit)
                else:
                    print(f"[d555] preset {preset!r} not offered; available: {sorted(names)}")
            except RuntimeError as e:
                print(f"[d555] could not set preset: {e}")
        if s.supports(rs.option.emitter_enabled):
            s.set_option(rs.option.emitter_enabled, 1.0 if emitter else 0.0)
            out["emitter_enabled"] = emitter
        if exposure and s.supports(rs.option.exposure):
            s.set_option(rs.option.enable_auto_exposure, 0.0)
            s.set_option(rs.option.exposure, float(exposure))
            out["exposure_us"] = exposure
        out["depth_scale_m"] = s.get_depth_scale()
    return out


def intrinsics(prof):
    out = {}
    for sp in prof.get_streams():
        name = sp.stream_name().lower()
        try:
            vp = sp.as_video_stream_profile()
            i = vp.get_intrinsics()
            out[name] = {"width": i.width, "height": i.height, "fx": i.fx, "fy": i.fy, "ppx": i.ppx, "ppy": i.ppy,
                         "model": str(i.model), "coeffs": list(i.coeffs), "fps": vp.fps()}
        except RuntimeError:
            out[name] = {"fps": sp.fps(), "format": str(sp.format())}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, epilog=EPILOG, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="recordings")
    ap.add_argument("--name", default="sidewalk")
    ap.add_argument("--serial", default=None)
    ap.add_argument("--depth", type=int, nargs=2, default=[848, 480], metavar=("W", "H"))
    ap.add_argument("--color", type=int, nargs=2, default=[848, 480], metavar=("W", "H"))
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--no-color", action="store_true", help="depth (+IMU) only: ~3x smaller files")
    ap.add_argument("--no-imu", action="store_true")
    ap.add_argument("--preset", default="high accuracy", help="depth visual preset (substring match), '' = keep")
    ap.add_argument("--exposure", type=float, default=None, help="fixed depth exposure [us]; default auto")
    ap.add_argument("--no-emitter", action="store_true", help="projector off (passive stereo, e.g. in direct sun)")
    ap.add_argument("--duration", type=float, default=None, help="stop after this many seconds")
    ap.add_argument("--height", type=float, default=None, help="camera height above the ground [m] (measured)")
    ap.add_argument("--pitch", type=float, default=None, help="camera pitch, nose down [deg] (measured)")
    ap.add_argument("--notes", default="")
    ap.add_argument("--no-preview", action="store_true")
    a = ap.parse_args(argv)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    free_gb = shutil.disk_usage(out).free / 1e9
    print(f"[d555] free disk at {out.resolve()}: {free_gb:.0f} GB")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    side = out / f"{a.name}_{stamp}.json"

    dev, info = pick_device(rs.context(), a.serial)
    opts = set_options(dev, a.preset, a.exposure, not a.no_emitter)
    for ext in (".db3", ".bag"):  # SDK 2.5x+ records ROS 2 bags (.db3) and refuses .bag; older SDKs only write .bag
        bag = out / f"{a.name}_{stamp}{ext}"
        try:
            pipe, prof, has_imu = try_start(info["serial_number"], bag, a.depth, a.color, a.fps, not a.no_imu,
                                            not a.no_color)
            break
        except RuntimeError as e:
            if "extension" not in str(e).lower() or ext == ".bag":
                sys.exit(f"could not start streaming: {e}")
    meta = {"device": info, "started": datetime.now().isoformat(timespec="seconds"), "bag": bag.name,
            "options": opts, "streams": intrinsics(prof), "imu": has_imu,
            "mount": {"height_m": a.height, "pitch_deg": a.pitch}, "notes": a.notes, "markers": []}
    side.write_text(json.dumps(meta, indent=2))
    print(f"[d555] recording -> {bag}")

    show = not a.no_preview
    if show:
        try:
            import cv2
        except ImportError:
            print("[d555] opencv not installed -> no preview (pip install opencv-python)")
            show = False
    stop = {"flag": False}
    signal.signal(signal.SIGINT, lambda *_: stop.__setitem__("flag", True))
    colorizer = rs.colorizer()
    t0 = time.time()
    n = {"depth": 0, "color": 0}
    last_fn, dropped, fills = None, 0, []
    try:
        while not stop["flag"]:
            if a.duration and time.time() - t0 > a.duration:
                break
            try:
                fs = pipe.wait_for_frames(2000)
            except RuntimeError:
                print("[d555] no frames for 2 s (link?)")
                continue
            d = fs.get_depth_frame()
            c = fs.get_color_frame() if not a.no_color else None
            if d:
                n["depth"] += 1
                fn = d.get_frame_number()
                if last_fn is not None and fn > last_fn + 1:
                    dropped += fn - last_fn - 1
                last_fn = fn
                z = np.asanyarray(d.get_data())
                fills.append(float((z > 0).mean()))
            if c:
                n["color"] += 1
            el = time.time() - t0
            if n["depth"] % a.fps == 0 and n["depth"]:
                mb = bag.stat().st_size / 1e6 if bag.exists() else 0.0
                print(f"\r[d555] {el:6.1f} s  depth {n['depth']}  color {n['color']}  dropped {dropped}  "
                      f"fill {np.mean(fills[-a.fps:]) * 100:4.1f} %  {mb:7.0f} MB", end="", flush=True)
            if show and d:
                img = np.asanyarray(colorizer.colorize(d).get_data())
                if c:
                    ci = cv2.cvtColor(np.asanyarray(c.get_data()), cv2.COLOR_RGB2BGR)
                    ci = cv2.resize(ci, (ci.shape[1] * img.shape[0] // ci.shape[0], img.shape[0]))
                    img = np.hstack([ci, img])
                cv2.putText(img, f"{el:5.1f}s fill {fills[-1] * 100:3.0f}% markers {len(meta['markers'])}  "
                            "[space] marker [q] stop", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.imshow("D555 recording", img)
                k = cv2.waitKey(1) & 0xFF
                if k in (ord("q"), 27):
                    break
                if k == ord(" "):
                    meta["markers"].append({"t_s": round(el, 3), "depth_frame": last_fn})
                    print(f"\n[d555] marker {len(meta['markers'])} at {el:.1f} s")
    finally:
        pipe.stop()
        if show:
            cv2.destroyAllWindows()
    el = time.time() - t0
    meta.update({"duration_s": round(el, 2), "frames": n, "dropped_depth_frames": dropped,
                 "mean_fill": round(float(np.mean(fills)), 4) if fills else None,
                 "bag_mb": round(bag.stat().st_size / 1e6, 1) if bag.exists() else None})
    side.write_text(json.dumps(meta, indent=2))
    print(f"\n[d555] done: {el:.1f} s, {n['depth']} depth frames ({dropped} dropped), "
          f"{meta['bag_mb']} MB -> {bag}\n[d555] sidecar -> {side}")
    if dropped > 0.02 * max(n["depth"], 1):
        print("[d555] WARNING: >2 % frames dropped (disk or network too slow?): try --no-color or --fps 15")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
