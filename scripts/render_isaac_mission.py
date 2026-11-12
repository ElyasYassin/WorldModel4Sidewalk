"""Phase 1: 3D chase-camera video of an Isaac-physics inspection mission (scripts/isaac_mission.py output).

  D:/go2wm/render_env/Scripts/python.exe scripts/render_isaac_mission.py \
      --mission outputs/isaac_missions/approach_tile8.json --out outputs/videos/14_isaac_inspection_tile8.mp4

Offline renderer (pyvista / VTK, no Isaac RTX needed): the tile's true surface with its concrete texture, puddles and
obstructions; the Go2 drawn from the recorded link poses of the physics run (trunk + legs + feet, so the gait is the
real one); the D555 and D405 measuring zones with their view rays; and every finding in memory as it appears, with a
label comparing the robot's MEASURED value against the TRUE defect size. A side panel keeps the running
measured-vs-true table; the last frames show the full scorecard, including missed defects.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import imageio.v2 as imageio  # noqa: E402
import pyvista as pv  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from go2wm.common.config import resolve_path  # noqa: E402
from go2wm.sim.sidewalk_texture import sidewalk_albedo  # noqa: E402
from go2wm.sim.terrain import params_with_defaults, sidewalk_height  # noqa: E402

KIND = {"depression": ("hole", (43, 108, 255)), "edge": ("upheaval / step", (255, 122, 26)),
        "dropout": ("possible water", (24, 195, 214))}
TYPE_NAME = {"hole": "hole", "upheaval": "upheaval", "puddle": "puddle"}


def quat_R(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def running_value(f, t):
    """The memory's estimate at time t (same rule as SpatialMemory._update: mean of the best-confidence class)."""
    meas = [o for o in f.get("observations", []) if o["t"] <= t and o.get("value_m") is not None]
    if not meas:
        return None, None
    c = max(o["confidence"] for o in meas)
    top = [o for o in meas if o["confidence"] >= c - 1e-9]
    return float(np.mean([o["value_m"] for o in top])), max(meas, key=lambda o: o["confidence"])["sensor"]


def font(size, bold=False):
    for name in (("arialbd.ttf" if bold else "arial.ttf"), "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mission", default="outputs/isaac_missions/approach_tile8.json")
    ap.add_argument("--out", default="outputs/videos/14_isaac_inspection_tile8.mp4")
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--size", type=int, nargs=2, default=[1920, 1080])
    ap.add_argument("--still", type=float, nargs="*", help="only save PNG stills at these times")
    ap.add_argument("--max-frames", type=int, default=0)
    a = ap.parse_args()

    r = json.loads(resolve_path(a.mission).read_text())
    tiles = json.loads(resolve_path(r["tiles_file"]).read_text())
    p = params_with_defaults(tiles[r["tile"]])
    L, half = float(p["size"][0]), 0.5 * float(p["width"])
    names = r["body_names"]
    links = r["links"]
    idx = {n: i for i, n in enumerate(names)}
    legs = [[idx[f"{s}_thigh"], idx[f"{s}_calf"], idx[f"{s}_foot"]] for s in ("FL", "FR", "RL", "RR")]
    hips = [idx[f"{s}_hip"] for s in ("FL", "FR", "RL", "RR")]
    base = idx["base"]
    sc = r["score"]
    gt_of = {d["finding"]: d for d in sc["per_defect"] if d.get("finding")}
    findings = r["memory"]["findings"]

    # ---------- static scene
    pv.OFF_SCREEN = True
    W, H = a.size
    pl = pv.Plotter(off_screen=True, window_size=(W, H))
    pl.set_background("#c9d6e3", top="#7d97b5")
    xs = np.arange(-L / 2, L / 2 + 1e-9, 0.02)
    ys = np.arange(-half - 1.0, half + 1.0 + 1e-9, 0.02)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    Z = sidewalk_height(X, Y, p, grid=0.02)
    grid = pv.StructuredGrid(X, Y, Z)
    rgb = sidewalk_albedo(X, Y, p, seed=r["tile"])
    rgb = np.asarray(rgb, dtype=np.float64)
    if rgb.max() <= 1.0:
        rgb = rgb * 255
    grid["rgb"] = rgb.reshape(-1, 3, order="F").astype(np.uint8)
    pl.add_mesh(grid, scalars="rgb", rgb=True, smooth_shading=True, ambient=0.25, diffuse=0.8, specular=0.05)
    for d in p["defects"]:
        if d["type"] == "puddle":
            t = np.linspace(0, 2 * np.pi, 60)
            c, s = np.cos(d["angle"]), np.sin(d["angle"])
            ex, ey = d["rx"] * np.cos(t), d["ry"] * np.sin(t)
            px, py = d["x"] + c * ex - s * ey, d["y"] + s * ex + c * ey
            z0 = float(np.max(sidewalk_height(px, py, p))) - 0.002
            poly = pv.PolyData(np.c_[px, py, np.full_like(px, z0)], faces=np.r_[len(px), np.arange(len(px))])
            pl.add_mesh(poly, color="#3b6f99", opacity=0.75, specular=0.9, specular_power=40)
        elif d["type"] == "obstruction":
            z0 = float(sidewalk_height(d["x"], d["y"], p))
            pl.add_mesh(pv.Cylinder(center=(d["x"], d["y"], z0 + 0.3), direction=(0, 0, 1), radius=d["r"], height=0.6),
                        color="#3d4a3d", smooth_shading=True)

    # ---------- robot (updated in place every frame)
    trunk = pv.Box(bounds=(-0.23, 0.21, -0.075, 0.075, -0.05, 0.06))
    trunk_actor = pl.add_mesh(trunk, color="#e9e9e9", smooth_shading=True, specular=0.4)
    head = pv.Box(bounds=(0.21, 0.29, -0.06, 0.06, -0.04, 0.04))
    head_actor = pl.add_mesh(head, color="#2a2a2a")
    cam555 = pv.Box(bounds=(0.26, 0.31, -0.045, 0.045, 0.045, 0.075))
    cam555_actor = pl.add_mesh(cam555, color="#202020")
    cam405 = pv.Box(bounds=(0.30, 0.335, -0.02, 0.02, -0.045, -0.02))
    cam405_actor = pl.add_mesh(cam405, color="#d4a017")
    leg_pts = np.zeros((16, 3))
    leg_lines = np.hstack([[2, 4 * k + j, 4 * k + j + 1] for k in range(4) for j in range(3)])
    leg_mesh = pv.PolyData(leg_pts.copy(), lines=leg_lines)
    pl.add_mesh(leg_mesh, color="#333333", line_width=18, render_lines_as_tubes=True)
    foot_mesh = pv.PolyData(np.zeros((4, 3)))
    pl.add_mesh(foot_mesh, color="#111111", point_size=22, render_points_as_spheres=True)
    trail = pv.PolyData(np.zeros((2, 3)), lines=[2, 0, 1])
    pl.add_mesh(trail, color="#ffffff", line_width=3, opacity=0.7)
    z555 = pv.PolyData(np.zeros((4, 3)), faces=[4, 0, 1, 2, 3])
    pl.add_mesh(z555, color="#ffffff", opacity=0.22)
    z405 = pv.PolyData(np.zeros((4, 3)), faces=[4, 0, 1, 2, 3])
    z405_actor = pl.add_mesh(z405, color="#ffd23f", opacity=0.45)
    rays = pv.PolyData(np.zeros((10, 3)), lines=np.hstack([[2, 0, k] for k in range(1, 5)] + [[2, 5, k] for k in range(6, 10)]))
    pl.add_mesh(rays, color="#ffd23f", line_width=1.5, opacity=0.6)

    def ground(x, y, dz=0.006):
        return np.c_[x, y, sidewalk_height(np.asarray(x), np.asarray(y), p) + dz]

    def zone(pose, x0, x1, hw):
        x, y, th = pose
        c, s = np.cos(th), np.sin(th)
        loc = np.array([[x0, -hw], [x1, -hw], [x1, hw], [x0, hw]])
        return ground(x + c * loc[:, 0] - s * loc[:, 1], y + s * loc[:, 0] + c * loc[:, 1])

    # findings: ring + pole, labels re-added per frame
    fmark = {}
    for fid, f in findings.items():
        lab, col = KIND.get(f["kind"], (f["kind"], (255, 255, 255)))
        cx, cy = f["xy"]
        ang = np.linspace(0, 2 * np.pi, 12, endpoint=False)  # rim level (a hole's centre is its bottom)
        z0 = float(np.median(sidewalk_height(cx + 0.45 * np.cos(ang), cy + 0.45 * np.sin(ang), p)))
        ring = pv.Disc(center=(cx, cy, z0 + 0.01), inner=0.2, outer=0.25, c_res=48)
        disc = pv.Disc(center=(cx, cy, z0 + 0.009), inner=0.0, outer=0.2, c_res=48)
        pole = pv.Line((cx, cy, z0), (cx, cy, z0 + 0.55))
        obs = sorted(f.get("observations", []), key=lambda o: o["t"])
        det = [o["t"] for o in obs if o.get("provenance") == "detected"]
        t_show = det[1] if len(det) > 1 else (det[0] if det else f.get("first_seen_t", 0.0))
        col_hex = "#%02x%02x%02x" % col
        fmark[fid] = dict(f=f, lab=lab, col=col, t=t_show, top=(cx, cy, z0 + 0.58),
                          ring=pl.add_mesh(ring, color=col_hex), disc=pl.add_mesh(disc, color=col_hex, opacity=0.55),
                          pole=pl.add_mesh(pole, color=col_hex, line_width=3))
    label_actor = [None]
    pl.enable_lightkit()

    traj = np.asarray(r["trajectory"])
    T = r["time_s"]
    dt = T / max(len(traj) - 1, 1)
    events = r["events"]
    F_big, F_med, F_small, F_mono = font(34, True), font(24, True), font(20), font(20)
    cam_state = {"yaw": traj[0, 2], "dist": 1.0}

    def state_at(t):
        k = min(int(round(t / dt)), len(links) - 1)
        return k, links[k]

    def mode_at(t):
        ev = [e for e in events if e["t"] <= t]
        if t >= T:
            return "GOAL REACHED", (110, 227, 110)
        if ev:
            e = ev[-1]
            if e["event"] == "approach":
                k, _ = state_at(t)
                k2 = min(k + 1, len(traj) - 1)
                v = np.linalg.norm(traj[k2, :2] - traj[k, :2]) / dt
                w = abs((traj[k2, 2] - traj[k, 2] + np.pi) % (2 * np.pi) - np.pi) / dt
                if v < 0.06 and w > 0.15:
                    return "TURNING to face the finding", (255, 210, 63)
                if v < 0.06:
                    return "INSPECTING: stopped, D405 close-up", (255, 210, 63)
                return "APPROACHING a finding for a close look", (255, 210, 63)
            if e["event"] == "approach_aborted" and t - e["t"] < 2.0:
                return "APPROACH ABORTED (stand-off not reached in time)", (255, 120, 120)
            if e["event"] == "inspected" and t - e["t"] < 2.0:
                return f"INSPECTED: {e['status']}", (110, 227, 110)
        return "WALKING the route · D555 scanning ahead", (232, 236, 240)

    def draw(t, final=False):
        k, ln = state_at(t)
        pos = np.asarray(ln["pos"])
        R = quat_R(ln["quat"][base])
        M = np.eye(4)
        M[:3, :3], M[:3, 3] = R, pos[base]
        for act in (trunk_actor, head_actor, cam555_actor, cam405_actor):
            act.user_matrix = M
        pts = []
        for h, (th, ca, fo) in zip(hips, legs):
            pts += [pos[h], pos[th], pos[ca], pos[fo]]
        leg_mesh.points = np.asarray(pts)
        foot_mesh.points = np.asarray([pos[l[2]] for l in legs]) + [0, 0, 0.0]
        tr = traj[: k + 1]
        if len(tr) >= 2:
            pts3 = ground(tr[:, 0], tr[:, 1], 0.01)
            trail.points = pts3
            trail.lines = np.hstack([[2, i, i + 1] for i in range(len(pts3) - 1)])
        pose = traj[k]
        z555.points = zone(pose, 0.6, 1.2, 0.6)
        zq = zone(pose, 0.42, 0.62, 0.2)
        z405.points = zq
        c555 = pos[base] + R @ np.array([0.30, 0.0, 0.08])
        c405 = pos[base] + R @ np.array([0.33, 0.0, -0.03])
        zq555 = z555.points
        rays.points = np.vstack([c405, zq, c555, zq555])
        inspecting = ln.get("target") is not None
        z405_actor.prop.opacity = 0.7 if inspecting else 0.35
        # findings
        lp, lt, lc = [], [], []
        rows = []
        for fid, m in fmark.items():
            vis = t >= m["t"]
            for key in ("ring", "pole"):
                m[key].visibility = vis
            val, sensor = running_value(m["f"], t)
            m["disc"].visibility = vis and val is not None
            if not vis:
                continue
            g = gt_of.get(fid)
            if m["f"]["kind"] == "dropout":
                txt = "possible water (no depth returns)"
                if g:
                    txt += f"\ntrue: puddle {g['true_value_m'] * 1000:.0f} mm deep"
            elif val is None:
                txt = f"{m['lab']}: detected, not yet measured"
            else:
                txt = f"{m['lab']}: measured {abs(val) * 1000:.1f} mm ({sensor.split('_')[-1].upper()})"
                if g:
                    err = abs(val) - abs(g["true_value_m"])
                    txt += f"\ntrue {abs(g['true_value_m']) * 1000:.1f} mm · error {err * 1000:+.1f} mm"
                else:
                    txt += "\nno true defect here (false finding)"
            lp.append(m["top"])
            lt.append(txt)
            rows.append((m, val, sensor, g))
        if label_actor[0] is not None:
            pl.remove_actor(label_actor[0])
            label_actor[0] = None
        if lp:
            label_actor[0] = pl.add_point_labels(np.asarray(lp), lt, font_size=17, point_size=1, shape_opacity=0.75,
                                                 shape_color="#101418", text_color="white", always_visible=True,
                                                 show_points=False, margin=4)
        # chase camera: behind-left, closer while inspecting
        yaw = pose[2]
        cam_state["yaw"] += 0.15 * ((yaw - cam_state["yaw"] + np.pi) % (2 * np.pi) - np.pi)
        cam_state["dist"] += 0.06 * ((0.62 if inspecting else 1.0) - cam_state["dist"])
        cy_, sy_ = np.cos(cam_state["yaw"]), np.sin(cam_state["yaw"])
        dd = cam_state["dist"]
        foc = np.array([pose[0] + 1.0 * cy_, pose[1] + 1.0 * sy_, pos[base][2] - 0.25])
        eye = foc + np.array([-3.0 * cy_ + 1.9 * sy_, -3.0 * sy_ - 1.9 * cy_, 0.0]) * dd + [0, 0, 1.9 * dd]
        cam = pl.camera
        cam.position, cam.focal_point, cam.up = tuple(eye), tuple(foc), (0.0, 0.0, 1.0)
        pl.renderer.ResetCameraClippingRange()
        pl.render()
        img = Image.fromarray(pl.screenshot(return_img=True))
        return hud(img, t, rows, final)

    def hud(img, t, rows, final=False):
        d = ImageDraw.Draw(img, "RGBA")
        d.rectangle([0, 0, W, 96], fill=(16, 20, 24, 215))
        strat = {"approach": "stop & look closely", "walkover": "walk past (no stops)"}.get(r["policy"], r["policy"])
        d.text((24, 12), f"Go2 sidewalk inspection: {strat} (Isaac Lab physics, rough-terrain policy)",
               font=F_big, fill=(232, 236, 240))
        prof = r["profile"] + (" + odometry drift + camera latency" if r.get("pose_errors") else "")
        d.text((24, 58), f"D555 + D405 simulated with {prof} errors · tile {r['tile']} (fresh test layout) · 2x speed",
               font=F_small, fill=(160, 172, 184))
        msg, col = mode_at(t)
        d.rectangle([0, H - 64, W, H], fill=(16, 20, 24, 215))
        d.text((24, H - 50), msg, font=F_med, fill=col)
        d.text((W - 230, H - 50), f"t = {min(t, T):5.1f} s", font=F_med, fill=(232, 236, 240))
        # measured-vs-true panel
        x0, y0, w = W - 640, 116, 620
        lines = [r_ for r_ in rows if r_[0]["f"]["kind"] != "dropout"]
        h = 70 + 34 * max(len(lines), 1)
        d.rectangle([x0, y0, x0 + w, y0 + h], fill=(16, 20, 24, 200))
        d.text((x0 + 16, y0 + 10), "Measured by the robot vs. true size", font=F_med, fill=(232, 236, 240))
        d.text((x0 + 16, y0 + 42), f"{'defect':<16}{'measured':>10}{'true':>9}{'error':>9}  sensor", font=F_mono,
               fill=(160, 172, 184))
        for i, (m, val, sensor, g) in enumerate(lines):
            yy = y0 + 70 + 34 * i
            name = m["lab"] if not g else TYPE_NAME.get(g["type"], g["type"])
            ms = f"{abs(val) * 1000:7.1f}mm" if val is not None else "      –  "
            tr_ = f"{abs(g['true_value_m']) * 1000:6.1f}mm" if g else "  none "
            er = f"{(abs(val) - abs(g['true_value_m'])) * 1000:+6.1f}mm" if (g and val is not None) else "    –  "
            sn = sensor.split("_")[-1].upper() if sensor else "–"
            d.text((x0 + 16, yy), f"{name:<16}{ms:>10}{tr_:>9}{er:>9}  {sn}", font=F_mono, fill=m["col"] + (255,))
        if not lines:
            d.text((x0 + 16, y0 + 70), "(nothing measured yet)", font=F_mono, fill=(160, 172, 184))
        if final:
            yy = y0 + h + 16
            missed = [x for x in sc["per_defect"] if not x["found"]]
            n_found = sum(x["found"] for x in sc["per_defect"])
            txt = (f"Ground truth: {n_found}/{len(sc['per_defect'])} defects found, {sc['false_findings']} false findings"
                   + "".join(f"\nmissed: {x['type']} ({abs(x['true_value_m']) * 1000:.0f} mm)" for x in missed))
            d.rectangle([x0, yy, x0 + w, yy + 40 + 28 * txt.count("\n")], fill=(16, 20, 24, 200))
            d.multiline_text((x0 + 16, yy + 8), txt, font=F_small, fill=(232, 236, 240), spacing=8)
        return np.asarray(img)

    out = resolve_path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if a.still:
        for t in a.still:
            Image.fromarray(draw(t)).save(out.with_name(out.stem + f"_t{t:.0f}.png"))
        return 0
    w = imageio.get_writer(out, fps=a.fps, codec="libx264", quality=8, macro_block_size=8, pixelformat="yuv420p")
    times = np.arange(0.0, T + 1e-9, 2.0 / a.fps)
    if a.max_frames:
        times = times[: a.max_frames]
    for i, t in enumerate(times):
        w.append_data(draw(float(t)))
        if i % 50 == 0:
            print(f"frame {i}/{len(times)}", flush=True)
    last = draw(T, final=True)
    for _ in range(6 * a.fps):
        w.append_data(last)
    w.close()
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
