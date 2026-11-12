"""Phase 1: side-by-side video, walk past vs. stop & look, Isaac Go2 physics, with live measured-vs-true plots.

  D:/go2wm/render_env/Scripts/python.exe scripts/compare_isaac_video.py \
      --walk outputs/isaac_missions/walkover_tile8.json --look outputs/isaac_missions/approach_tile8.json \
      --meshes outputs/go2_meshes.npz --out outputs/videos/15_walk_vs_look_tile8.mp4

Top: two time-synced 3D chase views of the same sidewalk tile. Each robot is the real Go2 visual model (meshes
exported from the Isaac stage by scripts/export_go2_meshes.py) posed with the link poses recorded during the Isaac
physics run, so the gait, body sway and the stops are the simulated ones. Bottom: for each measured defect, every raw
depth measurement (dots: hollow = D555 from walking distance, filled = D405 close-up) and the robot's running estimate
(line, same rule as SpatialMemory), against the ground-truth size (dashed) and a +-2 mm band. Playback is 1x while the
stop & look robot is approaching / inspecting, faster otherwise; the clock is shared by both views.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import imageio.v2 as imageio  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pyvista as pv  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from go2wm.common.config import resolve_path  # noqa: E402
from go2wm.sim.sidewalk_texture import sidewalk_albedo  # noqa: E402
from go2wm.sim.terrain import params_with_defaults, sidewalk_height  # noqa: E402
from render_isaac_mission import KIND, font, quat_R, running_value  # noqa: E402

LINK_COL = {"base": "#d4d6d9", "hip": "#2e3033", "thigh": "#c9cbce", "calf": "#26282b", "foot": "#101112"}
STRAT = {"walkover": ("WALK PAST (no stops)", "#9aa5b1"), "approach": ("STOP & LOOK closely", "#f08c00")}


def load_meshes(path, target=6000):
    z = np.load(resolve_path(path))
    out = []
    for i in range(int(z["n"])):
        v, f = z[f"m{i}_v"].astype(np.float64), z[f"m{i}_f"]
        m = pv.PolyData(v, np.c_[np.full(len(f), 3), f].ravel())
        if m.n_cells > target:
            m = m.decimate_pro(1.0 - target / m.n_cells, preserve_topology=True)
        link = str(z[f"m{i}_link"])
        part = link.split("_")[-1] if "_" in link else link
        out.append((link, m.compute_normals(auto_orient_normals=False, split_vertices=True, feature_angle=40),
                    LINK_COL.get(part, "#888888")))
    return out


class View:
    """One 3D chase-camera view of one mission."""

    def __init__(self, r, meshes, W, H):
        self.r, self.W, self.H = r, W, H
        tiles = json.loads(resolve_path(r["tiles_file"]).read_text())
        p = self.p = params_with_defaults(tiles[r["tile"]])
        L, half = float(p["size"][0]), 0.5 * float(p["width"])
        self.idx = {n: i for i, n in enumerate(r["body_names"])}
        self.links, self.traj = r["links"], np.asarray(r["trajectory"])
        self.T = r["time_s"]
        self.dt = self.T / max(len(self.traj) - 1, 1)
        self.events = r["events"]
        sc = r["score"]
        self.gt_of = {d["finding"]: d for d in sc["per_defect"] if d.get("finding")}
        pl = self.pl = pv.Plotter(off_screen=True, window_size=(W, H))
        pl.set_background("#dfe7ee", top="#8aa4c0")
        xs = np.arange(-L / 2 - 0.5, L / 2 + 0.5 + 1e-9, 0.02)
        ys = np.arange(-half - 1.2, half + 1.2 + 1e-9, 0.02)
        X, Y = np.meshgrid(xs, ys, indexing="ij")
        Z = sidewalk_height(X, Y, p, grid=0.02)
        grid = pv.StructuredGrid(X, Y, Z)
        rgb = np.asarray(sidewalk_albedo(X, Y, p, seed=r["tile"]), dtype=np.float64)
        if rgb.max() <= 1.0:
            rgb = rgb * 255
        off = np.abs(Y) > half + 0.02  # verge: grass either side of the walk
        rng = np.random.default_rng(1)
        grass = np.stack([70 + 25 * rng.random(X.shape), 110 + 30 * rng.random(X.shape), 55 + 15 * rng.random(X.shape)], -1)
        rgb[off] = grass[off]
        grid["rgb"] = rgb.reshape(-1, 3, order="F").astype(np.uint8)
        pl.add_mesh(grid, scalars="rgb", rgb=True, smooth_shading=True, ambient=0.3, diffuse=0.75, specular=0.05)
        for d in p["defects"]:
            if d["type"] == "puddle":
                t = np.linspace(0, 2 * np.pi, 60)
                c, s = np.cos(d["angle"]), np.sin(d["angle"])
                ex, ey = d["rx"] * np.cos(t), d["ry"] * np.sin(t)
                px, py = d["x"] + c * ex - s * ey, d["y"] + s * ex + c * ey
                z0 = float(np.max(sidewalk_height(px, py, p))) - 0.002
                poly = pv.PolyData(np.c_[px, py, np.full_like(px, z0)], faces=np.r_[len(px), np.arange(len(px))])
                pl.add_mesh(poly, color="#3b6f99", opacity=0.7, specular=0.9, specular_power=40)
            elif d["type"] == "obstruction":
                z0 = float(sidewalk_height(d["x"], d["y"], p))
                pl.add_mesh(pv.Cylinder(center=(d["x"], d["y"], z0 + 0.3), direction=(0, 0, 1), radius=d["r"],
                                        height=0.6), color="#3d4a3d", smooth_shading=True)
        # robot: real Go2 visual meshes, one actor per link
        self.robot = [(self.idx[link], pl.add_mesh(m, color=col, smooth_shading=True, specular=0.35, specular_power=20))
                      for link, m, col in meshes if link in self.idx]
        self.trail = pv.PolyData(np.zeros((2, 3)), lines=[2, 0, 1])
        pl.add_mesh(self.trail, color=STRAT[r["policy"]][1], line_width=4, opacity=0.85)
        self.z555 = pv.PolyData(np.zeros((4, 3)), faces=[4, 0, 1, 2, 3])
        pl.add_mesh(self.z555, color="#ffffff", opacity=0.2)
        self.z405 = pv.PolyData(np.zeros((4, 3)), faces=[4, 0, 1, 2, 3])
        self.z405_actor = pl.add_mesh(self.z405, color="#ffd23f", opacity=0.4)
        self.rays = pv.PolyData(np.zeros((5, 3)), lines=np.hstack([[2, 0, k] for k in range(1, 5)]))
        self.rays_actor = pl.add_mesh(self.rays, color="#ffd23f", line_width=2, opacity=0.8)
        self.fmark = {}
        for fid, f in r["memory"]["findings"].items():
            lab, col = KIND.get(f["kind"], (f["kind"], (255, 255, 255)))
            cx, cy = f["xy"]
            ang = np.linspace(0, 2 * np.pi, 12, endpoint=False)
            z0 = float(np.median(sidewalk_height(cx + 0.45 * np.cos(ang), cy + 0.45 * np.sin(ang), p)))
            obs = sorted(f.get("observations", []), key=lambda o: o["t"])
            det = [o["t"] for o in obs if o.get("provenance") == "detected"]
            t_show = det[1] if len(det) > 1 else (det[0] if det else f.get("first_seen_t", 0.0))
            hexc = "#%02x%02x%02x" % col
            self.fmark[fid] = dict(
                f=f, lab=lab, t=t_show, top=(cx, cy, z0 + 0.5),
                ring=pl.add_mesh(pv.Disc(center=(cx, cy, z0 + 0.01), inner=0.2, outer=0.25, c_res=48), color=hexc),
                pole=pl.add_mesh(pv.Line((cx, cy, z0), (cx, cy, z0 + 0.47)), color=hexc, line_width=3))
        self.label_actor = None
        pl.enable_lightkit()
        self.cam = {"yaw": self.traj[0, 2], "dist": 1.0}

    def ground(self, x, y, dz=0.006):
        return np.c_[x, y, sidewalk_height(np.asarray(x), np.asarray(y), self.p) + dz]

    def zone(self, pose, x0, x1, hw):
        x, y, th = pose
        c, s = np.cos(th), np.sin(th)
        loc = np.array([[x0, -hw], [x1, -hw], [x1, hw], [x0, hw]])
        return self.ground(x + c * loc[:, 0] - s * loc[:, 1], y + s * loc[:, 0] + c * loc[:, 1])

    def k_at(self, t):
        return min(int(round(min(t, self.T) / self.dt)), len(self.links) - 1)

    def mode(self, t):
        if t >= self.T:
            return "GOAL REACHED", (110, 227, 110)
        ev = [e for e in self.events if e["t"] <= t and e["event"] in ("approach", "approach_aborted", "inspected")]
        if ev:
            e = ev[-1]
            if e["event"] == "approach":
                k = self.k_at(t)
                k2 = min(k + 1, len(self.traj) - 1)
                v = np.linalg.norm(self.traj[k2, :2] - self.traj[k, :2]) / self.dt
                w = abs((self.traj[k2, 2] - self.traj[k, 2] + np.pi) % (2 * np.pi) - np.pi) / self.dt
                if v < 0.06 and w > 0.15:
                    return "TURNING to face the defect", (255, 210, 63)
                if v < 0.06:
                    return "STOPPED: D405 close-up measurement", (255, 210, 63)
                return "APPROACHING the defect", (255, 210, 63)
            if e["event"] == "approach_aborted" and t - e["t"] < 2.0:
                return "APPROACH ABORTED", (255, 120, 120)
            if e["event"] == "inspected" and t - e["t"] < 2.0:
                return "INSPECTED, resuming route", (110, 227, 110)
        return "WALKING, D555 scanning ahead", (232, 236, 240)

    def inspecting(self, t):
        return self.links[self.k_at(t)].get("target") is not None and t < self.T

    def render(self, t):
        k = self.k_at(t)
        ln = self.links[k]
        pos, quat = np.asarray(ln["pos"]), ln["quat"]
        for i, act in self.robot:
            M = np.eye(4)
            M[:3, :3], M[:3, 3] = quat_R(quat[i]), pos[i]
            act.user_matrix = M
        base = self.idx["base"]
        R = quat_R(quat[base])
        tr = self.traj[: k + 1]
        if len(tr) >= 2:
            pts = self.ground(tr[:, 0], tr[:, 1], 0.01)
            self.trail.points = pts
            self.trail.lines = np.hstack([[2, i, i + 1] for i in range(len(pts) - 1)])
        pose = self.traj[k]
        self.z555.points = self.zone(pose, 0.6, 1.2, 0.6)
        zq = self.zone(pose, 0.42, 0.62, 0.2)
        self.z405.points = zq
        insp = self.inspecting(t)
        self.rays.points = np.vstack([pos[base] + R @ np.array([0.30, 0.0, -0.03]), zq])
        self.rays_actor.visibility = insp
        self.z405_actor.prop.opacity = 0.7 if insp else 0.3
        lp, lt = [], []
        for fid, m in self.fmark.items():
            vis = t >= m["t"]
            m["ring"].visibility = m["pole"].visibility = vis
            if not vis:
                continue
            val, sensor = running_value(m["f"], t)
            g = self.gt_of.get(fid)
            if m["f"]["kind"] == "dropout":
                txt = "possible water"
            elif val is None:
                txt = f"{m['lab'].split(' ')[0]}: not measured yet"
            else:
                txt = f"{m['lab'].split(' ')[0]} {abs(val) * 1000:.1f} mm ({sensor.split('_')[-1].upper()})"
                txt += f"\ntrue {abs(g['true_value_m']) * 1000:.1f} mm" if g else "\n(false finding)"
            lp.append(m["top"])
            lt.append(txt)
        if self.label_actor is not None:
            self.pl.remove_actor(self.label_actor)
            self.label_actor = None
        if lp:
            self.label_actor = self.pl.add_point_labels(np.asarray(lp), lt, font_size=13, point_size=1,
                                                        shape_opacity=0.75, shape_color="#101418", text_color="white",
                                                        always_visible=True, show_points=False, margin=3)
        yaw = pose[2]
        self.cam["yaw"] += 0.12 * ((yaw - self.cam["yaw"] + np.pi) % (2 * np.pi) - np.pi)
        self.cam["dist"] += 0.05 * ((0.55 if insp else 0.85) - self.cam["dist"])
        cy_, sy_ = np.cos(self.cam["yaw"]), np.sin(self.cam["yaw"])
        dd = self.cam["dist"]
        foc = np.array([pose[0] + 0.9 * cy_, pose[1] + 0.9 * sy_, pos[base][2] - 0.25])
        eye = foc + np.array([-3.0 * cy_ + 1.7 * sy_, -3.0 * sy_ - 1.7 * cy_, 0.0]) * dd + [0, 0, 1.6 * dd]
        c = self.pl.camera
        c.position, c.focal_point, c.up = tuple(eye), tuple(foc), (0.0, 0.0, 1.0)
        self.pl.renderer.ResetCameraClippingRange()
        self.pl.render()
        img = Image.fromarray(self.pl.screenshot(return_img=True))
        d = ImageDraw.Draw(img, "RGBA")
        name, col = STRAT[self.r["policy"]]
        d.rectangle([0, 0, self.W, 44], fill=(16, 20, 24, 215))
        d.text((16, 8), name, font=font(26, True), fill=col)
        msg, mcol = self.mode(t)
        d.rectangle([0, self.H - 42, self.W, self.H], fill=(16, 20, 24, 215))
        d.text((16, self.H - 34), msg, font=font(22, True), fill=mcol)
        if t >= self.T:
            d.text((self.W - 250, self.H - 34), f"finished at {self.T:.1f} s", font=font(20), fill=(200, 210, 220))
        return np.asarray(img)


def series(r, typ):
    """Raw measurements and the running estimate of the finding matched to the true defect of type typ."""
    d = next(x for x in r["score"]["per_defect"] if x["type"] == typ)
    f = r["memory"]["findings"].get(d.get("finding")) if d.get("finding") else None
    obs = [o for o in (f or {}).get("observations", []) if o.get("value_m") is not None]
    raw = [(o["t"], abs(o["value_m"]) * 1000, o["sensor"]) for o in obs]
    ts = sorted({o["t"] for o in obs})
    run = [(t, abs(running_value(f, t)[0]) * 1000) for t in ts]
    return abs(d["true_value_m"]) * 1000, raw, run


class Plots:
    def __init__(self, walk, look, W, H, tmax):
        self.W, self.H = W, H
        self.typs = [t for t in ("hole", "upheaval") if any(x["type"] == t for x in walk["score"]["per_defect"])]
        self.data = {t: {"walkover": series(walk, t), "approach": series(look, t)} for t in self.typs}
        self.tmax = tmax
        first = min([x[0] for s in self.data.values() for v in s.values() for x in v[1]] or [0.0])
        self.t0 = max(0.0, np.floor(first - 3.0))
        self.T = {"walkover": walk["time_s"], "approach": look["time_s"]}

    def render(self, t):
        n = len(self.typs)
        fig, axs = plt.subplots(1, n, figsize=(self.W / 100, self.H / 100), dpi=100)
        fig.patch.set_facecolor("#14181c")
        axs = np.atleast_1d(axs)
        for ax, typ in zip(axs, self.typs):
            ax.set_facecolor("#1b2026")
            true = self.data[typ]["walkover"][0]
            lo = min([true] + [v for s in self.data[typ].values() for (_, v, _) in s[1]]) - 8
            hi = max([true] + [v for s in self.data[typ].values() for (_, v, _) in s[1]]) + 8
            lo = max(lo, 0)
            ax.axhspan(true - 2, true + 2, color="#2b8a3e", alpha=0.25, lw=0)
            ax.axhline(true, color="#e8ecf0", ls="--", lw=1.4)
            ax.text(self.t0 + 0.2, true + 1.0, f"true {typ} size: {true:.1f} mm  (green: within 2 mm)", color="#e8ecf0",
                    fontsize=10, va="bottom")
            notes = []
            for pol in ("walkover", "approach"):
                _, raw, run = self.data[typ][pol]
                col = STRAT[pol][1]
                r_ = [x for x in raw if x[0] <= t]
                for sens, mk in (("d555", "o"), ("d405", "D")):
                    pts = [x for x in r_ if sens in x[2]]
                    if pts:
                        ax.scatter([x[0] for x in pts], [x[1] for x in pts], s=22 if sens == "d555" else 30, marker=mk,
                                   facecolor="none" if sens == "d555" else col, edgecolor=col, lw=1.1, alpha=0.75,
                                   zorder=3)
                ru = [x for x in run if x[0] <= t]
                if ru:
                    xs = [x[0] for x in ru] + [min(t, self.T[pol])]
                    ys = [x[1] for x in ru] + [ru[-1][1]]
                    ax.step(xs, ys, where="post", color=col, lw=2.6, zorder=4)
                    err = ru[-1][1] - true
                    notes.append((pol, ru[-1][1], err))
            ax.axvline(max(t, self.t0), color="#ffffff", lw=0.8, alpha=0.5)
            ax.set_xlim(self.t0, self.tmax)
            ax.set_ylim(lo, hi)
            ax.set_title(f"{typ}: robot's estimate vs. ground truth", color="#e8ecf0", fontsize=13, loc="left")
            ax.set_xlabel("mission time [s]", color="#a0acb8", fontsize=10)
            ax.set_ylabel(f"{'depth' if typ == 'hole' else 'height'} [mm]", color="#a0acb8", fontsize=10)
            ax.tick_params(colors="#a0acb8", labelsize=9)
            for s in ax.spines.values():
                s.set_color("#3a424a")
            ax.grid(alpha=0.12)
            y = 0.06
            for pol, v, e in notes[::-1]:
                ax.text(0.015, y, f"{STRAT[pol][0].split(' (')[0].lower()}: {v:.1f} mm, error {e:+.1f} mm",
                        transform=ax.transAxes, color=STRAT[pol][1], fontsize=12, fontweight="bold")
                y += 0.11
            if t < self.t0:
                ax.text(0.5, 0.5, f"no defect measured yet (first measurement at {self.t0 + 3:.0f} s)",
                        transform=ax.transAxes, color="#a0acb8", fontsize=12, ha="center")
            elif not notes:
                ax.text(0.015, 0.06, "not measured yet", transform=ax.transAxes, color="#a0acb8", fontsize=12)
        h = [plt.Line2D([], [], color=STRAT["walkover"][1], lw=2.6, label="walk past: running estimate"),
             plt.Line2D([], [], color=STRAT["approach"][1], lw=2.6, label="stop & look: running estimate"),
             plt.Line2D([], [], color="#cccccc", marker="o", mfc="none", ls="", label="raw D555 reading"),
             plt.Line2D([], [], color="#cccccc", marker="D", ls="", label="raw D405 close-up reading")]
        leg = fig.legend(handles=h, loc="lower center", ncol=4, fontsize=10, frameon=False)
        for tx in leg.get_texts():
            tx.set_color("#d0d8e0")
        fig.tight_layout(rect=(0, 0.08, 1, 1))
        fig.canvas.draw()
        img = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
        plt.close(fig)
        return img


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--walk", default="outputs/isaac_missions/walkover_tile8.json")
    ap.add_argument("--look", default="outputs/isaac_missions/approach_tile8.json")
    ap.add_argument("--meshes", default="outputs/go2_meshes.npz")
    ap.add_argument("--out", default="outputs/videos/15_walk_vs_look_tile8.mp4")
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--fast", type=float, default=2.5, help="playback speed while nobody is inspecting")
    ap.add_argument("--still", type=float, nargs="*")
    a = ap.parse_args()
    walk = json.loads(resolve_path(a.walk).read_text())
    look = json.loads(resolve_path(a.look).read_text())
    W, H = 1920, 1080
    VW, VH, TOP = 956, 600, 64
    meshes = load_meshes(a.meshes)
    views = [View(walk, meshes, VW, VH), View(look, meshes, VW, VH)]
    T = max(walk["time_s"], look["time_s"])
    plots = Plots(walk, look, W, H - TOP - VH - 8, T + 0.5)
    F_big, F_small = font(30, True), font(18)

    def frame(t, speed, final=False):
        img = Image.new("RGB", (W, H), (20, 24, 28))
        d = ImageDraw.Draw(img)
        d.text((20, 10), "Go2 sidewalk inspection: walk past vs. stop & look  (Isaac Lab physics, same tile, same "
                         "sensors)", font=F_big, fill=(232, 236, 240))
        prof = walk["profile"] + (" + odometry drift + camera latency" if walk.get("pose_errors") else "")
        d.text((22, 46), f"D555 + D405 depth simulated with {prof} errors  ·  tile {walk['tile']} (test layout)",
               font=F_small, fill=(150, 162, 174))
        d.text((W - 330, 14), f"t = {min(t, T):5.1f} s", font=font(26, True), fill=(232, 236, 240))
        d.text((W - 330, 46), "playback 1x (inspecting)" if speed <= 1 else f"playback {speed:g}x", font=F_small,
               fill=(255, 210, 63) if speed <= 1 else (150, 162, 174))
        for i, v in enumerate(views):
            img.paste(Image.fromarray(v.render(t)), (i * (VW + 8), TOP))
        img.paste(Image.fromarray(plots.render(t)), (0, TOP + VH + 8))
        if final:
            d = ImageDraw.Draw(img, "RGBA")
            x0, y0 = 560, 250
            d.rectangle([x0, y0, x0 + 800, y0 + 250], fill=(16, 20, 24, 235), outline=(240, 140, 0), width=3)
            lines = ["Final measurements vs. ground truth"]
            for typ in plots.typs:
                tr = plots.data[typ]["walkover"][0]
                vals = []
                for pol in ("walkover", "approach"):
                    run = plots.data[typ][pol][2]
                    vals.append(f"{run[-1][1]:.1f} mm ({run[-1][1] - tr:+.1f})" if run else "not measured")
                lines.append(f"{typ:<9} true {tr:5.1f} mm | walk past {vals[0]:<18} | stop & look {vals[1]}")
            lines.append(f"time to goal: walk past {walk['time_s']:.1f} s, stop & look {look['time_s']:.1f} s")
            d.text((x0 + 24, y0 + 20), lines[0], font=font(28, True), fill=(240, 140, 0))
            for j, s in enumerate(lines[1:]):
                d.text((x0 + 24, y0 + 80 + 46 * j), s, font=font(20), fill=(232, 236, 240))
        return np.asarray(img)

    out = resolve_path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if a.still:
        for t in a.still:
            Image.fromarray(frame(t, 1.0)).save(out.with_name(out.stem + f"_t{t:.0f}.png"))
        return 0
    wr = imageio.get_writer(out, fps=a.fps, codec="libx264", quality=8, macro_block_size=8, pixelformat="yuv420p")
    t, i = 0.0, 0
    while t <= T + 1e-9:
        speed = 1.0 if views[1].inspecting(t) or views[0].inspecting(t) else a.fast
        wr.append_data(frame(t, speed))
        t += speed / a.fps
        i += 1
        if i % 50 == 0:
            print(f"frame {i}  t={t:.1f}/{T:.1f}", flush=True)
    last = frame(T, a.fast, final=True)
    for _ in range(7 * a.fps):
        wr.append_data(last)
    wr.close()
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
