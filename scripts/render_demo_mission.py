"""Phase 1 demo video: a full inspection mission on the demo street (Isaac Go2 physics, rendered offline).

  D:/go2wm/render_env/Scripts/python.exe scripts/render_demo_mission.py \
      --mission outputs/demo_mission/isaac/approach_posture_tile0.json \
      --compare "walk past=outputs/demo_mission/isaac/walkover_tile0.json" \
                "stop & look (standing)=outputs/demo_mission/isaac/approach_tile0.json" \
      --out outputs/videos/16_demo_mission.mp4

The street of mission/site.py:demo_tile (five blocks, ten defects, a construction work zone with workers, two
pedestrians, buildings, road, trees) with the real Go2 model posed from the recorded Isaac link poses, so walking,
turning, crouching and pitching to look down are the simulated ones. HUD: mission status, the robot's action set
with the active action, the findings log (measured vs. true), a live plot of the measurement in progress, and a
route minimap. Opens with a briefing, ends with the mission report (and the same route run with other strategies).
"""

from __future__ import annotations

import argparse
import json
import math
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

from compare_isaac_video import load_meshes  # noqa: E402
from go2wm.common.config import resolve_path  # noqa: E402
from go2wm.sim.sensors import SENSOR_SPECS, pitch_matrix  # noqa: E402
from go2wm.sim.sidewalk_texture import sidewalk_albedo  # noqa: E402
from go2wm.sim.terrain import params_with_defaults, sidewalk_height  # noqa: E402
from render_isaac_mission import KIND, ROBUST, font, quat_R, running_value  # noqa: E402

W, H = 1920, 1080
TOP, VW, VH = 52, 1344, 756
PX0, PW = VW + 8, W - VW - 8
MAP_Y = TOP + VH + 8
ADA = (6.4, 13.0)
ACTIONS = [("walk", "Walk the route (A* + MPC)"), ("turn", "Turn in place to face a defect"),
           ("stop", "Stop for a close-up"), ("crouch", "Crouch + level body (look closer)"),
           ("yield", "Yield to a pedestrian"), ("reroute", "Reroute (holes, work zone)")]
WHITE, GREY, AMBER, GREEN, RED = (232, 236, 240), (150, 162, 174), (255, 210, 63), (110, 227, 110), (255, 120, 120)


def mono(size, bold=False):
    from PIL import ImageFont

    for name in (("consolab.ttf" if bold else "consola.ttf"), "courbd.ttf" if bold else "cour.ttf", "DejaVuSansMono.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return font(size, bold)


def ada_class(mm):
    return "ok" if mm <= ADA[0] else ("bevel" if mm <= ADA[1] else "violation")


# ----------------------------------------------------------------------------------------------- scene props
def merged(parts):
    m = parts[0]
    for q in parts[1:]:
        m = m.merge(q)
    return m


def box(x0, x1, y0, y1, z0, z1):
    return pv.Box(bounds=(x0, x1, y0, y1, z0, z1))


def add_buildings(pl, L, rng):
    x = -L / 2 - 2
    cols = ["#b5654a", "#c9b79c", "#8d8f94", "#a3876a", "#d8d2c4", "#6f7d8c"]
    while x < L / 2 + 2:
        w = float(rng.uniform(6, 10))
        h = float(rng.uniform(3.6, 7.5))
        y0 = 3.35
        pl.add_mesh(box(x, x + w - 0.15, y0, y0 + 6, 0, h), color=cols[int(rng.integers(len(cols)))], ambient=0.25)
        wins = []
        for fl in range(int((h - 0.6) // 2.6) + 1):
            z = 0.9 + fl * 2.6
            if z + 1.3 > h:
                break
            for k in np.arange(x + 0.8, x + w - 1.0, 1.6):
                if fl == 0 and abs(k - (x + w / 2)) < 0.9:
                    continue
                wins.append(box(k, k + 0.95, y0 - 0.03, y0, z, z + 1.3))
        if wins:
            pl.add_mesh(merged(wins), color="#2c3a48", specular=0.8, specular_power=30)
        d = x + w / 2 - 0.5
        pl.add_mesh(box(d, d + 1.0, y0 - 0.04, y0, 0, 2.2), color="#4a3426")
        pl.add_mesh(box(x, x + w - 0.15, y0 - 0.3, y0, h - 0.25, h), color="#3e3e3e")
        x += w


def add_tree(pl, x, y, z, s=1.0):
    pl.add_mesh(pv.Cylinder(center=(x, y, z + 1.3 * s), direction=(0, 0, 1), radius=0.12 * s, height=2.6 * s),
                color="#5b4030")
    for dx, dy, dz, r in ((0, 0, 3.2, 1.3), (0.5, 0.3, 2.8, 0.9), (-0.5, -0.2, 2.9, 0.95)):
        pl.add_mesh(pv.Sphere(radius=r * s, center=(x + dx * s, y + dy * s, z + dz * s)), color="#4f7a3a",
                    smooth_shading=True, ambient=0.2)


def add_lamp(pl, x, y, z):
    pl.add_mesh(pv.Cylinder(center=(x, y, z + 2.4), direction=(0, 0, 1), radius=0.06, height=4.8), color="#3a3f44")
    pl.add_mesh(pv.Cylinder(center=(x, y - 0.5, z + 4.75), direction=(0, 1, 0), radius=0.04, height=1.0), color="#3a3f44")
    pl.add_mesh(box(x - 0.18, x + 0.18, y - 1.15, y - 0.85, z + 4.62, z + 4.75), color="#e8e2c0", ambient=0.6)


def add_car(pl, x, y, z, col):
    pl.add_mesh(box(x - 2.2, x + 2.2, y - 0.85, y + 0.85, z + 0.3, z + 1.0), color=col, specular=0.6)
    pl.add_mesh(box(x - 1.2, x + 1.0, y - 0.78, y + 0.78, z + 1.0, z + 1.5), color="#2a3440", specular=0.9)
    for dx in (-1.4, 1.4):
        for dy in (-0.85, 0.85):
            pl.add_mesh(pv.Cylinder(center=(x + dx, y + dy, z + 0.33), direction=(0, 1, 0), radius=0.33, height=0.22),
                        color="#151515")


def add_cone(pl, x, y, z):
    pl.add_mesh(box(x - 0.2, x + 0.2, y - 0.2, y + 0.2, z, z + 0.04), color="#1a1a1a")
    pl.add_mesh(pv.Cone(center=(x, y, z + 0.38), direction=(0, 0, 1), height=0.7, radius=0.16, resolution=24),
                color="#ff6a00", smooth_shading=True)
    pl.add_mesh(pv.Cylinder(center=(x, y, z + 0.42), direction=(0, 0, 1), radius=0.085, height=0.1), color="#f2f2f2")


def add_barriers(pl, posts, H_):
    posts = sorted(posts)
    for (x, y) in posts:
        z = float(H_(x, y))
        pl.add_mesh(box(x - 0.025, x + 0.025, y - 0.025, y + 0.025, z, z + 1.0), color="#9a9a9a")
        pl.add_mesh(box(x - 0.08, x + 0.08, y - 0.18, y + 0.18, z, z + 0.05), color="#2d2d2d")
    for (xa, ya), (xb, yb) in zip(posts[:-1], posts[1:]):
        z = float(H_(0.5 * (xa + xb), ya))
        n = 4
        for i in range(n):
            u0, u1 = xa + (xb - xa) * i / n, xa + (xb - xa) * (i + 1) / n
            c = "#ff6a00" if i % 2 == 0 else "#f4f4f4"
            for z0, z1 in ((0.72, 0.92), (0.38, 0.52)):
                pl.add_mesh(box(u0, u1, ya - 0.015, ya + 0.015, z + z0, z + z1), color=c)


def add_sign(pl, x, y, z, text):
    pl.add_mesh(box(x - 0.45, x + 0.45, y - 0.03, y + 0.03, z + 0.5, z + 1.25), color="#ff8c1a")
    pl.add_mesh(box(x - 0.03, x + 0.03, y - 0.25, y + 0.25, z, z + 0.55), color="#555555")
    words = text.split(" ")
    lines = [" ".join(words[:2]), " ".join(words[2:])]
    for i, ln in enumerate(lines):
        t = pv.Text3D(ln, depth=0.005)
        b = t.bounds
        sc = 0.8 / max(b[1] - b[0], 1e-6)
        t = t.scale([sc, sc, 1], inplace=False)
        b = t.bounds
        t = t.translate([-(b[0] + b[1]) / 2, 0, 0], inplace=False).rotate_x(90, inplace=False)
        t = t.translate([x, y - 0.035, z + 1.02 - i * 0.22], inplace=False)
        pl.add_mesh(t, color="#111111")


# ----------------------------------------------------------------------------------------------- people
class Person:
    def __init__(self, pl, style):
        skin = "#c99a7a"
        if style == "worker":
            shirt, pants, hat, vest = "#7a7f86", "#2b3a55", "#ffd000", "#ff7a00"
        elif style == "ped1":
            shirt, pants, hat, vest = "#3c6e9e", "#33302d", None, None
        else:
            shirt, pants, hat, vest = "#9e3c5a", "#5a5a66", None, None
        P = []  # (part, mesh, colour, pivot)
        for s in (1, -1):
            leg = merged([pv.Cylinder(center=(0, 0.1 * s, 0.45), direction=(0, 0, 1), radius=0.075, height=0.86),
                          box(-0.06, 0.16, 0.1 * s - 0.055, 0.1 * s + 0.055, 0.0, 0.07)])
            P.append(("legL" if s > 0 else "legR", leg, pants, (0, 0, 0.88)))
            arm = merged([pv.Cylinder(center=(0, 0.26 * s, 1.14), direction=(0, 0, 1), radius=0.05, height=0.6),
                          pv.Sphere(radius=0.055, center=(0, 0.26 * s, 0.83))])
            P.append(("armL" if s > 0 else "armR", arm, shirt, (0, 0.26 * s, 1.43)))
        torso = [box(-0.11, 0.11, -0.2, 0.2, 0.86, 1.46)]
        P.append(("torso", merged(torso), shirt, (0, 0, 0.88)))
        P.append(("head", merged([pv.Sphere(radius=0.11, center=(0.01, 0, 1.6)),
                                  pv.Cylinder(center=(0, 0, 1.49), direction=(0, 0, 1), radius=0.05, height=0.06)]),
                  skin, (0, 0, 0.88)))
        if vest:
            P.append(("vest", box(-0.125, 0.125, -0.215, 0.215, 0.98, 1.44), vest, (0, 0, 0.88)))
            P.append(("stripe", merged([box(-0.13, 0.13, -0.22, 0.22, 1.10, 1.13), box(-0.13, 0.13, -0.22, 0.22, 1.25, 1.28)]),
                      "#e6e6e6", (0, 0, 0.88)))
        if hat:
            P.append(("hat", merged([pv.Sphere(radius=0.125, center=(0.01, 0, 1.64), start_phi=0, end_phi=90),
                                     pv.Disc(center=(0.03, 0, 1.64), inner=0, outer=0.16, c_res=24)]), hat, (0, 0, 0.88)))
        else:
            P.append(("hair", pv.Sphere(radius=0.115, center=(-0.005, 0, 1.62), start_phi=0, end_phi=80), "#2a211c",
                      (0, 0, 0.88)))
        self.shovel = None
        if style == "worker":
            sh = merged([pv.Cylinder(center=(0.48, 0, 0.62), direction=(0.35, 0, -0.94), radius=0.018, height=1.05),
                         box(0.6, 0.78, -0.1, 0.1, 0.0, 0.02)])
            P.append(("shovel", sh, "#6b4a2b", (0, 0, 0.88)))
        self.parts = [(n, pl.add_mesh(m, color=c, smooth_shading=True), np.asarray(pv_)) for n, m, c, pv_ in P]
        self.has_shovel = style == "worker"

    @staticmethod
    def _rot_y(a, pivot):
        c, s = math.cos(a), math.sin(a)
        R = np.array([[c, 0, s, 0], [0, 1, 0, 0], [-s, 0, c, 0], [0, 0, 0, 1]])
        T, Ti = np.eye(4), np.eye(4)
        T[:3, 3], Ti[:3, 3] = pivot, -pivot
        return T @ R @ Ti

    def pose(self, x, y, z, yaw, phase, walking, mode="idle", t=0.0, visible=True):
        Mb = np.eye(4)
        c, s = math.cos(yaw), math.sin(yaw)
        Mb[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
        Mb[:3, 3] = (x, y, z)
        swing = 0.45 * math.sin(phase) if walking else 0.0
        lean = 0.0
        arm_fwd = 0.0
        if mode == "dig":
            lean = 0.35 + 0.12 * math.sin(1.6 * t)
            arm_fwd = -1.1
        for name, act, pivot in self.parts:
            act.visibility = visible and (name != "shovel" or mode == "dig")
            if name == "legL":
                M = self._rot_y(-swing, pivot)
            elif name == "legR":
                M = self._rot_y(swing, pivot)
            elif name in ("armL", "armR"):
                a = (0.6 * swing if name == "armL" else -0.6 * swing) + arm_fwd
                M = self._rot_y(lean, np.array([0, 0, 0.88])) @ self._rot_y(a, pivot)
            elif name == "shovel":
                M = self._rot_y(lean * 0.5, np.array([0, 0, 0.0]))
            else:
                M = self._rot_y(lean, pivot)
            act.user_matrix = Mb @ M


# ----------------------------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mission", default="outputs/demo_mission/isaac/approach_posture_tile0.json")
    ap.add_argument("--compare", nargs="*", default=[], help="label=mission.json (same tile, other strategies)")
    ap.add_argument("--meshes", default="outputs/go2_meshes.npz")
    ap.add_argument("--out", default="outputs/videos/16_demo_mission.mp4")
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--fast", type=float, default=3.0)
    ap.add_argument("--still", type=float, nargs="*")
    ap.add_argument("--grid", type=float, default=0.025)
    ap.add_argument("--timeline-out", default=None, help="write the camera path per rendered view (for Isaac RTX)")
    ap.add_argument("--view-dir", default=None, help="use Isaac RTX frames <dir>/<index>.png as the 3D view")
    ap.add_argument("--title", default="Autonomous sidewalk inspection mission: Unitree Go2 (Isaac Lab physics)")
    a = ap.parse_args()

    r = json.loads(resolve_path(a.mission).read_text())
    ROBUST["on"] = bool(r.get("robust_memory"))
    comps = [(lab, json.loads(resolve_path(f).read_text())) for lab, f in (c.split("=", 1) for c in a.compare)]
    p = params_with_defaults(json.loads(resolve_path(r["tiles_file"]).read_text())[r["tile"]])
    site = p.get("site", {})
    L, half = float(p["size"][0]), 0.5 * float(p["width"])
    H_ = lambda x, y: sidewalk_height(np.asarray(x), np.asarray(y), p)  # noqa: E731
    links, traj = r["links"], np.asarray(r["trajectory"])
    T = r["time_s"]
    dt = T / max(len(traj) - 1, 1)
    idx = {n: i for i, n in enumerate(r["body_names"])}
    base_i = idx["base"]
    events = r["events"]
    track = r.get("site_track") or []
    plog = r.get("posture_log") or []
    sc = r["score"]
    gt_of = {d["finding"]: d for d in sc["per_defect"] if d.get("finding")}
    findings = r["memory"]["findings"]
    start_x, goal_x = -L / 2 + 1.0, L / 2 - 1.5
    edges = np.linspace(start_x, goal_x, int(p.get("n_waypoints", 5)) + 1)
    blocks = p.get("blocks") or [f"Block {i + 1}" for i in range(len(edges) - 1)]
    zone = (site.get("zones") or [None])[0]

    # ---------------- 3D scene
    pl = pv.Plotter(off_screen=True, window_size=(VW, VH))
    pl.set_background("#e4ebf2", top="#7f9dbd")
    g = a.grid
    xs = np.arange(-L / 2, L / 2 + 1e-9, g)
    ys = np.arange(-3.5, 3.5 + 1e-9, g)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    Z = sidewalk_height(X, Y, p, grid=g)
    rgb = np.asarray(sidewalk_albedo(X, Y, p, seed=7), dtype=np.float64)
    rgb = rgb * 255 if rgb.max() <= 1.0 else rgb
    grid = pv.StructuredGrid(X, Y, Z)
    grid["rgb"] = rgb.reshape(-1, 3, order="F").astype(np.uint8)
    pl.add_mesh(grid, scalars="rgb", rgb=True, smooth_shading=True, ambient=0.3, diffuse=0.75, specular=0.04)
    # beyond the tile: more road / verge so the horizon is not empty
    pl.add_mesh(box(-L / 2 - 40, L / 2 + 40, -12, -3.5, -0.2, -0.15), color="#4a4c50")
    pl.add_mesh(box(-L / 2 - 40, L / 2 + 40, 3.5, 3.4 + 0.0001 + 6, -0.02, 0.0), color="#6f8f4f")
    rng = np.random.default_rng(3)
    add_buildings(pl, L, rng)
    for x in np.arange(-L / 2 + 3, L / 2, 9.0):
        if zone and zone["x"][0] - 1 < x < zone["x"][1] + 1:
            continue
        add_tree(pl, x + 1.5, 2.45, float(H_(x, 2.45)), float(rng.uniform(0.85, 1.1)))
    for x in (-20.0, -8.5, 8.0, 20.0):
        add_lamp(pl, x, 2.0, float(H_(x, 2.0)))
    add_car(pl, 13.0, -2.75, -0.15, "#9b1c1c")
    add_car(pl, -16.0, -2.75, -0.15, "#2f5d8a")
    for d in p["defects"]:
        if d["type"] == "puddle":
            t = np.linspace(0, 2 * np.pi, 60)
            c, s = np.cos(d["angle"]), np.sin(d["angle"])
            ex, ey = d["rx"] * np.cos(t), d["ry"] * np.sin(t)
            px, py = d["x"] + c * ex - s * ey, d["y"] + s * ex + c * ey
            z0 = float(np.max(H_(px, py))) - 0.002
            pl.add_mesh(pv.PolyData(np.c_[px, py, np.full_like(px, z0)], faces=np.r_[len(px), np.arange(len(px))]),
                        color="#3b6f99", opacity=0.7, specular=0.9, specular_power=40)
        elif d["type"] == "obstruction":
            z0 = float(H_(d["x"], d["y"]))
            pl.add_mesh(pv.Cylinder(center=(d["x"], d["y"], z0 + 3.5), direction=(0, 0, 1), radius=max(d["r"], 0.1),
                                    height=7.0), color="#6b5640")
        if d.get("tree") and d["tree"][1] > half:
            add_tree(pl, d["tree"][0], d["tree"][1], float(H_(*d["tree"])), 1.1)
    posts = [tuple(o["xy"]) for o in site.get("objects", []) if o["kind"] == "barrier"]
    if posts:
        add_barriers(pl, posts, H_)
    people_static, movers = [], []
    for o in site.get("objects", []):
        if "path" in o:
            movers.append((o, Person(pl, "worker" if o["kind"] == "worker" else ("ped1" if len(movers) % 2 else "ped2"))))
            continue
        x, y = o["xy"]
        z = float(H_(x, y))
        if o["kind"] == "cone":
            add_cone(pl, x, y, z)
        elif o["kind"] == "worker":
            people_static.append((o, Person(pl, "worker")))
        elif o["kind"] == "pile":
            pl.add_mesh(pv.ParametricEllipsoid(0.75, 0.45, 0.38).translate([x, y, z], inplace=False), color="#7a5a3c",
                        smooth_shading=True)
        elif o["kind"] == "generator":
            pl.add_mesh(box(x - 0.4, x + 0.4, y - 0.25, y + 0.25, z, z + 0.55), color="#f2c200")
            pl.add_mesh(box(x - 0.3, x + 0.1, y - 0.26, y - 0.25, z + 0.15, z + 0.42), color="#222222")
        elif o["kind"] == "sign":
            add_sign(pl, x, y, z, o.get("text", "SIDEWALK CLOSED"))
    # walking phase per mover
    phases = []
    for j in range(len(movers)):
        ph, acc = [0.0], 0.0
        for k in range(1, len(track)):
            acc += math.hypot(track[k][j][0] - track[k - 1][j][0], track[k][j][1] - track[k - 1][j][1])
            ph.append(2 * math.pi * acc / 1.3)
        phases.append(ph)

    meshes = load_meshes(a.meshes)
    robot = [(idx[l], pl.add_mesh(m, color=c, smooth_shading=True, specular=0.35, specular_power=20))
             for l, m, c in meshes if l in idx]
    trail = pv.PolyData(np.zeros((2, 3)), lines=[2, 0, 1])
    pl.add_mesh(trail, color="#f08c00", line_width=4, opacity=0.85)
    z555 = pv.PolyData(np.zeros((4, 3)), faces=[4, 0, 1, 2, 3])
    pl.add_mesh(z555, color="#ffffff", opacity=0.18)
    z405 = pv.PolyData(np.zeros((4, 3)), faces=[4, 0, 1, 2, 3])
    z405_act = pl.add_mesh(z405, color="#ffd23f", opacity=0.4)
    rays = pv.PolyData(np.zeros((5, 3)), lines=np.hstack([[2, 0, k] for k in range(1, 5)]))
    rays_act = pl.add_mesh(rays, color="#ffd23f", line_width=2, opacity=0.85)
    fmark = {}
    for fid, f in findings.items():
        lab, col = KIND.get(f["kind"], (f["kind"], (255, 255, 255)))
        cx, cy = f["xy"]
        ang = np.linspace(0, 2 * np.pi, 12, endpoint=False)
        z0 = float(np.median(H_(cx + 0.45 * np.cos(ang), cy + 0.45 * np.sin(ang))))
        obs = sorted(f.get("observations", []), key=lambda o: o["t"])
        det = [o["t"] for o in obs if o.get("provenance") == "detected"]
        t_show = det[1] if len(det) > 1 else (det[0] if det else f.get("first_seen_t", 0.0))
        hexc = "#%02x%02x%02x" % col
        fmark[fid] = dict(f=f, lab=lab.split(" ")[0], col=col, t=t_show, top=(cx, cy, z0 + 0.5), xy=(cx, cy),
                          ring=pl.add_mesh(pv.Disc(center=(cx, cy, z0 + 0.012), inner=0.2, outer=0.25, c_res=48), color=hexc),
                          pole=pl.add_mesh(pv.Line((cx, cy, z0), (cx, cy, z0 + 0.47)), color=hexc, line_width=3))
    label_act = [None]
    pl.enable_lightkit()
    spec = SENSOR_SPECS["depth_d405"]

    # ---------------- timeline helpers
    def k_at(t):
        return min(int(round(min(t, T) / dt)), len(links) - 1)

    def ev_window(name_on, name_off, t):
        on = [e["t"] for e in events if e["event"] == name_on and e["t"] <= t]
        off = [e["t"] for e in events if e["event"] == name_off and e["t"] <= t]
        return bool(on) and (not off or on[-1] > off[-1])

    def posture_at(t):
        k = k_at(t)
        return plog[min(k, len(plog) - 1)] if plog else 0.0

    def state(t):
        """(action key, banner text, colour)."""
        if t >= T:
            return "walk", "MISSION COMPLETE: goal reached", GREEN
        k = k_at(t)
        if ev_window("yield_pedestrian", "resume_after_yield", t):
            return "yield", "YIELDING: pedestrian crossing ahead, waiting", AMBER
        if posture_at(t) > 0.005:
            return "crouch", "CROUCHED + BODY LEVELLED: D405 close-up from ~30 cm", AMBER
        rec = [e for e in events if e["t"] <= t]
        if rec and rec[-1]["event"].startswith("replan") and t - rec[-1]["t"] < 2.5:
            return "reroute", "REROUTING around a confirmed hole", AMBER
        tgt = links[k].get("target")
        if tgt is not None:
            k2 = min(k + 1, len(traj) - 1)
            v = np.linalg.norm(traj[k2, :2] - traj[k, :2]) / dt
            w = abs((traj[k2, 2] - traj[k, 2] + np.pi) % (2 * np.pi) - np.pi) / dt
            if v < 0.06 and w > 0.15:
                return "turn", "TURNING to face the defect", AMBER
            if v < 0.06:
                return "stop", "STOPPED at the defect", AMBER
            return "stop", "APPROACHING a defect for a close look", AMBER
        x = traj[k, 0]
        if zone and zone["x"][0] - 1.5 < x < zone["x"][1] + 1.0:
            return "reroute", "WORK ZONE: sidewalk half closed, keeping to the open lane", AMBER
        if rec and rec[-1]["event"] == "inspected" and t - rec[-1]["t"] < 2.0:
            return "walk", "INSPECTED: finding logged, resuming route", GREEN
        return "walk", "WALKING the route, D555 scanning ahead", WHITE

    def slow_now(t):
        k = k_at(t)
        return (links[k].get("target") is not None or posture_at(t) > 0.001
                or ev_window("yield_pedestrian", "resume_after_yield", t + 1.0)
                or ev_window("yield_pedestrian", "resume_after_yield", t))

    # ---------------- per-frame 3D
    cam = {"eye": None, "foc": None}
    views = []  # camera per rendered view, in order (timeline for the Isaac RTX renderer)

    def geom(t):
        """Robot / sensor geometry and the (smoothed) camera for mission time t; one call per rendered view."""
        k = k_at(t)
        ln = links[k]
        pos, quat = np.asarray(ln["pos"]), ln["quat"]
        R = quat_R(quat[base_i])
        pose = traj[k]
        hd = np.array([math.cos(pose[2]), math.sin(pose[2])])
        lf = np.array([-hd[1], hd[0]])
        camp = pos[base_i] + R @ np.asarray(spec["mount_pos"])
        ax = R @ pitch_matrix(spec["pitch_deg"]) @ np.array([1.0, 0.0, 0.0])
        gz = float(H_(pos[base_i][0], pos[base_i][1]))
        sray = (gz - camp[2]) / min(ax[2], -0.05)
        hit = camp + sray * ax
        c405 = float((hit[:2] - pos[base_i][:2]) @ hd)
        key, _, _ = state(t)
        rb = np.array([pose[0], pose[1], gz])
        right = np.r_[-lf, 0.0]
        h3 = np.r_[hd, 0.0]
        if key == "crouch":
            foc = rb + 0.45 * h3 + [0, 0, 0.18]
            eye = foc + 1.55 * right - 0.35 * h3 + [0, 0, 0.45]
        elif key == "yield":
            foc = rb + 1.4 * h3 + [0, 0, 0.5]
            eye = foc - 3.8 * h3 + 2.6 * right + [0, 0, 2.0]
        elif key in ("stop", "turn"):
            foc = rb + 0.6 * h3 + [0, 0, 0.1]
            eye = foc - 1.9 * h3 + 1.3 * right + [0, 0, 1.15]
        else:
            foc = rb + 1.1 * h3 + [0, 0, 0.1]
            eye = foc - 3.4 * h3 + 2.1 * right + [0, 0, 1.8]
        if cam["eye"] is None:
            cam["eye"], cam["foc"] = eye, foc
        cam["eye"] = cam["eye"] + 0.07 * (eye - cam["eye"])
        cam["foc"] = cam["foc"] + 0.12 * (foc - cam["foc"])
        views.append({"t": float(t), "eye": [float(v) for v in cam["eye"]], "foc": [float(v) for v in cam["foc"]]})
        return dict(k=k, ln=ln, pos=pos, quat=quat, R=R, pose=pose, hd=hd, lf=lf, camp=camp, hit=hit, c405=c405,
                    gz=gz, eye=cam["eye"].copy(), foc=cam["foc"].copy(), idx=len(views) - 1,
                    rng405=float(np.linalg.norm(hit - camp)))

    def quad_at(g_, x0, x1, hw):
        q = [g_["pose"][:2] + g_["hd"] * u + g_["lf"] * v for u, v in ((x0, -hw), (x1, -hw), (x1, hw), (x0, hw))]
        q = np.asarray(q)
        return np.c_[q, H_(q[:, 0], q[:, 1]) + 0.006]

    def project(P, eye, foc):
        """World points [n, 3] -> pixel coords in the 3D view (VTK-style 30 deg vertical field of view)."""
        f_ = (foc - eye) / np.linalg.norm(foc - eye)
        r_ = np.cross(f_, [0.0, 0.0, 1.0])
        r_ /= np.linalg.norm(r_)
        u_ = np.cross(r_, f_)
        d_ = np.asarray(P, dtype=np.float64).reshape(-1, 3) - eye
        x, y, z = d_ @ r_, d_ @ u_, d_ @ f_
        fpx = (VH / 2) / math.tan(math.radians(15.0))
        ok = z > 0.1
        z = np.where(ok, z, 1.0)
        return np.stack([VW / 2 + fpx * x / z, VH / 2 - fpx * y / z], 1), ok

    def isaac_view(t):
        """The Isaac RTX frame for this view, with the trail, sensor zones and finding labels drawn on it."""
        g_ = geom(t)
        fp = Path(a.view_dir) / f"{g_['idx']:05d}.png"
        im = Image.open(fp).convert("RGB") if fp.exists() else Image.new("RGB", (VW, VH), (40, 40, 40))
        im = im.resize((VW, VH)) if im.size != (VW, VH) else im
        ov = Image.new("RGBA", (VW, VH), (0, 0, 0, 0))
        d_ = ImageDraw.Draw(ov)
        eye, foc = g_["eye"], g_["foc"]

        def poly(P, fill=None, outline=None, width=2):
            uv, ok = project(P, eye, foc)
            if ok.all():
                pts = [tuple(v) for v in uv]
                if fill:
                    d_.polygon(pts, fill=fill)
                if outline:
                    d_.line(pts + [pts[0]], fill=outline, width=width)

        k = g_["k"]
        tr = traj[max(0, k - 400): k + 1]
        if len(tr) >= 2:
            P = np.c_[tr[:, 0], tr[:, 1], H_(tr[:, 0], tr[:, 1]) + 0.01]
            uv, ok = project(P, eye, foc)
            seg = [tuple(v) for v, o in zip(uv, ok) if o]
            if len(seg) >= 2:
                d_.line(seg, fill=(240, 140, 0, 200), width=4)
        crouch = posture_at(t) > 0.005
        insp = g_["ln"].get("target") is not None
        poly(quad_at(g_, 0.6, 1.2, 0.6), fill=(255, 255, 255, 38))
        q4 = quad_at(g_, g_["c405"] - 0.1, g_["c405"] + 0.1, 0.12 if crouch else 0.17)
        poly(q4, fill=(255, 210, 63, 150 if (crouch or insp) else 70), outline=(255, 210, 63, 220))
        if crouch or insp:
            uv, ok = project(np.vstack([g_["camp"], q4]), eye, foc)
            if ok.all():
                for v in uv[1:]:
                    d_.line([tuple(uv[0]), tuple(v)], fill=(255, 210, 63, 170), width=2)
        labels = []
        for fid, m in fmark.items():
            if t < m["t"] or np.hypot(m["xy"][0] - g_["pose"][0], m["xy"][1] - g_["pose"][1]) > 9:
                continue
            cx, cy = m["xy"]
            ang = np.linspace(0, 2 * np.pi, 40, endpoint=False)
            z0 = m["top"][2] - 0.5
            col = m["col"]
            for rr_, wid in ((0.25, 3), (0.2, 2)):
                P = np.c_[cx + rr_ * np.cos(ang), cy + rr_ * np.sin(ang), np.full(40, z0 + 0.012)]
                uv, ok = project(P, eye, foc)
                if ok.all():
                    pts = [tuple(v) for v in uv]
                    d_.line(pts + [pts[0]], fill=col + (230,), width=wid)
            uv, ok = project(np.array([[cx, cy, z0], m["top"]]), eye, foc)
            if ok.all():
                d_.line([tuple(uv[0]), tuple(uv[1])], fill=col + (230,), width=3)
                val, sensor = running_value(m["f"], t)
                gt = gt_of.get(fid)
                if m["f"]["kind"] == "dropout":
                    txt = "possible water"
                elif val is None:
                    txt = f"{m['lab']}: not measured yet"
                else:
                    txt = f"{m['lab']} {abs(val) * 1000:.1f} mm ({sensor.split('_')[-1].upper()})"
                    txt += f"\ntrue {abs(gt['true_value_m']) * 1000:.1f} mm" if gt else "\n(false finding)"
                labels.append((tuple(uv[1]), txt))
        img_ = Image.alpha_composite(im.convert("RGBA"), ov)
        d2 = ImageDraw.Draw(img_)
        fl = font(16, True)
        for (lx, ly), txt in labels:
            bb = d2.multiline_textbbox((lx, ly), txt, font=fl, spacing=2)
            w_, h_ = bb[2] - bb[0], bb[3] - bb[1]
            x0_, y0_ = lx - 4, ly - h_ - 10
            d2.rectangle([x0_, y0_, x0_ + w_ + 10, y0_ + h_ + 8], fill=(16, 20, 24, 200))
            d2.multiline_text((x0_ + 5, y0_ + 3), txt, font=fl, fill=(255, 255, 255), spacing=2)
        return np.asarray(img_.convert("RGB")), g_["c405"], g_["rng405"]

    def render3d(t):
        if a.view_dir:
            return isaac_view(t)
        g_ = geom(t)
        if a.timeline_out:
            return None, g_["c405"], g_["rng405"]
        k, ln, pos, quat, R = g_["k"], g_["ln"], g_["pos"], g_["quat"], g_["R"]
        for i, act in robot:
            M = np.eye(4)
            M[:3, :3], M[:3, 3] = quat_R(quat[i]), pos[i]
            act.user_matrix = M
        tr = traj[: k + 1]
        if len(tr) >= 2:
            pts = np.c_[tr[:, 0], tr[:, 1], H_(tr[:, 0], tr[:, 1]) + 0.01]
            trail.points = pts
            trail.lines = np.hstack([[2, i, i + 1] for i in range(len(pts) - 1)])
        pose = traj[k]
        hd = np.array([math.cos(pose[2]), math.sin(pose[2])])
        lf = np.array([-hd[1], hd[0]])

        def quad(x0, x1, hw):
            q = [pose[:2] + hd * u + lf * v for u, v in ((x0, -hw), (x1, -hw), (x1, hw), (x0, hw))]
            q = np.asarray(q)
            return np.c_[q, H_(q[:, 0], q[:, 1]) + 0.006]

        z555.points = quad(0.6, 1.2, 0.6)
        camp = pos[base_i] + R @ np.asarray(spec["mount_pos"])
        ax = R @ pitch_matrix(spec["pitch_deg"]) @ np.array([1.0, 0.0, 0.0])
        gz = float(H_(pos[base_i][0], pos[base_i][1]))
        sray = (gz - camp[2]) / min(ax[2], -0.05)
        hit = camp + sray * ax
        c405 = float((hit[:2] - pos[base_i][:2]) @ hd)
        crouch = posture_at(t) > 0.005
        q4 = quad(c405 - 0.1, c405 + 0.1, 0.17 if not crouch else 0.12)
        z405.points = q4
        insp = ln.get("target") is not None
        rays.points = np.vstack([camp, q4])
        rays_act.visibility = insp or crouch
        z405_act.prop.opacity = 0.75 if crouch else (0.55 if insp else 0.28)
        # people
        for o, per in people_static:
            x, y = o["xy"]
            per.pose(x, y, float(H_(x, y)), o.get("yaw", 0.0), 0.0, False, o.get("pose", "idle"), t)
        if track:
            kk = min(k, len(track) - 1)
            for j, (o, per) in enumerate(movers):
                x, y, yaw, walking = track[kk][j]
                per.pose(x, y, float(H_(x, y)), yaw, phases[j][kk], bool(walking), "walk", t)
        # findings
        lp, lt = [], []
        for fid, m in fmark.items():
            vis = t >= m["t"]
            m["ring"].visibility = m["pole"].visibility = vis
            if not vis or np.hypot(m["xy"][0] - pose[0], m["xy"][1] - pose[1]) > 9:
                continue
            val, sensor = running_value(m["f"], t)
            gt = gt_of.get(fid)
            if m["f"]["kind"] == "dropout":
                txt = "possible water"
            elif val is None:
                txt = f"{m['lab']}: not measured yet"
            else:
                txt = f"{m['lab']} {abs(val) * 1000:.1f} mm ({sensor.split('_')[-1].upper()})"
                txt += f"\ntrue {abs(gt['true_value_m']) * 1000:.1f} mm" if gt else "\n(false finding)"
            lp.append(m["top"])
            lt.append(txt)
        if label_act[0] is not None:
            pl.remove_actor(label_act[0])
            label_act[0] = None
        if lp:
            label_act[0] = pl.add_point_labels(np.asarray(lp), lt, font_size=15, point_size=1, shape_opacity=0.75,
                                               shape_color="#101418", text_color="white", always_visible=True,
                                               show_points=False, margin=3)
        c = pl.camera
        c.position, c.focal_point, c.up = tuple(cam["eye"]), tuple(cam["foc"]), (0.0, 0.0, 1.0)
        pl.renderer.ResetCameraClippingRange()
        pl.render()
        return np.asarray(pl.screenshot(return_img=True)), c405, float(np.linalg.norm(hit - camp))

    # ---------------- minimap base
    mw, mh = W - 40, int((W - 40) * 7.0 / L)
    mh = min(mh, H - MAP_Y - 34)
    base_img = Image.fromarray(np.ascontiguousarray(np.flipud(rgb.astype(np.uint8).transpose(1, 0, 2)))).resize((mw, mh))

    def m_xy(x, y):
        return 20 + (x + L / 2) / L * mw, MAP_Y + 26 + (3.5 - y) / 7.0 * mh

    # ---------------- live plot of the measurement in progress
    def active_finding(t):
        k = k_at(t)
        tgt = links[k].get("target")
        if tgt in findings:
            return tgt
        last = [(o["t"], fid) for fid, f in findings.items() for o in f["observations"]
                if o.get("value_m") is not None and o["t"] <= t and f["kind"] != "dropout"]
        return max(last)[1] if last else None

    def crouch_t(tt):
        return posture_at(tt) > 0.03

    def live_plot(t, w, h):
        fid = active_finding(t)
        fig, ax = plt.subplots(figsize=(w / 100, h / 100), dpi=100)
        fig.patch.set_facecolor("#14181c")
        ax.set_facecolor("#1b2026")
        for s_ in ax.spines.values():
            s_.set_color("#3a424a")
        ax.tick_params(colors="#a0acb8", labelsize=8)
        if fid is None:
            ax.text(0.5, 0.5, "no measurement yet", color="#a0acb8", ha="center", va="center", transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            f = findings[fid]
            gt = gt_of.get(fid)
            obs = [o for o in f["observations"] if o.get("value_m") is not None and o["t"] <= t]
            if obs:
                t0 = obs[0]["t"] - 0.5
                for o in obs:
                    v = abs(o["value_m"]) * 1000
                    if "d405" in o["sensor"]:
                        if crouch_t(o["t"]):
                            ax.scatter(o["t"], v, marker="*", s=70, color="#ffd23f", zorder=4)
                        else:
                            ax.scatter(o["t"], v, marker="D", s=22, color="#f08c00", zorder=3)
                    else:
                        ax.scatter(o["t"], v, marker="o", s=18, facecolor="none", edgecolor="#9aa5b1", zorder=3)
                ts = sorted({o["t"] for o in obs})
                run = [abs(running_value(f, x)[0]) * 1000 for x in ts]
                ax.step(ts + [t], run + [run[-1]], where="post", color="#f08c00", lw=2.2, zorder=5)
                vals = [abs(o["value_m"]) * 1000 for o in obs]
                lo, hi = min(vals), max(vals)
                if gt:
                    tv = abs(gt["true_value_m"]) * 1000
                    ax.axhspan(tv - 2, tv + 2, color="#2b8a3e", alpha=0.3, lw=0)
                    ax.axhline(tv, color="#e8ecf0", ls="--", lw=1.2)
                    lo, hi = min(lo, tv), max(hi, tv)
                pad = max(4.0, 0.15 * (hi - lo))
                ax.set_ylim(max(0, lo - pad), hi + pad)
                ax.set_xlim(t0, max(t + 0.5, t0 + 4))
                name = (gt or {}).get("type") or KIND.get(f["kind"], (f["kind"],))[0].split(" ")[0]
                err = f" | error {run[-1] - abs(gt['true_value_m']) * 1000:+.1f} mm" if gt else ""
                ax.set_title(f"{name}: estimate {run[-1]:.1f} mm" + (f", true {abs(gt['true_value_m']) * 1000:.1f}" if gt
                                                                       else "") + err,
                             color="#e8ecf0", fontsize=10, loc="left")
            ax.grid(alpha=0.12)
        fig.tight_layout(pad=0.4)
        fig.canvas.draw()
        img = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
        plt.close(fig)
        return Image.fromarray(img)

    # ---------------- HUD
    F = {k: font(s, b) for k, (s, b) in dict(title=(28, True), h=(19, True), b=(17, False), bb=(17, True),
                                                s=(15, False), big=(30, True), banner=(22, True)).items()}

    def counts(t):
        ev = [e for e in events if e["t"] <= t]
        return (sum(e["event"] == "inspected" for e in ev), sum(e["event"] == "yield_pedestrian" for e in ev),
                sum(e["event"].startswith("replan_around") for e in ev),
                sum(1 for fid, m in fmark.items() if t >= m["t"]))

    def frame(t, speed, overlay=None):
        view, c405, rng405 = render3d(t)
        if view is None:  # --timeline-out: only the camera path is wanted
            return None
        img = Image.new("RGB", (W, H), (20, 24, 28))
        d = ImageDraw.Draw(img, "RGBA")
        img.paste(Image.fromarray(view), (0, TOP))
        d.text((18, 10), a.title, font=F["title"], fill=WHITE)
        d.text((W - 360, 8), f"mission time {min(t, T):6.1f} s", font=F["h"], fill=WHITE)
        d.text((W - 360, 30), "playback 1x (action in progress)" if speed <= 1.01 else f"playback {speed:.1f}x",
               font=F["s"], fill=AMBER if speed <= 1.01 else GREY)
        key, msg, col = state(t)
        d.rectangle([0, TOP + VH - 44, VW, TOP + VH], fill=(16, 20, 24, 215))
        d.text((16, TOP + VH - 36), msg, font=F["banner"], fill=col)
        if posture_at(t) > 0.005:
            d.text((VW - 470, TOP + VH - 34), f"body {-100 * posture_at(t):.0f} cm · D405 {rng405 * 100:.0f} cm from ground",
                   font=F["b"], fill=AMBER)
        # right panel
        x0 = PX0
        y = TOP
        d.rectangle([x0, y, W, TOP + VH], fill=(24, 29, 34))
        k = k_at(t)
        px = traj[k, 0]
        bi = int(np.clip(np.searchsorted(edges, px, side="right") - 1, 0, len(blocks) - 1))
        d.text((x0 + 14, y + 8), "MISSION STATUS", font=F["h"], fill=AMBER)
        prog = float(np.clip((px - start_x) / (goal_x - start_x), 0, 1)) if t < T else 1.0
        d.text((x0 + 14, y + 34), f"{blocks[bi]}  ·  {prog * (goal_x - start_x):4.1f} / {goal_x - start_x:.0f} m",
               font=F["b"], fill=WHITE)
        d.rectangle([x0 + 14, y + 60, W - 16, y + 72], fill=(52, 60, 68))
        d.rectangle([x0 + 14, y + 60, x0 + 14 + prog * (W - 30 - x0), y + 72], fill=(240, 140, 0))
        ni, ny, nr, nf = counts(t)
        d.text((x0 + 14, y + 80), f"findings {nf}   close inspections {ni}   yields {ny}   reroutes {nr}",
               font=F["s"], fill=GREY)
        y += 108
        d.text((x0 + 14, y), "ROBOT ACTIONS", font=F["h"], fill=AMBER)
        y += 26
        for ak, name in ACTIONS:
            on = ak == key and t < T
            if on:
                d.rectangle([x0 + 10, y - 2, W - 12, y + 21], fill=(240, 140, 0, 70))
            if on:
                d.polygon([(x0 + 18, y + 3), (x0 + 18, y + 17), (x0 + 30, y + 10)], fill=(255, 210, 63))
            d.text((x0 + 38, y), name, font=F["bb"] if on else F["b"], fill=WHITE if on else GREY)
            y += 25
        y += 8
        d.text((x0 + 14, y), "FINDINGS LOG  (measured vs. true)", font=F["h"], fill=AMBER)
        y += 26
        rows = []
        for fid, m in fmark.items():
            if t < m["t"]:
                continue
            val, sensor = running_value(m["f"], t)
            gt = gt_of.get(fid)
            rows.append((m["t"], m, val, sensor, gt))
        rows.sort(key=lambda z: z[0])
        for _, m, val, sensor, gt in rows[-8:]:
            name = (gt["type"] if gt else m["lab"])[:9]
            if m["f"]["kind"] == "dropout":
                txt = f"{name:<9} water: no depth returns" + (f" (true {abs(gt['true_value_m']) * 1000:.0f} mm)" if gt else "")
            elif val is None:
                txt = f"{name:<9} detected, not measured"
            else:
                txt = f"{name:<9} {abs(val) * 1000:5.1f} mm"
                txt += f"  true {abs(gt['true_value_m']) * 1000:5.1f}  {abs(val) * 1000 - abs(gt['true_value_m']) * 1000:+5.1f}" if gt \
                    else "  (no true defect)"
                txt += f"  {sensor.split('_')[-1].upper()}"
            d.text((x0 + 18, y), txt, font=mono(14), fill=m["col"] + (255,))
            y += 20
        ph = TOP + VH - (y + 10) - 6
        if ph > 120:
            img.paste(live_plot(t, PW - 20, ph), (x0 + 10, y + 8))
        # minimap
        d.rectangle([0, MAP_Y, W, H], fill=(20, 24, 28))
        d.text((20, MAP_Y + 2), "ROUTE MAP", font=F["h"], fill=AMBER)
        img.paste(base_img, (20, MAP_Y + 26))
        for e in edges:
            ex, _ = m_xy(e, 0)
            d.line([ex, MAP_Y + 26, ex, MAP_Y + 26 + mh], fill=(255, 255, 255, 120), width=1)
        for i, b in enumerate(blocks):
            bx, by_ = m_xy(0.5 * (edges[i] + edges[i + 1]), 3.3)
            tw = d.textlength(b, font=F["bb"])
            d.rectangle([bx - tw / 2 - 6, by_ - 2, bx + tw / 2 + 6, by_ + 21], fill=(16, 20, 24, 170))
            d.text((bx - tw / 2, by_), b, font=F["bb"], fill=(255, 210, 63) if i == bi else WHITE)
        if zone:
            a0, b0 = m_xy(zone["x"][0], zone["y"][1])
            a1, b1 = m_xy(zone["x"][1], zone["y"][0])
            d.rectangle([a0, b0, a1, b1], outline=(255, 120, 0), width=2, fill=(255, 120, 0, 50))
        pts = [m_xy(x, y_) for x, y_ in traj[: k + 1, :2]]
        if len(pts) > 1:
            d.line(pts, fill=(240, 140, 0), width=3)
        for fid, m in fmark.items():
            if t < m["t"]:
                continue
            cx, cy = m_xy(*m["xy"])
            d.ellipse([cx - 5, cy - 5, cx + 5, cy + 5], fill=m["col"] + (255,), outline=(0, 0, 0))
            if any(e["event"] == "inspected" and e.get("finding") == fid and e["t"] <= t for e in events):
                d.ellipse([cx - 9, cy - 9, cx + 9, cy + 9], outline=(255, 210, 63), width=2)
        for o, _ in people_static:
            cx, cy = m_xy(*o["xy"])
            d.ellipse([cx - 4, cy - 4, cx + 4, cy + 4], fill=(255, 122, 0))
        if track:
            kk = min(k, len(track) - 1)
            for j, (o, _) in enumerate(movers):
                cx, cy = m_xy(track[kk][j][0], track[kk][j][1])
                c_ = (255, 122, 0) if o["kind"] == "worker" else (90, 170, 255)
                d.ellipse([cx - 4, cy - 4, cx + 4, cy + 4], fill=c_)
        rx, ry = m_xy(*traj[k, :2])
        th = traj[k, 2]
        tri = [(rx + 11 * math.cos(th), ry - 11 * math.sin(th)), (rx + 7 * math.cos(th + 2.5), ry - 7 * math.sin(th + 2.5)),
               (rx + 7 * math.cos(th - 2.5), ry - 7 * math.sin(th - 2.5))]
        d.polygon(tri, fill=(255, 255, 255), outline=(0, 0, 0))
        lx = 300
        for lab_, c_ in (("robot", (255, 255, 255)), ("hole", KIND["depression"][1]), ("upheaval / step", KIND["edge"][1]),
                         ("water", KIND["dropout"][1]), ("worker", (255, 122, 0)), ("pedestrian", (90, 170, 255)),
                         ("inspected", None), ("work zone", "zone")):
            if c_ is None:
                d.ellipse([lx, MAP_Y + 7, lx + 14, MAP_Y + 21], outline=(255, 210, 63), width=2)
            elif c_ == "zone":
                d.rectangle([lx, MAP_Y + 7, lx + 16, MAP_Y + 21], outline=(255, 120, 0), width=2, fill=(255, 120, 0, 50))
            else:
                d.ellipse([lx + 2, MAP_Y + 9, lx + 12, MAP_Y + 19], fill=c_)
            d.text((lx + 20, MAP_Y + 4), lab_, font=F["s"], fill=GREY)
            lx += 30 + d.textlength(lab_, font=F["s"]) + 14
        if overlay:
            overlay(img)
        return np.asarray(img)

    # ---------------- briefing and report cards
    def briefing(img):
        d = ImageDraw.Draw(img, "RGBA")
        d.rectangle([150, 120, 1190, 720], fill=(14, 18, 22, 245), outline=(240, 140, 0), width=3)
        d.text((180, 140), "MISSION BRIEFING", font=F["big"], fill=(240, 140, 0))
        lines = [f"Route: {goal_x - start_x:.0f} m of sidewalk, {len(blocks)} blocks, start to goal, one pass",
                 "Task: find trip hazards (upheavals, holes, cracks, standing water),",
                 "      measure them and log every finding with its evidence",
                 "Sensors: RealSense D555 (scanning ahead) + D405 (close-up, on the chin)",
                 "Rules: stop and look closely at every confirmed defect;",
                 ("       crouch and level the body for the D405 close-up;" if r.get("posture")
                  else "       stand still facing it for the D405 close-up;"),
                 "       route around confirmed holes and the construction work zone;",
                 "       yield to people; never enter the closed area",
                 "Output: findings log + mission report (coverage, ADA classes, open issues)",
                 "",
                 f"Simulation: Isaac Lab, Go2 rough-terrain walking policy; depth sensors with",
                 f"{r['profile']} error model" + (" + odometry drift + camera latency" if r.get("pose_errors") else "")]
        for i, s in enumerate(lines):
            d.text((180, 200 + 40 * i), s, font=F["h"] if i < 9 else F["b"], fill=WHITE if i < 9 else GREY)

    def report(img):
        d = ImageDraw.Draw(img, "RGBA")
        d.rectangle([40, 70, 1880, 1040], fill=(14, 18, 22, 250), outline=(240, 140, 0), width=3)
        d.text((70, 88), "MISSION REPORT", font=F["big"], fill=(240, 140, 0))
        cov = r["memory"].get("coverage", {}).get("depth_d555") or []
        covm = np.mean([c_["seen"] for c_ in cov]) * 100 if cov else float("nan")
        ni, ny, nr, nf = counts(T + 1)
        d.text((70, 132), f"Outcome: {r['outcome']} in {T:.1f} s · path {r['path_m']:.1f} m · D555 coverage {covm:.0f} % of the walk · "
                          f"{ni} close inspections · {ny} yield(s) · {nr} reroute(s) · {sc['false_findings']} false finding(s)",
               font=F["b"], fill=WHITE)
        runs = [("this mission", r)] + comps
        hdr = f"{'defect':<10}{'true':>9}  {'ADA (true)':<11}" + "".join(f"{lab[:22]:>24}" for lab, _ in runs)
        y = 180
        d.text((70, y), "Ground truth vs. measured (mm; ADA class of the true size: <= 6.4 ok, <= 13 bevel, above: violation)", font=F["h"], fill=AMBER)
        y += 34
        d.text((70, y), hdr, font=mono(17, True), fill=GREY)
        y += 28
        for i, gtd in enumerate(sc["per_defect"]):
            tv = abs(gtd["true_value_m"]) * 1000
            row = f"{gtd['type']:<10}{tv:>7.1f}  " + f"{ada_class(tv) if gtd['type'] in ('upheaval', 'hole') else '-':<11}"
            for _, rr in runs:
                x = rr["score"]["per_defect"][i]
                if x.get("value_m") is not None:
                    v = abs(x["value_m"]) * 1000
                    cell = f"{v:.1f} ({v - tv:+.1f}) {(x.get('best_sensor') or '')[-4:].upper()}"
                elif x["found"]:
                    cell = "found, not measured"
                else:
                    cell = "MISSED"
                row += f"{cell:>24}"
            d.text((70, y), row, font=mono(17), fill=WHITE)
            y += 26
        y += 14
        d.text((70, y), "Per run:", font=F["h"], fill=AMBER)
        y += 30
        for lab, rr in runs:
            s2 = rr["score"]
            ev = rr["events"]
            nf_ = sum(x["found"] for x in s2["per_defect"])
            errs = [abs(abs(x["value_m"]) - abs(x["true_value_m"])) * 1000 for x in s2["per_defect"]
                    if x.get("value_m") is not None and x["type"] in ("upheaval", "hole")]
            d.text((70, y), f"{lab:<24} {rr['outcome']} in {rr['time_s']:5.1f} s · found {nf_}/{len(s2['per_defect'])} · "
                            f"median error {np.median(errs) if errs else float('nan'):.1f} mm, worst {max(errs) if errs else float('nan'):.1f} mm · "
                            f"false findings {s2['false_findings']} · stops {sum(e['event'] == 'inspected' for e in ev)}",
                   font=mono(16), fill=WHITE)
            y += 26
        y += 14
        missed = [x for x in sc["per_defect"] if not x["found"]]
        if missed:
            d.text((70, y), "Open issues: " + "; ".join(f"{x['type']} {abs(x['true_value_m']) * 1000:.0f} mm not detected"
                                                         for x in missed) + "  (revisit with a slower pass)",
                   font=F["b"], fill=RED)
            y += 28
        note = "Cracks are detected as findings but not scored by width yet"
        if site.get("objects"):
            note += ("; people and cones are simulated obstacles (people known from ground truth, standing in for a "
                     "person detector)")
        d.text((70, y), note + ".", font=F["s"], fill=GREY)
        if r.get("score_reference"):  # "true" values measured, not authored (e.g. tiles recreated from a real scan)
            d.text((70, y + 20), "True values: " + r["score_reference"] + ".", font=F["s"], fill=GREY)

    out = resolve_path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if a.still:
        for t in a.still:
            Image.fromarray(frame(t, 1.0)).save(out.with_name(out.stem + f"_t{t:.0f}.png"))
        Image.fromarray(frame(T, 1.0, report)).save(out.with_name(out.stem + "_report.png"))
        return 0
    if a.timeline_out:
        class _Null:
            def append_data(self, x):
                pass

            def close(self):
                pass
        wr = _Null()
    else:
        wr = imageio.get_writer(out, fps=a.fps, codec="libx264", quality=8, macro_block_size=8, pixelformat="yuv420p")
    first = frame(0.0, a.fast, briefing)
    for _ in range(7 * a.fps):
        wr.append_data(first)
    t, i, speed = 0.0, 0, a.fast
    while t <= T + 1e-9:
        want = 1.0 if slow_now(t) else a.fast
        speed += 0.25 * (want - speed)
        wr.append_data(frame(t, speed))
        t += speed / a.fps
        i += 1
        if i % 100 == 0:
            print(f"frame {i}  t={t:.1f}/{T:.1f}", flush=True)
    last = frame(T, a.fast)
    for _ in range(2 * a.fps):
        wr.append_data(last)
    rep = frame(T, a.fast, report)
    for _ in range(12 * a.fps):
        wr.append_data(rep)
    wr.close()
    if a.timeline_out:
        Path(a.timeline_out).write_text(json.dumps({"mission": a.mission, "fps": a.fps, "frames": views}))
        print(f"{a.timeline_out}: {len(views)} views")
        return 0
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
