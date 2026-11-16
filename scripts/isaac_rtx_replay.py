"""Phase 1 demo: Isaac Sim RTX rendering of a recorded Isaac-physics inspection mission (replay, no physics).

  D:\\Downloads\\isaac-sim-standalone-6.0.1-windows-x86_64\\python.bat scripts\\isaac_rtx_replay.py \
      --mission D:/go2wm/outputs/demo_mission/isaac/approach_tile0.json \
      --timeline D:/go2wm/outputs/demo_mission/timeline.json --out-dir D:/go2wm/outputs/demo_mission/rtx_frames

Runs in the standalone Isaac Sim 6.0.1 (its RTX renderer works on driver 610.88; Isaac Sim 5.1's crashes at start-up).
Builds the demo street in USD: the sidewalk surface from the same height function the physics ran on (holes,
upheavals, curb) with its concrete texture, asphalt road and grass verge with NVIDIA MDL materials, buildings, trees
from the NVIDIA vegetation library, water in the puddles, work-zone cones / barriers / dirt pile, and NVIDIA
characters (construction workers, pedestrians) with idle / walk-cycle animations, under an HDR sky with a sun. The
Unitree Go2 is NVIDIA's Go2 USD with every link posed from the recorded physics run. Each timeline entry (camera eye
and focus from scripts/render_demo_mission.py --timeline-out) is rendered with RTX real-time ray tracing to
<out-dir>/<index>.png; render_demo_mission.py --view-dir then composes the HUD around these frames.
--test T renders single frames at mission times T (chase camera) for look development.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--mission", default="D:/go2wm/outputs/demo_mission/isaac/approach_tile0.json")
ap.add_argument("--timeline", default=None)
ap.add_argument("--out-dir", default="D:/go2wm/outputs/demo_mission/rtx_frames")
ap.add_argument("--test", type=float, nargs="*", default=None)
ap.add_argument("--range", type=int, nargs=2, default=None, help="render only timeline entries [a, b)")
ap.add_argument("--width", type=int, default=1344)
ap.add_argument("--height", type=int, default=756)
ap.add_argument("--subframes", type=int, default=4)
ap.add_argument("--sky", default="NVIDIA/Assets/Skies/Cloudy/kloofendal_48d_partly_cloudy_4k.hdr")
ap.add_argument("--no-people", action="store_true")
ap.add_argument("--robot-usd", default=None, help="robot USD (default: NVIDIA's Go2); Go2-W: D:/go2wm/cache/go2w_usd/go2w_description.usd")
ap.add_argument("--no-trees", action="store_true")
a = ap.parse_args()

from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": True, "renderer": "RayTracedLighting", "width": a.width, "height": a.height})

import numpy as np  # noqa: E402
import omni.kit.commands  # noqa: E402
import omni.replicator.core as rep  # noqa: E402
import omni.usd  # noqa: E402
from isaacsim.storage.native import get_assets_root_path  # noqa: E402
from PIL import Image  # noqa: E402
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdShade, UsdSkel, Vt  # noqa: E402

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "src"))
from go2wm.sim.sidewalk_texture import CURB_W, GUTTER_W, sidewalk_albedo  # noqa: E402
from go2wm.sim.terrain import params_with_defaults, sidewalk_height  # noqa: E402

ASSETS = get_assets_root_path()
VFOV = 30.0  # deg, same as the offline renderer (VTK default view angle) so overlays line up

r = json.loads(Path(a.mission).read_text())
tiles_file = r["tiles_file"]
if not Path(tiles_file).is_absolute():
    tiles_file = str(Path("D:/go2wm") / tiles_file)
p = params_with_defaults(json.loads(Path(tiles_file).read_text())[r["tile"]])
site = p.get("site", {})
L, half = float(p["size"][0]), 0.5 * float(p["width"])
H_ = lambda x, y: sidewalk_height(np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64), p)  # noqa: E731
links, traj = r["links"], np.asarray(r["trajectory"])
T = r["time_s"]
dt = T / max(len(traj) - 1, 1)
track = r.get("site_track") or []
out_dir = Path(a.out_dir)
out_dir.mkdir(parents=True, exist_ok=True)
cache = Path("D:/go2wm/cache/rtx")
cache.mkdir(parents=True, exist_ok=True)

ctx = omni.usd.get_context()
ctx.new_stage()
stage = ctx.get_stage()
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
UsdGeom.SetStageMetersPerUnit(stage, 1.0)
UsdGeom.Xform.Define(stage, "/World")
mat_cache: dict[str, str] = {}


# ---------------------------------------------------------------------------------------------- helpers
def mdl(name: str, url: str) -> str:
    """MDL material from the NVIDIA base library (cached per name)."""
    if name in mat_cache:
        return mat_cache[name]
    path = f"/World/Looks/{name}"
    omni.kit.commands.execute("CreateMdlMaterialPrim", mtl_url=f"{ASSETS}/{url}", mtl_name=Path(url).stem,
                              mtl_path=path)
    mat_cache[name] = path
    return path


def preview(name: str, rgb=(0.5, 0.5, 0.5), rough=0.6, metal=0.0, tex: str | None = None, opacity=1.0,
            emissive=None) -> str:
    if name in mat_cache:
        return mat_cache[name]
    path = f"/World/Looks/{name}"
    m = UsdShade.Material.Define(stage, path)
    sh = UsdShade.Shader.Define(stage, path + "/Shader")
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(rough)
    sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metal)
    if opacity < 1.0:
        sh.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(opacity)
    if emissive is not None:
        sh.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*emissive))
    if tex:
        st = UsdShade.Shader.Define(stage, path + "/st")
        st.CreateIdAttr("UsdPrimvarReader_float2")
        st.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
        tx = UsdShade.Shader.Define(stage, path + "/tex")
        tx.CreateIdAttr("UsdUVTexture")
        tx.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(tex)
        tx.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
        tx.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(st.ConnectableAPI(), "result")
        tx.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
        sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(tx.ConnectableAPI(), "rgb")
    else:
        sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgb))
    m.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    mat_cache[name] = path
    return path


def bind(prim, mat_path: str):
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(UsdShade.Material(stage.GetPrimAtPath(mat_path)))


def grid_mesh(path, xs, ys, zfun, st_fun=None, mat=None):
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    Z = zfun(X, Y)
    nx, ny = X.shape
    pts = np.stack([X, Y, Z], -1).reshape(-1, 3)
    i = np.arange(nx * ny).reshape(nx, ny)
    q = np.stack([i[:-1, :-1], i[1:, :-1], i[1:, 1:], i[:-1, 1:]], -1).reshape(-1, 4)
    m = UsdGeom.Mesh.Define(stage, path)
    m.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(pts.astype(np.float32)))
    m.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(q), 4, dtype=np.int32)))
    m.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(q.reshape(-1).astype(np.int32)))
    m.CreateSubdivisionSchemeAttr("none")
    st = st_fun(X, Y) if st_fun else np.stack([X, Y], -1)
    pv = UsdGeom.PrimvarsAPI(m).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex)
    pv.Set(Vt.Vec2fArray.FromNumpy(st.reshape(-1, 2).astype(np.float32)))
    if mat:
        bind(m.GetPrim(), mat)
    return m


def box(path, x0, x1, y0, y1, z0, z1, mat, uv=1.0):
    """Box mesh with per-face UVs in metres (so tiled MDL materials keep their real scale)."""
    v = np.array([[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
                  [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]], dtype=np.float32)
    faces = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
    pts, st = [], []
    for f in faces:
        P = v[list(f)]
        n = np.cross(P[1] - P[0], P[2] - P[0])
        ax = int(np.argmax(np.abs(n)))
        uvax = [k for k in range(3) if k != ax]
        for q in P:
            pts.append(q)
            st.append((q[uvax[0]] / uv, q[uvax[1]] / uv))
    m = UsdGeom.Mesh.Define(stage, path)
    m.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(np.asarray(pts, dtype=np.float32)))
    m.CreateFaceVertexCountsAttr(Vt.IntArray([4] * 6))
    m.CreateFaceVertexIndicesAttr(Vt.IntArray(list(range(24))))
    m.CreateSubdivisionSchemeAttr("none")
    UsdGeom.PrimvarsAPI(m).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying).Set(
        Vt.Vec2fArray.FromNumpy(np.asarray(st, dtype=np.float32)))
    bind(m.GetPrim(), mat)
    return m


def place(path, url, xyz, yaw_deg=0.0, height=None, scale=None):
    """Reference an asset; scale it to `height` metres tall (assets come in cm or m) or by `scale`."""
    prim = stage.DefinePrim(path, "Xform")  # our transform on a parent; the asset (with its own ops) below it
    child = stage.DefinePrim(path + "/asset", "Xform")
    child.GetReferences().AddReference(f"{ASSETS}/{url}" if not url.startswith(("http", "D:", "C:")) else url)
    xf = UsdGeom.Xformable(prim)
    t = xf.AddTranslateOp()
    rz = xf.AddRotateZOp()
    s = xf.AddScaleOp()
    t.Set(Gf.Vec3d(*xyz))
    rz.Set(yaw_deg)
    if height is not None:
        bb = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
        rng = bb.ComputeUntransformedBound(prim).ComputeAlignedRange()
        hz = rng.GetMax()[2] - rng.GetMin()[2]
        k = height / hz if hz > 1e-6 else 1.0
        print(f"[rtx] place {path}: bbox z {rng.GetMin()[2]:.2f}..{rng.GetMax()[2]:.2f} -> scale {k:.4f}", flush=True)
        s.Set(Gf.Vec3f(k, k, k))
        t.Set(Gf.Vec3d(xyz[0], xyz[1], xyz[2] - rng.GetMin()[2] * k))
    elif scale is not None:
        s.Set(Gf.Vec3f(scale, scale, scale))
    else:
        s.Set(Gf.Vec3f(1, 1, 1))
    return prim


# ---------------------------------------------------------------------------------------------- lighting
dome = UsdLux.DomeLight.Define(stage, "/World/Sky")
dome.CreateTextureFileAttr(f"{ASSETS}/{a.sky}")
dome.CreateIntensityAttr(450.0)
dome.CreateTextureFormatAttr("latlong")
sun = UsdLux.DistantLight.Define(stage, "/World/Sun")
sun.CreateIntensityAttr(1100.0)
sun.CreateAngleAttr(0.6)
sun.CreateColorAttr(Gf.Vec3f(1.0, 0.96, 0.9))
UsdGeom.Xformable(sun).AddRotateXYZOp().Set(Gf.Vec3f(50.0, 0.0, -35.0))  # from the street side, behind

# ---------------------------------------------------------------------------------------------- ground
y_road = -half - CURB_W - GUTTER_W  # walk texture covers curb + gutter; asphalt beyond
tex_res = 0.008
txs = np.arange(-L / 2, L / 2 + 1e-9, tex_res)
tys = np.arange(y_road, half + 1e-9, tex_res)
tex_png = cache / f"walk_albedo_{r['tile']}_{len(txs)}x{len(tys)}.png"
if not tex_png.exists():
    TX, TY = np.meshgrid(txs, tys, indexing="xy")
    img = np.asarray(sidewalk_albedo(TX, TY, p, seed=7), dtype=np.float64)
    img = np.clip(img * 255 if img.max() <= 1.0 else img, 0, 255).astype(np.uint8)
    Image.fromarray(np.flipud(img)).save(tex_png)
walk_mat = preview("WalkConcrete", tex=str(tex_png).replace("\\", "/"), rough=0.88)
g = 0.02
grid_mesh("/World/Ground/Walk", np.arange(-L / 2, L / 2 + 1e-9, g), np.arange(y_road, half + 1e-9, g),
          lambda X, Y: H_(X, Y),
          st_fun=lambda X, Y: np.stack([(X + L / 2) / L, (Y - y_road) / (half - y_road)], -1), mat=walk_mat)
asphalt = mdl("Asphalt", "NVIDIA/Materials/Base/Natural/Asphalt.mdl")
grass = mdl("Grass", "NVIDIA/Materials/Base/Natural/Grass_Cut.mdl")
z_road = float(H_(0.0, y_road - 0.05))
grid_mesh("/World/Ground/Road", np.linspace(-L / 2 - 60, L / 2 + 60, 60), np.linspace(-13.0, y_road, 12),
          lambda X, Y: np.full_like(X, z_road), st_fun=lambda X, Y: np.stack([X / 4, Y / 4], -1), mat=asphalt)
paint = preview("RoadPaint", rgb=(0.85, 0.84, 0.78), rough=0.7)
ypaint = preview("RoadPaintYellow", rgb=(0.85, 0.65, 0.12), rough=0.7)
for x0 in np.arange(-L / 2 - 60, L / 2 + 60, 6.0):  # dashed centre line and the edge line
    box(f"/World/Ground/Dash_{int(x0 + 100)}", x0, x0 + 3.0, -7.08, -6.92, z_road, z_road + 0.004, ypaint)
box("/World/Ground/EdgeLine", -L / 2 - 60, L / 2 + 60, y_road - 0.42, y_road - 0.32, z_road, z_road + 0.004, paint)
z_verge = float(H_(0.0, half + 0.05))
grid_mesh("/World/Ground/Verge", np.linspace(-L / 2 - 60, L / 2 + 60, 80), np.linspace(half, 3.35, 8),
          lambda X, Y: np.full_like(X, z_verge) + 0.0, st_fun=lambda X, Y: np.stack([X / 2, Y / 2], -1), mat=grass)
for side in (-1, 1):  # walk continues beyond the tile (plain concrete)
    x0, x1 = (-L / 2 - 60, -L / 2) if side < 0 else (L / 2, L / 2 + 60)
    box(f"/World/Ground/WalkExt{side + 1}", x0, x1, y_road, half, -0.05, float(H_(side * L / 2 * 0.999, 0.0)),
        mdl("Concrete", "NVIDIA/Materials/Base/Masonry/Concrete_Rough.mdl"), uv=1.5)
water = mdl("Water", "NVIDIA/Materials/Base/Natural/Water.mdl")
for k, d in enumerate(p["defects"]):
    if d["type"] == "puddle":
        t = np.linspace(0, 2 * np.pi, 48, endpoint=False)
        c, s = np.cos(d["angle"]), np.sin(d["angle"])
        ex, ey = d["rx"] * np.cos(t), d["ry"] * np.sin(t)
        px, py = d["x"] + c * ex - s * ey, d["y"] + s * ex + c * ey
        z0 = float(np.max(H_(px, py))) - 0.003
        m = UsdGeom.Mesh.Define(stage, f"/World/Ground/Puddle_{k}")
        m.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(float(u), float(v), z0) for u, v in zip(px, py)]))
        m.CreateFaceVertexCountsAttr(Vt.IntArray([len(px)]))
        m.CreateFaceVertexIndicesAttr(Vt.IntArray(list(range(len(px)))))
        bind(m.GetPrim(), water)
    elif d["type"] == "obstruction":
        z0 = float(H_(d["x"], d["y"]))
        box(f"/World/Props/Pole_{k}", d["x"] - 0.11, d["x"] + 0.11, d["y"] - 0.11, d["y"] + 0.11, z0, z0 + 7.5,
            preview("PoleWood", rgb=(0.36, 0.27, 0.18), rough=0.9))

# ---------------------------------------------------------------------------------------------- buildings
rng = np.random.default_rng(3)
facades = [("Brick", "NVIDIA/Materials/Base/Masonry/Brick_Wall_Red.mdl"),
           ("Stucco", "NVIDIA/Materials/Base/Masonry/Stucco.mdl"),
           ("BrickBrown", "NVIDIA/Materials/Base/Masonry/Brick_Wall_Brown.mdl"),
           ("ConcreteSmooth", "NVIDIA/Materials/Base/Masonry/Concrete_Smooth.mdl")]
glass = preview("WindowGlass", rgb=(0.08, 0.11, 0.14), rough=0.05, metal=0.6)
frame_m = preview("WindowFrame", rgb=(0.15, 0.15, 0.16), rough=0.4, metal=0.5)
door_m = preview("Door", rgb=(0.22, 0.14, 0.09), rough=0.5)
x, bi = -L / 2 - 20, 0
while x < L / 2 + 20:
    w, h = float(rng.uniform(6, 10)), float(rng.uniform(4.0, 9.0))
    y0 = 3.35
    nm, url = facades[int(rng.integers(len(facades)))]
    box(f"/World/Buildings/B{bi}", x, x + w - 0.2, y0, y0 + 8, -0.1, h, mdl(nm, url), uv=2.0)
    box(f"/World/Buildings/B{bi}_cornice", x - 0.05, x + w - 0.15, y0 - 0.25, y0 + 0.1, h - 0.3, h,
        preview("Cornice", rgb=(0.55, 0.53, 0.5), rough=0.7))
    for fl in range(int((h - 0.6) // 2.8) + 1):
        z = 0.9 + fl * 2.8
        if z + 1.5 > h - 0.4:
            break
        for k, wx in enumerate(np.arange(x + 0.9, x + w - 1.2, 1.7)):
            if fl == 0 and abs(wx + 0.5 - (x + w / 2)) < 1.0:
                continue
            box(f"/World/Buildings/B{bi}_w{fl}_{k}f", wx - 0.06, wx + 1.06, y0 - 0.06, y0 + 0.02, z - 0.06, z + 1.56,
                frame_m)
            box(f"/World/Buildings/B{bi}_w{fl}_{k}", wx, wx + 1.0, y0 - 0.08, y0 + 0.02, z, z + 1.5, glass)
    dx = x + w / 2 - 0.55
    box(f"/World/Buildings/B{bi}_door", dx, dx + 1.1, y0 - 0.06, y0 + 0.02, 0.0, 2.3, door_m)
    x += w
    bi += 1

# ---------------------------------------------------------------------------------------------- vegetation, lamps
if not a.no_trees:
    kinds = ["Red_Maple", "American_Beech", "Honey_Locust", "Red_Ash"]
    zone = (site.get("zones") or [None])[0]
    for k, tx in enumerate(np.arange(-L / 2 + 3, L / 2 + 10, 9.0)):
        if zone and zone["x"][0] - 1.5 < tx < zone["x"][1] + 1.5:
            continue
        place(f"/World/Trees/T{k}", f"NVIDIA/Assets/Vegetation/Trees/{kinds[k % len(kinds)]}.usd",
              (tx + 1.5, 2.5, z_verge), yaw_deg=float(rng.uniform(0, 360)), height=float(rng.uniform(6.5, 8.5)))
    for d in p["defects"]:
        if d.get("tree") and d["tree"][1] > half:
            place("/World/Trees/RootTree", "NVIDIA/Assets/Vegetation/Trees/American_Beech.usd",
                  (d["tree"][0], d["tree"][1], z_verge), height=8.5)
    for k, sx in enumerate(np.arange(-L / 2, L / 2, 3.1)):
        place(f"/World/Shrubs/S{k}", "NVIDIA/Assets/Vegetation/Shrub/Boxwood.usd", (sx + 0.7, 3.05, z_verge),
              yaw_deg=float(rng.uniform(0, 360)), height=0.7)
metal = preview("LampMetal", rgb=(0.2, 0.22, 0.24), rough=0.35, metal=0.8)
for k, lx in enumerate((-20.0, -8.5, 8.0, 20.0)):
    z0 = float(H_(lx, 2.0))
    box(f"/World/Lamps/L{k}", lx - 0.06, lx + 0.06, 2.0 - 0.06, 2.0 + 0.06, z0, z0 + 5.0, metal)
    box(f"/World/Lamps/L{k}a", lx - 0.04, lx + 0.04, 0.9, 2.0, z0 + 4.9, z0 + 5.0, metal)
    box(f"/World/Lamps/L{k}h", lx - 0.18, lx + 0.18, 0.75, 1.1, z0 + 4.8, z0 + 4.95,
        preview("LampHead", rgb=(0.9, 0.9, 0.85), rough=0.3))

# ---------------------------------------------------------------------------------------------- work zone props
orange = preview("SafetyOrange", rgb=(0.95, 0.33, 0.02), rough=0.45)
white = preview("SafetyWhite", rgb=(0.92, 0.92, 0.9), rough=0.4)
black = preview("RubberBlack", rgb=(0.03, 0.03, 0.03), rough=0.8)
steel = preview("Galvanized", rgb=(0.55, 0.56, 0.57), rough=0.35, metal=0.9)


def cone(path, x, y):
    z = float(H_(x, y))
    box(path + "_base", x - 0.19, x + 0.19, y - 0.19, y + 0.19, z, z + 0.035, black)
    c = UsdGeom.Cone.Define(stage, path)
    c.CreateRadiusAttr(0.15)
    c.CreateHeightAttr(0.68)
    c.CreateAxisAttr("Z")
    UsdGeom.Xformable(c).AddTranslateOp().Set(Gf.Vec3d(x, y, z + 0.035 + 0.34))
    bind(c.GetPrim(), orange)
    band = UsdGeom.Cylinder.Define(stage, path + "_band")
    band.CreateRadiusAttr(0.092)
    band.CreateHeightAttr(0.1)
    band.CreateAxisAttr("Z")
    UsdGeom.Xformable(band).AddTranslateOp().Set(Gf.Vec3d(x, y, z + 0.40))
    bind(band.GetPrim(), white)


posts = sorted(tuple(o["xy"]) for o in site.get("objects", []) if o["kind"] == "barrier")
for k, (bx, by) in enumerate(posts):
    z = float(H_(bx, by))
    box(f"/World/Zone/Post{k}", bx - 0.025, bx + 0.025, by - 0.025, by + 0.025, z, z + 1.0, steel)
    box(f"/World/Zone/Foot{k}", bx - 0.06, bx + 0.06, by - 0.22, by + 0.22, z, z + 0.06, black)
for k, ((xa, ya), (xb, yb)) in enumerate(zip(posts[:-1], posts[1:])):
    z = float(H_(0.5 * (xa + xb), ya))
    for i in range(4):
        u0, u1 = xa + (xb - xa) * i / 4, xa + (xb - xa) * (i + 1) / 4
        for j, (z0, z1) in enumerate(((0.72, 0.92), (0.38, 0.52))):
            box(f"/World/Zone/Rail{k}_{i}_{j}", u0, u1, ya - 0.012, ya + 0.012, z + z0, z + z1,
                orange if i % 2 == 0 else white)
people_static, movers = [], []
for k, o in enumerate(site.get("objects", [])):
    if "path" in o:
        movers.append(o)
        continue
    x0, y0 = o["xy"]
    z = float(H_(x0, y0))
    if o["kind"] == "cone":
        cone(f"/World/Zone/Cone{k}", x0, y0)
    elif o["kind"] == "worker":
        people_static.append(o)
    elif o["kind"] == "pile":
        e = UsdGeom.Sphere.Define(stage, f"/World/Zone/Pile{k}")
        e.CreateRadiusAttr(1.0)
        xf = UsdGeom.Xformable(e)
        xf.AddTranslateOp().Set(Gf.Vec3d(x0, y0, z - 0.05))
        xf.AddScaleOp().Set(Gf.Vec3f(0.8, 0.5, 0.42))
        bind(e.GetPrim(), mdl("Dirt", "NVIDIA/Materials/Base/Natural/Soil_Rocky.mdl"))
    elif o["kind"] == "generator":
        box(f"/World/Zone/Gen{k}", x0 - 0.42, x0 + 0.42, y0 - 0.27, y0 + 0.27, z + 0.08, z + 0.62,
            preview("GenYellow", rgb=(0.95, 0.72, 0.02), rough=0.4))
        for wx in (-0.3, 0.3):
            box(f"/World/Zone/Gen{k}w{int(wx > 0)}", x0 + wx - 0.06, x0 + wx + 0.06, y0 - 0.3, y0 + 0.3, z, z + 0.12, black)
    elif o["kind"] == "sign":
        box(f"/World/Zone/Sign{k}", x0 - 0.45, x0 + 0.45, y0 - 0.02, y0 + 0.02, z + 0.55, z + 1.3, orange)
        box(f"/World/Zone/Sign{k}l", x0 - 0.03, x0 + 0.03, y0 - 0.3, y0 + 0.3, z, z + 0.6, steel)

# ---------------------------------------------------------------------------------------------- people
# NVIDIA characters (RL_BoneRoot skeleton; the stock skelanim clips use a different "Root/..." skeleton, so they do
# not bind). Each character gets its own SkelAnimation driving a few joints procedurally: a walk cycle phased by
# the distance actually walked (no foot sliding), arms lowered from the bind T-pose, idle sway, digging, looking
# around. Skeleton space: Z up, the character faces -Y, its left is +X.
CHAR = {"worker": ["Isaac/People/Characters/male_adult_construction_01_new/male_adult_construction_01_new.usd",
                   "Isaac/People/Characters/male_adult_construction_05_new/male_adult_construction_05_new.usd",
                   "Isaac/People/Characters/male_adult_construction_03/male_adult_construction_03.usd"],
        "pedestrian": ["Isaac/People/Characters/F_Business_02/F_Business_02.usd",
                       "Isaac/People/Characters/female_adult_police_01_new/female_adult_police_01_new.usd"]}
DRIVEN = ["Hip", "Waist", "Spine01", "NeckTwist02", "L_Thigh", "R_Thigh", "L_Calf", "R_Calf", "L_Foot", "R_Foot",
          "L_Upperarm", "R_Upperarm", "L_Forearm", "R_Forearm"]
persons = []
X_AX, Y_AX, Z_AX = Gf.Vec3d(1, 0, 0), Gf.Vec3d(0, 1, 0), Gf.Vec3d(0, 0, 1)


class SkelDriver:
    def __init__(self, skel_prim, anim_path):
        sk = UsdSkel.Skeleton(skel_prim)
        self.joints = list(sk.GetJointsAttr().Get())
        rest = list(sk.GetRestTransformsAttr().Get())
        self.short = [str(j).split("/")[-1] for j in self.joints]
        self.parent = {}
        idx = {str(j): i for i, j in enumerate(self.joints)}
        for i, j in enumerate(self.joints):
            pth = str(j).rsplit("/", 1)
            self.parent[i] = idx.get(pth[0]) if len(pth) > 1 else None
        self.rest = [Gf.Matrix4d(m) for m in rest]
        self.world_rest = []
        for i in range(len(self.joints)):
            p_ = self.parent[i]
            self.world_rest.append(self.rest[i] * self.world_rest[p_] if p_ is not None else Gf.Matrix4d(self.rest[i]))
        self.drive = [i for i, s in enumerate(self.short) if s in DRIVEN]
        self.by = {self.short[i]: i for i in self.drive}
        self.anim = UsdSkel.Animation.Define(stage, anim_path)
        self.anim.CreateJointsAttr([self.joints[i] for i in self.drive])
        UsdSkel.BindingAPI.Apply(skel_prim).CreateAnimationSourceRel().SetTargets([self.anim.GetPath()])

    def apply(self, deltas: dict[str, list]):
        """deltas: short joint name -> list of (axis Vec3d in skeleton space, angle deg), applied in order as
        world-space rotations about the joint, on top of what its parent chain already does."""
        world = {}

        def W(i):  # animated skel-space transform of joint i
            if i in world:
                return world[i]
            p_ = self.parent[i]
            wp = W(p_) if p_ is not None else Gf.Matrix4d(1.0)
            inh = self.rest[i] * wp
            rot = inh.ExtractRotationMatrix()
            for ax, deg in deltas.get(self.short[i], []):
                rot = rot * Gf.Matrix3d(Gf.Rotation(ax, deg))
            world[i] = Gf.Matrix4d(rot, inh.ExtractTranslation())
            return world[i]

        rots, trans, scales = [], [], []
        for i in self.drive:
            p_ = self.parent[i]
            wp = W(p_) if p_ is not None else Gf.Matrix4d(1.0)
            loc = W(i) * wp.GetInverse()
            q = loc.ExtractRotationQuat()
            rots.append(Gf.Quatf(q.GetReal(), *[float(v) for v in q.GetImaginary()]))
            trans.append(Gf.Vec3f(*[float(v) for v in self.rest[i].ExtractTranslation()]))
            scales.append(Gf.Vec3h(1, 1, 1))
        self.anim.CreateRotationsAttr().Set(Vt.QuatfArray(rots))
        self.anim.CreateTranslationsAttr().Set(Vt.Vec3fArray(trans))
        self.anim.CreateScalesAttr().Set(Vt.Vec3hArray(scales))


def body_pose(mode: str, phase: float, t: float) -> dict:
    """Procedural pose. Rotations about skeleton axes: +X angle swings a leg / arm BACKWARD (the character faces
    -Y), +Y lowers the left arm (+X side), -Y the right arm."""
    d: dict[str, list] = {}
    lower = 78.0
    if mode == "walk":
        s = math.sin(phase)
        thigh = 24.0
        d["L_Thigh"] = [(X_AX, -thigh * s)]
        d["R_Thigh"] = [(X_AX, thigh * s)]
        kl = 8 + 38 * max(0.0, math.sin(phase + 1.3)) ** 1.5  # knee bends in the swing phase
        kr = 8 + 38 * max(0.0, math.sin(phase + 1.3 + math.pi)) ** 1.5
        d["L_Calf"] = [(X_AX, kl)]
        d["R_Calf"] = [(X_AX, kr)]
        d["L_Foot"] = [(X_AX, -0.5 * kl + 6 * s)]
        d["R_Foot"] = [(X_AX, -0.5 * kr - 6 * s)]
        d["L_Upperarm"] = [(Y_AX, lower), (X_AX, 18 * s)]
        d["R_Upperarm"] = [(Y_AX, -lower), (X_AX, -18 * s)]
        d["L_Forearm"] = [(X_AX, -14 - 8 * max(0, -s))]
        d["R_Forearm"] = [(X_AX, -14 - 8 * max(0, s))]
        d["Spine01"] = [(Z_AX, 4 * s), (X_AX, -3)]
    elif mode == "dig":
        c = math.sin(1.4 * t)
        d["Waist"] = [(X_AX, -(22 + 8 * c))]  # lean forward
        d["L_Thigh"] = [(X_AX, -10)]
        d["R_Thigh"] = [(X_AX, 6)]
        d["L_Calf"] = [(X_AX, 18)]
        d["R_Calf"] = [(X_AX, 10)]
        d["L_Upperarm"] = [(Y_AX, 60), (X_AX, -(55 + 15 * c))]
        d["R_Upperarm"] = [(Y_AX, -60), (X_AX, -(40 + 15 * c))]
        d["L_Forearm"] = [(X_AX, -35)]
        d["R_Forearm"] = [(X_AX, -45)]
    else:  # idle / look
        b = math.sin(0.9 * t)
        d["L_Upperarm"] = [(Y_AX, lower), (X_AX, -4 + 2 * b)]
        d["R_Upperarm"] = [(Y_AX, -lower), (X_AX, -4 - 2 * b)]
        d["L_Forearm"] = [(X_AX, -12)]
        d["R_Forearm"] = [(X_AX, -12)]
        d["Spine01"] = [(X_AX, 1.5 * b)]
        if mode == "look":
            d["NeckTwist02"] = [(Z_AX, 35 * math.sin(0.35 * t))]
    return d


def add_person(path, url):
    prim = place(path, url, (0, 0, -50), scale=1.0)  # characters are in metres, feet at the origin
    skel = None
    for q in Usd.PrimRange(prim):
        if q.IsA(UsdSkel.Skeleton):
            skel = q
            break
    drv = SkelDriver(skel, path + "_anim") if skel is not None else None
    return prim, drv


def set_pose(prim, x, y, z, yaw):
    ops = UsdGeom.Xformable(prim).GetOrderedXformOps()
    ops[0].Set(Gf.Vec3d(x, y, z))
    ops[1].Set(math.degrees(yaw) + 90.0)  # the character's forward (-Y) turned to the heading


if not a.no_people:
    wk = 0
    for k, o in enumerate(people_static):
        prim, drv = add_person(f"/World/People/Static{k}", CHAR["worker"][wk % 3])
        wk += 1
        mode = "dig" if o.get("pose") == "dig" else ("look" if k % 2 else "idle")
        persons.append(dict(prim=prim, drv=drv, mode=mode, xy=o["xy"], yaw=o.get("yaw", 0.0)))
    for j, o in enumerate(movers):
        url = CHAR["worker"][wk % 3] if o["kind"] == "worker" else CHAR["pedestrian"][j % 2]
        if o["kind"] == "worker":
            wk += 1
        prim, drv = add_person(f"/World/People/Mover{j}", url)
        persons.append(dict(prim=prim, drv=drv, mover=j))
    # distance walked per mover and step (walk-cycle phase)
    walked = []
    for j in range(len(movers)):
        acc, out = 0.0, [0.0]
        for q in range(1, len(track)):
            acc += math.hypot(track[q][j][0] - track[q - 1][j][0], track[q][j][1] - track[q - 1][j][1])
            out.append(acc)
        walked.append(out)
    print(f"[rtx] people: {len(persons)} (procedural skeleton animation: "
          f"{sum(p_['drv'] is not None for p_ in persons)} rigged)", flush=True)

# ---------------------------------------------------------------------------------------------- robot
go2 = stage.DefinePrim("/World/Go2", "Xform")
go2.GetReferences().AddReference(a.robot_usd or f"{ASSETS}/Isaac/Robots/Unitree/Go2/go2.usd")
for _ in range(3):
    app.update()
link_ops = {}
for i, name in enumerate(r["body_names"]):
    prim = stage.GetPrimAtPath(f"/World/Go2/{name}")
    if not prim.IsValid():
        continue
    xf = UsdGeom.Xformable(prim)
    xf.ClearXformOpOrder()
    link_ops[i] = (xf.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble), xf.AddOrientOp(UsdGeom.XformOp.PrecisionDouble))
# no physics: the replay poses links directly (deactivate the articulation's physics schemas' effect by never playing)
print(f"[rtx] go2 links posed: {len(link_ops)} / {len(r['body_names'])}", flush=True)

# ---------------------------------------------------------------------------------------------- camera
cam = UsdGeom.Camera.Define(stage, "/World/Cam")
hap = 20.955
vap = hap * a.height / a.width
cam.CreateHorizontalApertureAttr(hap)
cam.CreateVerticalApertureAttr(vap)
cam.CreateFocalLengthAttr(vap / (2 * math.tan(math.radians(VFOV / 2))))
cam.CreateClippingRangeAttr(Gf.Vec2f(0.05, 500.0))
cam_op = UsdGeom.Xformable(cam).AddTransformOp()


def set_camera(eye, foc):
    m = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*foc), Gf.Vec3d(0, 0, 1))
    cam_op.Set(m.GetInverse())


def k_at(t):
    return min(int(round(min(t, T) / dt)), len(links) - 1)


phase_acc = {}


def pose_scene(t):
    k = k_at(t)
    ln = links[k]
    for i, (top, oop) in link_ops.items():
        x, y, z = ln["pos"][i]
        w, qx, qy, qz = ln["quat"][i]
        top.Set(Gf.Vec3d(x, y, z))
        oop.Set(Gf.Quatd(w, qx, qy, qz))
    for ps in persons:
        if "mover" in ps and track:
            kk = min(k, len(track) - 1)
            x, y, yaw, walking = track[kk][ps["mover"]]
            set_pose(ps["prim"], x, y, float(H_(x, y)), yaw)
            if ps["drv"] is not None:  # one walk cycle (two steps) per 1.35 m walked
                mode = "walk" if walking else "idle"
                ps["drv"].apply(body_pose(mode, 2 * math.pi * walked[ps["mover"]][kk] / 1.35, t))
        elif "xy" in ps:
            x, y = ps["xy"]
            set_pose(ps["prim"], x, y, float(H_(x, y)), ps["yaw"])
            if ps["drv"] is not None:
                ps["drv"].apply(body_pose(ps["mode"], 0.0, t))


def chase_camera(t):
    k = k_at(t)
    pose = traj[k]
    hd = np.array([math.cos(pose[2]), math.sin(pose[2]), 0.0])
    right = np.array([hd[1], -hd[0], 0.0])
    gz = float(H_(pose[0], pose[1]))
    rb = np.array([pose[0], pose[1], gz])
    foc = rb + 1.1 * hd + [0, 0, 0.1]
    eye = foc - 3.4 * hd + 2.1 * right + [0, 0, 1.8]
    return eye, foc


# ---------------------------------------------------------------------------------------------- render
rp = rep.create.render_product("/World/Cam", (a.width, a.height))
ann = rep.AnnotatorRegistry.get_annotator("rgb")
ann.attach([rp])
for _ in range(20):  # let assets stream in and shaders compile
    app.update()


def render_to(path):
    rep.orchestrator.step(rt_subframes=a.subframes)
    img = np.asarray(ann.get_data())
    Image.fromarray(img[..., :3].astype(np.uint8)).save(path)


if a.test:
    for t in a.test:
        pose_scene(t)
        set_camera(*chase_camera(t))
        for _ in range(30):
            app.update()
        render_to(out_dir / f"test_t{t:.0f}.png")
        print(f"[rtx] test frame t={t}", flush=True)
else:
    tl = json.loads(Path(a.timeline).read_text())
    frames = tl["frames"]
    lo, hi = a.range if a.range else (0, len(frames))
    for i in range(lo, min(hi, len(frames))):
        f = frames[i]
        dst = out_dir / f"{i:05d}.png"
        if dst.exists():
            continue
        pose_scene(f["t"])
        set_camera(f["eye"], f["foc"])
        render_to(dst)
        if i % 50 == 0:
            print(f"[rtx] frame {i}/{len(frames)}", flush=True)
app.close()
