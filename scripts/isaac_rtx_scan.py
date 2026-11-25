"""Photoreal Isaac Sim RTX stills of a scanned-sidewalk tile (heightmap tile from scripts/scan_to_tile.py).

  D:\\Downloads\\isaac-sim-standalone-6.0.1-windows-x86_64\\python.bat scripts\\isaac_rtx_scan.py \
      --mission D:/go2wm/outputs/scan_missions/approach_scan2_tile0.json --out-dir D:/go2wm/outputs/scan_missions/rtx

Runs in the standalone Isaac Sim 6.0.1 (RTX works there on driver 610.88; 5.1's RTX crashes). Unlike
isaac_rtx_replay.py (built around the hand-authored demo street: curb, road, buildings), this scene is only what the
scan measured:
  * terrain = the measured heightmap itself as a 2 cm mesh (grade, cross-slope, joints, the faulted slab, verges);
  * photo mode (default when <tile dir>/photo exists, from scripts/scan_photo_textures.py): the ground is textured
    with the true orthophoto made from the X7's own photos (the real slabs, stains, leaves, verge), the surroundings
    are the lidar ground model with the same projection out to --apron metres, and the sky / backdrop is the HDR
    panorama of the scan station nearest the camera (real buildings, trees, road, skyline, dusk light; no added sun).
    The backdrop is exact from that station and shows parallax the farther the camera is from it, so hero shots sit
    near stations (scene x -24.1, -13.6, -4.3, 4.8, 14.7);
  * otherwise (--no-photo): concrete albedo generated from the scan (joints drawn where the heightmap has its
    grooves), tiled crops of the photos for grass, a stock HDR sky and a low autumn sun;
  * the Go2 posed from the recorded Isaac physics run.
Views: --views chase:T (behind the robot at mission time T), follow:T (low, behind, horizon in frame), eye:X[:H]
(eye height H, default 1.6 m, at scene x looking along +x), look:X:Y:AZ:H:T (free camera), fault (close-up of the
faulted slab at scene x 10.7), aerial.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--mission", default="D:/go2wm/outputs/scan_missions/approach_scan2_tile0.json")
ap.add_argument("--out-dir", default="D:/go2wm/outputs/scan_missions/rtx")
ap.add_argument("--views", nargs="+", default=["chase:20", "eye:-14", "fault", "aerial"])
ap.add_argument("--width", type=int, default=1920)
ap.add_argument("--height", type=int, default=1080)
ap.add_argument("--subframes", type=int, default=32)
ap.add_argument("--sky", default="NVIDIA/Assets/Skies/Cloudy/kloofendal_48d_partly_cloudy_4k.hdr")
ap.add_argument("--grass-photo", default="4:3:2300:3100:1150:2550",
                help="station:imageset:row0:row1:col0:col1 crop of the scan's own photos used as the grass texture")
ap.add_argument("--concrete-photo", default="3:20:2250:2950:1250:2550", help="same, for the concrete detail")
ap.add_argument("--sun-elev", type=float, default=24.0, help="sun elevation [deg] (Denver, late Nov afternoon ~20-28)")
ap.add_argument("--sun-az", type=float, default=-35.0, help="sun azimuth in scene frame [deg] (0 = from +x)")
ap.add_argument("--no-robot", action="store_true")
ap.add_argument("--photo-dir", default="", help="photo textures from scripts/scan_photo_textures.py (default: "
                "<tile dir>/photo if present): orthophoto ground + the station panoramas as sky and backdrop")
ap.add_argument("--no-photo", action="store_true", help="use the procedural / tiled-crop look instead")
ap.add_argument("--dome-intensity", type=float, default=None, help="default 500 (stock sky) / 1.0 x calibration (photo)")
ap.add_argument("--sun-intensity", type=float, default=None, help="default 1500 (stock sky) / 0 (photo: dusk, no sun)")
ap.add_argument("--apron", type=float, default=8.0, help="photo mode: projected ground fades out this far "
                "beyond the tile [m]")
ap.add_argument("--station", type=int, default=0, help="panorama station for the backdrop (0 = nearest the camera)")
ap.add_argument("--video", type=float, default=0.0, help="also render a frame sequence of the whole run at this fps "
                "into <out-dir>/frames_<view>/ (stitch with ffmpeg)")
ap.add_argument("--video-view", default="follow", choices=["follow", "chase"], help="camera for --video")
ap.add_argument("--timeline", default=None, help="camera timeline from scripts/render_demo_mission.py --timeline-out: "
                "render each entry to <out-dir>/<index>.png for render_demo_mission.py --view-dir (HUD video)")
ap.add_argument("--range", type=int, nargs=2, default=None, help="--timeline: render only entries [a, b)")
ap.add_argument("--settle", type=int, default=2, help="app updates after posing each timeline / video frame")
ap.add_argument("--robot-usd", default=None, help="robot USD (default: NVIDIA's Go2); Go2-W: "
                "D:/go2wm/cache/go2w_usd/go2w_description.usd")
a = ap.parse_args()

from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": True, "renderer": "RayTracedLighting", "width": a.width, "height": a.height})

import numpy as np  # noqa: E402
import omni.kit.commands  # noqa: E402
import omni.replicator.core as rep  # noqa: E402
import omni.usd  # noqa: E402
from isaacsim.storage.native import get_assets_root_path  # noqa: E402
from PIL import Image  # noqa: E402
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade, Vt  # noqa: E402

ASSETS = get_assets_root_path()
VFOV = 34.0
r = json.loads(Path(a.mission).read_text())
tiles_file = r["tiles_file"] if Path(r["tiles_file"]).is_absolute() else str(Path("D:/go2wm") / r["tiles_file"])
tile = json.loads(Path(tiles_file).read_text())[r["tile"]]
hm_file = tile["heightmap"]["file"] if isinstance(tile["heightmap"], dict) else tile["heightmap"]
H = np.load(hm_file)
Z = H["z"].astype(np.float64)
CLS = H["cls"]
RES, X0, Y0 = float(H["res"]), float(H["x0"]), float(H["y0"])
NX, NY = Z.shape
XS, YS = X0 + np.arange(NX) * RES, Y0 + np.arange(NY) * RES
L, W = float(tile["size"][0]), float(tile["size"][1])
out_dir = Path(a.out_dir)
out_dir.mkdir(parents=True, exist_ok=True)
cache = Path("D:/go2wm/cache/rtx_scan")
cache.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(3)
PHOTO = None
if not a.no_photo:
    pdir = Path(a.photo_dir) if a.photo_dir else Path(hm_file).parent / "photo"
    if (pdir / "photo.json").exists():
        PHOTO = json.loads((pdir / "photo.json").read_text())
        PHOTO["dir"] = pdir
        print(f"[rtx-scan] photo textures from {pdir}", flush=True)


def hz(x, y):
    """Bilinear terrain height (edge-clamped), same as go2wm.sim.terrain.heightmap_height."""
    x, y = np.broadcast_arrays(np.asarray(x, float), np.asarray(y, float))
    u = np.clip((x - X0) / RES, 0, NX - 1)
    v = np.clip((y - Y0) / RES, 0, NY - 1)
    i = np.minimum(np.floor(u).astype(int), NX - 2)
    j = np.minimum(np.floor(v).astype(int), NY - 2)
    fu, fv = u - i, v - j
    return ((1 - fu) * (1 - fv) * Z[i, j] + fu * (1 - fv) * Z[i + 1, j] + (1 - fu) * fv * Z[i, j + 1]
            + fu * fv * Z[i + 1, j + 1])


# --------------------------------------------------------------------------------------- numpy-only image helpers
def blur(img, sigma_px):
    """Gaussian blur via FFT (wrap-around; fine for textures)."""
    if sigma_px <= 0:
        return img
    ky = np.fft.fftfreq(img.shape[0])[:, None]
    kx = np.fft.fftfreq(img.shape[1])[None, :]
    g = np.exp(-2 * (math.pi * sigma_px) ** 2 * (kx ** 2 + ky ** 2))
    return np.real(np.fft.ifft2(np.fft.fft2(img) * g))


def noise(shape, sigma_px, seed):
    n = blur(np.random.default_rng(seed).standard_normal(shape), sigma_px)
    return n / (n.std() + 1e-9)


def resample(arr, xq, yq):
    """Bilinear sample of a [NX, NY] grid at scene coords (2-D query arrays)."""
    u = np.clip((xq - X0) / RES, 0, NX - 1)
    v = np.clip((yq - Y0) / RES, 0, NY - 1)
    i = np.minimum(np.floor(u).astype(int), NX - 2)
    j = np.minimum(np.floor(v).astype(int), NY - 2)
    fu, fv = u - i, v - j
    return ((1 - fu) * (1 - fv) * arr[i, j] + fu * (1 - fv) * arr[i + 1, j] + (1 - fu) * fv * arr[i, j + 1]
            + fu * fv * arr[i + 1, j + 1])


def photo_crop(spec: str, size: int = 1024) -> np.ndarray:
    """A crop of one of the scanner's own photos (JPEG blobs in the .tcf next to the scan), as float RGB [0, 1],
    resized to size x size and made tileable (blend with a half-offset copy under a cosine window)."""
    import ast
    import io
    import sqlite3

    st, iset, r0, r1, c0, c1 = (int(v) for v in spec.split(":"))
    src = Path(ast.literal_eval(str(H["meta"]))["source"])
    tcf = src.parent / f"{src.stem}.tdf" / f"{src.stem}_{st}.tcf"
    con = sqlite3.connect(f"file:{tcf.as_posix()}?mode=ro", uri=True)
    rows = con.execute("select i.ExposureTime, b.BlobbyValue from TzfImage i join Blobby b using(BlobbyId) "
                       "where i.ImageSetId = ? order by i.ExposureTime", (iset,)).fetchall()
    con.close()
    im = Image.open(io.BytesIO(rows[len(rows) // 2][1])).convert("RGB")  # middle exposure
    im = im.crop((c0, r0, c1, r1)).resize((size, size), Image.LANCZOS)
    a_ = np.asarray(im, dtype=np.float64) / 255.0
    w1 = np.sin(np.linspace(0, np.pi, size)) ** 2
    w = np.outer(w1, w1)[..., None]
    b_ = np.roll(a_, (size // 2, size // 2), (0, 1))
    wb = np.roll(w, (size // 2, size // 2), (0, 1))
    return (a_ * w + b_ * wb) / (w + wb + 1e-9)


# walkway extent from the measured hard-surface class (contiguous band around y = 0)
hard_frac = (CLS == 0).mean(0)
j0 = int(np.argmin(np.abs(YS)))
lo = j0
while lo > 0 and hard_frac[lo - 1] > 0.6:
    lo -= 1
hi = j0
while hi < NY - 1 and hard_frac[hi + 1] > 0.6:
    hi += 1
WALK_LO, WALK_HI = float(YS[lo]) - RES / 2, float(YS[hi]) + RES / 2
print(f"[rtx-scan] walkway y in [{WALK_LO:.2f}, {WALK_HI:.2f}] (measured), tile {L} x {W} m", flush=True)

# groove / relief map from the heightmap (joints, the faulted slab edge): z minus a 12 cm box mean
k = 6
pad = np.pad(Z, k, mode="edge")
cs = pad.cumsum(0).cumsum(1)
cs = np.pad(cs, ((1, 0), (1, 0)))
box = (cs[2 * k + 1:, 2 * k + 1:] - cs[:-2 * k - 1, 2 * k + 1:] - cs[2 * k + 1:, :-2 * k - 1] + cs[:-2 * k - 1, :-2 * k - 1])
RELIEF = Z - box / (2 * k + 1) ** 2

# --------------------------------------------------------------------------------------- render geometry
# the physics tile is kept exact; for rendering only, take the mm-level sensor noise off the concrete (3x3 mean keeps
# the 13 mm fault as a 4 cm ramp) and the residual returns off the grass (5x5 median, then at most 4 cm above trend)
def box_mean(a, k):
    pad = np.pad(a, k, mode="edge")
    c = np.pad(pad.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    n = 2 * k + 1
    return (c[n:, n:] - c[:-n, n:] - c[n:, :-n] + c[:-n, :-n]) / n ** 2


def median5(a):
    pad = np.pad(a, 2, mode="edge")
    st_ = np.stack([pad[i:i + a.shape[0], j:j + a.shape[1]] for i in range(5) for j in range(5)])
    return np.median(st_, 0)


hard = CLS == 0
walk_band = (YS[None, :] > WALK_LO) & (YS[None, :] < WALK_HI)
ZR = Z.copy()
conc_s = box_mean(Z, 1)
ZR = np.where(walk_band, conc_s, ZR)
grass_m = median5(Z)
trend = box_mean(grass_m, 15)
ZR = np.where(~walk_band, np.minimum(grass_m, trend + 0.03), ZR)
if PHOTO:  # the photo texture carries the grass detail; measured blade-level relief only adds odd shading bands
    ZR = np.where(~walk_band, np.minimum(box_mean(grass_m, 4), trend + 0.02), ZR)

# --------------------------------------------------------------------------------------- textures
TEX_RES = 0.005
tag = f"wb2_{Path(hm_file).parent.name}_{TEX_RES}_{a.grass_photo}_{a.concrete_photo}".replace(":", "-")
tex_png = cache / f"ground_{tag}.png"
grass_tile = cache / f"wb2_grass_tile_{a.grass_photo.replace(':', '-')}.png"
conc_tile = cache / f"wb2_conc_tile_{a.concrete_photo.replace(':', '-')}.png"
if not tex_png.exists():
    # white balance: the X7's photos have a strong cool cast, so keep each photo's detail (per-pixel ratio to its
    # mean) and set the mean colour to a daylight-balanced target for the material
    def balance(ph, target, contrast=1.0):
        m = ph.reshape(-1, 3).mean(0)
        lum = ph.mean(-1, keepdims=True) / m.mean()
        chroma = ph / (ph.mean(-1, keepdims=True) + 1e-6) / (m / m.mean())  # residual colour variation
        return np.clip(np.asarray(target) * (1 + contrast * (lum - 1)) * chroma ** 0.5, 0, 1)

    gphoto = balance(photo_crop(a.grass_photo), (0.45, 0.385, 0.29), 1.3)  # dormant grass, leaves
    cphoto = balance(photo_crop(a.concrete_photo), (0.62, 0.605, 0.575), 1.0)
    Image.fromarray((gphoto * 255).astype(np.uint8)).save(grass_tile)
    Image.fromarray((cphoto * 255).astype(np.uint8)).save(conc_tile)
    tx = np.arange(-L / 2, L / 2, TEX_RES)
    ty = np.arange(-W / 2, W / 2, TEX_RES)
    TX, TY = np.meshgrid(tx, ty, indexing="xy")  # image rows = y
    shp = TX.shape
    print(f"[rtx-scan] building ground texture {shp[1]} x {shp[0]} px from the scan's photos", flush=True)

    def tiled(photo, metres, rot_seed):
        """Sample a tileable photo with `metres` per repeat; vary orientation per repeat to hide tiling."""
        n = photo.shape[0]
        u = (TX / metres) % 1.0
        v = (TY / metres) % 1.0
        cell = (np.floor(TX / metres) * 7 + np.floor(TY / metres) * 13 + rot_seed).astype(int) % 4
        uu = np.where(cell % 2 == 1, 1 - u, u)
        vv = np.where(cell >= 2, 1 - v, v)
        return photo[(vv * (n - 1)).astype(int), (uu * (n - 1)).astype(int)]

    # concrete: the photo's colour and fine aggregate texture, its slab-scale tone removed (we add our own mottling)
    c = tiled(cphoto, 0.9, 1)
    c_lum = c.mean(-1, keepdims=True)
    c_detail = c / (blur(c_lum[..., 0], 40)[..., None] + 1e-6)
    base = cphoto.reshape(-1, 3).mean(0) * 1.05
    mott = 0.045 * noise(shp, 70, 1) + 0.02 * noise(shp, 14, 2)
    stains = np.clip(noise(shp, 120, 4) - 1.3, 0, None) * 0.07
    rel = resample(RELIEF, TX, TY)
    filled = resample((CLS == 2).astype(float), TX, TY) > 0.05
    filled = blur(filled.astype(float), 6) > 0.02
    groove = np.clip((-rel - 0.0006) / 0.0025, 0, 1) * (~filled)  # joints from the measured grooves
    edge_dirt = np.clip(1 - np.minimum(TY - WALK_LO, WALK_HI - TY) / 0.10, 0, 1) * 0.12
    conc = base[None, None, :] * c_detail * (1 + mott - stains - edge_dirt)[..., None]
    conc *= (1 - 0.5 * groove)[..., None]
    # grass: the site's own dormant grass, two scales blended, slight large-scale tone variation
    g = 0.6 * tiled(gphoto, 1.6, 2) + 0.4 * tiled(np.rot90(gphoto), 2.3, 3)
    g *= (1 + 0.10 * noise(shp, 150, 5))[..., None]
    near = np.clip(1 - np.minimum(np.abs(TY - WALK_LO), np.abs(TY - WALK_HI)) / 0.08, 0, 1)
    soil = gphoto.reshape(-1, 3).mean(0) * 0.8
    g = g * (1 - 0.5 * near[..., None]) + soil * 0.5 * near[..., None]
    edge_n = 0.012 * noise(shp, 3, 9)
    inside = blur(((TY > WALK_LO + edge_n) & (TY < WALK_HI - edge_n)).astype(float), 0.6)
    img = conc * inside[..., None] + g * (1 - inside[..., None])
    Image.fromarray(np.flipud((np.clip(img, 0, 1) * 255).astype(np.uint8))).save(tex_png)  # row 0 = +y edge

# --------------------------------------------------------------------------------------- stage
ctx = omni.usd.get_context()
ctx.new_stage()
stage = ctx.get_stage()
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
UsdGeom.SetStageMetersPerUnit(stage, 1.0)
UsdGeom.Xform.Define(stage, "/World")
mats: dict[str, str] = {}


def preview(name, rgb=(0.5, 0.5, 0.5), rough=0.85, tex=None, alpha=False):
    if name in mats:
        return mats[name]
    path = f"/World/Looks/{name}"
    m = UsdShade.Material.Define(stage, path)
    sh = UsdShade.Shader.Define(stage, path + "/Shader")
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(rough)
    sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    if tex:
        st = UsdShade.Shader.Define(stage, path + "/st")
        st.CreateIdAttr("UsdPrimvarReader_float2")
        st.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
        tx = UsdShade.Shader.Define(stage, path + "/tex")
        tx.CreateIdAttr("UsdUVTexture")
        tx.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(str(tex).replace("\\", "/"))
        tx.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
        tx.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
        tx.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
        tx.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(st.ConnectableAPI(), "result")
        tx.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
        sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(tx.ConnectableAPI(), "rgb")
        if alpha:  # fractional cutout from the texture's alpha
            tx.CreateOutput("a", Sdf.ValueTypeNames.Float)
            sh.CreateInput("opacity", Sdf.ValueTypeNames.Float).ConnectToSource(tx.ConnectableAPI(), "a")
            sh.CreateInput("opacityThreshold", Sdf.ValueTypeNames.Float).Set(0.0)
    else:
        sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgb))
    m.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    mats[name] = path
    return path


def mesh(path, X, Y, Zm, st, mat, normals=False):
    nx, ny = X.shape
    pts = np.stack([X, Y, Zm], -1).reshape(-1, 3)
    i = np.arange(nx * ny).reshape(nx, ny)
    q = np.stack([i[:-1, :-1], i[1:, :-1], i[1:, 1:], i[:-1, 1:]], -1).reshape(-1, 4)
    m = UsdGeom.Mesh.Define(stage, path)
    m.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(pts.astype(np.float32)))
    m.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(q), 4, dtype=np.int32)))
    m.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(q.reshape(-1).astype(np.int32)))
    m.CreateSubdivisionSchemeAttr("none")
    UsdGeom.PrimvarsAPI(m).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex).Set(
        Vt.Vec2fArray.FromNumpy(st.reshape(-1, 2).astype(np.float32)))
    if normals:  # smooth vertex normals from the grid (faceted flat shading otherwise)
        dzx = np.gradient(Zm, axis=0) / np.gradient(X, axis=0)
        dzy = np.gradient(Zm, axis=1) / np.gradient(Y, axis=1)
        nrm = np.stack([-dzx, -dzy, np.ones_like(Zm)], -1)
        nrm /= np.linalg.norm(nrm, axis=-1, keepdims=True)
        m.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(nrm.reshape(-1, 3).astype(np.float32)))
        m.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
    UsdShade.MaterialBindingAPI.Apply(m.GetPrim()).Bind(UsdShade.Material(stage.GetPrimAtPath(mat)))
    return m


# lighting.  Photo mode: the dome is the station's own HDR panorama (scene frame, lat-long), normalised so a white
# lambert ground under it has radiance 1/pi; DOME_CAL brings that to the renderer's exposure
DOME_CAL = 2200.0
dome = UsdLux.DomeLight.Define(stage, "/World/Sky")
dome_tex = dome.CreateTextureFileAttr(f"{ASSETS}/{a.sky}")
dome.CreateIntensityAttr(a.dome_intensity if a.dome_intensity is not None else (DOME_CAL if PHOTO else 500.0))
dome.CreateTextureFormatAttr("latlong")


def set_backdrop(eye):
    """Photo mode: switch the dome to the panorama of the station nearest the camera (least parallax)."""
    if not PHOTO:
        return
    sts = {int(k): v for k, v in PHOTO["stations_scene"].items()}
    k = a.station or min(sts, key=lambda q: (sts[q][0] - eye[0]) ** 2 + (sts[q][1] - eye[1]) ** 2)
    dome_tex.Set(str(PHOTO["dir"] / f"pano_{k}.hdr").replace("\\", "/"))
    return k
sun = UsdLux.DistantLight.Define(stage, "/World/Sun")
sun.CreateIntensityAttr(a.sun_intensity if a.sun_intensity is not None else (0.0 if PHOTO else 1500.0))
sun.CreateAngleAttr(0.53)
sun.CreateColorAttr(Gf.Vec3f(1.0, 0.90, 0.78))
# DistantLight shines along its -Z; rotate so light comes from (az, elev)
xf = UsdGeom.Xformable(sun)
xf.AddRotateZOp().Set(a.sun_az + 90.0)
xf.AddRotateXOp().Set(90.0 - a.sun_elev)

# terrain: the measured tile (2 cm), texture over the whole tile
X, Y = np.meshgrid(XS, YS, indexing="ij")
st = np.stack([(X + L / 2) / L, (Y + W / 2) / W], -1)
ground_mat = preview("ScanGround", tex=PHOTO["dir"] / "ortho_tile.png" if PHOTO else tex_png, rough=0.9)
mesh("/World/Ground/Tile", X, Y, ZR, st, ground_mat, normals=True)

# surroundings: grass continuing the tile's edge heights (and the walk continuing its grade beyond the ends)
if PHOTO:
    LO, WO = PHOTO["outer_size"]
    ax_ = min(a.apron * 1.05, LO / 2 - L / 2)
    ay_ = min(a.apron * 1.05, WO / 2 - W / 2)
    gx = np.concatenate([-L / 2 - np.geomspace(ax_, 0.3, 50), np.linspace(-L / 2, L / 2, 401),
                         L / 2 + np.geomspace(0.3, ax_, 50)])
    gy = np.concatenate([-W / 2 - np.geomspace(ay_, 0.3, 50), np.linspace(-W / 2, W / 2, 71),
                         W / 2 + np.geomspace(0.3, ay_, 50)])
else:
  gx = np.concatenate([-L / 2 - np.geomspace(600, 0.5, 40), np.linspace(-L / 2, L / 2, 81), L / 2 + np.geomspace(0.5, 600, 40)])
  gy = np.concatenate([-W / 2 - np.geomspace(600, 0.5, 40), np.linspace(-W / 2, W / 2, 15), W / 2 + np.geomspace(0.5, 600, 40)])
GX, GY = np.meshgrid(np.unique(gx), np.unique(gy), indexing="ij")
slope_end = (hz(L / 2 - 0.1, 0) - hz(L / 2 - 3.1, 0)) / 3.0
slope_start = (hz(-L / 2 + 3.1, 0) - hz(-L / 2 + 0.1, 0)) / 3.0
GZ = hz(GX, GY) + np.clip(GX - L / 2, 0, 30) * slope_end + np.clip(GX + L / 2, -30, 0) * slope_start  # grade levels off 30 m out
inside = (np.abs(GX) < L / 2 - 0.05) & (np.abs(GY) < W / 2 - 0.05)
if PHOTO and (PHOTO["dir"] / "dem.npz").exists():  # the lidar ground model (road lower, real slopes)
    DM = np.load(PHOTO["dir"] / "dem.npz")
    dzr, dx0, dy0 = float(DM["res"]), float(DM["x0"]), float(DM["y0"])
    dzz = DM["z"].astype(np.float64)
    u_ = np.clip((GX - dx0) / dzr, 0, dzz.shape[0] - 1)
    v_ = np.clip((GY - dy0) / dzr, 0, dzz.shape[1] - 1)
    i_ = np.minimum(np.floor(u_).astype(int), dzz.shape[0] - 2)
    j_ = np.minimum(np.floor(v_).astype(int), dzz.shape[1] - 2)
    fu_, fv_ = u_ - i_, v_ - j_
    GZ = ((1 - fu_) * (1 - fv_) * dzz[i_, j_] + fu_ * (1 - fv_) * dzz[i_ + 1, j_] + (1 - fu_) * fv_ * dzz[i_, j_ + 1]
          + fu_ * fv_ * dzz[i_ + 1, j_ + 1])
GZ = np.where(inside, GZ - 0.05, GZ - 0.004)  # hidden under the tile; seamless at its rim
if PHOTO:
    # the projected ground is only right near the stations (anything standing smears outward), so fade it out
    # beyond --apron metres from the tile and let the dome (the station's own panorama) carry the distance
    LO, WO = PHOTO["outer_size"]
    o_png = cache / (f"outer_alpha_{a.apron:g}_{Path(hm_file).parent.name}_"
                     f"{int((PHOTO['dir'] / 'ortho_outer.png').stat().st_mtime)}.png")
    if not o_png.exists():
        oim = np.asarray(Image.open(PHOTO["dir"] / "ortho_outer.png").convert("RGB"))
        oh, ow = oim.shape[:2]
        ox_ = -LO / 2 + (np.arange(ow) + 0.5) * LO / ow
        oy_ = WO / 2 - (np.arange(oh) + 0.5) * WO / oh
        ex = np.maximum(np.abs(ox_)[None, :] - L / 2, 0)
        ey = np.maximum(np.abs(oy_)[:, None] - W / 2, 0)
        dd = np.hypot(ex, ey * 1.0)
        al = np.clip(1 - (dd - 0.5 * a.apron) / (0.5 * a.apron), 0, 1) ** 1.5
        Image.fromarray(np.dstack([oim, (al * 255).astype(np.uint8)])).save(o_png)
    outer_mat = preview("OrthoOuter", tex=o_png, rough=0.95, alpha=True)
    mesh("/World/Ground/Outer", GX, GY, GZ, np.stack([GX / LO + 0.5, GY / WO + 0.5], -1), outer_mat, normals=True)
else:
  grass_mat = preview("GrassTile", tex=grass_tile, rough=0.95)
  mesh("/World/Ground/Grass", GX, GY, GZ, np.stack([GX / 1.6, GY / 1.6], -1), grass_mat, normals=True)
conc_plain = preview("ConcretePhoto", tex=conc_tile, rough=0.9)
for side, (xa, xb) in ((-1, (-L / 2 - 60, -L / 2 + 0.02)), (1, (L / 2 - 0.02, L / 2 + 60))):
    ex = np.linspace(xa, xb, 600)
    ey = np.array([WALK_LO, WALK_HI])
    EX, EY = np.meshgrid(ex, ey, indexing="ij")
    ref = L / 2 * side
    EZ = hz(ref - side * 0.1, EY) + np.clip(EX - ref, -30, 30) * (slope_end if side > 0 else slope_start) + 0.002
    if not PHOTO:  # the outer orthophoto already shows the real walk continuing
        mesh(f"/World/Ground/WalkExt{side + 1}", EX, EY, EZ, np.stack([EX / 0.9, EY / 0.9], -1), conc_plain)

# robot posed from the recorded physics run
links, traj = r["links"], np.asarray(r["trajectory"])
T = r["time_s"]
dt = T / max(len(traj) - 1, 1)
link_ops = {}
if not a.no_robot:
    go2 = stage.DefinePrim("/World/Go2", "Xform")
    go2.GetReferences().AddReference(a.robot_usd or f"{ASSETS}/Isaac/Robots/Unitree/Go2/go2.usd")
    for prim in Usd.PrimRange(go2):
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            UsdPhysics.RigidBodyAPI(prim).CreateRigidBodyEnabledAttr(False)
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)
        if prim.IsA(UsdPhysics.Joint):
            prim.SetActive(False)
    for _ in range(3):
        app.update()
    for i, name in enumerate(r["body_names"]):
        prim = stage.GetPrimAtPath(f"/World/Go2/{name}")
        if prim.IsValid():
            x_ = UsdGeom.Xformable(prim)
            x_.ClearXformOpOrder()
            link_ops[i] = (x_.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble),
                           x_.AddOrientOp(UsdGeom.XformOp.PrecisionDouble))


def k_at(t):
    return min(int(round(min(t, T) / dt)), len(links) - 1)


def pose_robot(t):
    ln = links[k_at(t)]
    for i, (top, oop) in link_ops.items():
        top.Set(Gf.Vec3d(*ln["pos"][i]))
        w, qx, qy, qz = ln["quat"][i]
        oop.Set(Gf.Quatd(w, qx, qy, qz))


# camera
cam = UsdGeom.Camera.Define(stage, "/World/Cam")
hap = 20.955
vap = hap * a.height / a.width
cam.CreateHorizontalApertureAttr(hap)
cam.CreateVerticalApertureAttr(vap)
cam.CreateFocalLengthAttr(vap / (2 * math.tan(math.radians(VFOV / 2))))
cam.CreateClippingRangeAttr(Gf.Vec2f(0.03, 800.0))
cam_op = UsdGeom.Xformable(cam).AddTransformOp()


def set_camera(eye, foc):
    cam_op.Set(Gf.Matrix4d().SetLookAt(Gf.Vec3d(*map(float, eye)), Gf.Vec3d(*map(float, foc)),
                                       Gf.Vec3d(0, 0, 1)).GetInverse())


def view(spec):
    name, _, arg = spec.partition(":")
    if name == "chase":
        t = float(arg or 20)
        pose_robot(t)
        x, y, yaw = traj[k_at(t)][:3]
        hd = np.array([math.cos(yaw), math.sin(yaw), 0.0])
        right = np.array([hd[1], -hd[0], 0.0])
        rb = np.array([x, y, hz(x, y)])
        foc = rb + 1.0 * hd + [0, 0, 0.15]
        eye = foc - 3.0 * hd + 1.6 * right + [0, 0, 1.25]
        return f"chase_t{t:.0f}", eye, foc, t
    if name == "follow":  # low camera behind the robot, horizon in frame (the backdrop is part of the shot)
        t = float(arg or 20)
        pose_robot(t)
        x, y, yaw = traj[k_at(t)][:3]
        hd = np.array([math.cos(yaw), math.sin(yaw), 0.0])
        right = np.array([hd[1], -hd[0], 0.0])
        rb = np.array([x, y, hz(x, y)])
        eye = rb - 2.6 * hd + 0.5 * right + [0, 0, 0.75]
        foc = rb + 6.0 * hd + [0, 0, 0.35]
        return f"follow_t{t:.0f}", eye, foc, t
    if name == "look":  # look:X:Y:AZ:H:T  camera at scene (X, Y), H above ground, azimuth AZ deg, robot at time T
        x, y, az, hgt, t = (float(v) for v in arg.split(":"))
        pose_robot(t)
        e = np.array([x, y, hz(x, y) + hgt])
        d = np.array([math.cos(math.radians(az)), math.sin(math.radians(az)), 0.0])
        f = e + 10 * d
        f[2] = hz(f[0], f[1]) + 0.6 * hgt
        return f"look_{x:g}_{y:g}_{az:g}", e, f, t
    if name == "eye":
        arg, _, hgt = arg.partition(":")
        x = float(arg or -14)
        hgt = float(hgt or 1.6)
        t = float(np.clip((x + 0.5 + L / 2 - 1) / 0.7, 0, T))
        pose_robot(min(T, t + 5.0))
        e = np.array([x, 0.35, hz(x, 0.35) + hgt])
        f = np.array([x + 12, 0.0, hz(x + 12, 0) + 0.2 * hgt / 1.6])
        return f"eye_x{x:.0f}" + (f"_h{hgt:g}" if hgt != 1.6 else ""), e, f, None
    if name == "fault":
        pose_robot(T)
        # from the +y side looking across: the background verge (-y) lies beside stations 4 and 5, so its photo
        # texture is sharp; the far +y verge is only seen obliquely and smears
        e = np.array([9.1, 1.05, hz(9.1, 1.05) + 0.75])
        f = np.array([10.8, 0.35, hz(10.8, 0.35)])
        return "fault", e, f, None
    if name == "aerial":
        pose_robot(25.0)
        e = np.array([-22.0, -9.0, hz(-14, 0) + 7.5])
        f = np.array([-2.0, 0.6, hz(-2, 0)])
        return "aerial", e, f, None
    raise ValueError(spec)


rp = rep.create.render_product("/World/Cam", (a.width, a.height))
ann = rep.AnnotatorRegistry.get_annotator("rgb")
ann.attach([rp])
for _ in range(30):
    app.update()
for spec in a.views:
    nm, eye, foc, _ = view(spec)
    set_camera(eye, foc)
    k_ = set_backdrop(eye)
    if k_:
        print(f"[rtx-scan] {nm}: backdrop = station {k_} panorama", flush=True)
    for _ in range(20):
        app.update()
    rep.orchestrator.step(rt_subframes=a.subframes)
    img = np.asarray(ann.get_data())
    Image.fromarray(img[..., :3].astype(np.uint8)).save(out_dir / f"{nm}.png")
    print(f"[rtx-scan] wrote {out_dir / (nm + '.png')}", flush=True)
if a.video > 0:
    fdir = out_dir / f"frames_{a.video_view}"
    fdir.mkdir(parents=True, exist_ok=True)
    times = np.arange(0.0, T + 1e-9, 1.0 / a.video)
    for i, t in enumerate(times):
        _, eye, foc, _ = view(f"{a.video_view}:{t}")
        set_camera(eye, foc)
        set_backdrop(eye)
        for _ in range(a.settle):
            app.update()
        rep.orchestrator.step(rt_subframes=a.subframes)
        img = np.asarray(ann.get_data())
        Image.fromarray(img[..., :3].astype(np.uint8)).save(fdir / f"f{i:05d}.png")
        if i % 25 == 0:
            print(f"[rtx-scan] video frame {i + 1}/{len(times)} (t = {t:.1f} s)", flush=True)
    print(f"[rtx-scan] wrote {len(times)} frames to {fdir}", flush=True)
if a.timeline:
    frames = json.loads(Path(a.timeline).read_text())["frames"]
    lo, hi = a.range if a.range else (0, len(frames))
    for i in range(lo, min(hi, len(frames))):
        f = frames[i]
        dst = out_dir / f"{i:05d}.png"
        if dst.exists():
            continue
        pose_robot(f["t"])
        set_camera(f["eye"], f["foc"])
        set_backdrop(np.asarray(f["eye"], dtype=float))
        for _ in range(a.settle):
            app.update()
        rep.orchestrator.step(rt_subframes=a.subframes)
        Image.fromarray(np.asarray(ann.get_data())[..., :3].astype(np.uint8)).save(dst)
        if i % 50 == 0:
            print(f"[rtx-scan] timeline frame {i}/{len(frames)}", flush=True)
    print(f"[rtx-scan] timeline done ({len(frames)} frames)", flush=True)
app.close()
