"""Texture the Isaac sidewalk terrain mesh with the procedural albedo (sim/sidewalk_texture.py).

The terrain generator merges all tiles (plus the flat border) into one mesh. This bakes one image over
the mesh's xy extent, painting each pixel with the tile whose origin is nearest (tile-local scene
coordinates = world xy - tile origin xy), writes it as a PNG (cached by content hash), adds planar
`st` texture coordinates to the mesh and binds a UsdPreviewSurface material that samples the image.
Rendering only: collision geometry and physics materials are untouched.

Import only after the Isaac app is running (needs pxr).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .sidewalk_texture import sidewalk_albedo


def bake_albedo(tiles: list[dict[str, Any]], origins_xy: np.ndarray, x_range, y_range, res: float,
                rows_per_block: int = 128) -> np.ndarray:
    """Image [H, W, 3] uint8 over [x_range] x [y_range]; row 0 = y_min (flip before saving as PNG)."""
    W = int(np.ceil((x_range[1] - x_range[0]) / res))
    H = int(np.ceil((y_range[1] - y_range[0]) / res))
    xs = x_range[0] + (np.arange(W) + 0.5) * res
    ys = y_range[0] + (np.arange(H) + 0.5) * res
    img = np.zeros((H, W, 3), dtype=np.uint8)
    o = np.asarray(origins_xy, dtype=np.float64).reshape(-1, 2)
    for r0 in range(0, H, rows_per_block):
        X, Y = np.meshgrid(xs, ys[r0 : r0 + rows_per_block])
        d = (X[..., None] - o[:, 0]) ** 2 + (Y[..., None] - o[:, 1]) ** 2
        near = np.argmin(d, axis=-1)
        blk = np.zeros(X.shape + (3,))
        for i in np.unique(near):
            m = near == i
            blk[m] = sidewalk_albedo(X[m] - o[i, 0], Y[m] - o[i, 1], tiles[i], seed=int(i))
        img[r0 : r0 + rows_per_block] = np.round(blk * 255).astype(np.uint8)
    return img


def generator_tile_origins_xy(n_tiles: int, size: tuple[float, float]) -> np.ndarray:
    """Tile-centre origins of a 1-row TerrainGenerator layout: tile (0, c) centred at (0.5 sx, (c + 0.5) sy),
    the whole grid shifted by -(sx rows / 2, sy cols / 2). Checked against terrain_origins in
    IsaacGo2Adapter (Isaac Lab 2.3: 6 tiles of 28 x 8 m -> x 0, y -20 ... 20)."""
    sx, sy = size
    return np.array([[0.0 * sx, sy * (c + 0.5 - 0.5 * n_tiles)] for c in range(n_tiles)], dtype=np.float64)


def texture_mesh_and_define_material(stage, mesh_prim, material_path: str, tiles: list[dict[str, Any]],
                                     origins_xy: np.ndarray, cache_dir: str | Path, res: float = 0.02,
                                     max_px: int = 8192, roughness: float = 0.85):
    """Bake (or reuse) the albedo PNG, add `st` to the mesh and define the UsdPreviewSurface material at
    material_path. Returns (material, png path). Binding is left to the caller."""
    from PIL import Image
    from pxr import Sdf, UsdGeom, UsdShade, Vt

    pts = np.asarray(UsdGeom.Mesh(mesh_prim).GetPointsAttr().Get(), dtype=np.float64)
    M = np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(mesh_prim), dtype=np.float64)
    wp = pts @ M[:3, :3] + M[3, :3]  # USD matrices act on row vectors
    lo, hi = wp[:, :2].min(0), wp[:, :2].max(0)
    res = max(res, float(np.max(hi - lo)) / max_px)

    key = hashlib.sha1(json.dumps({"tiles": tiles, "o": np.round(origins_xy, 4).tolist(), "lo": np.round(lo, 4).tolist(),
                                   "hi": np.round(hi, 4).tolist(), "res": res, "v": 1},
                                  sort_keys=True, default=float).encode()).hexdigest()[:16]
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    png = cache_dir / f"sidewalk_{key}.png"
    if not png.exists():
        img = bake_albedo(tiles, origins_xy, (lo[0], hi[0]), (lo[1], hi[1]), res)
        Image.fromarray(img[::-1]).save(png)  # PNG row 0 = top = v 1 = y max
    W, H = Image.open(png).size
    st = np.stack([(wp[:, 0] - lo[0]) / (W * res), (wp[:, 1] - lo[1]) / (H * res)], axis=1).astype(np.float32)
    pv = UsdGeom.PrimvarsAPI(mesh_prim).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex)
    pv.Set(Vt.Vec2fArray.FromNumpy(st))

    mp = material_path
    mat = UsdShade.Material.Define(stage, mp)
    pbr = UsdShade.Shader.Define(stage, f"{mp}/PBR")
    pbr.CreateIdAttr("UsdPreviewSurface")
    pbr.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(float(roughness))
    pbr.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    reader = UsdShade.Shader.Define(stage, f"{mp}/stReader")
    reader.CreateIdAttr("UsdPrimvarReader_float2")
    reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    tex = UsdShade.Shader.Define(stage, f"{mp}/albedo")
    tex.CreateIdAttr("UsdUVTexture")
    tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(png.resolve().as_posix())
    tex.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
    tex.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("clamp")
    tex.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("clamp")
    tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(), "result")
    tex.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
    pbr.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(tex.ConnectableAPI(), "rgb")
    mat.CreateSurfaceOutput().ConnectToSource(pbr.ConnectableAPI(), "surface")
    return mat, png


def sidewalk_material_cfg(tiles: list[dict[str, Any]], size: tuple[float, float], cache_dir: str, res: float = 0.02):
    """A visual-material config for TerrainImporterCfg.visual_material. The importer calls its func right
    after creating the terrain mesh (before the sim starts and before any camera renders), with the
    material path next to the mesh, and then binds the material to the mesh. Import only after the
    Isaac app is running."""
    from collections.abc import Callable

    from isaaclab.sim.spawners.materials import VisualMaterialCfg
    from isaaclab.utils import configclass

    def spawn(prim_path: str, cfg):
        import omni.usd

        stage = omni.usd.get_context().get_stage()
        mesh_prim = stage.GetPrimAtPath(prim_path.rsplit("/", 1)[0] + "/mesh")
        mat, png = texture_mesh_and_define_material(stage, mesh_prim, prim_path, cfg.tiles,
                                                    generator_tile_origins_xy(len(cfg.tiles), cfg.size), cfg.cache_dir,
                                                    cfg.res)
        cfg.png = str(png)
        return mat.GetPrim()

    @configclass
    class SidewalkTextureCfg(VisualMaterialCfg):
        func: Callable = spawn
        tiles: list = None
        size: tuple = (28.0, 8.0)
        cache_dir: str = ""
        res: float = 0.02
        png: str = ""

    return SidewalkTextureCfg(tiles=tiles, size=tuple(size), cache_dir=cache_dir, res=res)


def apply_sidewalk_texture(terrain_prim_path: str, tiles: list[dict[str, Any]], origins_xy: np.ndarray,
                           cache_dir: str | Path, res: float = 0.02, max_px: int = 8192,
                           roughness: float = 0.85) -> Path:
    """Post-hoc variant (after the scene exists). Prefer sidewalk_material_cfg: changing a material after
    the RTX scene is up left tiled-camera frames black in most launches."""
    from pxr import Usd, UsdGeom, UsdShade

    import omni.usd

    stage = omni.usd.get_context().get_stage()
    root = stage.GetPrimAtPath(terrain_prim_path)
    mesh_prim = next(p for p in Usd.PrimRange(root) if p.IsA(UsdGeom.Mesh))
    mat, png = texture_mesh_and_define_material(stage, mesh_prim, f"{terrain_prim_path}/sidewalkMaterial", tiles,
                                                origins_xy, cache_dir, res, max_px, roughness)
    for prim in (mesh_prim, root):  # the importer binds its material on the mesh, stronger than descendants
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(mat, UsdShade.Tokens.strongerThanDescendants)
    return png
