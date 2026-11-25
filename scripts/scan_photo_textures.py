"""Photo textures for a scanned-sidewalk tile from the Trimble X7's own photos (for photoreal rendering).

  PYTHONPATH=src python scripts/scan_photo_textures.py --tile-dir scan/sidewalk_tdx

Writes to <tile-dir>/photo/:
  ortho_tile.png    true orthophoto of the tile (scene frame, row 0 = +y edge, column 0 = -x edge): every
                    station's photos projected onto the measured heightmap, blended with a robust vote across
                    stations (rejects people, tripod legs, the scanner's own footprint, trunk occlusions)
  dem.npz           ground model beyond the tile from the lidar (lowest return per 10 cm cell, trunks / posts /
                    cars removed by a 3 m morphological opening, gaps filled), joined to the tile heightmap at its rim
  ortho_outer.png   same as ortho_tile beyond the tile, projected on dem.npz (coarser), for the surroundings
  pano_<k>.hdr      HDR 360 deg panorama of station k in the scene frame, as an Omniverse dome lat-long texture
                    (z up; drop it on a DomeLight unrotated), same colour / exposure calibration as the orthophotos
  photo.json        frames, station positions, gains, irradiance normalisation

Colour: the X7 shoots with white balance off (cool cast) and the light changed between stations (dusk), so
per-station gains are solved from the ground they share, then one white balance makes the measured concrete a
neutral light grey. Albedo = radiance / (E / pi), with E the sky irradiance from the panorama, so rendering the
orthophoto under the panorama dome reproduces the photographed brightness.
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

from go2wm.common.config import resolve_path
from go2wm.scan.photos import StationPhotos, equirect_dirs, linear_to_srgb, tcf_path, write_hdr
from go2wm.scan.tzf import station_poses


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tile-dir", default="scan/sidewalk_tdx")
    ap.add_argument("--tile-res", type=float, default=0.01, help="orthophoto pixel over the tile [m]")
    ap.add_argument("--outer-res", type=float, default=0.05)
    ap.add_argument("--outer-size", type=float, nargs=2, default=[120.0, 70.0], help="outer orthophoto extent [m]")
    ap.add_argument("--pano-width", type=int, default=4096)
    ap.add_argument("--min-dist", type=float, default=0.75, help="ignore a station's photos within this radius [m]")
    ap.add_argument("--concrete", type=float, nargs=3, default=[0.30, 0.29, 0.27], help="target linear albedo")
    ap.add_argument("--tile-only", action="store_true", help="only the tile orthophoto (quick look; no panoramas)")
    ap.add_argument("--outer-only", action="store_true",
                    help="only rebuild dem.npz + ortho_outer.png, reusing photo.json's calibration")
    ap.add_argument("--dem-res", type=float, default=0.10)
    args = ap.parse_args(argv)

    tdir = resolve_path(args.tile_dir)
    out = tdir / "photo"
    out.mkdir(exist_ok=True)
    H = np.load(tdir / "heightmap.npz")
    meta = ast.literal_eval(str(H["meta"]))
    tdx = Path(meta["source"])
    fr = meta["frame"]
    cx, cy = fr["centre_xy"]
    hd = fr["heading"]
    Z, CLS = H["z"].astype(np.float64), H["cls"]
    res, x0, y0 = float(H["res"]), float(H["x0"]), float(H["y0"])
    NX, NY = Z.shape
    L, W = fr["size"]
    c, s = np.cos(hd), np.sin(hd)
    stations = station_poses(tdx)

    # scene <-> world.  heightmap z is world z minus the frame's z0
    z0 = float(meta["z0_world"])

    def scene_to_world(P):
        return np.stack([cx + c * P[..., 0] - s * P[..., 1], cy + s * P[..., 0] + c * P[..., 1], P[..., 2] + z0], -1)

    def hz(x, y):
        u = np.clip((x - x0) / res, 0, NX - 1)
        v = np.clip((y - y0) / res, 0, NY - 1)
        i = np.minimum(np.floor(u).astype(int), NX - 2)
        j = np.minimum(np.floor(v).astype(int), NY - 2)
        fu, fv = u - i, v - j
        return ((1 - fu) * (1 - fv) * Z[i, j] + fu * (1 - fv) * Z[i + 1, j] + (1 - fu) * fv * Z[i, j + 1]
                + fu * fv * Z[i + 1, j + 1])

    st_scene = {}
    for k, st in stations.items():
        d = st["p"] - np.array([cx, cy, z0])
        st_scene[k] = [float(c * d[0] + s * d[1]), float(-s * d[0] + c * d[1]), float(d[2])]
    print("stations (scene):", {k: np.round(v, 2).tolist() for k, v in st_scene.items()}, flush=True)

    def per_station(P_scene, chunk=400_000):
        """Per-station radiance [S, N, 3] and weight [S, N] for scene points [N, 3]."""
        Pw = scene_to_world(P_scene)
        ks = sorted(stations)
        rgb = np.zeros((len(ks), len(Pw), 3), np.float32)
        wgt = np.zeros((len(ks), len(Pw)), np.float32)
        for a, k in enumerate(ks):
            t = time.time()
            st = stations[k]
            ph = StationPhotos(tcf_path(tdx, k))
            v = Pw - st["p"]
            dist = np.linalg.norm(v, axis=-1)
            dirs = (v / dist[:, None]) @ st["R"].T  # world -> scanner frame
            hdist = np.linalg.norm(v[:, :2], axis=-1)
            for s_ in ph.sets:  # image by image, all points (keeps memory flat)
                col, row, w = ph.lookup(s_["id"], dirs)
                m = w > 0
                if not m.any():
                    continue
                img = ph.hdr(s_["id"])
                idx = np.nonzero(m)[0]
                for b in range(0, len(idx), chunk):
                    ii = idx[b:b + chunk]
                    from go2wm.scan.photos import bilinear
                    rgb[a, ii] += bilinear(img, col[ii], row[ii]) * w[ii, None]
                    wgt[a, ii] += w[ii]
            ph.close()
            nz = wgt[a] > 1e-30
            rgb[a, nz] /= wgt[a, nz, None]
            wgt[a] = np.where(nz, 1.0, 0.0)  # coverage only; the photo-centre weights were for the in-station blend
            # resolution: prefer near stations (ground footprint of a pixel grows ~ d^2 / h at grazing view)
            # and fade a station's own footprint out (tripod legs, scanner head shadow) between 1.6x and 1x min-dist
            own = np.clip((hdist - args.min_dist) / (0.6 * args.min_dist), 0.0, 1.0)
            wgt[a] = np.where(nz, own ** 2 / (1.0 + (hdist / 4.0) ** 2) ** 4, 0.0)
            print(f"  station {k}: {nz.mean():.0%} of points seen, {time.time() - t:.0f} s", flush=True)
        return rgb, wgt

    def robust_blend(rgb, wgt, gains):
        rgb = rgb * gains[:, None, :]
        lum = np.log(np.maximum(rgb.mean(-1), 1e-9))
        # weighted median of log luminance across stations
        order = np.argsort(lum, 0)
        ls = np.take_along_axis(lum, order, 0)
        ws = np.take_along_axis(wgt, order, 0)
        cw = np.cumsum(ws, 0)
        half = cw[-1] / 2
        mi = np.argmax(cw >= half[None], 0)
        med = np.take_along_axis(ls, mi[None], 0)[0]
        w2 = wgt * np.exp(-((lum - med[None]) / 0.18) ** 2)
        tot = w2.sum(0)
        out_ = (rgb * w2[..., None]).sum(0) / np.maximum(tot, 1e-12)[:, None]
        return out_, tot

    t0 = time.time()

    def to_png(rad, shape, path):
        alb = rad / (E_ref / np.pi) * wb
        Image.fromarray((np.clip(linear_to_srgb(alb), 0, 1) * 255).astype(np.uint8).reshape(shape + (3,))).save(path)

    if args.outer_only:
        info = json.loads((out / "photo.json").read_text())
        ks = sorted(stations)
        gains = np.array([info["gains"][str(k)] for k in ks], np.float32)
        wb = np.asarray(info["wb"])
        E_ref = float(info["E_ref"])
        outer(args, out, tdx, stations, per_station, robust_blend, gains, to_png, (cx, cy, z0, c, s), H)
        info["dem_res"] = args.dem_res
        (out / "photo.json").write_text(json.dumps(info, indent=1))
        print(f"outer rebuilt in {time.time() - t0:.0f} s", flush=True)
        return 0
    # ------------------------------------------------------------------ tile orthophoto (on the measured surface)
    tx = -L / 2 + (np.arange(int(round(L / args.tile_res))) + 0.5) * args.tile_res
    ty = W / 2 - (np.arange(int(round(W / args.tile_res))) + 0.5) * args.tile_res  # row 0 = +y edge
    TX, TY = np.meshgrid(tx, ty)
    Pt = np.stack([TX, TY, hz(TX, TY)], -1).reshape(-1, 3)
    print(f"tile orthophoto {TX.shape[1]} x {TX.shape[0]} px", flush=True)
    rgb_t, w_t = per_station(Pt)

    # per-station, per-channel gains from shared ground (the light dimmed and reddened through the dusk scan):
    # log-ratio least squares over ground both stations see from 1.2-12 m, mean log gain 0
    ks = sorted(stations)
    S = len(ks)
    hd_ = np.stack([np.hypot(Pt[:, 0] - st_scene[k][0], Pt[:, 1] - st_scene[k][1]) for k in ks])
    good = (w_t > 0) & (hd_ > 1.2) & (hd_ < 12.0)
    loose = (w_t > 0) & (hd_ > 1.2) & (hd_ < 35.0)  # for a station that shares no near ground (station 6)
    linked = {a for a in range(S) for b in range(S) if a != b and (good[a] & good[b]).sum() > 2000}
    lg = np.zeros((S, 3))
    for ch in range(3):
        lum = np.log(np.maximum(rgb_t[..., ch], 1e-12))
        rows, rhs = [], []
        for a in range(S):
            for b in range(a + 1, S):
                m = good[a] & good[b] if (a in linked and b in linked) else loose[a] & loose[b]
                if m.sum() > 2000:
                    r_ = np.zeros(S)
                    r_[a], r_[b] = 1, -1
                    rows.append(r_ * np.sqrt(m.sum()))
                    rhs.append(np.median(lum[b, m] - lum[a, m]) * np.sqrt(m.sum()))
        rows.append(np.ones(S) * 1e3)
        rhs.append(0.0)
        lg[:, ch] = np.linalg.lstsq(np.array(rows), np.array(rhs), rcond=None)[0]
    del hd_, good, loose
    gains = np.exp(lg).astype(np.float32)
    print("station gains:", dict(zip(ks, np.round(gains, 3).tolist())), flush=True)
    ortho_t, cov_t = robust_blend(rgb_t, w_t, gains)
    del rgb_t, w_t

    # ------------------------------------------------------------------ panoramas -> irradiance normalisation
    D = equirect_dirs(512 if args.tile_only else args.pano_width)  # scene frame
    Dw = np.stack([c * D[..., 0] - s * D[..., 1], s * D[..., 0] + c * D[..., 1], D[..., 2]], -1)
    panos, E = {}, {}
    h_ = D.shape[0]
    el = np.pi / 2 - (np.arange(h_) + 0.5) / h_ * np.pi
    dO = (2 * np.pi / args.pano_width) * (np.pi / h_) * np.cos(el)  # solid angle per pixel row
    for a, k in enumerate(ks):
        t = time.time()
        ph = StationPhotos(tcf_path(tdx, k))
        rgb, w = ph.project(Dw @ stations[k]["R"].T, img_scale=8 if args.tile_only else 2)
        ph.close()
        rgb *= gains[a]
        up = (el > 0)[:, None] & (w > 0)
        E[k] = float(((rgb.mean(-1) * np.sin(el)[:, None] * dO[:, None]) * up).sum())
        # unseen nadir: fill with the mean ground colour (hidden under the ground in renders anyway)
        g = rgb[(el < -0.6)[:, None] & (w > 0)].mean(0)
        rgb[w <= 0] = g
        panos[k] = rgb
        print(f"  pano {k}: E = {E[k]:.4g}, {time.time() - t:.0f} s", flush=True)
    E_ref = float(np.median(list(E.values())))

    # white balance + albedo scale from the measured concrete
    hard = np.asarray(Image.fromarray(CLS.astype(np.uint8).T[::-1]).resize(TX.shape[::-1], Image.NEAREST)) == 0
    walk = hard.reshape(-1) & (np.abs(Pt[:, 1]) < 0.9) & (cov_t > 0.05)
    conc = np.median(ortho_t[walk], 0) / (E_ref / np.pi)
    wb = np.asarray(args.concrete) / conc
    print("concrete albedo before calibration:", np.round(conc, 4), "-> gains", np.round(wb, 3), flush=True)

    # station footprints (tripod, scanner, whatever was set down there) are seen only obliquely from the other
    # stations: clone them from the same row of the walk/verge beside them, from whichever side (+-0.8 .. 1.6 m)
    # has the least edge energy (no slab joint), feathered in
    img = ortho_t.reshape(TX.shape + (3,))
    from go2wm.scan.photos import gauss_blur
    hp_e = np.abs(img.mean(-1) - gauss_blur(img.mean(-1), 3))
    rad = 1.1 * args.min_dist
    for q in st_scene.values():
        if abs(q[0]) > L / 2 + rad or abs(q[1]) > W / 2 + rad:
            continue
        dd = np.hypot(TX - q[0], TY - q[1])
        m = np.clip((rad + 0.15 - dd) / 0.3, 0.0, 1.0)
        if not m.any():
            continue
        r0, r1 = np.nonzero(m.any(1))[0][[0, -1]]
        c0, c1 = np.nonzero(m.any(0))[0][[0, -1]]
        best = None
        for sh_m in (0.9, 1.1, 1.3, 1.5, -0.9, -1.1, -1.3, -1.5):
            sh = int(round((2 * rad + abs(sh_m) - 0.9) / args.tile_res)) * int(np.sign(sh_m))
            if c0 + sh < 0 or c1 + sh >= TX.shape[1]:
                continue
            e = float(hp_e[r0:r1 + 1, c0 + sh:c1 + sh + 1][m[r0:r1 + 1, c0:c1 + 1] > 0].mean())
            if best is None or e < best[0]:
                best = (e, sh)
        if best is None:
            continue
        sh = best[1]
        patch = img[r0:r1 + 1, c0 + sh:c1 + sh + 1].copy()
        mm = m[r0:r1 + 1, c0:c1 + 1, None]
        img[r0:r1 + 1, c0:c1 + 1] = img[r0:r1 + 1, c0:c1 + 1] * (1 - mm) + patch * mm
    ortho_t = np.clip(img, 0, None).reshape(-1, 3)
    to_png(ortho_t, TX.shape, out / "ortho_tile.png")
    Image.fromarray((np.clip(cov_t / 0.3, 0, 1) * 255).astype(np.uint8).reshape(TX.shape)).save(out / "ortho_tile_cov.png")

    if args.tile_only:
        return 0
    LO, WO = args.outer_size
    outer(args, out, tdx, stations, per_station, robust_blend, gains, to_png, (cx, cy, z0, c, s), H)

    # ------------------------------------------------------------------ panoramas, same calibration
    for k, rgb in panos.items():
        # white lambert ground under it -> radiance 1/pi.  Stored in Kit's dome lat-long convention (checked with a
        # marker render, Z-up stage): direction azimuth az reads our column of 180 - az, i.e. mirrored + half-turn
        write_hdr(out / f"pano_{k}.hdr", np.roll((rgb / E_ref * wb)[:, ::-1], rgb.shape[1] // 2, 1))
    info = {"tile_res": args.tile_res, "tile_size": [L, W], "outer_res": args.outer_res, "outer_size": [LO, WO],
            "stations_scene": st_scene, "gains": dict(zip(map(str, ks), gains.tolist())), "wb": wb.tolist(),
            "E": {str(k): v / E_ref for k, v in E.items()}, "E_ref": E_ref, "dem_res": args.dem_res,
            "seconds": round(time.time() - t0)}
    (out / "photo.json").write_text(json.dumps(info, indent=1))
    print(info, flush=True)
    return 0


def ground_dem(stations, frame, extent, res, H):
    """Ground elevation (scene z) on a res grid over extent = (LO, WO) centred on the scene origin, from the lidar:
    lowest return per cell, a morphological opening (3 m footprint) rejecting what stands on the ground
    (trunks, posts, people, cars), gaps filled by normalized convolution; inside the tile the measured heightmap, joined over a 0.6 m band."""
    from scipy import ndimage

    from go2wm.scan.tzf import TzfScan

    cx, cy, z0, c, s = frame
    LO, WO = extent
    nx, ny = int(round(LO / res)), int(round(WO / res))
    zmin = np.full(nx * ny, np.inf)
    for k, st in sorted(stations.items()):
        with TzfScan(st["scan"]) as scan:
            for tx, ty in scan.tiles:
                X = scan.tile_points(tx, ty, 2)
                if not len(X):
                    continue
                Wp = X @ st["R"] + st["p"]
                dx, dy = Wp[:, 0] - cx, Wp[:, 1] - cy
                qx, qy = c * dx + s * dy, -s * dx + c * dy
                i = np.floor((qx + LO / 2) / res).astype(int)
                j = np.floor((qy + WO / 2) / res).astype(int)
                ok = (i >= 0) & (i < nx) & (j >= 0) & (j < ny) & (np.hypot(X[:, 0], X[:, 1]) > 0.8)
                np.minimum.at(zmin, i[ok] * ny + j[ok], Wp[ok, 2] - z0)
        print(f"  dem: station {k} done", flush=True)
    z = zmin.reshape(nx, ny)
    have = np.isfinite(z)

    def fill(zz, mask):
        out_ = np.where(mask, zz, np.nan)
        todo = ~mask
        for sg in (3, 10, 30, 100):  # widening passes for bigger gaps
            num = ndimage.gaussian_filter(np.where(mask, zz, 0.0), sg)
            den = ndimage.gaussian_filter(mask.astype(float), sg)
            ok = todo & (den > 1e-4)
            out_[ok] = num[ok] / den[ok]
            todo &= ~ok
            if not todo.any():
                break
        out_[todo] = np.nanmedian(zz[mask])
        return out_

    # morphological ground filter: an opening with a 3 m footprint removes everything narrower than 3 m standing
    # on the ground (trunks, posts, signs, people, cars across their width) but keeps slopes and curb steps
    zf = fill(z, have)
    keep = have.copy()
    for size_m, tol in ((3.0, 0.20), (3.0, 0.10)):
        zo = ndimage.grey_opening(zf, size=(int(size_m / res) | 1,) * 2)
        keep &= z < zo + tol
        zf = fill(z, keep)
    zf = ndimage.gaussian_filter(zf, 0.7)
    # join the tile heightmap
    xs = -LO / 2 + (np.arange(nx) + 0.5) * res
    ys = -WO / 2 + (np.arange(ny) + 0.5) * res
    Xg, Yg = np.meshgrid(xs, ys, indexing="ij")
    Zt, rt, x0, y0 = H["z"].astype(np.float64), float(H["res"]), float(H["x0"]), float(H["y0"])
    L, W = Zt.shape[0] * rt, Zt.shape[1] * rt
    u = np.clip(np.round((Xg - x0) / rt), 0, Zt.shape[0] - 1).astype(int)
    v = np.clip(np.round((Yg - y0) / rt), 0, Zt.shape[1] - 1).astype(int)
    inside = np.minimum(L / 2 - np.abs(Xg), W / 2 - np.abs(Yg))
    wt = np.clip(inside / 0.6, 0, 1)
    zf = wt * Zt[u, v] + (1 - wt) * zf
    return zf.astype(np.float32), -LO / 2 + res / 2, -WO / 2 + res / 2


def outer(args, out, tdx, stations, per_station, robust_blend, gains, to_png, frame, H):
    LO, WO = args.outer_size
    t = time.time()
    dem, dx0, dy0 = ground_dem(stations, frame, (LO, WO), args.dem_res, H)
    np.savez_compressed(out / "dem.npz", z=dem, res=args.dem_res, x0=dx0, y0=dy0)
    print(f"dem {dem.shape} in {time.time() - t:.0f} s, z range {dem.min():.2f} .. {dem.max():.2f}", flush=True)
    r = args.dem_res

    def dz(x, y):
        u = np.clip((x - dx0) / r, 0, dem.shape[0] - 1)
        v = np.clip((y - dy0) / r, 0, dem.shape[1] - 1)
        i = np.minimum(np.floor(u).astype(int), dem.shape[0] - 2)
        j = np.minimum(np.floor(v).astype(int), dem.shape[1] - 2)
        fu, fv = u - i, v - j
        return ((1 - fu) * (1 - fv) * dem[i, j] + fu * (1 - fv) * dem[i + 1, j] + (1 - fu) * fv * dem[i, j + 1]
                + fu * fv * dem[i + 1, j + 1])

    ox = -LO / 2 + (np.arange(int(round(LO / args.outer_res))) + 0.5) * args.outer_res
    oy = WO / 2 - (np.arange(int(round(WO / args.outer_res))) + 0.5) * args.outer_res
    OX, OY = np.meshgrid(ox, oy)
    Po = np.stack([OX, OY, dz(OX, OY)], -1).reshape(-1, 3)
    print(f"outer orthophoto {OX.shape[1]} x {OX.shape[0]} px (on the lidar ground model)", flush=True)
    rgb_o, w_o = per_station(Po)
    ortho_o, cov_o = robust_blend(rgb_o, w_o, gains)
    del rgb_o, w_o
    to_png(ortho_o, OX.shape, out / "ortho_outer.png")
    Image.fromarray((np.clip(cov_o / 0.3, 0, 1) * 255).astype(np.uint8).reshape(OX.shape)).save(
        out / "ortho_outer_cov.png")


if __name__ == "__main__":
    sys.exit(main())
