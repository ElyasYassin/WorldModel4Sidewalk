"""The Trimble X7's own photos (.tcf next to each .tzf): HDR merge and projection onto any viewing direction.

Each station's .tcf (SQLite) holds 30 image sets (3 cameras x 10 head azimuths) of 3 bracketed JPEG exposures,
2748 x 3840 px, plus a calibrated camera model per camera. The geometry was worked out on our 2026-11-18 scan
(seam correlation between overlapping photos peaks with the stored calibration unchanged):
  * ImageSet orientation rows U, V, W are the camera axes in the scanner frame; the optical axis is +U,
    image columns run along -V and rows along -W (cameras at -31, +12 and +52 deg elevation);
  * cameraModelExt: rectilinear, |principalDistance| = focal length in px, principal point (col, row), and an
    "angle-symmetric" radial correction dr(r) = k3 r^3 + k5 r^5 + ... (r in px from the principal point):
    undistorted radius = r - dr(r), i.e. theta = atan((r - dr(r)) / f). Valid out to r ~ 1300 px (the lens
    vignette circle); beyond that the series diverges.
  * camera offsets from the scanner origin are ~8 cm, ignored (fine beyond ~3 m).
Exposure times are relative (merged as radiance = linear / time). Lens vignetting is small (~5 % at r = 1200 px,
measured on ground overlaps). Blending overlapping photos evenly leaves dark radial wedges in ground projections;
taking (almost) only the most central photo per direction (weight ** BLEND_POWER) is seamless and sharp, while
per-photo gain compensation (fit_gains; off by default) made the seams worse. The photos have a strong cool cast
(white balance "Off"); calibrate colour against a known surface (e.g. concrete) after merging.
"""

from __future__ import annotations

import io
import sqlite3
from pathlib import Path

import numpy as np

R_MAX = 1300.0  # px: usable radius (vignette circle)
BLEND_POWER = 8  # within a station: favour the most central photo, feather only a narrow band at seams


def tcf_path(tdx: str | Path, station: int) -> Path:
    tdx = Path(tdx)
    return tdx.parent / f"{tdx.stem}.tdf" / f"{tdx.stem}_{station}.tcf"


def srgb_to_linear(v: np.ndarray) -> np.ndarray:
    return np.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(v: np.ndarray) -> np.ndarray:
    v = np.clip(v, 0, None)
    return np.where(v <= 0.0031308, 12.92 * v, 1.055 * v ** (1 / 2.4) - 0.055)


class StationPhotos:
    """The photo rig of one station. Directions are in the scanner frame (world = scanner @ R + p)."""

    def __init__(self, tcf: str | Path):
        from PIL import Image  # noqa: F401  (fail early if missing)

        self.path = Path(tcf)
        self._con = sqlite3.connect(f"file:{self.path.as_posix()}?mode=ro", uri=True)
        cams = {}
        for r in self._con.execute(
                "select cameraModelExtId, principalPointColumn, principalPointRow, principalDistance, "
                "angleSymmetricDistortionOrder3, angleSymmetricDistortionOrder5, angleSymmetricDistortionOrder7, "
                "angleSymmetricDistortionOrder9, angleSymmetricDistortionOrder11, sensorWidth, sensorHeight "
                "from cameraModelExt"):
            f = abs(r[3])
            rr = np.linspace(0, R_MAX, 1301)
            dr = sum(k * rr ** n for k, n in zip(r[4:9], (3, 5, 7, 9, 11)))
            th = np.arctan((rr - dr) / f)
            cams[r[0]] = {"c": r[1], "r": r[2], "rr": rr, "th": th, "w": r[9], "h": r[10]}
        self.sets = []
        for r in self._con.execute(
                "select ImageSetId, orientationUx, orientationUy, orientationUz, orientationVx, orientationVy, "
                "orientationVz, orientationWx, orientationWy, orientationWz, cameraModelExtId from ImageSet"):
            self.sets.append({"id": r[0], "R": np.array(r[1:10]).reshape(3, 3), "cam": cams[r[10]]})
        self._gains = None

    def fit_gains(self, width: int = 1024) -> np.ndarray:
        """Per-photo log gains (mean 0) that make overlapping photos agree: least squares on the median
        log-luminance difference of every overlapping pair, on a coarse panorama."""
        d = equirect_dirs(width)
        ground = d[..., 2] < -0.08  # the response differs between sky and ground; ground overlaps count 10x
        lum = []
        for s in self.sets:
            col, row, w = self.lookup(s["id"], d, 4)
            img = self.hdr(s["id"], 4, compensate=False)
            v = np.full(w.shape, np.nan)
            m = w > 0
            v[m] = np.log(np.maximum(bilinear(img, col[m], row[m]).mean(-1), 1e-12))
            lum.append(v)
        n = len(lum)
        rows, rhs = [], []
        for i in range(n):
            for j in range(i + 1, n):
                both = ~np.isnan(lum[i]) & ~np.isnan(lum[j])
                for m, f in ((both & ground, 10.0), (both & ~ground, 1.0)):
                    if m.sum() < 50:
                        continue
                    r = np.zeros(n)
                    r[i], r[j] = 1.0, -1.0
                    wt = f * np.sqrt(m.sum())
                    rows.append(r * wt)
                    rhs.append(np.median(lum[j][m] - lum[i][m]) * wt)
        rows.append(np.full(n, 1000.0))
        rhs.append(0.0)
        return np.linalg.lstsq(np.array(rows), np.array(rhs), rcond=None)[0]

    def hdr(self, set_id: int, scale: int = 1, compensate: bool = False) -> np.ndarray:
        """Linear radiance (relative) float32 [h, w, 3] merged from the bracketed exposures (and, with
        compensate, scaled by the photo's overlap gain on the ground)."""
        if compensate and self._gains is None:
            self._gains = self.fit_gains()
        from PIL import Image

        rows = self._con.execute(
            "select i.ExposureTime, b.BlobbyValue from TzfImage i join Blobby b using(BlobbyId) "
            "where i.ImageSetId = ? order by i.ExposureTime", (set_id,)).fetchall()
        num = den = None
        for t, blob in rows:
            im = Image.open(io.BytesIO(blob)).convert("RGB")
            if scale > 1:
                im = im.resize((im.width // scale, im.height // scale), Image.BILINEAR)
            v = np.asarray(im, dtype=np.float32) / 255.0
            w = np.clip(1.0 - np.abs(2.0 * v.max(-1, keepdims=True) - 1.0) ** 4, 0.0, 1.0)
            w = np.where(v.max(-1, keepdims=True) > 0.98, 1e-4, w + 1e-4)  # clipped: almost no say
            lin = srgb_to_linear(v) / np.float32(t)
            num = lin * w if num is None else num + lin * w
            den = w if den is None else den + w
        out = num / den
        if compensate:
            out *= np.float32(np.exp(self._gains[set_id]))
        return out.astype(np.float32)

    def lookup(self, set_id: int, dirs: np.ndarray, img_scale: int = 1):
        """Pixel coords (col, row) in a 1/img_scale image and a blend weight for scanner-frame unit dirs [..., 3]."""
        s = self.sets[set_id]
        cam = s["cam"]
        c = dirs @ s["R"].T
        cu, cv, cw = c[..., 0], -c[..., 1], -c[..., 2]
        th = np.arccos(np.clip(cu, -1.0, 1.0))
        ph = np.arctan2(cw, cv)
        rd = np.interp(th, cam["th"], cam["rr"], right=np.inf)
        col = cam["c"] + rd * np.cos(ph)
        row = cam["r"] + rd * np.sin(ph)
        ok = (cu > 0) & (rd < R_MAX) & (col >= 0) & (col < cam["w"] - 1) & (row >= 0) & (row < cam["h"] - 1)
        wgt = np.where(ok, np.clip(1.0 - (rd / R_MAX) ** 2, 0.0, 1.0) ** (2 * BLEND_POWER), 0.0)
        return col / img_scale, row / img_scale, wgt

    def project(self, dirs: np.ndarray, img_scale: int = 1) -> tuple[np.ndarray, np.ndarray]:
        """Weighted-blend radiance for scanner-frame unit directions [..., 3] -> (rgb [..., 3], weight [...])."""
        acc = np.zeros(dirs.shape[:-1] + (3,), np.float64)
        wsum = np.zeros(dirs.shape[:-1], np.float64)
        for s in self.sets:
            col, row, w = self.lookup(s["id"], dirs, img_scale)
            m = w > 0
            if not m.any():
                continue
            img = self.hdr(s["id"], img_scale)
            acc[m] += bilinear(img, col[m], row[m]) * w[m, None]
            wsum[m] += w[m]
        rgb = np.where(wsum[..., None] > 0, acc / np.maximum(wsum, 1e-300)[..., None], 0.0)
        return rgb, wsum

    def close(self) -> None:
        self._con.close()


def bilinear(img: np.ndarray, col: np.ndarray, row: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    c0 = np.clip(np.floor(col).astype(int), 0, w - 2)
    r0 = np.clip(np.floor(row).astype(int), 0, h - 2)
    fc = (col - c0)[:, None]
    fr = (row - r0)[:, None]
    return ((1 - fc) * (1 - fr) * img[r0, c0] + fc * (1 - fr) * img[r0, c0 + 1]
            + (1 - fc) * fr * img[r0 + 1, c0] + fc * fr * img[r0 + 1, c0 + 1])


def equirect_dirs(width: int) -> np.ndarray:
    """Unit directions [h, w, 3] of a lat-long image (z up, column 0 = azimuth -180 deg, row 0 = zenith)."""
    h = width // 2
    az = (np.arange(width) + 0.5) / width * 2 * np.pi - np.pi
    el = np.pi / 2 - (np.arange(h) + 0.5) / h * np.pi
    A, E = np.meshgrid(az, el)
    return np.stack([np.cos(E) * np.cos(A), np.cos(E) * np.sin(A), np.sin(E)], -1)


def write_hdr(path: str | Path, rgb: np.ndarray) -> None:
    """Radiance RGBE (.hdr) writer, uncompressed scanlines."""
    rgb = np.clip(rgb.astype(np.float64), 0, None)
    mx = rgb.max(-1)
    e = np.where(mx > 1e-32, np.floor(np.log2(np.maximum(mx, 1e-32))) + 1, -128)
    scale = np.where(mx > 1e-32, 256.0 / np.exp2(e), 0.0)
    rgbe = np.empty(rgb.shape[:2] + (4,), np.uint8)
    rgbe[..., :3] = np.clip(rgb * scale[..., None], 0, 255).astype(np.uint8)
    rgbe[..., 3] = np.clip(e + 128, 0, 255).astype(np.uint8)
    h, w = rgb.shape[:2]
    with open(path, "wb") as f:
        f.write(f"#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y {h} +X {w}\n".encode())
        f.write(rgbe.tobytes())


def gauss_blur(img: np.ndarray, sigma_px: float) -> np.ndarray:
    """Separable Gaussian blur of an [h, w, ...] array (edge-clamped), numpy only."""
    r = int(3 * sigma_px) + 1
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma_px) ** 2)
    k /= k.sum()
    out = img.astype(np.float64)
    for ax in (0, 1):
        pad = [(0, 0)] * out.ndim
        pad[ax] = (r, r)
        p_ = np.pad(out, pad, mode="edge")
        out = sum(k[i] * np.take(p_, np.arange(i, i + out.shape[ax]), axis=ax) for i in range(2 * r + 1))
    return out
