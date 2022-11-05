"""Terrain feature extraction from open Terrarium elevation tiles.

Tiles: AWS Open Data "elevation-tiles-prod" (Mapzen terrain, SRTM/NASA derived),
PNG-encoded elevation: h = R*256 + G + B/256 - 32768 (meters). No API key needed.

For each query point we stitch the 3x3 tile neighborhood, compute terrain
rasters on the stitched array, and sample the pixel nearest the point.
"""
import io
import math
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import requests
from PIL import Image
from numpy.lib.stride_tricks import sliding_window_view

from .config import TILE_CACHE, TILE_URL, TILE_ZOOM

TILE_PX = 256
_WIN = 9  # window (px) for roughness / local relief


def prefetch_tiles(tiles, z: int = TILE_ZOOM, workers: int = 16):
    """Download a set of (x, y) tiles in parallel into the disk cache."""
    TILE_CACHE.mkdir(parents=True, exist_ok=True)
    todo = [(z, x, y) for x, y in tiles if not (TILE_CACHE / f"{z}_{x}_{y}.npy").exists()]
    if not todo:
        return
    print(f"prefetching {len(todo)} tiles ({workers} workers) ...")
    with ThreadPoolExecutor(workers) as ex:
        got = [t for t in ex.map(lambda k: fetch_tile(*k), todo) if t is not None]
    print(f"fetched {len(got)}/{len(todo)} tiles")


def lonlat_to_webmerc_px(lon: float, lat: float, z: int = TILE_ZOOM):
    scale = 256 * 2 ** z
    x = (lon + 180.0) / 360.0 * scale
    y = (1.0 - math.log(math.tan(math.radians(lat)) + 1.0 / math.cos(math.radians(lat))) / math.pi) / 2.0 * scale
    return x, y


def webmerc_px_to_lonlat(x: float, y: float, z: int = TILE_ZOOM):
    scale = 256 * 2 ** z
    lon = x / scale * 360.0 - 180.0
    n = math.pi * (1.0 - 2.0 * y / scale)
    lat = math.degrees(math.atan(math.sinh(n)))
    return lon, lat


def meters_per_px(lat: float, z: int = TILE_ZOOM) -> float:
    return 156543.03392 * math.cos(math.radians(lat)) / 2 ** z


def fetch_tile(z: int, x: int, y: int, retries: int = 3, timeout: int = 30) -> np.ndarray | None:
    """Return (256,256) float32 elevation in meters, or None if unavailable."""
    cache = TILE_CACHE / f"{z}_{x}_{y}.npy"
    if cache.exists():
        return np.load(cache)
    url = TILE_URL.format(z=z, x=x, y=y)
    for attempt in range(retries):
        try:
            r = requests.get(url, timeout=timeout)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            img = np.asarray(Image.open(io.BytesIO(r.content)).convert("RGB"), dtype=np.float32)
            elev = img[..., 0] * 256.0 + img[..., 1] + img[..., 2] / 256.0 - 32768.0
            TILE_CACHE.mkdir(parents=True, exist_ok=True)
            np.save(cache, elev.astype(np.float32))
            return elev.astype(np.float32)
        except Exception:
            if attempt == retries - 1:
                return None
            time.sleep(1.0 + attempt)


def stitch_3x3(z: int, x: int, y: int) -> np.ndarray | None:
    """Stitch the 3x3 tile neighborhood around (x, y); returns (768,768) or None."""
    tiles = {}
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            t = fetch_tile(z, x + dx, y + dy)
            if t is None:
                return None
            tiles[(dx, dy)] = t
    rows = []
    for dy in (-1, 0, 1):
        rows.append(np.concatenate([tiles[(dx, dy)] for dx in (-1, 0, 1)], axis=1))
    return np.concatenate(rows, axis=0)


def terrain_rasters(stitched: np.ndarray, lat_center: float, z: int = TILE_ZOOM) -> dict[str, np.ndarray]:
    """Compute terrain feature rasters on a stitched (768,768) elevation array."""
    res = meters_per_px(lat_center, z)
    gy, gx = np.gradient(stitched, res)  # dh/dy (north->south rows), dh/dx
    slope_rad = np.arctan(np.hypot(gx, gy))
    aspect = np.arctan2(-gx, gy)  # downslope azimuth from north

    win = sliding_window_view(stitched, (_WIN, _WIN))
    roughness = win.std(axis=(-2, -1))
    relief = win.max(axis=(-2, -1)) - win.min(axis=(-2, -1))

    lap = (
        stitched[:-2, 1:-1] + stitched[2:, 1:-1] + stitched[1:-1, :-2] + stitched[1:-1, 2:]
        - 4.0 * stitched[1:-1, 1:-1]
    ) / (res ** 2)
    pad = np.pad(lap, 1, mode="edge")

    twi = np.log(1.0 / (np.tan(slope_rad) + 1e-6))
    return {
        "elevation": stitched,
        "slope_deg": np.degrees(slope_rad),
        "northness": np.cos(aspect),
        "eastness": np.sin(aspect),
        "curvature": pad,
        "roughness": np.pad(roughness, _WIN // 2, mode="edge"),
        "relief_9x9": np.pad(relief, _WIN // 2, mode="edge"),
        "twi_proxy": twi,
    }


class TerrainSampler:
    """Sample terrain features for many lon/lat points, grouped by containing tile."""

    def __init__(self, z: int = TILE_ZOOM):
        self.z = z
        self._stitched: tuple[int, int, np.ndarray, dict] | None = None

    def _rasters_for(self, tx: int, ty: int):
        if self._stitched is not None and self._stitched[0] == tx and self._stitched[1] == ty:
            return self._stitched[2], self._stitched[3]
        st = stitch_3x3(self.z, tx, ty)
        if st is None:
            return None, None
        lat_c, _ = webmerc_px_to_lonlat((tx + 0.5) * TILE_PX, (ty + 0.5) * TILE_PX, self.z)
        ras = terrain_rasters(st, lat_c, self.z)
        self._stitched = (tx, ty, st, ras)
        return st, ras

    def sample(self, lons: np.ndarray, lats: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (feature_matrix [n, 8], ok_mask [n]) for the given points."""
        pxs = np.zeros(len(lons))
        pys = np.zeros(len(lons))
        for i in range(len(lons)):
            pxs[i], pys[i] = lonlat_to_webmerc_px(lons[i], lats[i], self.z)
        txs = (pxs // TILE_PX).astype(int)
        tys = (pys // TILE_PX).astype(int)
        containing = set(zip(txs.tolist(), tys.tolist()))
        needed = {
            (int(tx + dx), int(ty + dy))
            for tx, ty in containing
            for dx in (-1, 0, 1)
            for dy in (-1, 0, 1)
        }
        prefetch_tiles(needed, self.z)
        out = np.zeros((len(lons), 8), dtype=np.float32)
        ok = np.zeros(len(lons), dtype=bool)
        for i in np.argsort(tys * 2 ** 22 + txs):  # group points on the same stitched tile
            tx, ty = int(txs[i]), int(tys[i])
            _, ras = self._rasters_for(tx, ty)
            if ras is None:
                continue
            lx = int(pxs[i] - (tx - 1) * TILE_PX)
            ly = int(pys[i] - (ty - 1) * TILE_PX)
            if not (TILE_PX <= lx < 2 * TILE_PX and TILE_PX <= ly < 2 * TILE_PX):
                continue
            for k, name in enumerate(
                ["elevation", "slope_deg", "northness", "eastness", "curvature", "roughness", "relief_9x9", "twi_proxy"]
            ):
                out[i, k] = ras[name][ly, lx]
            ok[i] = True
        return out, ok
