"""Multi-channel terrain patch extraction.

A "patch" is an 8-channel window of the terrain rasters (elevation, slope,
northness, eastness, curvature, roughness, local relief, TWI proxy) centered on
a site — the CNN input. Patches are float16 on disk/transfer and float32 on GPU.
"""
import numpy as np

from .config import TILE_ZOOM
from .elevation import TILE_PX, TerrainSampler, lonlat_to_webmerc_px, prefetch_tiles

CHANNELS = [
    "elevation", "slope_deg", "northness", "eastness",
    "curvature", "roughness", "relief_9x9", "twi_proxy",
]


def _stack(ras: dict) -> np.ndarray:
    return np.stack([ras[c] for c in CHANNELS])  # (8, 768, 768) float32


class PatchSampler(TerrainSampler):
    """Extract patches for scattered points (training data)."""

    def sample(self, lons, lats, size: int = 33):
        h = size // 2
        pxs = np.zeros(len(lons))
        pys = np.zeros(len(lons))
        for i in range(len(lons)):
            pxs[i], pys[i] = lonlat_to_webmerc_px(lons[i], lats[i], self.z)
        txs = (pxs // TILE_PX).astype(int)
        tys = (pys // TILE_PX).astype(int)
        containing = set(zip(txs.tolist(), tys.tolist()))
        prefetch_tiles(
            {(int(tx + dx), int(ty + dy)) for tx, ty in containing for dx in (-1, 0, 1) for dy in (-1, 0, 1)},
            self.z,
        )
        patches = np.zeros((len(lons), len(CHANNELS), size, size), dtype=np.float16)
        ok = np.zeros(len(lons), dtype=bool)
        for i in np.argsort(tys * 2 ** 22 + txs):
            tx, ty = int(txs[i]), int(tys[i])
            _, ras = self._rasters_for(tx, ty)
            if ras is None:
                continue
            lx = int(pxs[i] - (tx - 1) * TILE_PX)
            ly = int(pys[i] - (ty - 1) * TILE_PX)
            # point must be inside the center tile; the 3x3 stitch guarantees the
            # +-h margin around [256, 512) stays within the (768, 768) array
            if not (TILE_PX <= lx < 2 * TILE_PX and TILE_PX <= ly < 2 * TILE_PX):
                continue
            st = _stack(ras)
            patches[i] = st[:, ly - h : ly + h + 1, lx - h : lx + h + 1].astype(np.float16)
            ok[i] = True
        return patches, ok


def tile_patch_lattice(ras: dict, tx: int, ty: int, stride: int, size: int, rows_per_chunk: int = 16):
    """Yield (lattice_px, lattice_py, patches[m,8,size,size] float16) chunks for one tile.

    Lattice points sit in the tile's center region [256,512) with the given stride;
    returned px/py are global pixel coordinates (tile origin included).
    """
    from numpy.lib.stride_tricks import sliding_window_view

    h = size // 2
    st = _stack(ras)
    win = sliding_window_view(st, (size, size), axis=(1, 2))  # (8, H', W', s, s)
    idx = np.arange(TILE_PX, 2 * TILE_PX, stride, dtype=int)
    top = idx - h
    for r0 in range(0, len(idx), rows_per_chunk):
        rows = idx[r0 : r0 + rows_per_chunk]
        tops = top[r0 : r0 + rows_per_chunk]
        block = win[:, tops[:, None], top[None, :]]  # (8, nr, nc, s, s)
        nr, nc = block.shape[1], block.shape[2]
        p = block.transpose(1, 2, 0, 3, 4).reshape(nr * nc, len(CHANNELS), size, size)
        px = np.tile(idx[:nc], nr)
        yield (
            tx * TILE_PX + px,
            ty * TILE_PX + np.repeat(rows, nc),
            p.astype(np.float16),
        )
