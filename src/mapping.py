"""Predict a landslide susceptibility map over an AOI and render it.

Usage:
  python -m src.mapping --model mlp --aoi default
  python -m src.mapping --model mlp --aoi smoke --stride 4
"""
import argparse
import json

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from .config import DATA_DIR, FEATURES, MAP_AOI_DEFAULT, MAP_AOI_SMOKE, RUNS_DIR, TILE_ZOOM
from .elevation import TerrainSampler, lonlat_to_webmerc_px, webmerc_px_to_lonlat

TILE_PX = 256


def grid_cells(aoi, stride):
    """Yield (lon, lat, feature_rows) for a subsampled lattice over the AOI."""
    lon_min, lat_min, lon_max, lat_max = aoi
    x0 = int(lonlat_to_webmerc_px(lon_min, lat_max, TILE_ZOOM)[0] // TILE_PX)
    x1 = int(lonlat_to_webmerc_px(lon_max, lat_min, TILE_ZOOM)[0] // TILE_PX)
    y0 = int(lonlat_to_webmerc_px(lon_min, lat_max, TILE_ZOOM)[1] // TILE_PX)
    y1 = int(lonlat_to_webmerc_px(lon_max, lat_min, TILE_ZOOM)[1] // TILE_PX)
    sampler = TerrainSampler()
    from .elevation import prefetch_tiles

    tile_list = [(tx, ty) for tx in range(x0, x1 + 1) for ty in range(y0, y1 + 1)]
    prefetch_tiles(
        {(tx + dx, ty + dy) for tx, ty in tile_list for dx in (-1, 0, 1) for dy in (-1, 0, 1)},
        TILE_ZOOM,
    )
    lons, lats, feats = [], [], []
    missed = 0
    for tx in range(x0, x1 + 1):
        for ty in range(y0, y1 + 1):
            st = sampler._rasters_for(tx, ty)[1]
            if st is None:
                missed += 1
                continue
            idx = np.arange(TILE_PX, 2 * TILE_PX, stride, dtype=int)
            yy = np.repeat(idx[:, None], len(idx), axis=1)
            xx = np.repeat(idx[None, :], len(idx), axis=0)
            for k, name in enumerate(
                ["elevation", "slope_deg", "northness", "eastness", "curvature", "roughness", "relief_9x9", "twi_proxy"]
            ):
                vals = st[name][yy, xx][:, :, None]
                if k == 0:
                    block = np.zeros((yy.size, len(FEATURES)), dtype=np.float32)
                block[:, k] = vals.ravel()
            # global pixel coords -> lon/lat
            gx = tx * TILE_PX + xx
            gy = ty * TILE_PX + yy
            lon = gx / (TILE_PX * 2 ** TILE_ZOOM) * 360.0 - 180.0
            n = np.pi * (1 - 2 * gy / (TILE_PX * 2 ** TILE_ZOOM))
            lat = np.degrees(np.arctan(np.sinh(n)))
            in_aoi = (lon >= lon_min) & (lon <= lon_max) & (lat >= lat_min) & (lat <= lat_max)
            if in_aoi.any():
                lons.append(lon[in_aoi])
                lats.append(lat[in_aoi])
                feats.append(block[in_aoi.ravel()])
    if not lons:
        raise SystemExit("no cells with terrain data in AOI")
    print(f"grid: {sum(len(a) for a in lons)} cells | tiles skipped: {missed}")
    return np.concatenate(lons), np.concatenate(lats), np.concatenate(feats)


def predict_grid_cnn(aoi, stride, patch_size, predict):
    """Sliding-window CNN inference over the AOI; predicts per-tile to bound memory."""
    lon_min, lat_min, lon_max, lat_max = aoi
    x0 = int(lonlat_to_webmerc_px(lon_min, lat_max, TILE_ZOOM)[0] // TILE_PX)
    x1 = int(lonlat_to_webmerc_px(lon_max, lat_min, TILE_ZOOM)[0] // TILE_PX)
    y0 = int(lonlat_to_webmerc_px(lon_min, lat_max, TILE_ZOOM)[1] // TILE_PX)
    y1 = int(lonlat_to_webmerc_px(lon_max, lat_min, TILE_ZOOM)[1] // TILE_PX)
    from .climate import ClimateSampler
    from .elevation import prefetch_tiles
    from .geo import scalar_covariates
    from .patches import tile_patch_lattice

    sampler = TerrainSampler()
    prefetch_tiles(
        {(tx + dx, ty + dy) for tx in range(x0, x1 + 1) for ty in range(y0, y1 + 1)
         for dx in (-1, 0, 1) for dy in (-1, 0, 1)},
        TILE_ZOOM,
    )
    climate = scalar_covariates  # (n,6): annual/monsoon/cv precip + road/river/fault dist
    lons_all, lats_all, p_all = [], [], []
    missed = 0
    scale = 360.0 / (TILE_PX * 2 ** TILE_ZOOM)
    for tx in range(x0, x1 + 1):
        for ty in range(y0, y1 + 1):
            ras = sampler._rasters_for(tx, ty)[1]
            if ras is None:
                missed += 1
                continue
            for gx, gy, patch in tile_patch_lattice(ras, tx, ty, stride, patch_size):
                lon = gx * scale - 180.0
                n = np.pi * (1 - 2 * gy * scale / 360.0)
                lat = np.degrees(np.arctan(np.sinh(n)))
                keep = (lon >= lon_min) & (lon <= lon_max) & (lat >= lat_min) & (lat <= lat_max)
                if not keep.any():
                    continue
                scal = climate(lon[keep], lat[keep])
                p_all.append(predict(patch[keep].astype(np.float32), scal))
                lons_all.append(lon[keep])
                lats_all.append(lat[keep])
    print(f"grid: {sum(len(a) for a in lons_all)} cells | tiles skipped: {missed}")
    return (
        np.concatenate(lons_all), np.concatenate(lats_all), np.concatenate(p_all),
    )


def load_predictor(name: str):
    if name == "cnn":
        ckpt = torch.load(RUNS_DIR / "final_cnn.pt", map_location="cpu", weights_only=False)
        from .models import HybridCNN

        model = HybridCNN(in_ch=ckpt["in_ch"], scalar_dim=ckpt["scalar_dim"], width=ckpt["width"])
        model.load_state_dict(ckpt["state_dict"])
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        model.to(device).eval()
        cm, cs, sm, ss = ckpt["ch_mean"], ckpt["ch_std"], ckpt["sc_mean"], ckpt["sc_std"]

        def predict(P, S, batch=2048):
            out = []
            with torch.no_grad():
                for i in range(0, len(P), batch):
                    p = torch.as_tensor((P[i : i + batch] - cm) / cs, dtype=torch.float32, device=device)
                    s = torch.as_tensor((S[i : i + batch] - sm) / ss, dtype=torch.float32, device=device)
                    out.append(torch.sigmoid(model(p, s)).float().cpu().numpy())
            return np.concatenate(out)

        return predict, ckpt["patch_size"]
    if name == "mlp":
        ckpt = torch.load(RUNS_DIR / "final_mlp.pt", map_location="cpu", weights_only=False)
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        from .models import make_mlp

        model = make_mlp(len(ckpt["features"]), tuple(ckpt["hidden"]))
        model.load_state_dict(ckpt["state_dict"])
        model.to(device).eval()
        mean, std = ckpt["mean"], ckpt["std"]

        def predict(X):
            Xn = (X - mean) / std
            out = []
            with torch.no_grad():
                for i in range(0, len(Xn), 65536):
                    xb = torch.as_tensor(Xn[i : i + 65536], dtype=torch.float32, device=device)
                    out.append(torch.sigmoid(model(xb).squeeze(-1)).float().cpu().numpy())
            return np.concatenate(out)

        return predict, None
    model = joblib.load(RUNS_DIR / f"final_{name}.joblib")
    return (lambda X: model.predict_proba(X)[:, 1]), None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlp", choices=["mlp", "logreg", "gbm", "cnn"])
    ap.add_argument("--aoi", default="default", choices=["default", "smoke"])
    ap.add_argument("--stride", type=int, default=None, help="px stride (default: 3 scalar / 6 cnn on default AOI)")
    args = ap.parse_args()

    aoi = MAP_AOI_DEFAULT if args.aoi == "default" else MAP_AOI_SMOKE
    default_stride = 6 if args.model == "cnn" else 3
    stride = args.stride or (4 if (args.aoi == "smoke" and args.model != "cnn") else default_stride)
    predict, patch_size = load_predictor(args.model)
    if args.model == "cnn":
        lons, lats, p = predict_grid_cnn(aoi, stride, patch_size, predict)
    else:
        lons, lats, X = grid_cells(aoi, stride)
        p = predict(X)
    print(f"predicted {len(p)} cells | P: min={p.min():.3f} mean={p.mean():.3f} max={p.max():.3f}")

    np.savez(
        RUNS_DIR / "susceptibility.npz", lon=lons, lat=lats, prob=p,
        extent=np.asarray(aoi),
    )

    # render
    fig, ax = plt.subplots(figsize=(11, 8), dpi=130)
    vmin, vmax = np.percentile(p, [2, 98])
    sc = ax.scatter(lons, lats, c=p, s=0.5, cmap="hot_r", vmin=vmin, vmax=vmax)
    ds = pd.read_parquet(DATA_DIR / "dataset.parquet")
    known = ds[(ds.label == 1) & ds.longitude.between(aoi[0], aoi[2]) & ds.latitude.between(aoi[1], aoi[3])]
    if len(known):
        ax.scatter(known.longitude, known.latitude, s=9, facecolors="none", edgecolors="cyan", linewidths=0.7, label=f"GLC events (n={len(known)})")
        ax.legend(loc="lower right")
    metrics = {}
    mp = RUNS_DIR / "metrics.json"
    if mp.exists():
        metrics = json.loads(mp.read_text()).get("cv", {})
    auc = metrics.get(args.model, {}).get("auc_mean")
    ax.set_title(
        f"LandSlide susceptibility — Nepal ({args.model}{f', CV AUC={auc:.3f}' if auc else ''})\n"
        f"AOI {aoi[0]:.2f}E-{aoi[2]:.2f}E, {aoi[1]:.2f}N-{aoi[3]:.2f}N | stride {stride}px (~{stride * 34} m)"
    )
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    fig.colorbar(sc, ax=ax, label=f"susceptibility (P, {vmin:.2f}-{vmax:.2f} pct range)")
    out = RUNS_DIR / f"susceptibility_map_{args.model}_{args.aoi}.png"
    fig.tight_layout()
    fig.savefig(out)
    print(f"map -> {out}")


if __name__ == "__main__":
    main()
