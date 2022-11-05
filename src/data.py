"""Build the Nepal landslide dataset (positives = NASA GLC events, negatives = pseudo-absences).

Usage: python -m src.data build [--max-points 120]
"""
import argparse
import json

import numpy as np
import pandas as pd
import requests
from scipy.spatial import cKDTree

from .climate import ClimateSampler
from .config import DATA_DIR, FEATURES, GLC_URL, NEPAL_BBOX, RANDOM_SEED, RUNS_DIR, SPATIAL_BLOCK_DEG
from .elevation import TerrainSampler
from .geo import GEO_FEATURES, GeoSampler, scalar_covariates
from .patches import CHANNELS as TERRAIN
from .patches import PatchSampler

MIN_NEG_DIST_DEG = 0.02  # ~2 km: negatives must be at least this far from any event


def load_glc(max_points: int | None = None) -> pd.DataFrame:
    path = DATA_DIR / "glc.csv"
    if not path.exists():
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        r = requests.get(GLC_URL, timeout=300)
        r.raise_for_status()
        path.write_bytes(r.content)
    df = pd.read_csv(path)
    lon_min, lat_min, lon_max, lat_max = NEPAL_BBOX
    df = df[
        df["country_name"].astype(str).str.contains("Nepal", case=False, na=False)
        & df["longitude"].between(lon_min, lon_max)
        & df["latitude"].between(lat_min, lat_max)
    ].dropna(subset=["longitude", "latitude"])
    df = df.drop_duplicates(subset=["longitude", "latitude"])
    if max_points and len(df) > max_points:
        df = df.sample(max_points, random_state=RANDOM_SEED)
    return df.reset_index(drop=True)


def build(max_points: int | None):
    rng = np.random.default_rng(RANDOM_SEED)
    pos = load_glc(max_points)
    print(f"Positive events (Nepal, GLC): {len(pos)}")

    n_neg = len(pos)  # 1:1 ratio, standard for pseudo-absence susceptibility models
    lon_min, lat_min, lon_max, lat_max = NEPAL_BBOX
    tree = cKDTree(pos[["longitude", "latitude"]].to_numpy())
    neg_lons, neg_lats = [], []
    attempts = 0
    while len(neg_lons) < n_neg and attempts < n_neg * 200:
        attempts += 1
        cand = rng.uniform([lon_min, lat_min], [lon_max, lat_max])
        if tree.query(cand[None, :], k=1)[0][0] >= MIN_NEG_DIST_DEG:
            neg_lons.append(cand[0])
            neg_lats.append(cand[1])
    print(f"Sampled pseudo-absences: {len(neg_lons)} (after {attempts} attempts)")

    sampler = TerrainSampler()
    feats_pos, ok_pos = sampler.sample(
        pos["longitude"].to_numpy().astype(float), pos["latitude"].to_numpy().astype(float)
    )
    feats_neg, ok_neg = sampler.sample(
        np.asarray(neg_lons, dtype=float), np.asarray(neg_lats, dtype=float)
    )
    n_drop = int((~ok_pos).sum() + (~ok_neg).sum())
    if n_drop:
        print(f"Dropping {n_drop} points with missing terrain data")

    rows = pd.DataFrame(
        np.concatenate([feats_pos[ok_pos], feats_neg[ok_neg]]), columns=TERRAIN
    )
    rows["label"] = np.concatenate(
        [np.ones(ok_pos.sum()), np.zeros(ok_neg.sum())]
    ).astype(int)
    rows["longitude"] = np.concatenate(
        [pos["longitude"].to_numpy()[ok_pos], np.asarray(neg_lons)[ok_neg]]
    )
    rows["latitude"] = np.concatenate(
        [pos["latitude"].to_numpy()[ok_pos], np.asarray(neg_lats)[ok_neg]]
    )
    n_pos = int(ok_pos.sum())
    kept_pos = pos.loc[np.where(ok_pos)[0], "landslide_trigger"].astype(str).to_numpy()
    rows["trigger"] = np.concatenate([kept_pos, np.full(len(rows) - n_pos, "")]).astype(object)
    rows["block_id"] = (
        (np.floor(rows["longitude"] / SPATIAL_BLOCK_DEG) * SPATIAL_BLOCK_DEG).astype(str)
        + "_"
        + (np.floor(rows["latitude"] / SPATIAL_BLOCK_DEG) * SPATIAL_BLOCK_DEG).astype(str)
    )
    geo = GeoSampler().sample(rows["longitude"].to_numpy(), rows["latitude"].to_numpy())
    for k, name in enumerate(GEO_FEATURES):
        rows[name] = geo[:, k]

    DATA_DIR.mkdir(exist_ok=True)
    out = DATA_DIR / "dataset.parquet"
    rows.to_parquet(out)
    print(f"Saved {len(rows)} rows -> {out}")

    RUNS_DIR.mkdir(exist_ok=True)
    trig = pos["landslide_trigger"].astype(str).value_counts().head(6).to_dict()
    summary = {
        "n_total": len(rows),
        "n_positive": int(rows["label"].sum()),
        "top_triggers": trig,
        "date_range": [
            str(pos["event_date"].min()),
            str(pos["event_date"].max()),
        ],
        "features": FEATURES,
        "notes": [
            "Positives: NASA Global Landslide Catalog (country_name=Nepal, bbox-cropped, coords deduped)",
            "Negatives: uniform pseudo-absences >= ~2 km from any event (1:1)",
            f"Spatial CV blocks: {SPATIAL_BLOCK_DEG} deg",
        ],
    }
    (RUNS_DIR / "data_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def build_patches(patch_size: int = 33):
    """Extract 8-channel terrain patches + climate scalars for every row of dataset.parquet."""
    df = pd.read_parquet(DATA_DIR / "dataset.parquet")
    sampler = PatchSampler()
    patches, ok = sampler.sample(
        df["longitude"].to_numpy().astype(float), df["latitude"].to_numpy().astype(float), size=patch_size
    )
    climate = scalar_covariates(
        df["longitude"].to_numpy().astype(float), df["latitude"].to_numpy().astype(float)
    )
    keep = df.index[ok].to_numpy()
    out = DATA_DIR / "patches.npz"
    np.savez_compressed(
        out,
        patches=patches[ok].astype(np.float16),
        scalars=climate[ok],
        label=df["label"].to_numpy()[ok],
        longitude=df["longitude"].to_numpy()[ok],
        latitude=df["latitude"].to_numpy()[ok],
        block_id=df["block_id"].astype(str).to_numpy()[ok],
        patch_size=patch_size,
    )
    print(f"saved {len(keep)} patches {tuple(patches.shape[1:])} -> {out}")
    print(f"climate covariates at first 3 sites:\n{climate[ok][:3]}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "build-patches"])
    ap.add_argument("--max-points", type=int, default=None, help="subsample positives (smoke runs)")
    ap.add_argument("--patch-size", type=int, default=33)
    args = ap.parse_args()
    if args.cmd == "build":
        build(args.max_points)
    else:
        build_patches(args.patch_size)
