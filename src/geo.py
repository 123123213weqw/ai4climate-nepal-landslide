"""Geographic proximity covariates: distance to roads, rivers, and active faults.

Sources (all open, no API keys):
  - Roads & rivers: OpenStreetMap via the Overpass API (motorway..tertiary,
    waterway=river), extracted once for the Nepal bounding box and cached.
  - Active faults: GEM Global Active Faults Database GeoJSON (github mirror),
    cropped to the bounding box.

Distances are Euclidean on an equirectangular projection around the Nepal
central latitude (good to <1% at this scale), computed to line vertices via a
cKDTree.
"""
import json
from pathlib import Path

import numpy as np
import requests
from scipy.spatial import cKDTree

from .config import DATA_DIR, NEPAL_BBOX

GEO_FEATURES = ["dist_road_km", "dist_river_km", "dist_fault_km"]
GEO_DIR = DATA_DIR / "geo"


def scalar_covariates(lons, lats) -> np.ndarray:
    """CNN scalar head inputs: climate (annual/monsoon/variability) + geo distances."""
    global _SCALAR_SAMPLERS
    from .climate import ClimateSampler

    if _SCALAR_SAMPLERS is None:
        _SCALAR_SAMPLERS = (ClimateSampler(), GeoSampler())
    clim, geo = _SCALAR_SAMPLERS
    return np.concatenate([clim.sample(lons, lats), geo.sample(lons, lats)], axis=1)


_SCALAR_SAMPLERS = None

OVERPASS_URL = "https://overpass-api.de/api/interpreter"
OVERPASS_QUERY = """
[out:json][timeout:600];
(
  way["highway"~"^(motorway|trunk|primary|secondary|tertiary)$"]({bbox});
  way["waterway"="river"]({bbox});
);
out geom;
"""
GEM_FAULTS_URL = (
    "https://raw.githubusercontent.com/GEMScienceTools/"
    "gem-global-active-faults/master/geojson/gem_active_faults.geojson"
)
# Fallback river layer when Overpass is unreachable (coarse but global, public).
NE_RIVERS_URL = (
    "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/"
    "geojson/ne_10m_rivers_lake_centerlines.geojson"
)

_KM_PER_DEG_LAT = 110.574


def _scale(lons, lats, lat0):
    """Equirectangular scaling to km around reference latitude lat0."""
    return np.column_stack([np.asarray(lons) * 111.320 * np.cos(np.radians(lat0)), np.asarray(lats) * _KM_PER_DEG_LAT])


def _collect_lines(features, bbox):
    lon_min, lat_min, lon_max, lat_max = bbox
    pts = []
    for f in features:
        g = f.get("geometry") or {}
        if g.get("type") == "LineString":
            coords = [g["coordinates"]]
        elif g.get("type") == "MultiLineString":
            coords = g["coordinates"]
        else:
            continue
        for line in coords:
            for pt in line:  # points may carry elevation; keep lon/lat
                lon, lat = pt[0], pt[1]
                if lon_min <= lon <= lon_max and lat_min <= lat <= lat_max:
                    pts.append((lon, lat))
    return np.asarray(pts, dtype=float)


def _overpass_lines(bbox, kind):
    """kind: 'road' or 'river' — separate cached extracts per kind."""
    cache = GEO_DIR / f"osm_{kind}.json"
    if cache.exists():
        return json.loads(cache.read_text())
    q = (
        f'[out:json][timeout:600];'
        f'(way["highway"~"^(motorway|trunk|primary|secondary|tertiary)$"]({bbox[1]},{bbox[0]},{bbox[3]},{bbox[2]}););'
        f'out geom;'
        if kind == "road"
        else f'[out:json][timeout:600];'
        f'(way["waterway"="river"]({bbox[1]},{bbox[0]},{bbox[3]},{bbox[2]}););'
        f'out geom;'
    )
    r = requests.post(OVERPASS_URL, data={"data": q}, timeout=120)
    r.raise_for_status()
    payload = r.json()
    GEO_DIR.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(payload))
    return payload


def _fault_lines(bbox):
    cache = GEO_DIR / "gem_faults_bbox.json"
    direct = GEO_DIR / "gem_active_faults.geojson"
    if direct.exists():  # pre-downloaded full file wins
        return json.loads(direct.read_text())
    if cache.exists():
        return json.loads(cache.read_text())
    r = requests.get(GEM_FAULTS_URL, timeout=660)
    r.raise_for_status()
    payload = r.json()
    feats = payload.get("features", [])
    GEO_DIR.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({"features": feats}))
    return payload


def _natural_earth_rivers(bbox):
    cache = GEO_DIR / "ne_rivers.geojson"
    if cache.exists():
        return json.loads(cache.read_text())
    r = requests.get(NE_RIVERS_URL, timeout=660)
    r.raise_for_status()
    payload = r.json()
    GEO_DIR.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(payload))
    return payload


class GeoSampler:
    """distances (km) to nearest road / river / active-fault line vertex.

    Roads and rivers come from OpenStreetMap (Overpass) where reachable; if the
    Overpass endpoint is blocked, rivers fall back to the Natural Earth 10m
    layer and roads are set to zero (recorded in self.source_log).
    """

    def __init__(self):
        lat0 = 0.5 * (NEPAL_BBOX[1] + NEPAL_BBOX[3])
        self.lat0 = lat0
        self.trees = {}
        self.source_log = {}
        for kind in ("road", "river"):
            pts = np.zeros((0, 2))
            try:
                payload = _overpass_lines(NEPAL_BBOX, kind)
                pts = np.asarray(
                    [(g["lon"], g["lat"]) for el in payload.get("elements", []) for g in el.get("geometry", [])],
                    dtype=float,
                ).reshape(-1, 2)
                self.source_log[kind] = f"OSM/Overpass ({len(pts)} vertices)"
            except Exception as e:
                if kind == "river":
                    ne = _natural_earth_rivers(NEPAL_BBOX)
                    pts = _collect_lines(ne.get("features", []), NEPAL_BBOX)
                    self.source_log[kind] = f"Natural Earth 10m fallback ({len(pts)} vertices)"
                else:
                    self.source_log[kind] = f"unavailable ({type(e).__name__}); zeros"
            self.trees[kind] = cKDTree(_scale(pts[:, 0], pts[:, 1], lat0)) if len(pts) else None
        try:
            fpts = _collect_lines(_fault_lines(NEPAL_BBOX).get("features", []), NEPAL_BBOX)
            self.trees["fault"] = cKDTree(_scale(fpts[:, 0], fpts[:, 1], lat0)) if len(fpts) else None
            self.source_log["fault"] = f"GEM GAF ({len(fpts)} vertices)"
        except Exception as e:  # faults are a bonus; never block the pipeline
            print(f"[geo] fault layer unavailable ({type(e).__name__}); using zeros")
            self.trees["fault"] = None
            self.source_log["fault"] = f"unavailable ({type(e).__name__}); zeros"
        print(f"[geo] sources: {self.source_log}")

    def sample(self, lons, lats) -> np.ndarray:
        X = _scale(lons, lats, self.lat0)
        out = np.zeros((len(lons), 3), dtype=np.float32)
        for i, kind in enumerate(("road", "river", "fault")):
            tree = self.trees.get(kind)
            out[:, i] = tree.query(X)[0] if tree is not None else 0.0
        return out
