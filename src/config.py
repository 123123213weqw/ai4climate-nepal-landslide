"""Shared configuration for the Nepal landslide susceptibility pipeline."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RUNS_DIR = ROOT / "runs"
TILE_CACHE = DATA_DIR / "tiles"

# Nepal bounding box used to crop events and sample pseudo-absences.
NEPAL_BBOX = (80.0, 26.3, 88.2, 30.5)  # lon_min, lat_min, lon_max, lat_max

# Terrarium elevation tiles (AWS Open Data, SRTM/NASA derived) at zoom 12 (~34 m/px in Nepal).
TILE_URL = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"
TILE_ZOOM = 12

# NASA Global Landslide Catalog export (frozen 2016 snapshot, 11k events, CC-BY).
GLC_URL = (
    "https://data.nasa.gov/docs/legacy/Global_Landslide_Catalog_Export/"
    "Global_Landslide_Catalog_Export_rows.csv"
)

FEATURES = [
    "elevation",
    "slope_deg",
    "northness",
    "eastness",
    "curvature",
    "roughness",
    "relief_9x9",
    "twi_proxy",
    # geographic proximity factors (src/geo.py): roads (OSM), rivers, active faults (GEM)
    "dist_road_km",
    "dist_river_km",
    "dist_fault_km",
]

# Default map AOI: Gorkha / Trishuli / Langtang corridor (2015 quake + monsoon hotspot).
MAP_AOI_DEFAULT = (84.0, 27.6, 85.5, 28.6)
MAP_AOI_SMOKE = (84.30, 27.90, 84.70, 28.20)

SPATIAL_BLOCK_DEG = 0.5  # block size for group CV

RANDOM_SEED = 42
