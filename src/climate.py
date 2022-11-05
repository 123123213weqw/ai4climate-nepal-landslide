"""Climate covariates from the NASA POWER API (no key required).

One regional request covers Nepal with a ~0.625 deg lon x 0.5 deg lat grid of
monthly precipitation climatology (PRECTOTCORR_SUM, mm/month). Sites are matched
to the nearest grid cell (coarse but real; see README for the upgrade path).

Covariates per site:
  annual_precip_mm  - annual total (sum of monthly means)
  monsoon_precip_mm - Jun-Sep total (South Asian monsoon window)
  precip_cv         - inter-monthly std/mean (seasonality)
"""
import json

import numpy as np
import requests

from .config import DATA_DIR

CLIMATE_FEATURES = ["annual_precip_mm", "monsoon_precip_mm", "precip_cv"]
MONTH_KEYS = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]
POWER_URL = (
    "https://power.larc.nasa.gov/api/temporal/climatology/regional"
    "?latitude-min=26&latitude-max=31&longitude-min=80&longitude-max=89"
    "&parameters=PRECTOTCORR_SUM&community=AG&format=JSON"
)  # regional endpoint caps requests at 10 deg span; this box covers all of Nepal
CACHE = DATA_DIR / "climate" / "power_precip_nepal.json"


class ClimateSampler:
    def __init__(self):
        if CACHE.exists():
            payload = json.loads(CACHE.read_text())
        else:
            CACHE.parent.mkdir(parents=True, exist_ok=True)
            r = requests.get(POWER_URL, timeout=300)
            r.raise_for_status()
            payload = r.json()
            CACHE.write_text(json.dumps(payload))
        lons, lats, months = [], [], []
        for f in payload["features"]:
            lon, lat, _ = f["geometry"]["coordinates"]
            m = f["properties"]["parameter"]["PRECTOTCORR_SUM"]
            lons.append(lon)
            lats.append(lat)
            months.append([m[k] for k in MONTH_KEYS])
        self.grid_lons = np.asarray(lons)
        self.grid_lats = np.asarray(lats)
        self.months = np.asarray(months, dtype=np.float32)  # (n_grid, 12)

    def sample(self, lons: np.ndarray, lats: np.ndarray) -> np.ndarray:
        lons = np.atleast_1d(np.asarray(lons, dtype=float))
        lats = np.atleast_1d(np.asarray(lats, dtype=float))
        d2 = (self.grid_lons[None, :] - lons[:, None]) ** 2 + (self.grid_lats[None, :] - lats[:, None]) ** 2
        nn = d2.argmin(axis=1)
        vals = self.months[nn]  # (n, 12)
        annual = vals.sum(axis=1)
        monsoon = vals[:, 5:9].sum(axis=1)  # JJA S
        cv = vals.std(axis=1) / (vals.mean(axis=1) + 1e-6)
        return np.stack([annual, monsoon, cv], axis=1).astype(np.float32)
