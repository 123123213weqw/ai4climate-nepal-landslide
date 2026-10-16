# Himalaya Rainfall-Landslide Susceptibility (AI4Climate Track 2)

Repository: <https://github.com/123123213weqw/ai4climate-nepal-landslide>

Open-source starter pipeline for **Track 2 (Himalayan communities, Nepal)** of the
[AI4Climate Global Youth Climate Solutions Challenge](https://www.ai4climate.world/en):
flash-flood / landslide / GLOF climate resilience.

The task: rank terrain by landslide susceptibility from open data, benchmark classic
statistical vs ML vs deep models under **spatial** cross-validation (the honest way —
random CV inflates AUC in susceptibility mapping), and render a decision-ready map for
communities and municipal disaster offices.

## What it does

1. **Data** — NASA Global Landslide Catalog (11k events; `country_name=Nepal` → 479
   events after dedup/bbox crop; ~84% rainfall-triggered) + pseudo-absences sampled
   uniformly ≥ ~2 km from events (1:1).
2. **Features**
   - *Tabular (8):* elevation, slope, northness/eastness, curvature, roughness,
     9×9 local relief, TWI proxy — from AWS Open Data Terrarium tiles
     (SRTM/NASA-derived, ~34 m/px at zoom 12). No API keys, no GEE signup.
   - *Patches:* 8-channel 33×33 terrain windows (~1.1 km context) per site.
   - *Climate (3):* annual precipitation, Jun–Sep monsoon total, rainfall
     seasonality — NASA POWER climatology API (keyless), nearest cell.
3. **Models**
   - Logistic regression (classic LSM baseline), HistGradientBoosting, MLP on the
     8 scalars;
   - **HybridCNN**: ResNet-style encoder over terrain patches fused with climate
     scalars in the head (AMP, early stopping).
   - Grouped 5-fold CV on 0.5° spatial blocks; metrics AUC-ROC + average precision.
4. **Map** — sliding-window susceptibility raster + PNG over the Gorkha/Trishuli/
   Langtang corridor with known events overlaid; patch inference runs on GPU.

## Quickstart (CPU, ~3 min, ~1500 small tile downloads)

```bash
pip install -r requirements.txt
bash scripts/smoke.sh   # build 240-row dataset → train 3 models → render map
```

## Results (5-fold spatial block CV, 479 events + 479 absences)

| Model | AUC-ROC | AP | Inputs |
|---|---|---|---|
| **HybridCNN** | **0.892 ± 0.021** | 0.825 | 8×33×33 terrain patches + climate |
| Gradient boosting | 0.885 ± 0.025 | 0.845 | 8 terrain scalars |
| MLP | 0.878 ± 0.028 | 0.824 | 8 terrain scalars |
| Logistic regression | 0.848 ± 0.042 | 0.799 | 8 terrain scalars |

84% of catalog events are rainfall-triggered; monsoon (Jun–Sep) precipitation is
~80% of the annual total at most sites — the direct climate link for Track 2.

## Full run on the V100 (or any CUDA box)

```bash
python -m src.data build                       # 479 events + 479 absences, ~5k tiles
python -m src.data build-patches               # 958 x (8,33,33) terrain patches + climate
python -m src.train --model all --device cuda --folds 5 --epochs 200
python -m src.mapping --model cnn  --aoi default          # sliding-window CNN map (~380k patches)
python -m src.mapping --model gbm  --aoi default          # compare models visually
```

`--device auto` (default) picks CUDA → MPS → CPU, so the same commands run anywhere.
The CNN's sliding-window map inference is the GPU-heavy step; the V100 also enables
wider nets (`--width 64+`) and larger patch sizes for ablations. Verify the GPU with:

```bash
python -c "import torch; print(torch.cuda.get_device_name(0), torch.cuda.get_arch_list())"
# V100 → 'sm_70' must appear in the arch list
```

## Outputs

| File | Content |
|------|---------|
| `paper/paper.pdf` | arXiv-style technical preprint (6 pages, compiled from `paper/main.tex`) |
| `runs/metrics.json` | per-model 5-fold AUC/AP (mean ± std, per-fold) |
| `runs/data_summary.json` | dataset provenance + trigger distribution |
| `runs/susceptibility_map_*.png` | rendered susceptibility map w/ events overlaid |
| `runs/susceptibility.npz` | lon/lat/prob grid for GIS import |
| `runs/final_mlp.pt`, `final_{logreg,gbm}.joblib`, `final_cnn.pt` | final fits for inference |

One-command pipeline: `bash scripts/run_all.sh` (or `--smoke` for a quick pass;
`DEVICE=cuda bash scripts/run_all.sh` on the V100).

## Data & licenses

- NASA Global Landslide Catalog export (frozen 2016-03 snapshot), via
  data.nasa.gov (Global_Landslide_Catalog_Export). Cite: Kirschbaum et al., NASA GSFC.
- Terrain tiles: Mapzen/Terrarium on AWS Open Data (SRTM + other DEMs, public).
- Climate normals: NASA POWER (PRECTOTCORR_SUM climatology, ~0.5° over Nepal).
- This code: MIT.

## Known limitations (be honest in the submission)

- GLC events are **reports**, not a complete inventory: biased toward populated/
  road corridors. Absences are pseudo-absences, not true non-landslides.
- Terrain-only tabular baselines vs CNN: the CNN also sees climate covariates, so
  the gap is not a pure architecture effect (ablation listed in the roadmap).
- POWER climatology is coarse (~0.5°, 1991-2020 normals): it captures regional
  gradients, not valley-scale rain shadows. WorldClim/CHIRPS would sharpen it.
- The 2016 catalog snapshot misses recent monsoon seasons (COOLR live feed or
  ICIMOD inventories would extend it).

## Roadmap

1. **Ablations** — CNN on patches only (no climate) vs + climate; patch size
   17/33/65; width 32/64/128. Quantifies what actually helps.
2. **Sharper climate covariates** — WorldClim 2.1 (~18 km) or CHIRPS (~5 km)
   in place of POWER (~55 km); antecedent 7/30-day rainfall for dynamic hazard.
3. **Move from static susceptibility to near-real-time** — couple the terrain
   prior with rainfall nowcasts → dynamic "next-72h" advisory per municipality.
4. **Better learning** — larger spatial blocks for CV, Geographically Weighted
   models, SHAP/CAM interpretability, calibrated probabilities (Platt/isotonic)
   so thresholds mean something for users.
5. **GLOF extension** — overlay ICIMOD glacial-lake inventory with downstream
   flow paths for a combined multi-hazard view.
6. **90-day pilot plan** — validate maps with 2–3 municipalities in Gandaki
   province (risk-management offices), measure hit-rate on next monsoon season.

## Competition submission checklist (deadline 2026-10-16 24:00 CST)

Portal: <https://ai4climate.deepseek.club> — needs: project statement, demo video
(≤3 min), runnable demo/repo (this repo + `runs/` artifacts), data & AI statement,
validation plan (see roadmap item 5). Same materials are reused through all rounds;
on-site final 2026-10-27 (Ningbo), showcase 2026-10-28.
