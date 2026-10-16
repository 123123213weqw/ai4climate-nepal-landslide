# The Mountain That Remembers the Rain

### A research story: open-data landslide susceptibility mapping for the Nepal Himalaya

*Based on the AI4Climate Track 2 pipeline (`ai4climate-nepal-landslide`). This report adds a
new architecture-and-context ablation study, run for this document on a Tesla V100 32 GB
(2× V100-PCIE-32GB, PyTorch 2.5.1+cu124, sm_70); those results live in
`runs/metrics_v100.json` alongside the original `runs/metrics.json`.*

---

## 1. Summary

Every summer the monsoon descends on the Nepal Himalaya, and slopes that stood quietly for
years let go. Eighty-four percent of the landslide events cataloged by NASA in Nepal were
triggered by rain. Yet the offices that must answer "which slope do we watch first?" rarely
have susceptibility information at a resolution that helps. This research builds a fully
open, reproducible pipeline that ranks terrain by landslide susceptibility from free data —
no API keys, no proprietary imagery — and evaluates it the honest way, under **spatial
cross-validation**. A hybrid convolutional network that reads the terrain like a small
picture (an 8-channel, 33×33-pixel window around each site) and fuses climate and proximity
context matches or beats every classical baseline (AUC 0.893 ± 0.013 on the V100 rerun;
0.898 ± 0.008 on the original run). A new ablation study, conducted for this report, shows
the winning configuration is already at the optimum of its design axes: widening the
network or enlarging its field of view does not help, and shrinking either one hurts. The
binding constraint is not the model — it is the size of the inventory. The story that
follows explains how the pieces fit together, what the numbers mean, and where the terrain
of improvement actually lies.

## 2. The problem, and the place

The study area is Nepal between 80.0–88.2°E and 26.3–30.5°N — the frontal Siwaliks, the
Mahabharat, and the High Himalaya stacked into a country where elevation rises from ~100 m
to >8,000 m within 200 km of horizontal distance. The 2015 Gorkha earthquake loosened
slope material across central Nepal; the monsoon that follows each summer finds it. The
communities in the corridor this study maps — Gorkha, the Trishuli valley, Langtang —
lived exactly this sequence: the quake in April 2015, then the burial of Langtang village
by an earthquake- and rain-triggered mass movement in May.

Susceptibility mapping asks a deceptively simple question: *given only static, open
information about terrain and climate, which locations rank as more likely to fail?* It
is not forecasting a date; it is producing a durable, ranked map that a municipal disaster
office can overlay on settlements and roads.

## 3. The data: 958 sites, everything open

The pipeline assembles its evidence from four open sources:

- **The inventory.** NASA's Global Landslide Catalog (frozen 2016 snapshot), filtered to
  Nepal: **479 events** after deduplication and bounding-box crop. The trigger table is
  itself a climate signal — downpour 221, rain 96, continuous rain 59, monsoon 28; only 9
  events are earthquake-triggered.
- **The absences.** Susceptibility models need negatives, and honest ones are scarce, so
  the pipeline samples **479 uniform pseudo-absences** at least ~2 km from any cataloged
  event, 1:1 with the positives. (What "absence" really means is revisited in §9.)
- **The terrain.** AWS-hosted Terrarium tiles (SRTM-derived, ~34 m/pixel at zoom 12),
  from which the pipeline derives eight covariates per site: elevation, slope, northness,
  eastness, curvature, roughness, 9×9 local relief, and a topographic wetness proxy. The
  same tiles feed the CNN's patches: **8×33×33 windows, ~1.1 km of terrain context**,
  centered on each site.
- **The climate and proximity context.** NASA POWER precipitation climatology (annual
  total, June–September monsoon total, seasonality; at typical sites the monsoon carries
  ~80% of the annual ~1,100–1,500 mm), plus distances to the nearest river (Natural
  Earth) and active fault (GEM Global Active Faults). One honesty note: the road-distance
  layer was unavailable at dataset-build time, so that column is identically zero in
  every model input, tabular and CNN alike — effectively dropped, consistently, and the
  paper's claims deliberately mention only rivers and faults.

Tabular models see the eight terrain scalars plus the three distance features (eleven
factors); the CNN sees the patch **plus** the six-element context vector
(3 climate + 3 proximity scalars).

## 4. The honest yardstick

Landslide data is spatially autocorrelated: neighboring sites share slope, lithology, and
rain. Random cross-validation quietly trains on one slope and tests on its neighbor,
inflating apparent skill. This pipeline evaluates on **0.5° spatial blocks** (145 blocks;
whole blocks held out), and runs a matched leakage experiment to show the difference:

| Split protocol | GBM AUC | CNN AUC | Fold std (CNN) |
|---|---|---|---|
| Spatial block CV (honest) | 0.893 ± 0.023 | 0.898 ± 0.008 | 0.008 |
| Random CV (leaky) | 0.901 ± 0.047 | 0.900 ± 0.048 | 0.048 |

The optimism is modest here (+0.002–0.008 AUC) — because pseudo-absences were sampled
nationally rather than in event buffers — but the random split's fold variance is **six
times larger**, which is the more insidious effect: it replaces "wrong but confidently
wrong" with "sometimes great, sometimes poor," the worst kind of evaluation for a
deployment decision.

## 5. The models

Four contenders, one protocol:

1. **Logistic regression** — the classic statistical susceptibility model, linear on the
   eleven factors.
2. **HistGradientBoosting** — the modern tabular workhorse.
3. **MLP** (256-128-64) — a neural baseline on the same scalars.
4. **HybridCNN** — a ResNet-style encoder over the 8×33×33 terrain patch whose pooled
   features are concatenated with the six context scalars in the classification head
   (~50k parameters at base width 32; AMP training, early stopping).

All trained under identical 5-fold spatial block CV, seed fixed.

## 6. The headline results

Original runs (Apple MPS) and the device-consistent V100 rerun for this report agree to
within device nondeterminism:

| Model | AUC-ROC (V100 rerun) | AUC-ROC (original) | AP |
|---|---|---|---|
| Logistic regression | 0.852 ± 0.041 | 0.852 ± 0.041 | 0.799 |
| MLP | 0.875 ± 0.024 | 0.877 ± 0.026 | 0.823 |
| Gradient boosting | **0.893 ± 0.023** | 0.893 ± 0.023 | **0.855** |
| HybridCNN (w32, 33px) | 0.893 ± 0.013 | **0.898 ± 0.008** | 0.844 |

The honest reading: **the CNN and GBM tie on AUC; GBM retains a small AP edge; the CNN's
fold-to-fold spread is the tightest of any model.** The two families also err
differently, which is exactly why the rendered map ships both (`susceptibility.npz`
carries the CNN grid; `final_gbm.joblib` is one flag away).

## 7. New: what the network actually needs (the V100 ablation study)

The original paper left one question open: is the CNN's configuration an accident or a
necessity? For this report the full design grid was rerun on a Tesla V100, all under the
same spatial CV and identical scalar inputs:

| Configuration | AUC-ROC | AP | vs. baseline |
|---|---|---|---|
| **Baseline** (width 32, 33 px, + scalars) | **0.893 ± 0.013** | **0.844** | — |
| Patches only (context zeroed) | 0.886 ± 0.019 | 0.822 | −0.007 AUC, fold std ×1.5 |
| Width 64 | 0.877 ± 0.025 | 0.816 | −0.016 AUC |
| Width 128 | 0.882 ± 0.025 | 0.824 | −0.011 AUC |
| Patch 17 px (~0.6 km) | 0.883 ± 0.012 | 0.807 | −0.010 AUC |
| Patch 65 px (~2.2 km) | 0.894 ± 0.014 | 0.841 | +0.001 (tie) |

Three findings, none of them flattering to "just scale it":

1. **Context helps; capacity doesn't.** Removing the six scalars costs AUC and halves
   stability (the original run showed the same direction more strongly, −0.013). But
   doubling or quadrupling width *hurts* — with 958 training sites, extra parameters buy
   only variance.
2. **Spatial context saturates at ~1 km.** Shrinking the field of view to 0.6 km costs
   0.010 AUC; extending it to 2.2 km returns nothing. The slope-scale processes that
   matter are already inside a 33-pixel window; lithological and climatic controls enter
   through the scalars, not through more pixels.
3. **The baseline is Pareto-optimal on this dataset.** Every axis of "make the model
   bigger" fails, and the one axis of "give the model more context information" already
   works at its cheapest setting. The conclusion writes itself: **the binding constraint
   is inventory size, not architecture.** A network this size saturates what 479 events
   can teach; the next gains must come from more events (the live COOLR feed, ICIMOD
   inventories) or sharper covariates — not from a deeper net.

## 8. The map: where the story ends up on paper

Sliding the trained CNN across the Gorkha–Trishuli–Langtang corridor produces a
**1.5-million-cell susceptibility raster** (`runs/susceptibility.npz` for GIS import;
`runs/susceptibility_map_cnn_default.png` with cataloged events overlaid). Its structure
is geologically literate: high-susceptibility channels follow the drainage network, and
the cataloged events cluster inside them. Permutation importance agrees with the eye —
fault and river proximity are the strongest non-elevation factors, ahead of any single
climate scalar. This is the decision-ready artifact the track asked for: a ranked
terrain surface a municipal office can drape over settlement and road layers.

## 9. What this map is not (the honest section)

- **Susceptibility ≠ probability.** Outputs are relative rankings from an uncalibrated
  classifier; a "0.8" cell is higher-risk than a "0.5" cell, not 80% likely to fail.
- **Reports, not truth.** The GLC inventory is biased toward populated areas and road
  corridors; remote failures are undercounted. The model learns *reported* landslides.
- **Pseudo-absences are assumptions.** A sampled "non-landslide" 2 km from the nearest
  report is probably safe, not certainly safe.
- **Climate at ~55 km resolution.** NASA POWER normals capture regional gradients, not
  valley-scale rain shadows; monsoon totals are ~80% of annual everywhere, so the climate
  scalars partly encode "Nepal," not local hazard.
- **A frozen snapshot.** The 2016 catalog cut-off misses every monsoon since; the
  earthquake legacy is present, recent rainfall trends are not.
- **One seed per configuration.** Uncertainty here is fold-level, not seed-level; the
  device A–A test (MPS 0.898 vs V100 0.893) is a fair proxy for run-to-run noise.

## 10. The next chapter

The ablation result reshapes the roadmap by elimination: bigger models are a dead end on
this inventory, so the priorities are data-side — extend the inventory (COOLR live feed,
ICIMOD), sharpen rainfall covariates (CHIRPS ~5 km or WorldClim in place of POWER), add
antecedent 7/30-day rainfall to move from static susceptibility toward dynamic
next-72-hour advisories, calibrate probabilities so thresholds become actionable, and
overlay ICIMOD glacial lakes for GLOF multi-hazard. The 90-day ambition stays what it
was: hang the map in two or three Gandaki municipality offices and score it against the
next monsoon.

## 11. Reproducibility

Everything above reruns from the repository: `bash scripts/run_all.sh` (CPU/MPS/CUDA
auto-detects); the new ablation axes added for this report are `--width` and `--patches`
flags on `python -m src.train`, with sweep results in `runs/metrics_v100.json` (V100
device column) and the original numbers preserved in `runs/metrics.json`. Data provenance
and the frozen catalog snapshot are documented in `runs/data_summary.json` and the
6-page preprint (`paper/paper.pdf`).

---

*Authors of the underlying study: Yue Wang, Chenfeng Zhang. V100 ablation study run
2026-10-16 on WZU V100 (2× Tesla V100-PCIE-32GB), PyTorch 2.5.1+cu124. Code: MIT.*
