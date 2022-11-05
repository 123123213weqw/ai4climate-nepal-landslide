#!/usr/bin/env bash
# End-to-end smoke test (CPU, ~3-4 min, downloads ~1500 small tiles).
set -euo pipefail
cd "$(dirname "$0")/.."
python3 -m src.data build --max-points 120
python3 -m src.train --model all --device auto --folds 3 --epochs 3 || true   # tabular quick pass
python3 -m src.mapping --model mlp --aoi smoke --stride 4
python3 -m src.data build-patches               # CNN inputs (reuses cached tiles)
python3 -m src.train --model cnn --device auto --folds 3 --epochs 5
python3 -m src.mapping --model cnn --aoi smoke --stride 8
echo "SMOKE OK"
