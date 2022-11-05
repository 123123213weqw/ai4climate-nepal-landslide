#!/usr/bin/env bash
# One-command pipeline: data -> tabular models -> CNN -> maps.
# Usage: bash scripts/run_all.sh [--device cuda] [--smoke]
set -euo pipefail
cd "$(dirname "$0")/.."

DEVICE="${DEVICE:-auto}"
SMOKE=0
for arg in "$@"; do
  case "$arg" in
    --device) shift; DEVICE="${1:-auto}" ;;
    --smoke) SMOKE=1 ;;
  esac
done

if [[ "$SMOKE" == "1" ]]; then
  python3 -m src.data build --max-points 120
  python3 -m src.train --model logreg --folds 3
  python3 -m src.train --model cnn --folds 3 --epochs 5
else
  python3 -m src.data build
  python3 -m src.train --model all --device "$DEVICE" --folds 5 --epochs 200
  python3 -m src.data build-patches
fi

python3 -m src.mapping --model cnn --aoi default
python3 -m src.mapping --model gbm --aoi default
echo "RUN_ALL OK — see runs/metrics.json and runs/susceptibility_map_*.png"
