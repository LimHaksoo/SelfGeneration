#!/usr/bin/env bash
set -euo pipefail

# Baseline-compatible IRON runtime measurement.
# Runs sequentially on one GPU to avoid cross-process timing interference.

GPU=${GPU:-0}
OUT=${OUT:-runtime_all.csv}
DATASETS=${DATASETS:-"PAM boiler epilepsy wafer freezer"}
FOLDS=${FOLDS:-"0 1 2 3 4"}
N_REP=${N_REP:-100}
WARMUP=${WARMUP:-5}
K=${K:-100}
PATH_BATCH=${PATH_BATCH:-0}

mkdir -p logs_runtime

for data in ${DATASETS}; do
  for fold in ${FOLDS}; do
    log="logs_runtime/${data}_f${fold}_iron.log"
    echo "[${data} fold${fold}] IRON"

    CUDA_VISIBLE_DEVICES="${GPU}" \
    PYTHONPATH=. \
    python real/benchmark_iron_runtime.py \
      --data "${data}" \
      --fold "${fold}" \
      --seed 42 \
      --device cuda:0 \
      --n-rep "${N_REP}" \
      --warmup "${WARMUP}" \
      --n-steps "${K}" \
      --path-batch "${PATH_BATCH}" \
      --output-file "${OUT}" \
      >"${log}" 2>&1
  done
done

echo "[DONE] IRON runtime rows written to ${OUT}"
echo "Aggregate with: python agg_runtime.py"
