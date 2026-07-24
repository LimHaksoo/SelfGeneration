#!/usr/bin/env bash
set -euo pipefail

# Baseline-compatible runtime protocol:
#   first 5 samples = warm-up
#   next 100 samples = measurement
#   batch size = 1
#   K = 100
#
# One GPU is used sequentially, matching runtime_all.sh and avoiding contention
# between concurrent attribution jobs.

GPU=${GPU:-0}
DATASETS=${DATASETS:-"PAM boiler epilepsy wafer freezer"}
FOLDS=${FOLDS:-"0 1 2 3 4"}
SEED=${SEED:-42}
N_REP=${N_REP:-100}
WARMUP=${WARMUP:-5}
K=${K:-100}
PATH_BATCH_SIZE=${PATH_BATCH_SIZE:-0}
OUTPUT_FILE=${OUTPUT_FILE:-runtime_all.csv}
RUNTIME_TABLE_OUTPUT=${RUNTIME_TABLE_OUTPUT:-runtime_table.csv}
FORCE_RECOMPUTE=${FORCE_RECOMPUTE:-0}

mkdir -p logs_runtime

force_args=()
if [[ "${FORCE_RECOMPUTE}" == "1" ]]; then
  force_args+=(--force)
fi

for data in ${DATASETS}; do
  for fold in ${FOLDS}; do
    log="logs_runtime/${data}_f${fold}_iron.log"
    echo "[LAUNCH] data=${data} fold=${fold} IRON gpu=${GPU}"

    CUDA_VISIBLE_DEVICES="${GPU}" \
    PYTHONPATH=. \
    python -u real/benchmark_iron_runtime.py \
      --data "${data}" \
      --fold "${fold}" \
      --seed "${SEED}" \
      --device cuda:0 \
      --n-steps "${K}" \
      --n-rep "${N_REP}" \
      --warmup "${WARMUP}" \
      --path-batch-size "${PATH_BATCH_SIZE}" \
      --output-file "${OUTPUT_FILE}" \
      "${force_args[@]}" \
      >"${log}" 2>&1

    echo "[DONE] data=${data} fold=${fold} log=${log}"
  done
done

echo "[AGGREGATE] ${OUTPUT_FILE}"
RUNTIME_INPUT="${OUTPUT_FILE}" \
RUNTIME_TABLE_OUTPUT="${RUNTIME_TABLE_OUTPUT}" \
PYTHONPATH=. \
python agg_runtime.py
