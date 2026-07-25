#!/usr/bin/env bash
set -euo pipefail

# Candidate jobs are distributed across GPUs; each GPU processes its assigned
# candidates sequentially.  Fold 0 and a fixed 50-sample screening subset are
# the defaults for fast hyperparameter search.
DATASETS=${DATASETS:-"state switch-feature"}
FOLDS=${FOLDS:-"0"}
GPUS=${GPUS:-"0 1 2 3"}
EPOCHS=${EPOCHS:-500}
MAX_SAMPLES=${MAX_SAMPLES:-50}
FORCE_RETRAIN=${FORCE_RETRAIN:-0}
FORCE_EVAL=${FORCE_EVAL:-0}
STRICT=${STRICT:-1}

read -r -a DATASET_ARRAY <<< "${DATASETS}"
read -r -a FOLD_ARRAY <<< "${FOLDS}"
read -r -a GPU_ARRAY <<< "${GPUS}"

if [[ ${#DATASET_ARRAY[@]} -eq 0 || ${#FOLD_ARRAY[@]} -eq 0 || \
      ${#GPU_ARRAY[@]} -eq 0 ]]; then
  echo "DATASETS/FOLDS/GPUS must not be empty" >&2
  exit 2
fi

for fold in "${FOLD_ARRAY[@]}"; do
  if (( fold < 0 || fold > 4 )); then
    echo "Invalid fold: ${fold}" >&2
    exit 2
  fi
done

mkdir -p logs/synthetic_iron_search
TASK_FILE=$(mktemp)
trap 'rm -f "${TASK_FILE}"' EXIT

DATASETS_ENV="${DATASETS}" FOLDS_ENV="${FOLDS}" PYTHONPATH=. \
python - <<'PY' > "${TASK_FILE}"
import os
from configs.synthetic_iron_search_space import (
    canonical_dataset,
    get_candidates,
)

datasets = [canonical_dataset(v) for v in os.environ["DATASETS_ENV"].split()]
folds = [int(v) for v in os.environ["FOLDS_ENV"].split()]
for data in datasets:
    for fold in folds:
        for candidate in get_candidates(data):
            print(f"{data}\t{fold}\t{candidate['name']}")
PY

mapfile -t TASK_ARRAY < "${TASK_FILE}"
if [[ ${#TASK_ARRAY[@]} -eq 0 ]]; then
  echo "No search tasks were generated" >&2
  exit 2
fi

retrain_args=()
eval_args=()
if [[ "${FORCE_RETRAIN}" == "1" ]]; then
  retrain_args+=(--force-retrain)
fi
if [[ "${FORCE_EVAL}" == "1" ]]; then
  eval_args+=(--force-eval)
fi

worker() {
  local worker_index=$1
  local gpu=${GPU_ARRAY[$worker_index]}
  local task_index

  for ((task_index=worker_index; task_index<${#TASK_ARRAY[@]}; task_index+=${#GPU_ARRAY[@]})); do
    local data fold candidate
    IFS=$'\t' read -r data fold candidate <<< "${TASK_ARRAY[$task_index]}"
    local safe_data=${data//-/_}
    local log="logs/synthetic_iron_search/${safe_data}_fold${fold}_${candidate}.log"

    echo "[LAUNCH] data=${data} fold=${fold} candidate=${candidate} gpu=${gpu}"
    if ! CUDA_VISIBLE_DEVICES="${gpu}" \
         PYTHONPATH=. \
         PYTHONUNBUFFERED=1 \
         python -u synthetic/search_iron_candidate.py \
           --data "${data}" \
           --fold "${fold}" \
           --candidate "${candidate}" \
           --device cuda:0 \
           --epochs "${EPOCHS}" \
           --max-samples "${MAX_SAMPLES}" \
           "${retrain_args[@]}" \
           "${eval_args[@]}" \
           >"${log}" 2>&1; then
      echo "[FAIL] data=${data} fold=${fold} candidate=${candidate} log=${log}" >&2
      tail -n 40 "${log}" >&2 || true
      return 1
    fi
    echo "[DONE] data=${data} fold=${fold} candidate=${candidate} log=${log}"
  done
}

pids=()
for ((worker_index=0; worker_index<${#GPU_ARRAY[@]}; worker_index++)); do
  if (( worker_index >= ${#TASK_ARRAY[@]} )); then
    break
  fi
  worker "${worker_index}" &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done
if [[ ${status} -ne 0 ]]; then
  echo "[ERROR] at least one synthetic IRON search worker failed" >&2
  exit "${status}"
fi

aggregate_args=()
if [[ "${STRICT}" == "1" ]]; then
  aggregate_args+=(--strict)
fi

PYTHONPATH=. python synthetic/aggregate_iron_search.py \
  --datasets "${DATASET_ARRAY[@]}" \
  --folds "${FOLD_ARRAY[@]}" \
  "${aggregate_args[@]}"
