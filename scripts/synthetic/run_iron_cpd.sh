#!/usr/bin/env bash
set -euo pipefail

# CPD only. Existing AUP/AUR/Completeness results are untouched.
# Existing iron_attributions.npy files are reused. If absent, only the
# TIMING-style IRON saliency map is regenerated from existing checkpoints.

DATASETS=${DATASETS:-"state switch-feature"}
FOLDS=${FOLDS:-"0 1 2 3 4"}
GPUS=${GPUS:-"0 1"}
BASELINES=${BASELINES:-"average zero"}
ATTRIBUTION_BATCH=${ATTRIBUTION_BATCH:-0}
PATH_BATCH=${PATH_BATCH:-0}
TESTBS=${TESTBS:-0}
RECOMPUTE_IF_MISSING=${RECOMPUTE_IF_MISSING:-1}
FORCE_RECOMPUTE=${FORCE_RECOMPUTE:-0}
STRICT=${STRICT:-1}

read -r -a DATASET_ARRAY <<< "${DATASETS}"
read -r -a FOLD_ARRAY <<< "${FOLDS}"
read -r -a GPU_ARRAY <<< "${GPUS}"
read -r -a BASELINE_ARRAY <<< "${BASELINES}"

if [[ ${#DATASET_ARRAY[@]} -eq 0 || ${#FOLD_ARRAY[@]} -eq 0 || \
      ${#GPU_ARRAY[@]} -eq 0 || ${#BASELINE_ARRAY[@]} -eq 0 ]]; then
  echo "DATASETS/FOLDS/GPUS/BASELINES must not be empty" >&2
  exit 2
fi

mkdir -p logs/synthetic_iron_cpd

recompute_args=()
force_args=()
if [[ "${RECOMPUTE_IF_MISSING}" == "1" ]]; then
  recompute_args+=(--recompute-if-missing)
fi
if [[ "${FORCE_RECOMPUTE}" == "1" ]]; then
  force_args+=(--force)
fi

worker() {
  local worker_index=$1
  local gpu=${GPU_ARRAY[$worker_index]}

  for ((dataset_index=worker_index; dataset_index<${#DATASET_ARRAY[@]}; dataset_index+=${#GPU_ARRAY[@]})); do
    local data=${DATASET_ARRAY[$dataset_index]}

    for fold in "${FOLD_ARRAY[@]}"; do
      local log="logs/synthetic_iron_cpd/${data}_fold${fold}.log"
      echo "[LAUNCH] data=${data} fold=${fold} gpu=${gpu}"

      CUDA_VISIBLE_DEVICES="${gpu}" \
      PYTHONPATH=. \
      python synthetic/eval_iron_cpd.py \
        --data "${data}" \
        --fold "${fold}" \
        --device cuda:0 \
        --attribution-batch "${ATTRIBUTION_BATCH}" \
        --path-batch "${PATH_BATCH}" \
        --testbs "${TESTBS}" \
        --baselines "${BASELINE_ARRAY[@]}" \
        "${recompute_args[@]}" \
        "${force_args[@]}" \
        >"${log}" 2>&1

      echo "[DONE] data=${data} fold=${fold} log=${log}"
    done
  done
}

pids=()
for ((worker_index=0; worker_index<${#GPU_ARRAY[@]}; worker_index++)); do
  worker "${worker_index}" &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done
if [[ "${status}" -ne 0 ]]; then
  echo "[ERROR] at least one synthetic IRON CPD worker failed" >&2
  exit "${status}"
fi

aggregate_args=()
if [[ "${STRICT}" == "1" ]]; then
  aggregate_args+=(--strict)
fi

PYTHONPATH=. python synthetic/aggregate_iron_cpd.py \
  --datasets "${DATASET_ARRAY[@]}" \
  --folds "${FOLD_ARRAY[@]}" \
  --baselines "${BASELINE_ARRAY[@]}" \
  "${aggregate_args[@]}"
