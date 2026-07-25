#!/usr/bin/env bash
set -euo pipefail

FOLDS=${FOLDS:-"0 1 2 3 4"}
GPUS=${GPUS:-"0 1 2 3 4"}
EPOCHS=${EPOCHS:-50}
FORCE_RETRAIN=${FORCE_RETRAIN:-0}

read -r -a FOLD_ARRAY <<< "${FOLDS}"
read -r -a GPU_ARRAY <<< "${GPUS}"

if [[ ${#FOLD_ARRAY[@]} -eq 0 || ${#GPU_ARRAY[@]} -eq 0 ]]; then
  echo "FOLDS and GPUS must not be empty" >&2
  exit 2
fi

mkdir -p model/switch_feature logs/synthetic_switch_classifier

force_args=()
if [[ "${FORCE_RETRAIN}" == "1" ]]; then
  force_args+=(--force)
fi

pids=()
for index in "${!FOLD_ARRAY[@]}"; do
  fold=${FOLD_ARRAY[$index]}
  gpu=${GPU_ARRAY[$((index % ${#GPU_ARRAY[@]}))]}
  log="logs/synthetic_switch_classifier/fold${fold}.log"

  echo "[LAUNCH] switch-feature classifier fold=${fold} gpu=${gpu}"
  CUDA_VISIBLE_DEVICES="${gpu}" \
  PYTHONPATH=. \
  python synthetic/train_whitebox_classifier.py \
    --data switch-feature \
    --fold "${fold}" \
    --seed 42 \
    --device cuda:0 \
    --epochs "${EPOCHS}" \
    --deterministic \
    "${force_args[@]}" \
    >"${log}" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done
if [[ "${status}" -ne 0 ]]; then
  echo "[ERROR] switch-feature classifier training failed" >&2
  exit "${status}"
fi

for fold in "${FOLD_ARRAY[@]}"; do
  checkpoint="model/switch_feature/classifier_${fold}_42"
  if [[ ! -f "${checkpoint}" ]]; then
    echo "[ERROR] missing checkpoint after training: ${checkpoint}" >&2
    exit 1
  fi
  echo "[READY] ${checkpoint}"
done
