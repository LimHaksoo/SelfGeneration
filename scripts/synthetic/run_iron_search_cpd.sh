#!/usr/bin/env bash
set -euo pipefail

# CPD-only evaluation for already-trained synthetic IRON search candidates.
DATASETS=${DATASETS:-"state switch-feature"}
FOLDS=${FOLDS:-"0"}
GPUS=${GPUS:-"0 1 2 3"}

N_STEPS=${N_STEPS:-20}
MAX_EVAL_SAMPLES=${MAX_EVAL_SAMPLES:-64}
ATTRIBUTION_BATCH=${ATTRIBUTION_BATCH:-0}
PATH_BATCH=${PATH_BATCH:-0}
METRIC_BATCH=${METRIC_BATCH:-0}
BASELINES=${BASELINES:-"zero average"}

# Blank means all configured candidates for that dataset.
CANDIDATES_STATE=${CANDIDATES_STATE:-""}
CANDIDATES_SWITCH_FEATURE=${CANDIDATES_SWITCH_FEATURE:-""}

CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-"model/synthetic_iron_search"}
RESULT_ROOT=${RESULT_ROOT:-"results_synthetic_iron_search_cpd"}
FFJORD_ROOT=${FFJORD_ROOT:-"third_party/ffjord_official"}

FORCE=${FORCE:-0}
FORCE_ATTRIBUTION=${FORCE_ATTRIBUTION:-0}
NO_ATTRIBUTION_CACHE=${NO_ATTRIBUTION_CACHE:-0}
STRICT=${STRICT:-1}
PRIMARY_BASELINE=${PRIMARY_BASELINE:-"zero"}

read -r -a DATASET_ARRAY <<< "${DATASETS}"
read -r -a FOLD_ARRAY <<< "${FOLDS}"
read -r -a GPU_ARRAY <<< "${GPUS}"
read -r -a BASELINE_ARRAY <<< "${BASELINES}"

if [[ ${#DATASET_ARRAY[@]} -eq 0 || ${#FOLD_ARRAY[@]} -eq 0 || \
      ${#GPU_ARRAY[@]} -eq 0 || ${#BASELINE_ARRAY[@]} -eq 0 ]]; then
  echo "DATASETS/FOLDS/GPUS/BASELINES must not be empty" >&2
  exit 2
fi

candidate_indices_for_dataset() {
  local data=$1
  local explicit=""

  case "${data}" in
    state|hmm)
      explicit="${CANDIDATES_STATE}"
      ;;
    switch-feature|switch_feature|switch|switchstate)
      explicit="${CANDIDATES_SWITCH_FEATURE}"
      ;;
    *)
      echo "Unsupported dataset=${data}" >&2
      return 2
      ;;
  esac

  if [[ -n "${explicit}" ]]; then
    echo "${explicit}"
    return 0
  fi

  local count
  count=$(PYTHONPATH=. python - "${data}" <<'PY'
import sys
from configs.synthetic_iron_search_space import get_candidates
print(len(get_candidates(sys.argv[1])))
PY
)
  seq 0 $((count - 1)) | paste -sd' ' -
}

candidate_name() {
  local data=$1
  local index=$2
  PYTHONPATH=. python - "${data}" "${index}" <<'PY'
import sys
from configs.synthetic_iron_search_space import get_candidate
print(get_candidate(sys.argv[1], int(sys.argv[2]))["name"])
PY
}

jobs=()
for data in "${DATASET_ARRAY[@]}"; do
  read -r -a candidate_array <<< "$(candidate_indices_for_dataset "${data}")"
  for fold in "${FOLD_ARRAY[@]}"; do
    for candidate_index in "${candidate_array[@]}"; do
      jobs+=("${data}|${fold}|${candidate_index}")
    done
  done
done

if [[ ${#jobs[@]} -eq 0 ]]; then
  echo "No CPD jobs were generated" >&2
  exit 2
fi

mkdir -p logs/synthetic_iron_search_cpd

extra_args=()
if [[ "${FORCE}" == "1" ]]; then
  extra_args+=(--force)
fi
if [[ "${FORCE_ATTRIBUTION}" == "1" ]]; then
  extra_args+=(--force-attribution)
fi
if [[ "${NO_ATTRIBUTION_CACHE}" == "1" ]]; then
  extra_args+=(--no-attribution-cache)
fi

worker() {
  local worker_index=$1
  local gpu=${GPU_ARRAY[$worker_index]}

  for ((job_index=worker_index; job_index<${#jobs[@]}; job_index+=${#GPU_ARRAY[@]})); do
    IFS='|' read -r data fold candidate_index <<< "${jobs[$job_index]}"
    local name
    name=$(candidate_name "${data}" "${candidate_index}")
    local safe_data=${data//-/_}
    local log="logs/synthetic_iron_search_cpd/${safe_data}_fold${fold}_c${candidate_index}_${name}.log"

    echo "[LAUNCH] data=${data} fold=${fold} candidate=${candidate_index}:${name} gpu=${gpu}"
    CUDA_VISIBLE_DEVICES="${gpu}" \
    PYTHONPATH=. \
    python synthetic/eval_iron_search_cpd.py \
      --data "${data}" \
      --fold "${fold}" \
      --candidate-index "${candidate_index}" \
      --device cuda:0 \
      --n-steps "${N_STEPS}" \
      --max-eval-samples "${MAX_EVAL_SAMPLES}" \
      --attribution-batch "${ATTRIBUTION_BATCH}" \
      --path-batch "${PATH_BATCH}" \
      --metric-batch "${METRIC_BATCH}" \
      --baselines "${BASELINE_ARRAY[@]}" \
      --checkpoint-root "${CHECKPOINT_ROOT}" \
      --output-root "${RESULT_ROOT}" \
      --ffjord-root "${FFJORD_ROOT}" \
      "${extra_args[@]}" \
      >"${log}" 2>&1

    echo "[DONE] data=${data} fold=${fold} candidate=${candidate_index}:${name} log=${log}"
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

aggregate_args=(
  --results-root "${RESULT_ROOT}"
  --primary-baseline "${PRIMARY_BASELINE}"
)
if [[ "${STRICT}" == "1" ]]; then
  aggregate_args+=(--strict)
fi
if [[ -n "${CANDIDATES_STATE}" ]]; then
  read -r -a selected_state <<< "${CANDIDATES_STATE}"
  aggregate_args+=(--state-candidates "${selected_state[@]}")
fi
if [[ -n "${CANDIDATES_SWITCH_FEATURE}" ]]; then
  read -r -a selected_switch <<< "${CANDIDATES_SWITCH_FEATURE}"
  aggregate_args+=(--switch-candidates "${selected_switch[@]}")
fi

PYTHONPATH=. python synthetic/aggregate_iron_search_cpd.py \
  --datasets "${DATASET_ARRAY[@]}" \
  --folds "${FOLD_ARRAY[@]}" \
  "${aggregate_args[@]}"
