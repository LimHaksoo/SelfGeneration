#!/usr/bin/env bash
set -euo pipefail

# Numerical x -> z -> x round-trip evaluation for trained IRON CNFs.
# Default: all five benchmark datasets and the two final synthetic models,
# folds 0-4, one sequential worker per listed GPU.

SUITES=${SUITES:-"benchmark synthetic"}
BENCHMARK_DATASETS=${BENCHMARK_DATASETS:-"PAM boiler epilepsy wafer freezer"}
SYNTHETIC_DATASETS=${SYNTHETIC_DATASETS:-"state switch-feature"}
SYNTHETIC_SOURCE=${SYNTHETIC_SOURCE:-"final"}  # final | search

FOLDS=${FOLDS:-"0 1 2 3 4"}
GPUS=${GPUS:-"0 1 2 3 4 5 6"}
SEED=${SEED:-42}
SPLIT=${SPLIT:-test}
BATCH_SIZE=${BATCH_SIZE:-0}
MAX_SAMPLES=${MAX_SAMPLES:-0}
OUTPUT_ROOT=${OUTPUT_ROOT:-results_cnf_reconstruction}
FFJORD_ROOT=${FFJORD_ROOT:-third_party/ffjord_official}
DATA_ROOT=${DATA_ROOT:-""}

# Used only when SYNTHETIC_SOURCE=search. Blank means every configured search
# candidate. Values are zero-based indices from synthetic_iron_search_space.py.
SYNTHETIC_CANDIDATES_STATE=${SYNTHETIC_CANDIDATES_STATE:-""}
SYNTHETIC_CANDIDATES_SWITCH_FEATURE=${SYNTHETIC_CANDIDATES_SWITCH_FEATURE:-""}

SKIP_MISSING=${SKIP_MISSING:-1}
FORCE=${FORCE:-0}
STRICT=${STRICT:-0}

read -r -a SUITE_ARRAY <<< "${SUITES}"
read -r -a BENCHMARK_ARRAY <<< "${BENCHMARK_DATASETS}"
read -r -a SYNTHETIC_ARRAY <<< "${SYNTHETIC_DATASETS}"
read -r -a FOLD_ARRAY <<< "${FOLDS}"
read -r -a GPU_ARRAY <<< "${GPUS}"

if [[ ${#GPU_ARRAY[@]} -eq 0 || ${#FOLD_ARRAY[@]} -eq 0 ]]; then
  echo "GPUS and FOLDS must not be empty" >&2
  exit 2
fi
if [[ "${SYNTHETIC_SOURCE}" != "final" && "${SYNTHETIC_SOURCE}" != "search" ]]; then
  echo "SYNTHETIC_SOURCE must be final or search" >&2
  exit 2
fi

declare -A seen_gpu=()
for gpu in "${GPU_ARRAY[@]}"; do
  if [[ -n "${seen_gpu[$gpu]:-}" ]]; then
    echo "Duplicate GPU id is not allowed: ${gpu}" >&2
    exit 2
  fi
  seen_gpu[$gpu]=1
done

benchmark_candidates() {
  local data=$1
  PYTHONPATH=. python - "${data}" <<'PY'
import sys
from configs.official_cnf_fullfold_by_dataset import get_dataset_config
for candidate in get_dataset_config(sys.argv[1])["candidates"]:
    print(candidate["name"])
PY
}

synthetic_candidates() {
  local data=$1
  local source=$2
  local indices=""

  case "${data}" in
    state|hmm)
      indices="${SYNTHETIC_CANDIDATES_STATE}"
      ;;
    switch-feature|switch_feature|switch|switchstate)
      indices="${SYNTHETIC_CANDIDATES_SWITCH_FEATURE}"
      ;;
    *)
      echo "Unsupported synthetic dataset: ${data}" >&2
      return 2
      ;;
  esac

  PYTHONPATH=. python - "${data}" "${source}" "${indices}" <<'PY'
import sys

data, source, raw_indices = sys.argv[1:]
if source == "final":
    from configs.synthetic_iron_whitebox import get_candidate
    print(get_candidate(data)["name"])
else:
    from configs.synthetic_iron_search_space import get_candidates
    candidates = get_candidates(data)
    if raw_indices.strip():
        selected = [int(value) for value in raw_indices.split()]
    else:
        selected = list(range(len(candidates)))
    for index in selected:
        if not 0 <= index < len(candidates):
            raise IndexError(
                f"candidate index {index} outside [0,{len(candidates)-1}]"
            )
        print(candidates[index]["name"])
PY
}

jobs=()
for suite in "${SUITE_ARRAY[@]}"; do
  case "${suite}" in
    benchmark)
      for data in "${BENCHMARK_ARRAY[@]}"; do
        [[ -z "${data}" ]] && continue
        mapfile -t candidates < <(benchmark_candidates "${data}")
        for fold in "${FOLD_ARRAY[@]}"; do
          for candidate in "${candidates[@]}"; do
            jobs+=("benchmark|final|${data}|${fold}|${candidate}")
          done
        done
      done
      ;;
    synthetic)
      for data in "${SYNTHETIC_ARRAY[@]}"; do
        [[ -z "${data}" ]] && continue
        mapfile -t candidates < <(
          synthetic_candidates "${data}" "${SYNTHETIC_SOURCE}"
        )
        for fold in "${FOLD_ARRAY[@]}"; do
          for candidate in "${candidates[@]}"; do
            jobs+=("synthetic|${SYNTHETIC_SOURCE}|${data}|${fold}|${candidate}")
          done
        done
      done
      ;;
    *)
      echo "Unsupported suite: ${suite}" >&2
      exit 2
      ;;
  esac
done

if [[ ${#jobs[@]} -eq 0 ]]; then
  echo "No reconstruction jobs were generated" >&2
  exit 2
fi

mkdir -p logs/cnf_reconstruction

common_args=(
  --seed "${SEED}"
  --split "${SPLIT}"
  --batch-size "${BATCH_SIZE}"
  --max-samples "${MAX_SAMPLES}"
  --output-root "${OUTPUT_ROOT}"
  --ffjord-root "${FFJORD_ROOT}"
)
if [[ -n "${DATA_ROOT}" ]]; then
  common_args+=(--data-root "${DATA_ROOT}")
fi
if [[ "${SKIP_MISSING}" == "1" ]]; then
  common_args+=(--skip-missing)
fi
if [[ "${FORCE}" == "1" ]]; then
  common_args+=(--force)
fi

worker() {
  local worker_index=$1
  local gpu=${GPU_ARRAY[$worker_index]}

  for ((job_index=worker_index; job_index<${#jobs[@]}; job_index+=${#GPU_ARRAY[@]})); do
    IFS='|' read -r suite source data fold candidate <<< "${jobs[$job_index]}"
    local safe_data=${data//-/_}
    local safe_candidate=${candidate//\//_}
    local log="logs/cnf_reconstruction/${suite}_${source}_${safe_data}_fold${fold}_${safe_candidate}.log"

    echo "[LAUNCH] suite=${suite} source=${source} data=${data} fold=${fold} candidate=${candidate} gpu=${gpu}"
    CUDA_VISIBLE_DEVICES="${gpu}" \
    PYTHONPATH=. \
    python real/eval_cnf_reconstruction.py \
      --suite "${suite}" \
      --synthetic-source "${source}" \
      --data "${data}" \
      --fold "${fold}" \
      --candidate-name "${candidate}" \
      --device cuda:0 \
      "${common_args[@]}" \
      >"${log}" 2>&1

    echo "[DONE] suite=${suite} data=${data} fold=${fold} candidate=${candidate} log=${log}"
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
  echo "[ERROR] at least one reconstruction worker failed" >&2
  exit "${status}"
fi

aggregate_args=(
  --results-root "${OUTPUT_ROOT}"
  --folds "${FOLD_ARRAY[@]}"
)
if [[ "${STRICT}" == "1" ]]; then
  aggregate_args+=(--strict)
fi

PYTHONPATH=. python real/aggregate_cnf_reconstruction.py "${aggregate_args[@]}"
