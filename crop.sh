#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# Build a Boiler/Wafer-only IRON paper-code archive.
#
# Default paths:
#   source repository : /data4/haksoo/pna_big
#   staged release    : /data4/haksoo/iron_boiler_wafer_code
#   output archive    : /data4/haksoo/iron_boiler_wafer_code.zip
#
# Run:
#   bash crop.sh
#
# Override paths:
#   REPO=/path/to/pna_big \
#   OUT=/path/to/release_dir \
#   ZIP=/path/to/release.zip \
#   bash crop.sh
#
# Include the final Boiler/Wafer classifier and flow checkpoints:
#   INCLUDE_CHECKPOINTS=1 bash crop.sh
#
# Raw datasets, generated attributions, result directories, and logs are never
# copied. By default, checkpoints are excluded as well.
# =============================================================================

REPO="${REPO:-/data4/haksoo/pna_big}"
OUT="${OUT:-/data4/haksoo/iron_boiler_wafer_code}"
ZIP="${ZIP:-/data4/haksoo/iron_boiler_wafer_code.zip}"
INCLUDE_CHECKPOINTS="${INCLUDE_CHECKPOINTS:-0}"

REPO="$(cd "$REPO" && pwd)"
OUT="$(python -c 'import os,sys; print(os.path.abspath(os.path.expanduser(sys.argv[1])))' "$OUT")"
ZIP="$(python -c 'import os,sys; print(os.path.abspath(os.path.expanduser(sys.argv[1])))' "$ZIP")"

cd "$REPO"

case "$OUT" in
    ""|"/"|"$REPO")
        echo "[ERROR] Unsafe OUT path: $OUT" >&2
        exit 2
        ;;
esac

case "$OUT/" in
    "$REPO/"*)
        echo "[ERROR] OUT must be outside REPO to avoid recursive copying." >&2
        echo "        REPO=$REPO" >&2
        echo "        OUT =$OUT" >&2
        exit 2
        ;;
esac

if [[ "$INCLUDE_CHECKPOINTS" != "0" && "$INCLUDE_CHECKPOINTS" != "1" ]]; then
    echo "[ERROR] INCLUDE_CHECKPOINTS must be 0 or 1." >&2
    exit 2
fi

rm -rf "$OUT" "$ZIP"
mkdir -p "$OUT"

copy_required_file() {
    local src="$1"

    if [[ ! -f "$src" ]]; then
        echo "[MISSING REQUIRED FILE] $src" >&2
        exit 1
    fi

    mkdir -p "$OUT/$(dirname "$src")"
    cp -a "$src" "$OUT/$src"
}

copy_optional_file() {
    local src="$1"

    if [[ ! -f "$src" ]]; then
        return 0
    fi

    mkdir -p "$OUT/$(dirname "$src")"
    cp -a "$src" "$OUT/$src"
}

copy_required_dir() {
    local src="$1"

    if [[ ! -d "$src" ]]; then
        echo "[MISSING REQUIRED DIRECTORY] $src" >&2
        exit 1
    fi

    mkdir -p "$OUT/$(dirname "$src")"
    cp -a "$src" "$OUT/$src"
}

# =============================================================================
# 1. IRON core and final Boiler/Wafer training/evaluation pipeline
# =============================================================================

CORE_FILES=(
    attribution/nf_ig.py
    attribution/official_ffjord.py
    attribution/official_ffjord_epoch.py

    configs/official_cnf_fullfold_by_dataset.py

    datasets/boiler.py
    datasets/wafer.py

    real/classifier.py
    real/cumulative_difference.py
    real/cnf_faithfulness_metrics.py
    real/cnf_dataset_registry.py
    real/train_official_cnf_epoch_cpd.py
    real/eval_cnf_checkpoint_metrics.py

    scripts/real/run_official_cnf_epoch_by_dataset.sh
)

for file in "${CORE_FILES[@]}"; do
    copy_required_file "$file"
done

# The current training module imports these loaders at module-import time.
# They are copied only to keep the reduced package importable. Their datasets,
# checkpoints, results, and launchers are not included.
copy_required_file datasets/PAM.py
copy_required_file datasets/epilepsy.py

# Optional checkpoint-metric aggregator/launcher variants.
shopt -s nullglob
for file in \
    real/aggregate_cnf_checkpoint_metrics.py \
    scripts/real/run_cnf_checkpoint_metrics.sh \
    scripts/real/run_cnf_checkpoint_metrics*.sh
do
    copy_optional_file "$file"
done
shopt -u nullglob

# =============================================================================
# 2. Paper appendix experiments
#    - finite-step convergence
#    - inference runtime
#    - CNF round-trip reconstruction
# =============================================================================

APPENDIX_FILES=(
    configs/finite_step_convergence.py

    real/eval_finite_step_convergence.py
    real/aggregate_finite_step_convergence.py
    scripts/real/run_finite_step_convergence.sh

    real/benchmark_runtime_table.py
    real/aggregate_runtime_table.py
    scripts/real/run_runtime_table.sh

    real/eval_cnf_reconstruction.py
    real/aggregate_cnf_reconstruction.py
    scripts/real/run_cnf_reconstruction.sh
)

for file in "${APPENDIX_FILES[@]}"; do
    copy_required_file "$file"
done

# =============================================================================
# 3. Package initializers and project metadata
# =============================================================================

for file in \
    attribution/__init__.py \
    configs/__init__.py \
    datasets/__init__.py \
    real/__init__.py \
    scripts/__init__.py \
    scripts/real/__init__.py \
    README.md \
    LICENSE \
    LICENSE.txt \
    requirements.txt \
    requirement.txt \
    environment.yml \
    environment.yaml \
    pyproject.toml \
    setup.py
do
    copy_optional_file "$file"
done

# =============================================================================
# 4. Minimal official FFJORD source
#
# attribution/official_ffjord.py loads train_misc.py and modules under lib/.
# The full external repository history, examples, datasets, and generated files
# are intentionally excluded.
# =============================================================================

copy_required_file third_party/ffjord_official/train_misc.py
copy_required_dir third_party/ffjord_official/lib

copy_optional_file third_party/ffjord_official/LICENSE
copy_optional_file third_party/ffjord_official/LICENSE.md
copy_optional_file third_party/ffjord_official/README.md
copy_optional_file scripts/setup_official_ffjord.sh

# =============================================================================
# 5. Boiler/Wafer-only classifier trainer
# =============================================================================

cat > "$OUT/real/train_boiler_wafer_classifier.py" <<'PYFILE'
#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from pytorch_lightning import Trainer, seed_everything

from datasets.boiler import Boiler
from datasets.wafer import Wafer
from real.classifier import MimicClassifierNet


def build(data: str, fold: int, seed: int):
    if data == "boiler":
        datamodule = Boiler(fold=fold, seed=seed)
        shape = (20, 2, 36)
    elif data == "wafer":
        datamodule = Wafer(n_folds=5, fold=fold, seed=seed)
        shape = (1, 2, 152)
    else:
        raise ValueError(data)

    feature_size, n_state, n_timesteps = shape
    classifier = MimicClassifierNet(
        feature_size=feature_size,
        n_state=n_state,
        n_timesteps=n_timesteps,
        hidden_size=200,
        regres=True,
        loss="cross_entropy",
        lr=1e-4,
        l2=1e-3,
        model_type="state",
    )
    return datamodule, classifier


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, choices=("boiler", "wafer"))
    parser.add_argument("--fold", type=int, required=True, choices=range(5))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    seed_everything(args.seed, workers=True)
    datamodule, classifier = build(args.data, args.fold, args.seed)

    accelerator = args.device.split(":", 1)[0]
    devices = 1
    if accelerator == "cuda" and ":" in args.device:
        devices = [int(args.device.split(":", 1)[1])]

    trainer = Trainer(
        max_epochs=args.epochs,
        accelerator=accelerator,
        devices=devices,
        deterministic=True,
        logger=False,
        enable_checkpointing=False,
    )
    trainer.fit(classifier, datamodule=datamodule)

    destination = (
        Path("model")
        / args.data
        / f"state_classifier_{args.fold}_{args.seed}_no_imputation"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(classifier.state_dict(), destination)
    print(f"[SAVE] {destination}")


if __name__ == "__main__":
    main()
PYFILE

cat > "$OUT/scripts/real/train_boiler_wafer_classifiers.sh" <<'RUNNER'
#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

read -r -a DATASET_LIST <<< "${DATASETS:-boiler wafer}"
read -r -a GPU_LIST <<< "${GPUS:-0 1}"
FOLDS="${FOLDS:-0 1 2 3 4}"
EPOCHS="${EPOCHS:-100}"
SEED="${SEED:-42}"

if (( ${#DATASET_LIST[@]} > ${#GPU_LIST[@]} )); then
    echo "[ERROR] Need at least one GPU per dataset." >&2
    exit 2
fi

pids=()
for index in "${!DATASET_LIST[@]}"; do
    data="${DATASET_LIST[$index]}"
    gpu="${GPU_LIST[$index]}"

    (
        set -euo pipefail
        for fold in $FOLDS; do
            CUDA_VISIBLE_DEVICES="$gpu" \
            PYTHONPATH=. \
            python real/train_boiler_wafer_classifier.py \
                --data "$data" \
                --fold "$fold" \
                --seed "$SEED" \
                --epochs "$EPOCHS" \
                --device cuda:0
        done
    ) &
    pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
    wait "$pid" || failed=1
done
(( failed == 0 )) || exit 1
RUNNER

chmod +x "$OUT/scripts/real/train_boiler_wafer_classifiers.sh"

# =============================================================================
# 6. Boiler/Wafer-only main-metric launcher
# =============================================================================

cat > "$OUT/scripts/real/run_iron_boiler_wafer_metrics.sh" <<'RUNNER'
#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

read -r -a DATASET_LIST <<< "${DATASETS:-boiler wafer}"
read -r -a GPU_LIST <<< "${GPUS:-0 1}"

FOLDS="${FOLDS:-0 1 2 3 4}"
SEED="${SEED:-42}"
DATA_ROOT="${DATA_ROOT:-${PNA_DATA_ROOT:-}}"
FORCE_RECOMPUTE="${FORCE_RECOMPUTE:-0}"

if (( ${#DATASET_LIST[@]} > ${#GPU_LIST[@]} )); then
    echo "[ERROR] Need at least one GPU per dataset." >&2
    exit 2
fi

extra_args=()
if [[ -n "$DATA_ROOT" ]]; then
    extra_args+=(--data-root "$DATA_ROOT")
fi
if [[ "$FORCE_RECOMPUTE" == "1" ]]; then
    extra_args+=(--force-recompute)
fi

pids=()
labels=()

for index in "${!DATASET_LIST[@]}"; do
    data="${DATASET_LIST[$index]}"
    gpu="${GPU_LIST[$index]}"

    case "$data" in
        boiler|wafer) ;;
        *)
            echo "[ERROR] This launcher supports only boiler and wafer: $data" >&2
            exit 2
            ;;
    esac

    (
        set -euo pipefail
        for fold in $FOLDS; do
            echo "[METRICS] data=$data fold=$fold gpu=$gpu"
            CUDA_VISIBLE_DEVICES="$gpu" \
            PYTHONPATH=. \
            python real/eval_cnf_checkpoint_metrics.py \
                --data "$data" \
                --fold "$fold" \
                --seed "$SEED" \
                --device cuda:0 \
                "${extra_args[@]}"
        done
    ) &

    pids+=("$!")
    labels+=("$data/gpu$gpu")
done

failed=0
for index in "${!pids[@]}"; do
    if wait "${pids[$index]}"; then
        echo "[DONE] ${labels[$index]}"
    else
        echo "[FAILED] ${labels[$index]}" >&2
        failed=1
    fi
done
(( failed == 0 )) || exit 1
RUNNER

chmod +x "$OUT/scripts/real/run_iron_boiler_wafer_metrics.sh"

# =============================================================================
# 7. Release README
# =============================================================================

cat > "$OUT/README_RELEASE.md" <<'README'
# IRON: Boiler and Wafer Reproduction Package

This archive contains the IRON implementation and paper-evaluation code for
Boiler and Wafer. Raw datasets, generated attribution arrays, results, and logs
are not included. Trained checkpoints are included only when the archive is
built with `INCLUDE_CHECKPOINTS=1`.

## Final configurations

| Dataset | Candidate | Hidden dims | Blocks | Solver | Step size | LR |
|---|---|---:|---:|---|---:|---:|
| Boiler | `rk4_64x2_b2_s010_lr5e-4` | 64-64 | 2 | RK4 | 0.10 | 5e-4 |
| Wafer | `rk4_64x2_b1_s010` | 64-64 | 1 | RK4 | 0.10 | 1e-3 |

Shared attribution protocol: predicted-class target, zero baseline, `K=100`,
attribution batch 256, path batch 24, folds 0--4, and seed 42.

## 1. Train classifiers

```bash
DATASETS="boiler wafer" GPUS="0 1" \
bash scripts/real/train_boiler_wafer_classifiers.sh
```

## 2. Train selected flow models

```bash
DATASETS="boiler wafer" GPUS="0 1" EPOCHS=1000 \
bash scripts/real/run_official_cnf_epoch_by_dataset.sh
```

## 3. Evaluate main IRON metrics

```bash
DATASETS="boiler wafer" GPUS="0 1" \
bash scripts/real/run_iron_boiler_wafer_metrics.sh
```

## 4. Finite-step convergence

```bash
DATASETS="boiler wafer" METHODS="IG IRON" \
K_VALUES="5 10 20 50 100 200" FOLDS="0 1 2 3 4" GPUS="0 1" \
bash scripts/real/run_finite_step_convergence.sh
```

## 5. IRON runtime

The reduced package does not include local MA-GIG and TIMING implementations.
Run the runtime evaluator with `METHODS=IRON`.

```bash
DATASETS="boiler wafer" METHODS="IRON" FOLDS="0 1 2 3 4" \
GPUS="0 1" N_SAMPLES=100 WARMUP=5 K=100 \
bash scripts/real/run_runtime_table.sh
```

## 6. CNF round-trip reconstruction

```bash
BENCHMARK_DATASETS="boiler wafer" SYNTHETIC_DATASETS="" \
FOLDS="0 1 2 3 4" GPUS="0 1" BENCHMARK_SOURCE="selected" STRICT=1 \
bash scripts/real/run_cnf_reconstruction.sh
```

Expected selected flow checkpoints:

```text
model/boiler/official_ffjord_epoch/fold<fold>_seed42/
    rk4_64x2_b2_s010_lr5e-4.pt
model/wafer/official_ffjord_epoch/fold<fold>_seed42/
    rk4_64x2_b1_s010.pt
```

`datasets/PAM.py` and `datasets/epilepsy.py` are present only because the
current flow-training module imports them at module-import time. Their data,
checkpoints, results, and experiment launchers are not included.
README

# =============================================================================
# 8. Optional Boiler/Wafer checkpoints
# =============================================================================

if [[ "$INCLUDE_CHECKPOINTS" == "1" ]]; then
    for fold in 0 1 2 3 4; do
        copy_required_file \
            "model/boiler/state_classifier_${fold}_42_no_imputation"
        copy_required_file \
            "model/wafer/state_classifier_${fold}_42_no_imputation"

        copy_required_file \
            "model/boiler/official_ffjord_epoch/fold${fold}_seed42/rk4_64x2_b2_s010_lr5e-4.pt"
        copy_required_file \
            "model/wafer/official_ffjord_epoch/fold${fold}_seed42/rk4_64x2_b1_s010.pt"
    done
fi

# =============================================================================
# 9. Remove generated/private artifacts
# =============================================================================

find "$OUT" -type d \
    \( -name '.git' \
       -o -name '__pycache__' \
       -o -name '.pytest_cache' \
       -o -name '.mypy_cache' \) \
    -prune -exec rm -rf {} +

find "$OUT" -type f \
    \( -name '*.pyc' \
       -o -name '*.pyo' \
       -o -name '*.tmp' \
       -o -name '*.log' \
       -o -name '.DS_Store' \) \
    -delete

if [[ "$INCLUDE_CHECKPOINTS" == "0" ]]; then
    if find "$OUT" -type f \
        \( -name '*.npy' \
           -o -name '*.npz' \
           -o -name '*.pt' \
           -o -name '*.pth' \
           -o -name '*.ckpt' \
           -o -name '*.pkl' \) \
        -print -quit | grep -q .; then
        echo "[ERROR] Generated arrays/checkpoints were accidentally included." >&2
        exit 1
    fi
else
    if find "$OUT" -type f \
        \( -name '*.npy' -o -name '*.npz' -o -name '*.pkl' \) \
        -print -quit | grep -q .; then
        echo "[ERROR] Generated arrays or raw pickle data were included." >&2
        exit 1
    fi
fi

# =============================================================================
# 10. Syntax validation
# =============================================================================

# Exclude pinned third-party FFJORD code from py_compile. It is kept unchanged
# for reproducibility and may contain compatibility constructs for its original
# environment.
while IFS= read -r -d '' file; do
    python -m py_compile "$file"
done < <(
    find "$OUT" \
        -path "$OUT/third_party" -prune -o \
        -type f -name '*.py' -print0
)

while IFS= read -r -d '' file; do
    bash -n "$file"
done < <(find "$OUT" -type f -name '*.sh' -print0)

find "$OUT" -type d -name '__pycache__' \
    -prune -exec rm -rf {} +

# =============================================================================
# 11. Manifest and ZIP
# =============================================================================

python - "$OUT" "$ZIP" <<'PY'
from __future__ import annotations

import hashlib
import sys
import zipfile
from pathlib import Path

release = Path(sys.argv[1]).resolve()
archive = Path(sys.argv[2]).resolve()

files = sorted(
    path
    for path in release.rglob("*")
    if path.is_file() and path.name != "SHA256SUMS"
)

manifest_lines = []
for path in files:
    relative = path.relative_to(release)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest_lines.append(f"{digest}  {relative.as_posix()}")

(release / "SHA256SUMS").write_text(
    "\n".join(manifest_lines) + "\n",
    encoding="utf-8",
)

archive.parent.mkdir(parents=True, exist_ok=True)
if archive.exists():
    archive.unlink()

with zipfile.ZipFile(
    archive,
    mode="w",
    compression=zipfile.ZIP_DEFLATED,
    compresslevel=9,
) as handle:
    root_name = release.name
    for path in sorted(p for p in release.rglob("*") if p.is_file()):
        arcname = Path(root_name) / path.relative_to(release)
        handle.write(path, arcname.as_posix())
PY

echo "[DONE] Release directory: $OUT"
echo "[DONE] ZIP: $ZIP"
du -h "$ZIP"
