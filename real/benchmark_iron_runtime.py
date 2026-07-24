#!/usr/bin/env python3
"""Measure IRON attribution runtime with the baseline runtime protocol.

Protocol reproduced from ``real/main_runtime.py``:

* use the first ``warmup + n_rep`` test samples in their original order;
* batch size is one;
* move inputs, masks, and timesteps to the GPU before timing;
* time predicted-target selection plus attribution;
* synchronize CUDA immediately before and after the timed region;
* discard the first ``warmup`` complete calls;
* report per-sample mean/std/median for the remaining calls.

Model loading, checkpoint loading, explainer construction, data transfer, and
explicit timestep construction are outside the timed region.  CNF encode,
decode, path construction, and classifier backward passes remain inside
``NFIntegratedGradients.attribute`` and are therefore timed.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import gc
import os
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch as th
from captum._utils.common import _run_forward
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from attribution.nf_ig import NFIntegratedGradients
from attribution.official_ffjord import load_checkpoint
from configs.official_cnf_fullfold_by_dataset import get_candidates, split_candidate
from real.cnf_dataset_registry import build_classifier, load_splits

try:
    from configs.official_cnf_fullfold_by_dataset import get_dataset_config
except ImportError:  # compatibility with older config revisions
    get_dataset_config = None


DATASETS: Tuple[str, ...] = (
    "PAM",
    "boiler",
    "epilepsy",
    "wafer",
    "freezer",
)

MAIN_CANDIDATE: Dict[str, str] = {
    "PAM": "rk4_64x2_b2_s010_lr5e-5",
    "boiler": "rk4_64x2_b2_s010_lr5e-4",
    "epilepsy": "rk4_128x2_b1_s005",
    "wafer": "rk4_64x2_b1_s010",
    "freezer": "rk4_64x2_b1_s010_lr1e-3",
}

DEFAULT_PATH_BATCH: Dict[str, int] = {
    "PAM": 8,
    "boiler": 24,
    "epilepsy": 24,
    "wafer": 24,
    "freezer": 24,
}

CSV_COLUMNS: Tuple[str, ...] = (
    "data",
    "fold",
    "seed",
    "explainer",
    "n_rep",
    "warmup",
    "batch_size",
    "mean_ms",
    "std_ms",
    "median_ms",
)


def canonical_data(value: str) -> str:
    token = str(value).strip().casefold()
    aliases = {
        "pam": "PAM",
        "boiler": "boiler",
        "epilepsy": "epilepsy",
        "wafer": "wafer",
        "freezer": "freezer",
    }
    if token not in aliases:
        raise ValueError(f"Unsupported dataset={value!r}; available={DATASETS}")
    return aliases[token]


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    if th.cuda.is_available():
        th.cuda.manual_seed_all(seed)


def sync_cuda(device: th.device) -> None:
    if device.type == "cuda":
        th.cuda.synchronize(device)


def candidate_name(candidate: Mapping[str, Any]) -> str:
    if "name" in candidate:
        return str(candidate["name"])
    model = candidate.get("model")
    if isinstance(model, Mapping) and "name" in model:
        return str(model["name"])
    raise ValueError(f"Candidate has no name: {candidate}")


def select_candidate(data: str) -> Tuple[int, Dict[str, Any]]:
    expected = MAIN_CANDIDATE[data]
    candidates = [dict(value) for value in get_candidates(data)]
    matches = [
        (index, candidate)
        for index, candidate in enumerate(candidates)
        if candidate_name(candidate) == expected
    ]
    if len(matches) != 1:
        available = [candidate_name(value) for value in candidates]
        raise RuntimeError(
            f"Expected exactly one candidate {expected!r} for {data}; "
            f"available={available}"
        )
    return matches[0]


def resolve_path_batch(data: str, override: int) -> int:
    if int(override) > 0:
        return int(override)

    if get_dataset_config is not None:
        config = get_dataset_config(data)
        compute = config.get("compute", {})
        value = compute.get("path_batch")
        if value is not None and int(value) > 0:
            return int(value)

    return int(DEFAULT_PATH_BATCH[data])


def checkpoint_path(
    checkpoint_root: Path,
    data: str,
    fold: int,
    seed: int,
    candidate: str,
) -> Path:
    return (
        checkpoint_root
        / data
        / "official_ffjord_epoch"
        / f"fold{fold}_seed{seed}"
        / f"{candidate}.pt"
    )


def benchmark_iron(
    *,
    classifier: nn.Module,
    flow: nn.Module,
    x_test: Tensor,
    mask_test: Tensor,
    device: th.device,
    n_steps: int,
    n_rep: int,
    warmup: int,
    path_batch_size: int,
) -> Dict[str, Any]:
    n_total = min(int(warmup) + int(n_rep), int(x_test.shape[0]))
    if n_total <= int(warmup):
        raise ValueError(
            f"Test split is too small: n_test={x_test.shape[0]}, warmup={warmup}"
        )

    # Baseline protocol: use the first samples, in their original order, and
    # place all inputs used by the benchmark on the device before timing.
    x_eval = x_test[:n_total].to(device=device, dtype=th.float32)
    mask_eval = mask_test[:n_total].to(device=device)
    timesteps = (
        th.linspace(
            0.0,
            1.0,
            x_eval.shape[1],
            device=device,
            dtype=x_eval.dtype,
        )
        .unsqueeze(0)
        .repeat(n_total, 1)
    )

    loader = DataLoader(
        TensorDataset(x_eval, mask_eval, timesteps),
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    classifier.eval()
    flow.eval()
    for parameter in classifier.parameters():
        parameter.requires_grad_(False)
    for parameter in flow.parameters():
        parameter.requires_grad_(False)

    explainer = NFIntegratedGradients(classifier.predict, flow)
    measured_seconds = []

    previous_cudnn = th.backends.cudnn.enabled
    if device.type == "cuda":
        th.backends.cudnn.enabled = False

    try:
        for batch_index, (x_batch, mask_batch, timestep_batch) in enumerate(loader):
            sync_cuda(device)
            start = time.perf_counter()

            # Match main_runtime.py: predicted-target selection is timed.
            with th.no_grad():
                prediction = _run_forward(
                    classifier,
                    x_batch,
                    additional_forward_args=(
                        mask_batch,
                        timestep_batch,
                        False,
                    ),
                )
            targets = prediction.argmax(dim=-1)

            attribution = explainer.attribute(
                x_batch,
                targets=targets,
                baseline=th.zeros_like(x_batch),
                additional_forward_args=(
                    mask_batch,
                    timestep_batch,
                    False,
                ),
                n_steps=int(n_steps),
                path_batch_size=int(path_batch_size),
                return_diagnostics=False,
            )
            if isinstance(attribution, tuple):
                attribution = attribution[0]
            attribution = attribution.abs()

            sync_cuda(device)
            elapsed = time.perf_counter() - start

            if batch_index >= int(warmup):
                measured_seconds.append(float(elapsed))

            del attribution, prediction, targets
    finally:
        if device.type == "cuda":
            th.backends.cudnn.enabled = previous_cudnn

    milliseconds = [value * 1000.0 for value in measured_seconds]
    if not milliseconds:
        raise RuntimeError("No measured IRON samples remain after warm-up")

    return {
        "n_rep": len(milliseconds),
        "warmup": int(warmup),
        "batch_size": 1,
        "mean_ms": float(statistics.mean(milliseconds)),
        "std_ms": float(
            statistics.stdev(milliseconds) if len(milliseconds) > 1 else 0.0
        ),
        "median_ms": float(statistics.median(milliseconds)),
        "n_steps": int(n_steps),
        "path_batch_size": int(path_batch_size),
    }


def write_runtime_row(
    output_file: Path,
    row: Mapping[str, Any],
    *,
    force: bool,
) -> bool:
    """Insert one fold row without creating duplicate IRON measurements."""

    output_file.parent.mkdir(parents=True, exist_ok=True)
    key = (
        str(row["data"]),
        str(row["fold"]),
        str(row["seed"]),
        str(row["explainer"]),
    )

    with output_file.open("a+", encoding="utf-8", newline="") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        handle.seek(0)
        existing = list(csv.DictReader(handle)) if output_file.stat().st_size else []

        def row_key(value: Mapping[str, Any]) -> Tuple[str, str, str, str]:
            return (
                str(value.get("data", "")),
                str(value.get("fold", "")),
                str(value.get("seed", "")),
                str(value.get("explainer", "")),
            )

        matched = [value for value in existing if row_key(value) == key]
        if matched and not force:
            return False

        retained = [value for value in existing if row_key(value) != key]
        retained.append({column: row[column] for column in CSV_COLUMNS})

        handle.seek(0)
        handle.truncate()
        writer = csv.DictWriter(handle, fieldnames=list(CSV_COLUMNS))
        writer.writeheader()
        writer.writerows(retained)
        handle.flush()
        os.fsync(handle.fileno())
        return True


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--fold", type=int, required=True, choices=range(5))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--n-steps", type=int, default=100)
    parser.add_argument("--n-rep", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--path-batch-size", type=int, default=0)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--checkpoint-root", default="model")
    parser.add_argument("--ffjord-root", default="third_party/ffjord_official")
    parser.add_argument("--output-file", default="runtime_all.csv")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    if args.n_steps < 1 or args.n_rep < 1 or args.warmup < 0:
        parser.error("n-steps/n-rep must be positive and warmup must be nonnegative")
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    data = canonical_data(args.data)
    seed_all(args.seed)

    device = th.device(args.device)
    if device.type == "cuda" and not th.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    candidate_index, candidate = select_candidate(data)
    candidate_label = candidate_name(candidate)
    checkpoint = checkpoint_path(
        Path(args.checkpoint_root),
        data,
        args.fold,
        args.seed,
        candidate_label,
    )
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    _train, _val, test = load_splits(data, args.fold, args.seed, args.data_root)
    classifier = build_classifier(data, args.fold, args.seed, device)
    classifier.eval()

    model_candidate, _training = split_candidate(candidate)
    flow, metadata = load_checkpoint(
        checkpoint,
        ffjord_root=args.ffjord_root,
        device=device,
        expected_input_shape=test["x"].shape[1:],
    )
    if hasattr(flow, "candidate") and dict(flow.candidate) != dict(model_candidate):
        raise RuntimeError(
            f"Checkpoint candidate mismatch: config={model_candidate}, "
            f"checkpoint={flow.candidate}"
        )

    path_batch_size = resolve_path_batch(data, args.path_batch_size)
    try:
        result = benchmark_iron(
            classifier=classifier,
            flow=flow,
            x_test=test["x"],
            mask_test=test["mask"],
            device=device,
            n_steps=args.n_steps,
            n_rep=args.n_rep,
            warmup=args.warmup,
            path_batch_size=path_batch_size,
        )

        row = {
            "data": data,
            "fold": int(args.fold),
            "seed": int(args.seed),
            "explainer": "iron",
            "n_rep": int(result["n_rep"]),
            "warmup": int(result["warmup"]),
            "batch_size": 1,
            "mean_ms": f"{result['mean_ms']:.4f}",
            "std_ms": f"{result['std_ms']:.4f}",
            "median_ms": f"{result['median_ms']:.4f}",
        }
        written = write_runtime_row(
            Path(args.output_file),
            row,
            force=bool(args.force),
        )

        status = "WROTE" if written else "SKIP existing"
        print(
            f"[{status}] data={data} fold={args.fold} IRON "
            f"candidate={candidate_label} index={candidate_index} "
            f"K={args.n_steps} path_batch={path_batch_size} "
            f"mean={result['mean_ms']:.4f} ms "
            f"std={result['std_ms']:.4f} ms "
            f"median={result['median_ms']:.4f} ms "
            f"best_epoch={metadata.get('best_epoch')}"
        )
        return 0
    finally:
        del flow
        del classifier
        gc.collect()
        if th.cuda.is_available():
            th.cuda.empty_cache()


if __name__ == "__main__":
    raise SystemExit(main())
