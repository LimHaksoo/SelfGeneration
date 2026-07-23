#!/usr/bin/env python3
"""Measure IRON attribution latency with the original baseline protocol.

Protocol copied from ``real/main_runtime.py``:
  * use the first ``warmup + n_rep`` test samples in their original order;
  * batch size is fixed to one;
  * move data and construct timesteps before timing;
  * synchronize CUDA immediately before and after timing;
  * include predicted-target forward plus the attribution call;
  * discard the first ``warmup`` measurements;
  * write the same CSV schema as ``runtime_all.csv``.
"""

from __future__ import annotations

import argparse
import csv
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import torch
from captum._utils.common import _run_forward


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from attribution.nf_ig import NFIntegratedGradients  # noqa: E402
from attribution.official_ffjord import load_checkpoint  # noqa: E402
from configs.official_cnf_fullfold_by_dataset import (  # noqa: E402
    get_candidates,
    get_dataset_config,
    split_candidate,
)
from real.cnf_dataset_registry import (  # noqa: E402
    build_classifier,
    load_splits,
)


DEFAULT_CANDIDATE: Dict[str, str] = {
    "PAM": "rk4_64x2_b2_s010_lr5e-5",
    "boiler": "rk4_64x2_b2_s010_lr5e-4",
    "epilepsy": "rk4_128x2_b1_s005",
    "wafer": "rk4_64x2_b1_s010",
    "freezer": "rk4_64x2_b1_s010_lr1e-3",
}

try:
    from configs.finite_step_convergence import MAIN_TABLE_IRON_CANDIDATE
except ImportError:
    MAIN_TABLE_IRON_CANDIDATE = DEFAULT_CANDIDATE
else:
    MAIN_TABLE_IRON_CANDIDATE = {
        **DEFAULT_CANDIDATE,
        **dict(MAIN_TABLE_IRON_CANDIDATE),
    }

CSV_FIELDS = (
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
    token = str(value).strip()
    if token.casefold() == "pam":
        return "PAM"
    token = token.casefold()
    if token not in {"boiler", "epilepsy", "wafer", "freezer"}:
        raise ValueError(f"Unsupported dataset: {value!r}")
    return token


def candidate_name(candidate: Mapping[str, Any]) -> str:
    if "name" in candidate:
        return str(candidate["name"])
    model = candidate.get("model")
    if isinstance(model, Mapping) and "name" in model:
        return str(model["name"])
    raise ValueError(f"Candidate has no name: {candidate}")


def select_candidate(
    data: str,
    override_name: Optional[str],
) -> Tuple[int, Dict[str, Any]]:
    expected = override_name or MAIN_TABLE_IRON_CANDIDATE[data]
    candidates = [dict(candidate) for candidate in get_candidates(data)]
    matches = [
        (index, candidate)
        for index, candidate in enumerate(candidates)
        if candidate_name(candidate) == expected
    ]
    if len(matches) != 1:
        available = [candidate_name(candidate) for candidate in candidates]
        raise RuntimeError(
            f"Expected exactly one candidate {expected!r} for {data}; "
            f"available={available}"
        )
    return matches[0]


def flow_checkpoint(
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


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def upsert_runtime_row(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = []
    if path.is_file():
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames and tuple(reader.fieldnames) != CSV_FIELDS:
                raise RuntimeError(
                    f"Unexpected runtime CSV columns in {path}: {reader.fieldnames}"
                )
            for current in reader:
                same_key = (
                    current.get("data") == str(row["data"])
                    and current.get("fold") == str(row["fold"])
                    and current.get("seed") == str(row["seed"])
                    and current.get("explainer") == str(row["explainer"])
                )
                if not same_key:
                    existing.append(current)

    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(existing)
        writer.writerow({key: row[key] for key in CSV_FIELDS})
    os.replace(temporary, path)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        required=True,
        choices=["PAM", "pam", "boiler", "epilepsy", "wafer", "freezer"],
    )
    parser.add_argument("--fold", required=True, type=int, choices=range(5))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--n-rep", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--n-steps", type=int, default=100)
    parser.add_argument(
        "--path-batch",
        type=int,
        default=0,
        help="0 uses get_dataset_config(data)['compute']['path_batch'].",
    )
    parser.add_argument("--candidate", default=None)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--checkpoint-root", default="model")
    parser.add_argument("--ffjord-root", default="third_party/ffjord_official")
    parser.add_argument("--output-file", default="runtime_all.csv")
    args = parser.parse_args(argv)

    if args.n_rep < 1:
        parser.error("--n-rep must be >= 1")
    if args.warmup < 0:
        parser.error("--warmup must be >= 0")
    if args.n_steps < 1:
        parser.error("--n-steps must be >= 1")
    if args.path_batch < 0:
        parser.error("--path-batch must be >= 0")
    return args


def run(args: argparse.Namespace) -> Dict[str, Any]:
    data = canonical_data(args.data)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA requested but unavailable: {device}")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    _train, _val, test = load_splits(
        data,
        args.fold,
        args.seed,
        args.data_root,
    )
    n_total = min(int(args.warmup + args.n_rep), int(test["x"].shape[0]))
    if n_total <= args.warmup:
        raise RuntimeError(
            f"Test split has {test['x'].shape[0]} samples, but warmup={args.warmup}"
        )

    # Match main_runtime.py: select the first samples in order and place all
    # timed tensors on GPU before starting the stopwatch.
    x_eval = test["x"][:n_total].float().contiguous().to(device)
    mask_eval = test["mask"][:n_total].float().contiguous().to(device)
    timesteps = (
        torch.linspace(
            0.0,
            1.0,
            x_eval.shape[1],
            device=device,
            dtype=x_eval.dtype,
        )
        .unsqueeze(0)
    )

    classifier = build_classifier(data, args.fold, args.seed, device)
    classifier.eval()
    for parameter in classifier.parameters():
        parameter.requires_grad_(False)

    candidate_index, candidate = select_candidate(data, args.candidate)
    candidate_label = candidate_name(candidate)
    checkpoint = flow_checkpoint(
        Path(args.checkpoint_root),
        data,
        args.fold,
        args.seed,
        candidate_label,
    )
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    flow, metadata = load_checkpoint(
        checkpoint,
        ffjord_root=args.ffjord_root,
        device=device,
        expected_input_shape=x_eval.shape[1:],
    )
    model_candidate, _training = split_candidate(candidate)
    if dict(flow.candidate) != model_candidate:
        raise RuntimeError(
            f"Checkpoint candidate mismatch:\n"
            f"config={model_candidate}\ncheckpoint={flow.candidate}"
        )
    flow.eval()
    for parameter in flow.parameters():
        parameter.requires_grad_(False)

    compute = dict(get_dataset_config(data)["compute"])
    path_batch = (
        int(args.path_batch)
        if int(args.path_batch) > 0
        else int(compute["path_batch"])
    )
    if path_batch < 1:
        raise ValueError("Resolved path_batch must be >= 1")

    explainer = NFIntegratedGradients(classifier.predict, flow)
    elapsed_seconds = []

    old_cudnn = torch.backends.cudnn.enabled
    if device.type == "cuda":
        # Same RNN-backward setting as main_runtime.py.
        torch.backends.cudnn.enabled = False

    try:
        for sample_index in range(n_total):
            x = x_eval[sample_index : sample_index + 1]
            mask = mask_eval[sample_index : sample_index + 1]

            synchronize(device)
            started = time.perf_counter()

            # Match the baseline timer boundary: target selection is included.
            with torch.no_grad():
                outputs = _run_forward(
                    classifier,
                    x,
                    additional_forward_args=(mask, timesteps, False),
                )
                targets = outputs.argmax(dim=-1)

            attribution = explainer.attribute(
                x,
                targets=targets,
                baseline=torch.zeros_like(x),
                additional_forward_args=(mask, timesteps, False),
                n_steps=int(args.n_steps),
                path_batch_size=path_batch,
                return_diagnostics=False,
            ).abs()

            synchronize(device)
            elapsed = time.perf_counter() - started

            if sample_index >= args.warmup:
                elapsed_seconds.append(elapsed)

            del attribution, outputs, targets
    finally:
        if device.type == "cuda":
            torch.backends.cudnn.enabled = old_cudnn

    measured_ms = [value * 1000.0 for value in elapsed_seconds]
    if not measured_ms:
        raise RuntimeError("No timed samples were collected")

    row = {
        "data": data,
        "fold": int(args.fold),
        "seed": int(args.seed),
        "explainer": "iron",
        "n_rep": len(measured_ms),
        "warmup": int(args.warmup),
        "batch_size": 1,
        "mean_ms": f"{statistics.mean(measured_ms):.4f}",
        "std_ms": f"{statistics.stdev(measured_ms) if len(measured_ms) > 1 else 0.0:.4f}",
        "median_ms": f"{statistics.median(measured_ms):.4f}",
    }
    output_path = Path(args.output_file)
    upsert_runtime_row(output_path, row)

    print(
        f"[IRON-RUNTIME] data={data} fold={args.fold} "
        f"candidate={candidate_label} candidate_index={candidate_index} "
        f"K={args.n_steps} path_batch={path_batch} n={len(measured_ms)}"
    )
    print(
        f"[RESULT] mean={row['mean_ms']} ms/sample "
        f"std={row['std_ms']} median={row['median_ms']}"
    )
    print(f"[SAVE] {output_path}")

    return {
        **row,
        "candidate": candidate_label,
        "candidate_index": candidate_index,
        "checkpoint": str(checkpoint),
        "path_batch": path_batch,
        "training_best_epoch": metadata.get("best_epoch"),
        "training_best_iteration": metadata.get("best_iteration"),
    }


if __name__ == "__main__":
    run(parse_args())
