#!/usr/bin/env python3
"""Evaluate numerical encode--decode reconstruction of trained IRON CNFs.

This script measures the raw numerical round trip

    x -> h_eta(x) -> g_eta(h_eta(x)) = x_hat

on an existing held-out split.  No endpoint clamping is applied.  The reported
quantities therefore measure solver / implementation round-trip error of the
trained invertible model, not density-fit or Gaussianization quality.  Use the
existing NLL, two-sample AUC, and sliced-W1 results alongside these values when
making a Gaussianization claim.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

BENCHMARK_DATASETS = ("PAM", "boiler", "epilepsy", "wafer", "freezer")
SYNTHETIC_DATASETS = ("state", "switch-feature")
EPS = 1e-12


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def canonical_benchmark(value: str) -> str:
    key = str(value).strip().casefold()
    mapping = {
        "pam": "PAM",
        "pamap2": "PAM",
        "boiler": "boiler",
        "epilepsy": "epilepsy",
        "wafer": "wafer",
        "freezer": "freezer",
        "freezerregulartrain": "freezer",
        "freezersmalltrain": "freezer",
    }
    if key not in mapping:
        raise ValueError(f"Unsupported benchmark dataset: {value!r}")
    return mapping[key]


def canonical_synthetic(value: str) -> str:
    key = str(value).strip().casefold().replace("_", "-")
    mapping = {
        "state": "state",
        "hmm": "state",
        "switch": "switch-feature",
        "switch-feature": "switch-feature",
        "switchfeature": "switch-feature",
        "switch-state": "switch-feature",
        "switchstate": "switch-feature",
    }
    if key not in mapping:
        raise ValueError(f"Unsupported synthetic dataset: {value!r}")
    return mapping[key]


def filesystem_name(value: str) -> str:
    return str(value).replace("-", "_")


def resolve_project_path(value: str | os.PathLike[str]) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def extract_x(value: Any) -> Tensor:
    if isinstance(value, Tensor):
        return value
    if isinstance(value, Mapping) and "x" in value:
        result = value["x"]
        if isinstance(result, Tensor):
            return result
    raise TypeError(
        "Expected a Tensor or mapping containing Tensor key 'x', "
        f"got {type(value)!r}"
    )


def load_benchmark_inputs(
    data: str,
    fold: int,
    seed: int,
    split: str,
    data_root: Optional[str],
) -> Tensor:
    from real.cnf_dataset_registry import load_splits

    train, val, test = load_splits(data, fold, seed, data_root)
    selected = {"train": train, "val": val, "test": test}[split]
    return extract_x(selected).detach().cpu().float().contiguous()


def load_synthetic_inputs(
    data: str,
    fold: int,
    seed: int,
    split: str,
) -> Tensor:
    from synthetic.iron_whitebox_common import (
        load_bundle,
        split_train_validation,
    )

    bundle = load_bundle(data, fold, seed)
    if split == "train":
        result = bundle.train_x
    elif split == "test":
        result = bundle.test_x
    elif split == "val":
        _train_x, result = split_train_validation(
            bundle.train_x,
            bundle.train_y,
            fold=fold,
            seed=seed,
        )
    else:  # argparse prevents this.
        raise AssertionError(split)
    return result.detach().cpu().float().contiguous()


def benchmark_candidate(data: str, candidate_name: Optional[str]) -> Dict[str, Any]:
    from configs.official_cnf_fullfold_by_dataset import get_dataset_config

    candidates = list(get_dataset_config(data)["candidates"])
    if candidate_name is None:
        if len(candidates) != 1:
            names = [str(item["name"]) for item in candidates]
            raise ValueError(
                f"Dataset {data} has {len(candidates)} configured candidates; "
                f"pass --candidate-name from {names}"
            )
        return dict(candidates[0])

    for candidate in candidates:
        if str(candidate["name"]) == candidate_name:
            return dict(candidate)
    names = [str(item["name"]) for item in candidates]
    raise ValueError(f"Unknown candidate {candidate_name!r} for {data}; choices={names}")


def synthetic_candidate(
    data: str,
    source: str,
    candidate_name: Optional[str],
) -> Tuple[Dict[str, Any], Dict[str, int]]:
    if source == "final":
        from configs.synthetic_iron_whitebox import get_candidate, get_compute

        candidate = dict(get_candidate(data))
        if candidate_name is not None and str(candidate["name"]) != candidate_name:
            raise ValueError(
                f"Final synthetic candidate for {data} is {candidate['name']!r}, "
                f"not {candidate_name!r}"
            )
        return candidate, dict(get_compute(data))

    from configs.synthetic_iron_search_space import get_candidates, get_compute

    candidates = list(get_candidates(data))
    if candidate_name is None:
        if len(candidates) != 1:
            names = [str(item["name"]) for item in candidates]
            raise ValueError(
                f"Synthetic search source has {len(candidates)} candidates; "
                f"pass --candidate-name from {names}"
            )
        candidate = dict(candidates[0])
    else:
        matches = [
            dict(item) for item in candidates
            if str(item["name"]) == candidate_name
        ]
        if not matches:
            names = [str(item["name"]) for item in candidates]
            raise ValueError(
                f"Unknown synthetic candidate {candidate_name!r}; choices={names}"
            )
        candidate = matches[0]
    return candidate, dict(get_compute(data))


def resolve_job(args: argparse.Namespace) -> Tuple[str, Dict[str, Any], Tensor, int, Path]:
    if args.suite == "benchmark":
        data = canonical_benchmark(args.data)
        candidate = benchmark_candidate(data, args.candidate_name)
        inputs = load_benchmark_inputs(
            data=data,
            fold=args.fold,
            seed=args.seed,
            split=args.split,
            data_root=args.data_root,
        )
        from configs.official_cnf_fullfold_by_dataset import get_dataset_config

        compute = dict(get_dataset_config(data)["compute"])
        default_batch = int(compute["nll_eval_batch"])
        root = (
            resolve_project_path(args.checkpoint_root)
            if args.checkpoint_root
            else PROJECT_ROOT / "model"
        )
        checkpoint = (
            root
            / data
            / "official_ffjord_epoch"
            / f"fold{args.fold}_seed{args.seed}"
            / f"{candidate['name']}.pt"
        )
    else:
        data = canonical_synthetic(args.data)
        candidate, compute = synthetic_candidate(
            data=data,
            source=args.synthetic_source,
            candidate_name=args.candidate_name,
        )
        inputs = load_synthetic_inputs(
            data=data,
            fold=args.fold,
            seed=args.seed,
            split=args.split,
        )
        default_batch = int(compute.get("nll_eval_batch", 256))
        if args.checkpoint_root:
            root = resolve_project_path(args.checkpoint_root)
        elif args.synthetic_source == "final":
            root = PROJECT_ROOT / "model" / "synthetic_iron"
        else:
            root = PROJECT_ROOT / "model" / "synthetic_iron_search"
        checkpoint = (
            root
            / filesystem_name(data)
            / f"fold{args.fold}_seed{args.seed}"
            / f"{candidate['name']}.pt"
        )

    if args.checkpoint:
        checkpoint = resolve_project_path(args.checkpoint)
    batch_size = int(args.batch_size) if args.batch_size > 0 else default_batch
    return data, candidate, inputs, batch_size, checkpoint


class RoundTripAccumulator:
    def __init__(self) -> None:
        self.n_samples = 0
        self.n_elements = 0
        self.total_abs = 0.0
        self.total_sq = 0.0
        self.global_max_abs = 0.0
        self.sample_mae: list[float] = []
        self.sample_rmse: list[float] = []
        self.sample_max_abs: list[float] = []
        self.sample_relative_l2: list[float] = []

    def update(self, inputs: Tensor, reconstructed: Tensor) -> None:
        if inputs.shape != reconstructed.shape:
            raise RuntimeError(
                f"Round-trip shape mismatch: x={tuple(inputs.shape)}, "
                f"x_hat={tuple(reconstructed.shape)}"
            )
        if not torch.isfinite(reconstructed).all():
            raise FloatingPointError("Non-finite values in CNF reconstruction")

        error = (reconstructed - inputs).detach().reshape(inputs.shape[0], -1).double()
        flat_inputs = inputs.detach().reshape(inputs.shape[0], -1).double()
        abs_error = error.abs()
        sq_error = error.square()

        self.n_samples += int(error.shape[0])
        self.n_elements += int(error.numel())
        self.total_abs += float(abs_error.sum().item())
        self.total_sq += float(sq_error.sum().item())
        self.global_max_abs = max(
            self.global_max_abs,
            float(abs_error.max().item()),
        )

        sample_mae = abs_error.mean(dim=1)
        sample_rmse = sq_error.mean(dim=1).sqrt()
        sample_max = abs_error.amax(dim=1)
        relative_l2 = error.norm(p=2, dim=1) / flat_inputs.norm(
            p=2, dim=1
        ).clamp_min(EPS)

        self.sample_mae.extend(sample_mae.cpu().tolist())
        self.sample_rmse.extend(sample_rmse.cpu().tolist())
        self.sample_max_abs.extend(sample_max.cpu().tolist())
        self.sample_relative_l2.extend(relative_l2.cpu().tolist())

    def summary(self) -> Dict[str, float | int]:
        if self.n_samples < 1 or self.n_elements < 1:
            raise RuntimeError("No reconstruction samples were accumulated")

        sample_mae = np.asarray(self.sample_mae, dtype=np.float64)
        sample_rmse = np.asarray(self.sample_rmse, dtype=np.float64)
        sample_max = np.asarray(self.sample_max_abs, dtype=np.float64)
        relative_l2 = np.asarray(self.sample_relative_l2, dtype=np.float64)

        return {
            "n_samples": int(self.n_samples),
            "n_elements": int(self.n_elements),
            "mae_per_element": float(self.total_abs / self.n_elements),
            "rmse_per_element": float(math.sqrt(self.total_sq / self.n_elements)),
            "mean_sample_mae": float(sample_mae.mean()),
            "median_sample_mae": float(np.median(sample_mae)),
            "mean_sample_rmse": float(sample_rmse.mean()),
            "median_sample_rmse": float(np.median(sample_rmse)),
            "mean_sample_max_abs": float(sample_max.mean()),
            "median_sample_max_abs": float(np.median(sample_max)),
            "p95_sample_max_abs": float(np.quantile(sample_max, 0.95)),
            "global_max_abs": float(self.global_max_abs),
            "mean_relative_l2": float(relative_l2.mean()),
            "median_relative_l2": float(np.median(relative_l2)),
            "p95_relative_l2": float(np.quantile(relative_l2, 0.95)),
        }


def architecture_summary(model: torch.nn.Module) -> Dict[str, Any]:
    candidate = dict(getattr(model, "candidate", {}))
    dims = str(candidate.get("dims", ""))
    widths = [int(value) for value in dims.split("-") if value.isdigit()]
    keys = (
        "dims",
        "num_blocks",
        "solver",
        "step_size",
        "atol",
        "rtol",
        "test_solver",
        "test_atol",
        "test_rtol",
        "layer_type",
        "nonlinearity",
        "divergence_fn",
        "rademacher",
        "residual",
        "batch_norm",
    )
    return {
        **{key: candidate.get(key) for key in keys},
        "hidden_width_max": max(widths) if widths else None,
        "hidden_layers": len(widths),
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
    }


def output_path(
    output_root: Path,
    suite: str,
    source: str,
    data: str,
    fold: int,
    seed: int,
    candidate_name: str,
) -> Path:
    suite_name = suite if suite == "benchmark" else f"synthetic_{source}"
    return (
        output_root
        / suite_name
        / filesystem_name(data)
        / f"fold{fold}_seed{seed}"
        / f"{candidate_name}.json"
    )


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def evaluate(args: argparse.Namespace) -> Optional[Dict[str, Any]]:
    from attribution.official_ffjord import load_checkpoint

    seed_all(args.seed)
    data, configured_candidate, inputs, batch_size, checkpoint = resolve_job(args)
    candidate_name = str(configured_candidate["name"])

    if not checkpoint.is_file():
        message = f"CNF checkpoint not found: {checkpoint}"
        if args.skip_missing:
            print(f"[SKIP] {message}", flush=True)
            return None
        raise FileNotFoundError(message)

    if args.max_samples > 0:
        inputs = inputs[: args.max_samples].contiguous()
    if len(inputs) < 1:
        raise RuntimeError("Selected reconstruction split is empty")

    batch_size = min(int(batch_size), int(len(inputs)))
    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    checkpoint_stat = checkpoint.stat()
    protocol = {
        "format_version": 1,
        "suite": args.suite,
        "synthetic_source": args.synthetic_source if args.suite == "synthetic" else None,
        "data": data,
        "fold": int(args.fold),
        "seed": int(args.seed),
        "split": args.split,
        "max_samples": int(args.max_samples),
        "actual_samples": int(len(inputs)),
        "batch_size": int(batch_size),
        "candidate_name": candidate_name,
        "checkpoint": str(checkpoint),
        "checkpoint_size_bytes": int(checkpoint_stat.st_size),
        "checkpoint_mtime_ns": int(checkpoint_stat.st_mtime_ns),
        "measurement": "raw float32 decode(encode(x)) without endpoint clamping",
    }

    result_file = output_path(
        output_root=resolve_project_path(args.output_root),
        suite=args.suite,
        source=args.synthetic_source,
        data=data,
        fold=args.fold,
        seed=args.seed,
        candidate_name=candidate_name,
    )
    if result_file.is_file() and not args.force:
        previous = json.loads(result_file.read_text(encoding="utf-8"))
        if previous.get("status") == "complete" and previous.get("protocol") == protocol:
            print(f"[SKIP] current result: {result_file}", flush=True)
            return previous

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    model, metadata = load_checkpoint(
        checkpoint,
        ffjord_root=resolve_project_path(args.ffjord_root),
        device=device,
        expected_input_shape=inputs.shape[1:],
    )
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    loader = DataLoader(
        TensorDataset(inputs),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    accumulator = RoundTripAccumulator()

    with torch.no_grad():
        for (batch_cpu,) in loader:
            batch = batch_cpu.to(
                device=device,
                dtype=torch.float32,
                non_blocking=True,
            )
            latent = model.encode(batch)
            reconstructed = model.decode(latent)
            accumulator.update(batch, reconstructed)
            del batch, latent, reconstructed

        zero = torch.zeros(
            1,
            *inputs.shape[1:],
            device=device,
            dtype=torch.float32,
        )
        zero_reconstructed = model.decode(model.encode(zero))
        zero_error = (zero_reconstructed - zero).reshape(-1).abs().double()
        zero_metrics = {
            "mae_per_element": float(zero_error.mean().item()),
            "rmse_per_element": float(zero_error.square().mean().sqrt().item()),
            "max_abs": float(zero_error.max().item()),
        }

    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak_memory_mib = float(
            torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        )
    else:
        peak_memory_mib = 0.0
    elapsed_seconds = float(time.perf_counter() - started)

    metrics = accumulator.summary()
    result = {
        "status": "complete",
        "protocol": protocol,
        "suite": args.suite,
        "synthetic_source": args.synthetic_source if args.suite == "synthetic" else None,
        "data": data,
        "fold": int(args.fold),
        "seed": int(args.seed),
        "split": args.split,
        "input_shape": [int(value) for value in inputs.shape[1:]],
        "ambient_dimension": int(np.prod(inputs.shape[1:])),
        "candidate": architecture_summary(model),
        "configured_candidate": dict(configured_candidate),
        "checkpoint": str(checkpoint),
        "training": {
            "best_epoch": metadata.get("best_epoch"),
            "last_epoch": metadata.get("last_epoch"),
            "best_val_nll_per_dim": metadata.get("best_val_nll_per_dim"),
            "training_seconds": metadata.get("training_seconds"),
        },
        "reconstruction": metrics,
        "zero_baseline_roundtrip": zero_metrics,
        "runtime": {
            "evaluation_seconds": elapsed_seconds,
            "ms_per_sample": float(1000.0 * elapsed_seconds / len(inputs)),
            "peak_memory_mib": peak_memory_mib,
        },
        "interpretation": (
            "These are numerical CNF round-trip errors. They establish stable "
            "invertibility of the shallow flow, but do not alone establish "
            "Gaussianization or density-model fidelity."
        ),
    }
    atomic_write_json(result_file, result)

    print(
        f"[RESULT] suite={args.suite} data={data} fold={args.fold} "
        f"candidate={candidate_name} d={result['ambient_dimension']} "
        f"MAE={metrics['mae_per_element']:.6e} "
        f"RMSE={metrics['rmse_per_element']:.6e} "
        f"mean-max={metrics['mean_sample_max_abs']:.6e} "
        f"global-max={metrics['global_max_abs']:.6e}",
        flush=True,
    )
    print(f"[SAVE] {result_file}", flush=True)

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("benchmark", "synthetic"), required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--fold", type=int, required=True, choices=range(5))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--synthetic-source", choices=("final", "search"), default="final")
    parser.add_argument("--candidate-name", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--checkpoint-root", default=None)
    parser.add_argument("--ffjord-root", default="third_party/ffjord_official")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--output-root", default="results_cnf_reconstruction")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--skip-missing", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    if args.batch_size < 0:
        parser.error("--batch-size must be non-negative")
    if args.max_samples < 0:
        parser.error("--max-samples must be non-negative")
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    evaluate(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
