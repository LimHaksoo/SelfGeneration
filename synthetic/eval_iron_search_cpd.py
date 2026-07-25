#!/usr/bin/env python3
"""Compute TIMING-protocol CPD@10% for one trained synthetic IRON candidate.

This script intentionally computes only CPD.  It does not recompute AUP, AUR,
Completeness, likelihood, or Gaussianity.  The attribution map follows the
same synthetic TIMING protocol used by the search evaluator: predicted target
for every prefix, zero attribution baseline, absolute prefix attribution, and
prefix accumulation into a full [N,T,D] saliency map.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from attribution.official_ffjord import load_checkpoint
from configs.official_cnf_fullfold_by_dataset import split_candidate
from configs.synthetic_iron_search_space import (
    CHECKPOINT_ROOT,
    FFJORD_ROOT,
    SCREEN_PROTOCOL,
    candidate_summary,
    canonical_dataset,
    filesystem_name,
    get_candidate,
    get_compute,
)
from synthetic.iron_whitebox_common import load_bundle, load_classifier
from synthetic.search_iron_candidate import (
    deterministic_indices,
    timing_style_iron_attribution,
)
from synthetic.switchstate.cumulative_difference import cumulative_difference


FORMAT_VERSION = 1
DEFAULT_OUTPUT_ROOT = "results_synthetic_iron_search_cpd"
TOPK = 0.10
TOP = 0


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def project_path(value: str | os.PathLike[str]) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_npy(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        np.save(handle, value)
    os.replace(temporary, path)


def candidate_checkpoint(
    checkpoint_root: Path,
    data: str,
    fold: int,
    seed: int,
    candidate_name: str,
) -> Path:
    return (
        checkpoint_root
        / filesystem_name(data)
        / f"fold{fold}_seed{seed}"
        / f"{candidate_name}.pt"
    )


def output_paths(
    output_root: Path,
    data: str,
    fold: int,
    seed: int,
    candidate_name: str,
    n_steps: int,
    n_samples: int,
) -> tuple[Path, Path, Path]:
    directory = (
        output_root
        / filesystem_name(data)
        / f"fold{fold}_seed{seed}"
    )
    result = directory / f"{candidate_name}.json"
    attribution = directory / (
        f"{candidate_name}_timing_saliency_K{n_steps}_N{n_samples}.npy"
    )
    attribution_metadata = attribution.with_suffix(".json")
    return result, attribution, attribution_metadata


def compute_saliency(
    *,
    test_x: Tensor,
    classifier: torch.nn.Module,
    flow: torch.nn.Module,
    device: torch.device,
    n_steps: int,
    attribution_batch: int,
    path_batch: int,
) -> Tensor:
    loader = DataLoader(
        TensorDataset(test_x),
        batch_size=min(attribution_batch, len(test_x)),
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    parts: List[Tensor] = []
    previous_cudnn = torch.backends.cudnn.enabled
    if device.type == "cuda":
        torch.backends.cudnn.enabled = False

    try:
        for (x_cpu,) in tqdm(loader, desc="IRON CPD attribution"):
            x = x_cpu.to(
                device=device,
                dtype=torch.float32,
                non_blocking=True,
            )
            accumulated, _signed, _output_delta = timing_style_iron_attribution(
                inputs=x,
                classifier=classifier,
                flow=flow,
                n_steps=n_steps,
                path_batch=path_batch,
            )
            parts.append(accumulated.detach().cpu())
            del accumulated, _signed, _output_delta
    finally:
        if device.type == "cuda":
            torch.backends.cudnn.enabled = previous_cudnn

    saliency = torch.cat(parts, dim=0)
    if tuple(saliency.shape) != tuple(test_x.shape):
        raise RuntimeError(
            f"Attribution shape={tuple(saliency.shape)} does not match "
            f"input shape={tuple(test_x.shape)}"
        )
    if not torch.isfinite(saliency).all():
        raise FloatingPointError("Non-finite synthetic IRON saliency")
    return saliency


def compute_cpd(
    *,
    classifier: torch.nn.Module,
    test_x: Tensor,
    saliency: Tensor,
    device: torch.device,
    metric_batch: int,
    baselines: Sequence[str],
) -> Dict[str, Dict[str, Any]]:
    """Call the repository's original synthetic cumulative_difference."""

    x = test_x.to(device=device, dtype=torch.float32)
    average = x.mean(dim=1, keepdim=True).repeat(1, x.shape[1], 1)
    actual_batch = min(metric_batch, len(test_x))

    result: Dict[str, Dict[str, Any]] = {}
    for baseline_name in baselines:
        if baseline_name == "zero":
            baseline: float | Tensor = 0.0
        elif baseline_name == "average":
            baseline = average
        else:
            raise ValueError(f"Unsupported baseline={baseline_name!r}")

        cpd, aucc, cumulative_50, curve = cumulative_difference(
            classifier,
            x,
            attributions=saliency,
            baselines=baseline,
            topk=TOPK,
            top=TOP,
            testbs=actual_batch,
            additional_forward_args=(None, None, True),
        )
        result[baseline_name] = {
            "cpd_10pct": float(cpd),
            "aucc_10pct": float(aucc),
            "cumulative_first_50": float(cumulative_50),
            "num_removed_cells": int(len(curve)),
            "curve": [float(value) for value in curve],
        }

    return result


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--fold", type=int, required=True, choices=range(5))
    parser.add_argument("--candidate-index", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")

    parser.add_argument(
        "--n-steps",
        type=int,
        default=int(SCREEN_PROTOCOL["n_steps"]),
    )
    parser.add_argument(
        "--max-eval-samples",
        type=int,
        default=int(SCREEN_PROTOCOL["max_eval_samples"]),
    )
    parser.add_argument("--attribution-batch", type=int, default=0)
    parser.add_argument("--path-batch", type=int, default=0)
    parser.add_argument("--metric-batch", type=int, default=0)
    parser.add_argument(
        "--baselines",
        nargs="+",
        choices=("zero", "average"),
        default=("zero", "average"),
    )

    parser.add_argument("--checkpoint-root", default=CHECKPOINT_ROOT)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--ffjord-root", default=FFJORD_ROOT)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--force-attribution", action="store_true")
    parser.add_argument("--no-attribution-cache", action="store_true")

    args = parser.parse_args(argv)
    if args.n_steps < 1:
        parser.error("--n-steps must be positive")
    if args.max_eval_samples < 0:
        parser.error("--max-eval-samples must be non-negative")
    return args


def run(args: argparse.Namespace) -> Dict[str, Any]:
    data = canonical_dataset(args.data)
    candidate = get_candidate(data, args.candidate_index)
    candidate_name = str(candidate["name"])
    model_candidate, _training = split_candidate(candidate)
    compute = get_compute(data)

    attribution_batch = int(args.attribution_batch) or int(
        compute["attribution_batch"]
    )
    path_batch = int(args.path_batch) or int(compute["path_batch"])
    metric_batch = int(args.metric_batch) or int(compute["metric_batch"])

    seed_all(int(args.seed) + 1009 * int(args.fold))
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    checkpoint_root = project_path(args.checkpoint_root)
    output_root = project_path(args.output_root)
    ffjord_root = project_path(args.ffjord_root)

    checkpoint = candidate_checkpoint(
        checkpoint_root,
        data,
        args.fold,
        args.seed,
        candidate_name,
    )
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"Synthetic IRON search checkpoint not found: {checkpoint}"
        )

    bundle = load_bundle(data, args.fold, args.seed)
    indices = deterministic_indices(
        len(bundle.test_x),
        int(args.max_eval_samples),
        int(args.seed) + 7919 * int(args.fold),
    )
    test_x = bundle.test_x[indices].contiguous()

    protocol = {
        "format_version": FORMAT_VERSION,
        "data": data,
        "fold": int(args.fold),
        "seed": int(args.seed),
        "candidate_index": int(args.candidate_index),
        "candidate": candidate_summary(candidate),
        "n_steps": int(args.n_steps),
        "max_eval_samples": int(args.max_eval_samples),
        "actual_eval_samples": int(len(test_x)),
        "eval_indices": indices.tolist(),
        "attribution_batch": int(attribution_batch),
        "path_batch": int(path_batch),
        "metric_batch": int(metric_batch),
        "baselines": list(args.baselines),
        "target": "predicted class for each temporal prefix",
        "attribution": "absolute prefix accumulation",
        "topk": TOPK,
        "top": TOP,
    }

    result_path, attribution_path, attribution_metadata_path = output_paths(
        output_root,
        data,
        args.fold,
        args.seed,
        candidate_name,
        args.n_steps,
        len(test_x),
    )

    if result_path.is_file() and not args.force:
        previous = json.loads(result_path.read_text(encoding="utf-8"))
        if (
            previous.get("status") == "complete"
            and previous.get("protocol") == protocol
        ):
            print(f"[SKIP] current result: {result_path}", flush=True)
            return previous

    classifier = load_classifier(data, args.fold, args.seed, device)
    flow, metadata = load_checkpoint(
        checkpoint,
        ffjord_root=ffjord_root,
        device=device,
        expected_input_shape=test_x.shape[1:],
    )
    if dict(flow.candidate) != model_candidate:
        raise RuntimeError(
            f"Candidate mismatch in {checkpoint}\n"
            f"expected={model_candidate}\nactual={flow.candidate}"
        )
    flow.eval()
    for parameter in flow.parameters():
        parameter.requires_grad_(False)

    cache_protocol = {
        key: protocol[key]
        for key in (
            "data",
            "fold",
            "seed",
            "candidate_index",
            "candidate",
            "n_steps",
            "actual_eval_samples",
            "eval_indices",
            "path_batch",
            "target",
            "attribution",
        )
    }

    use_cache = False
    if (
        not args.no_attribution_cache
        and not args.force_attribution
        and attribution_path.is_file()
        and attribution_metadata_path.is_file()
    ):
        cache_metadata = json.loads(
            attribution_metadata_path.read_text(encoding="utf-8")
        )
        use_cache = cache_metadata.get("protocol") == cache_protocol

    if use_cache:
        saliency = torch.from_numpy(np.load(attribution_path)).float()
        if tuple(saliency.shape) != tuple(test_x.shape):
            raise RuntimeError("Cached attribution shape mismatch")
        print(f"[CACHE] loaded {attribution_path}", flush=True)
    else:
        saliency = compute_saliency(
            test_x=test_x,
            classifier=classifier,
            flow=flow,
            device=device,
            n_steps=args.n_steps,
            attribution_batch=attribution_batch,
            path_batch=path_batch,
        )
        if not args.no_attribution_cache:
            atomic_npy(attribution_path, saliency.numpy())
            atomic_json(
                attribution_metadata_path,
                {
                    "status": "complete",
                    "protocol": cache_protocol,
                    "attribution_path": str(attribution_path),
                },
            )
            print(f"[CACHE] saved {attribution_path}", flush=True)

    metrics = compute_cpd(
        classifier=classifier,
        test_x=test_x,
        saliency=saliency,
        device=device,
        metric_batch=metric_batch,
        baselines=args.baselines,
    )

    result = {
        "status": "complete",
        "protocol": protocol,
        "data": data,
        "fold": int(args.fold),
        "seed": int(args.seed),
        "candidate_index": int(args.candidate_index),
        "candidate": candidate_summary(candidate),
        "checkpoint": str(checkpoint),
        "flow_best_epoch": metadata.get("best_epoch"),
        "attribution_cache": (
            None if args.no_attribution_cache else str(attribution_path)
        ),
        "metrics": metrics,
    }
    atomic_json(result_path, result)

    values = " ".join(
        f"CPD-{name}={block['cpd_10pct']:.6f}"
        for name, block in metrics.items()
    )
    print(
        f"[RESULT] data={data} fold={args.fold} "
        f"candidate={candidate_name} {values}",
        flush=True,
    )
    print(f"[SAVE] {result_path}", flush=True)

    del saliency, classifier, flow
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


if __name__ == "__main__":
    run(parse_args())
