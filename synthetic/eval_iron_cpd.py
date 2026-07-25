#!/usr/bin/env python3
"""Compute only CPD@10% for synthetic IRON attributions.

This follows the original synthetic TIMING evaluation path:
1. use the absolute prefix-accumulated saliency map;
2. rank cells by descending attribution magnitude;
3. mask the top 10% one cell at a time; and
4. call synthetic.switchstate.cumulative_difference unchanged.

AUP, AUR, and Completeness are not recomputed.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from attribution.official_ffjord import load_checkpoint
from configs.official_cnf_fullfold_by_dataset import split_candidate
from configs.synthetic_iron_whitebox import (
    N_STEPS,
    RESULT_ROOT,
    canonical_dataset,
    filesystem_name,
    get_candidate,
    get_compute,
)
from synthetic.iron_whitebox_common import (
    ffjord_root,
    flow_checkpoint,
    load_bundle,
    load_classifier,
)

CPD_TOPK = 0.10
CPD_TOP = 0
DEFAULT_OUTPUT_ROOT = "results_synthetic_iron_cpd"


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def fold_dir(root: Path, data: str, fold: int, seed: int) -> Path:
    return root / filesystem_name(data) / f"fold{fold}_seed{seed}"


def output_path(root: Path, data: str, fold: int, seed: int) -> Path:
    return fold_dir(root, data, fold, seed) / "iron_cpd.json"


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def atomic_npy(path: Path, tensor: Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        np.save(handle, tensor.detach().cpu().numpy())
    os.replace(temporary, path)


def cached_saliency_path(cache_root: Path, data: str, fold: int, seed: int) -> Optional[Path]:
    directory = fold_dir(cache_root, data, fold, seed)
    result_json = directory / "iron.json"
    if result_json.is_file():
        payload = json.loads(result_json.read_text(encoding="utf-8"))
        value = payload.get("attribution_path")
        if value:
            candidate = project_path(str(value))
            if candidate.is_file():
                return candidate

    candidate = directory / "iron_attributions.npy"
    return candidate if candidate.is_file() else None


def extract_saliency(value: Any) -> Tensor:
    # v1 returns saliency; v2+ returns (saliency, signed_attr, output_delta).
    value = value[0] if isinstance(value, tuple) else value
    if not isinstance(value, Tensor):
        raise TypeError(f"Unexpected attribution output: {type(value)}")
    return value


def recompute_saliency(
    *,
    data: str,
    fold: int,
    seed: int,
    test_x: Tensor,
    classifier,
    device: torch.device,
    attribution_batch: int,
    path_batch: int,
    n_steps: int,
    destination: Path,
) -> Tensor:
    from synthetic.eval_iron_whitebox import timing_style_iron_attribution

    full_candidate = get_candidate(data)
    model_candidate, _ = split_candidate(full_candidate)
    checkpoint = flow_checkpoint(data, fold, seed)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Synthetic IRON flow not found: {checkpoint}")

    flow, _ = load_checkpoint(
        checkpoint,
        ffjord_root=ffjord_root(),
        device=device,
        expected_input_shape=test_x.shape[1:],
    )
    if dict(flow.candidate) != model_candidate:
        raise RuntimeError(f"Flow candidate mismatch: {checkpoint}")
    flow.eval()
    for parameter in flow.parameters():
        parameter.requires_grad_(False)

    loader = DataLoader(
        TensorDataset(test_x),
        batch_size=attribution_batch,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    parts: List[Tensor] = []
    old_cudnn = torch.backends.cudnn.enabled
    if device.type == "cuda":
        torch.backends.cudnn.enabled = False
    try:
        for (x_cpu,) in tqdm(loader, desc=f"IRON saliency {data}/f{fold}"):
            x = x_cpu.to(device=device, dtype=torch.float32, non_blocking=True)
            result = timing_style_iron_attribution(
                inputs=x,
                classifier=classifier,
                flow=flow,
                n_steps=n_steps,
                path_batch=path_batch,
            )
            parts.append(extract_saliency(result).detach().cpu().float())
            del result, x
    finally:
        if device.type == "cuda":
            torch.backends.cudnn.enabled = old_cudnn

    saliency = torch.cat(parts, dim=0)
    if tuple(saliency.shape) != tuple(test_x.shape):
        raise RuntimeError(
            f"Saliency shape={tuple(saliency.shape)} != test shape={tuple(test_x.shape)}"
        )
    if not torch.isfinite(saliency).all():
        raise FloatingPointError("Non-finite IRON saliency")

    atomic_npy(destination, saliency)
    print(f"[CACHE] saved {destination}", flush=True)
    del flow
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return saliency


def load_saliency(
    *,
    data: str,
    fold: int,
    seed: int,
    test_x: Tensor,
    classifier,
    device: torch.device,
    cache_root: Path,
    attribution_batch: int,
    path_batch: int,
    n_steps: int,
    recompute_if_missing: bool,
) -> Tuple[Tensor, Path, str]:
    cached = cached_saliency_path(cache_root, data, fold, seed)
    if cached is not None:
        saliency = torch.from_numpy(np.load(cached)).float()
        if tuple(saliency.shape) != tuple(test_x.shape):
            raise RuntimeError(f"Cached saliency shape mismatch: {cached}")
        print(f"[CACHE] loaded {cached}", flush=True)
        return saliency, cached, "loaded"

    destination = fold_dir(cache_root, data, fold, seed) / "iron_attributions.npy"
    if not recompute_if_missing:
        raise FileNotFoundError(
            f"IRON saliency cache not found: {destination}. "
            "Pass --recompute-if-missing or generate it with --save-attributions."
        )

    saliency = recompute_saliency(
        data=data,
        fold=fold,
        seed=seed,
        test_x=test_x,
        classifier=classifier,
        device=device,
        attribution_batch=attribution_batch,
        path_batch=path_batch,
        n_steps=n_steps,
        destination=destination,
    )
    return saliency, destination, "recomputed"


def as_curve(value: Any) -> List[float]:
    if value is None:
        return []
    if isinstance(value, Tensor):
        value = value.detach().cpu().numpy()
    return [float(v) for v in np.asarray(value, dtype=np.float64).reshape(-1)]


def compute_cpd(
    *,
    classifier,
    x_test: Tensor,
    saliency: Tensor,
    baseline_name: str,
    testbs: int,
) -> Dict[str, Any]:
    from synthetic.switchstate.cumulative_difference import cumulative_difference

    if baseline_name == "average":
        baseline = x_test.mean(1, keepdim=True).repeat(1, x_test.shape[1], 1)
        display_name = "Average"
    elif baseline_name == "zero":
        baseline = 0.0
        display_name = "Zeros"
    else:
        raise ValueError(baseline_name)

    cpd, _aucc, _cum50, curve = cumulative_difference(
        classifier,
        x_test,
        attributions=saliency.abs().cpu(),
        baselines=baseline,
        topk=CPD_TOPK,
        top=CPD_TOP,
        testbs=testbs,
        additional_forward_args=(None, None, True),
    )
    return {"baseline": display_name, "cpd": float(cpd), "curve": as_curve(curve)}


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--fold", type=int, required=True, choices=range(5))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--n-steps", type=int, default=N_STEPS)
    parser.add_argument("--attribution-batch", type=int, default=0)
    parser.add_argument("--path-batch", type=int, default=0)
    parser.add_argument("--testbs", type=int, default=0)
    parser.add_argument(
        "--baselines",
        nargs="+",
        default=["average", "zero"],
        choices=["average", "zero"],
    )
    parser.add_argument("--cache-root", default=RESULT_ROOT)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--recompute-if-missing", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    if args.n_steps < 1:
        parser.error("--n-steps must be positive")
    return args


def run(args: argparse.Namespace) -> Dict[str, Any]:
    data = canonical_dataset(args.data)
    seed_all(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    compute = get_compute(data)
    attribution_batch = (
        args.attribution_batch
        if args.attribution_batch > 0
        else int(compute["attribution_batch"])
    )
    path_batch = args.path_batch if args.path_batch > 0 else int(compute["path_batch"])

    bundle = load_bundle(data, args.fold, args.seed)
    test_x_cpu = bundle.test_x.detach().cpu().float().contiguous()
    classifier = load_classifier(data, args.fold, args.seed, device)

    cache_root = project_path(args.cache_root)
    result_root = project_path(args.output_root)
    destination = output_path(result_root, data, args.fold, args.seed)

    protocol = {
        "dataset": data,
        "fold": args.fold,
        "seed": args.seed,
        "method": "IRON",
        "metric": "CPD",
        "topk": CPD_TOPK,
        "top": CPD_TOP,
        "ranking": "descending abs(TIMING-style accumulated saliency)",
        "masking": "one cell at a time",
        "additional_forward_args": [None, None, True],
        "baselines": list(args.baselines),
        "n_steps": args.n_steps,
        "attribution_batch": attribution_batch,
        "path_batch": path_batch,
        "n_test": len(test_x_cpu),
    }
    if destination.is_file() and not args.force:
        previous = json.loads(destination.read_text(encoding="utf-8"))
        if previous.get("status") == "complete" and previous.get("protocol") == protocol:
            print(f"[SKIP] {destination}", flush=True)
            return previous

    saliency, saliency_path, cache_status = load_saliency(
        data=data,
        fold=args.fold,
        seed=args.seed,
        test_x=test_x_cpu,
        classifier=classifier,
        device=device,
        cache_root=cache_root,
        attribution_batch=attribution_batch,
        path_batch=path_batch,
        n_steps=args.n_steps,
        recompute_if_missing=args.recompute_if_missing,
    )

    x_test = test_x_cpu.to(device=device, dtype=torch.float32)
    testbs = args.testbs if args.testbs > 0 else len(x_test)
    testbs = min(testbs, len(x_test))

    old_cudnn = torch.backends.cudnn.enabled
    if device.type == "cuda":
        torch.backends.cudnn.enabled = False
    try:
        results = {
            name: compute_cpd(
                classifier=classifier,
                x_test=x_test,
                saliency=saliency,
                baseline_name=name,
                testbs=testbs,
            )
            for name in args.baselines
        }
    finally:
        if device.type == "cuda":
            torch.backends.cudnn.enabled = old_cudnn

    payload = {
        "status": "complete",
        "protocol": protocol,
        "data": data,
        "fold": args.fold,
        "seed": args.seed,
        "method": "IRON",
        "metric": "CPD",
        "results": results,
        "saliency_path": str(saliency_path),
        "saliency_cache_status": cache_status,
    }
    atomic_json(destination, payload)

    values = " ".join(
        f"{item['baseline']}={item['cpd']:.6f}" for item in results.values()
    )
    print(f"[RESULT] {data}/fold{args.fold} CPD {values}", flush=True)
    print(f"[SAVE] {destination}", flush=True)

    del x_test, saliency, classifier
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return payload


if __name__ == "__main__":
    run(parse_args())
