#!/usr/bin/env python3
"""Compute only TIMING-style CPD@10% for synthetic IRON attributions.

The script reuses the absolute prefix-accumulated attribution produced by
``synthetic/eval_iron_whitebox.py``.  If that cache is absent, it can recompute
only the attribution map and save it for later CPD runs.  AUP, AUR, and
Completeness are not evaluated here.
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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def whitebox_fold_dir(root: Path, data: str, fold: int, seed: int) -> Path:
    return root / filesystem_name(data) / f"fold{fold}_seed{seed}"


def cpd_result_path(root: Path, data: str, fold: int, seed: int) -> Path:
    return root / filesystem_name(data) / f"fold{fold}_seed{seed}" / "iron_cpd.json"


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_save_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        np.save(handle, array)
    os.replace(temporary, path)


def _json_attribution_path(fold_dir: Path) -> Optional[Path]:
    result_json = fold_dir / "iron.json"
    if not result_json.is_file():
        return None
    payload = json.loads(result_json.read_text(encoding="utf-8"))
    value = payload.get("attribution_path")
    if not value:
        return None
    path = resolve_project_path(str(value))
    return path if path.is_file() else None


def find_cached_saliency(
    *,
    cache_root: Path,
    data: str,
    fold: int,
    seed: int,
) -> Optional[Path]:
    fold_dir = whitebox_fold_dir(cache_root, data, fold, seed)
    candidates = [
        _json_attribution_path(fold_dir),
        fold_dir / "iron_attributions.npy",
    ]
    for candidate in candidates:
        if candidate is not None and candidate.is_file():
            return candidate
    return None


def _extract_saliency(output: Any) -> Tensor:
    # v1 returns only the accumulated saliency.  v2+ returns
    # (accumulated_saliency, signed_full_sequence, output_delta).
    value = output[0] if isinstance(output, tuple) else output
    if not isinstance(value, Tensor):
        raise TypeError(f"Unexpected IRON attribution result: {type(value)}")
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
    model_candidate, _optimizer = split_candidate(full_candidate)
    checkpoint = flow_checkpoint(data, fold, seed)
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"Synthetic IRON flow not found: {checkpoint}. "
            "Run synthetic/train_iron_flow.py first."
        )

    flow, _metadata = load_checkpoint(
        checkpoint,
        ffjord_root=ffjord_root(),
        device=device,
        expected_input_shape=test_x.shape[1:],
    )
    if dict(flow.candidate) != model_candidate:
        raise RuntimeError(
            f"Flow candidate mismatch: {checkpoint}\n"
            f"expected={model_candidate}\nactual={flow.candidate}"
        )
    flow.eval()
    for parameter in flow.parameters():
        parameter.requires_grad_(False)

    loader = DataLoader(
        TensorDataset(test_x),
        batch_size=int(attribution_batch),
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    parts: List[Tensor] = []
    old_cudnn = torch.backends.cudnn.enabled
    if device.type == "cuda":
        torch.backends.cudnn.enabled = False
    try:
        for (x_cpu,) in tqdm(
            loader,
            desc=f"IRON-saliency {data}/f{fold}",
        ):
            x = x_cpu.to(
                device=device,
                dtype=torch.float32,
                non_blocking=True,
            )
            output = timing_style_iron_attribution(
                inputs=x,
                classifier=classifier,
                flow=flow,
                n_steps=int(n_steps),
                path_batch=int(path_batch),
            )
            saliency = _extract_saliency(output).detach().cpu().float()
            parts.append(saliency)
            del output, saliency, x
    finally:
        if device.type == "cuda":
            torch.backends.cudnn.enabled = old_cudnn

    result = torch.cat(parts, dim=0)
    if tuple(result.shape) != tuple(test_x.shape):
        raise RuntimeError(
            f"Recomputed saliency shape={tuple(result.shape)} "
            f"!= test shape={tuple(test_x.shape)}"
        )
    if not torch.isfinite(result).all():
        raise FloatingPointError("Non-finite synthetic IRON saliency")

    atomic_save_npy(destination, result.numpy())
    print(f"[CACHE] saved {destination}", flush=True)

    del flow
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def load_or_compute_saliency(
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
    cached = find_cached_saliency(
        cache_root=cache_root,
        data=data,
        fold=fold,
        seed=seed,
    )
    if cached is not None:
        array = np.load(cached)
        saliency = torch.from_numpy(np.asarray(array)).float()
        if tuple(saliency.shape) != tuple(test_x.shape):
            raise RuntimeError(
                f"Cached saliency shape={tuple(saliency.shape)} "
                f"!= test shape={tuple(test_x.shape)}: {cached}"
            )
        if not torch.isfinite(saliency).all():
            raise FloatingPointError(f"Non-finite cached saliency: {cached}")
        print(f"[CACHE] loaded {cached}", flush=True)
        return saliency, cached, "loaded"

    if not recompute_if_missing:
        expected = whitebox_fold_dir(cache_root, data, fold, seed) / "iron_attributions.npy"
        raise FileNotFoundError(
            f"IRON saliency cache not found: {expected}. "
            "Rerun eval_iron_whitebox.py with --save-attributions or pass "
            "--recompute-if-missing."
        )

    destination = whitebox_fold_dir(cache_root, data, fold, seed) / "iron_attributions.npy"
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


def _curve_to_list(value: Any) -> List[float]:
    if isinstance(value, Tensor):
        value = value.detach().cpu().numpy()
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    return [float(item) for item in array]


def compute_cpd(
    *,
    classifier,
    x_test: Tensor,
    saliency: Tensor,
    baseline_name: str,
    testbs: int,
) -> Dict[str, Any]:
    # This is the exact function called by the original synthetic TIMING code.
    from synthetic.switchstate.cumulative_difference import cumulative_difference

    if baseline_name == "average":
        baseline = x_test.mean(1, keepdim=True).repeat(1, x_test.shape[1], 1)
        display_name = "Average"
    elif baseline_name == "zero":
        baseline = 0.0
        display_name = "Zeros"
    else:  # pragma: no cover
        raise ValueError(baseline_name)

    cpd, _aucc, _cum50, curve = cumulative_difference(
        classifier,
        x_test,
        attributions=saliency.abs().cpu(),
        baselines=baseline,
        topk=CPD_TOPK,
        top=CPD_TOP,
        testbs=int(testbs),
        additional_forward_args=(None, None, True),
    )
    return {
        "baseline": display_name,
        "cpd": float(cpd),
        "curve": _curve_to_list(curve),
    }


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
        int(args.attribution_batch)
        if int(args.attribution_batch) > 0
        else int(compute["attribution_batch"])
    )
    path_batch = (
        int(args.path_batch)
        if int(args.path_batch) > 0
        else int(compute["path_batch"])
    )

    bundle = load_bundle(data, args.fold, args.seed)
    test_x_cpu = bundle.test_x.detach().cpu().float().contiguous()
    classifier = load_classifier(data, args.fold, args.seed, device)

    cache_root = resolve_project_path(args.cache_root)
    output_root = resolve_project_path(args.output_root)
    output = cpd_result_path(output_root, data, args.fold, args.seed)

    protocol = {
        "dataset": data,
        "fold": int(args.fold),
        "seed": int(args.seed),
        "method": "IRON",
        "metric": "CPD",
        "topk": CPD_TOPK,
        "top": CPD_TOP,
        "ranking": "descending absolute TIMING-style accumulated saliency",
        "masking": "one cell at a time",
        "probability_difference": "L1 between consecutive class-probability vectors",
        "additional_forward_args": [None, None, True],
        "baselines": list(args.baselines),
        "n_steps": int(args.n_steps),
        "attribution_batch": attribution_batch,
        "path_batch": path_batch,
        "n_test": int(len(test_x_cpu)),
    }
    if output.is_file() and not args.force:
        previous = json.loads(output.read_text(encoding="utf-8"))
        if previous.get("status") == "complete" and previous.get("protocol") == protocol:
            print(f"[SKIP] current CPD result: {output}", flush=True)
            return previous

    saliency, saliency_path, cache_status = load_or_compute_saliency(
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
    testbs = int(args.testbs) if int(args.testbs) > 0 else int(len(x_test))
    testbs = min(testbs, int(len(x_test)))

    old_cudnn = torch.backends.cudnn.enabled
    if device.type == "cuda":
        torch.backends.cudnn.enabled = False
    try:
        baseline_results = {
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

    result = {
        "status": "complete",
        "protocol": protocol,
        "data": data,
        "fold": int(args.fold),
        "seed": int(args.seed),
        "method": "IRON",
        "metric": "CPD",
        "results": baseline_results,
        "saliency_path": str(saliency_path),
        "saliency_cache_status": cache_status,
    }
    atomic_write_json(output, result)

    summary = " ".join(
        f"{item['baseline']}={item['cpd']:.6f}"
        for item in baseline_results.values()
    )
    print(f"[RESULT] data={data} fold={args.fold} CPD {summary}", flush=True)
    print(f"[SAVE] {output}", flush=True)

    del x_test, saliency, classifier
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


if __name__ == "__main__":
    run(parse_args())
