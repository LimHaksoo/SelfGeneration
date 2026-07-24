#!/usr/bin/env python3
"""Evaluate synthetic IRON with TIMING-style AUP/AUR and Completeness.

TIMING's synthetic implementation predicts a class for every temporal prefix,
computes one prefix attribution, takes its absolute value, and accumulates the
prefix attribution into a full [N,T,D] saliency tensor. IRON follows the same
prefix-target and accumulation protocol for AUP/AUR.

Because the official FFJORD flow is fixed-dimensional, each prefix is embedded
in the full sequence space by zero-padding its unobserved suffix. The
classifier wrapper ignores that suffix and evaluates only the active prefix.
Completeness is computed from the signed attribution at the final full-length
prefix, using the standard CE/NCE definitions.
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
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from attribution.nf_ig import NFIntegratedGradients
from attribution.official_ffjord import load_checkpoint
from configs.official_cnf_fullfold_by_dataset import split_candidate
from configs.synthetic_iron_whitebox import (
    FORMAT_VERSION,
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


COMPLETENESS_EPS = 1e-8


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class PrefixClassifier(nn.Module):
    """Expose a full-shape input while evaluating only its active prefix."""

    def __init__(self, classifier: nn.Module) -> None:
        super().__init__()
        self.classifier = classifier
        self.prefix_length = 1

    def set_prefix_length(self, value: int) -> None:
        if int(value) < 1:
            raise ValueError("prefix_length must be positive")
        self.prefix_length = int(value)

    def forward(self, inputs: Tensor) -> Tensor:
        prefix = inputs[:, : self.prefix_length, :]
        return self.classifier(
            prefix,
            mask=None,
            timesteps=None,
            return_all=False,
        )


def timing_style_iron_attribution(
    *,
    inputs: Tensor,
    classifier: nn.Module,
    flow: nn.Module,
    n_steps: int,
    path_batch: int,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Return TIMING-style saliency and full-sequence Completeness arrays.

    Returns:
        accumulated_absolute:
            Prefix-accumulated absolute saliency used for AUP/AUR.
        attribution_sum:
            Signed full-sequence IRON attribution summed over all input cells.
        output_delta:
            Full-sequence target output difference F_c(x)-F_c(0).
    """

    wrapper = PrefixClassifier(classifier)
    explainer = NFIntegratedGradients(wrapper, flow)

    padded = torch.zeros_like(inputs)
    baseline = torch.zeros_like(inputs)
    accumulated_absolute = torch.zeros_like(inputs)
    final_attribution_sum: Optional[Tensor] = None
    final_output_delta: Optional[Tensor] = None

    for prefix_length in range(1, inputs.shape[1] + 1):
        padded[:, prefix_length - 1 : prefix_length, :] = inputs[
            :, prefix_length - 1 : prefix_length, :
        ]
        wrapper.set_prefix_length(prefix_length)

        with torch.no_grad():
            targets = wrapper(padded).argmax(dim=-1)

        is_final_prefix = prefix_length == inputs.shape[1]
        if is_final_prefix:
            signed, diagnostics = explainer.attribute(
                padded,
                targets=targets,
                baseline=baseline,
                n_steps=int(n_steps),
                path_batch_size=int(path_batch),
                enforce_exact_endpoints=True,
                return_diagnostics=True,
            )
            final_attribution_sum = diagnostics[
                "attribution_sum"
            ].detach()
            final_output_delta = diagnostics["output_delta"].detach()
        else:
            signed = explainer.attribute(
                padded,
                targets=targets,
                baseline=baseline,
                n_steps=int(n_steps),
                path_batch_size=int(path_batch),
                enforce_exact_endpoints=True,
                return_diagnostics=False,
            )

        accumulated_absolute[:, :prefix_length, :] += signed[
            :, :prefix_length, :
        ].abs()

    if final_attribution_sum is None or final_output_delta is None:
        raise RuntimeError("The final synthetic prefix was not evaluated")

    return (
        accumulated_absolute,
        final_attribution_sum,
        final_output_delta,
    )


def completeness_metrics(
    attribution_sum: Tensor,
    output_delta: Tensor,
    eps: float = COMPLETENESS_EPS,
) -> Tuple[Dict[str, float], Tensor, Tensor, Tensor]:
    """Compute the repository's standard CE and NCE metrics.

    residual_i = sum_j A_ij - DeltaF_i
    CE  = mean_i |residual_i|
    NCE = sum_i |residual_i| / (sum_i |DeltaF_i| + eps)
    """

    sum_attr = attribution_sum.detach().cpu().double().reshape(-1)
    delta = output_delta.detach().cpu().double().reshape(-1)

    if sum_attr.numel() == 0 or sum_attr.numel() != delta.numel():
        raise ValueError(
            f"Invalid Completeness arrays: sum_attr={tuple(sum_attr.shape)}, "
            f"output_delta={tuple(delta.shape)}"
        )

    finite = torch.isfinite(sum_attr) & torch.isfinite(delta)
    sum_attr = sum_attr[finite]
    delta = delta[finite]
    if sum_attr.numel() == 0:
        raise ValueError("No finite Completeness samples remain")

    residual = sum_attr - delta
    absolute = residual.abs()
    metrics = {
        "n_samples": int(absolute.numel()),
        "ce": float(absolute.mean().item()),
        "nce": float(
            absolute.sum().item()
            / (delta.abs().sum().item() + float(eps))
        ),
        "median_abs_error": float(absolute.median().item()),
        "max_abs_error": float(absolute.max().item()),
    }
    return metrics, sum_attr, delta, residual


def _to_float(value: Any) -> float:
    if isinstance(value, Tensor):
        value = value.detach().cpu().item()
    return float(value)


def result_path(root: Path, data: str, fold: int, seed: int) -> Path:
    return (
        root
        / filesystem_name(data)
        / f"fold{int(fold)}_seed{int(seed)}"
        / "iron.json"
    )


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--fold", type=int, required=True, choices=range(5))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--n-steps", type=int, default=N_STEPS)
    parser.add_argument("--attribution-batch", type=int, default=0)
    parser.add_argument("--path-batch", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--output-root", default=RESULT_ROOT)
    parser.add_argument("--save-attributions", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    if args.n_steps < 1:
        parser.error("--n-steps must be positive")
    if args.max_samples < 0:
        parser.error("--max-samples must be non-negative")
    return args


def run(args: argparse.Namespace) -> Dict[str, Any]:
    from tint.metrics.white_box import aup, aur

    dataset = canonical_dataset(args.data)
    seed_all(args.seed)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    compute = get_compute(dataset)
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

    full_candidate = get_candidate(dataset)
    model_candidate, _optimizer = split_candidate(full_candidate)
    checkpoint = flow_checkpoint(dataset, args.fold, args.seed)
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"Synthetic IRON flow not found: {checkpoint}. "
            "Run synthetic/train_iron_flow.py first."
        )

    bundle = load_bundle(dataset, args.fold, args.seed)
    test_x = bundle.test_x
    true_saliency = bundle.true_saliency
    if args.max_samples > 0:
        test_x = test_x[: args.max_samples]
        true_saliency = true_saliency[: args.max_samples]

    protocol = {
        "format_version": max(int(FORMAT_VERSION), 2),
        "dataset": dataset,
        "fold": int(args.fold),
        "seed": int(args.seed),
        "method": "IRON",
        "synthetic_evaluation": "TIMING prefix accumulation",
        "prefix_target": "predicted class",
        "prefix_embedding": "zero-padded full FFJORD input",
        "baseline": "zero",
        "n_steps": int(args.n_steps),
        "attribution_batch": attribution_batch,
        "path_batch": path_batch,
        "candidate": full_candidate["name"],
        "n_samples": int(len(test_x)),
        "completeness": "full-sequence signed IRON CE/NCE",
    }
    output = result_path(Path(args.output_root), dataset, args.fold, args.seed)
    if output.is_file() and not args.force:
        previous = json.loads(output.read_text(encoding="utf-8"))
        if (
            previous.get("status") == "complete"
            and previous.get("protocol") == protocol
        ):
            print(f"[SKIP] current result: {output}", flush=True)
            return previous

    classifier = load_classifier(dataset, args.fold, args.seed, device)
    flow, flow_metadata = load_checkpoint(
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
        batch_size=attribution_batch,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    saliency_parts: List[Tensor] = []
    attribution_sum_parts: List[Tensor] = []
    output_delta_parts: List[Tensor] = []

    old_cudnn = torch.backends.cudnn.enabled
    if device.type == "cuda":
        torch.backends.cudnn.enabled = False

    try:
        for (x_cpu,) in tqdm(
            loader,
            desc=f"IRON-whitebox {dataset}/f{args.fold}",
        ):
            x = x_cpu.to(
                device=device,
                dtype=torch.float32,
                non_blocking=True,
            )
            saliency, attribution_sum, output_delta = (
                timing_style_iron_attribution(
                    inputs=x,
                    classifier=classifier,
                    flow=flow,
                    n_steps=args.n_steps,
                    path_batch=path_batch,
                )
            )
            saliency_parts.append(saliency.detach().cpu())
            attribution_sum_parts.append(attribution_sum.detach().cpu())
            output_delta_parts.append(output_delta.detach().cpu())
            del saliency, attribution_sum, output_delta
    finally:
        if device.type == "cuda":
            torch.backends.cudnn.enabled = old_cudnn

    attributions = torch.cat(saliency_parts, dim=0)
    attribution_sum_all = torch.cat(attribution_sum_parts, dim=0)
    output_delta_all = torch.cat(output_delta_parts, dim=0)

    if tuple(attributions.shape) != tuple(true_saliency.shape):
        raise RuntimeError(
            f"Attribution shape={tuple(attributions.shape)} does not match "
            f"true saliency={tuple(true_saliency.shape)}"
        )
    if not torch.isfinite(attributions).all():
        raise FloatingPointError("Non-finite synthetic IRON attribution")

    metric_attr = attributions.to(device)
    metric_true = true_saliency.to(device)
    aup_value = _to_float(aup(metric_attr, metric_true))
    aur_value = _to_float(aur(metric_attr, metric_true))

    completeness, sum_attr, delta, residual = completeness_metrics(
        attribution_sum_all,
        output_delta_all,
    )

    completeness_arrays_path = output.with_name(
        "iron_completeness_arrays.npz"
    )
    completeness_arrays_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        completeness_arrays_path,
        attribution_sum=sum_attr.numpy(),
        output_delta=delta.numpy(),
        residual=residual.numpy(),
    )

    attribution_path = None
    if args.save_attributions:
        attribution_path = output.with_name("iron_attributions.npy")
        attribution_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(attribution_path, attributions.numpy())

    result = {
        "status": "complete",
        "protocol": protocol,
        "data": dataset,
        "fold": int(args.fold),
        "seed": int(args.seed),
        "method": "IRON",
        "aup": aup_value,
        "aur": aur_value,
        "completeness": completeness,
        "checkpoint": str(checkpoint),
        "training_best_epoch": flow_metadata.get("best_epoch"),
        "attribution_path": (
            None if attribution_path is None else str(attribution_path)
        ),
        "completeness_arrays_path": str(completeness_arrays_path),
    }
    atomic_write_json(output, result)
    print(
        f"[RESULT] data={dataset} fold={args.fold} "
        f"AUP={aup_value:.6f} AUR={aur_value:.6f} "
        f"CE={completeness['ce']:.6f} "
        f"NCE={completeness['nce']:.6f}",
        flush=True,
    )
    print(f"[SAVE] {output}", flush=True)

    del (
        metric_attr,
        metric_true,
        attributions,
        attribution_sum_all,
        output_delta_all,
        flow,
        classifier,
    )
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


if __name__ == "__main__":
    run(parse_args())
