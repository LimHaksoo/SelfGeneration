#!/usr/bin/env python3
"""Evaluate IRON with the existing TIMING synthetic white-box protocol.

TIMING predicts a class for every temporal prefix, computes one prefix
attribution, takes its absolute value, and accumulates the prefix attribution
into a full [N,T,D] saliency tensor. IRON follows the same prefix-target and
accumulation protocol for AUP/AUR.

Completeness is evaluated separately from the final full-sequence *signed*
IRON attribution:

    residual_i = sum_j A_ij - [F_target(x_i) - F_target(0)]
    CE  = mean_i |residual_i|
    NCE = sum_i |residual_i| / (sum_i |DeltaF_i| + eps)

The absolute, prefix-accumulated saliency used by AUP/AUR is never used for
Completeness.
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
EVALUATION_REVISION = 2


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
    """Return TIMING saliency and final signed IRON quantities.

    Returns:
        accumulated_absolute:
            TIMING-style absolute prefix accumulation used for AUP/AUR.
        final_signed:
            Signed IRON attribution at the final full-length prefix.
        final_output_delta:
            F_target(x) - F_target(0) for the final predicted target.
    """

    wrapper = PrefixClassifier(classifier)
    explainer = NFIntegratedGradients(wrapper, flow)

    padded = torch.zeros_like(inputs)
    baseline = torch.zeros_like(inputs)
    accumulated = torch.zeros_like(inputs)
    final_signed: Optional[Tensor] = None
    final_output_delta: Optional[Tensor] = None

    for prefix_length in range(1, inputs.shape[1] + 1):
        padded[:, prefix_length - 1 : prefix_length, :] = inputs[
            :, prefix_length - 1 : prefix_length, :
        ]
        wrapper.set_prefix_length(prefix_length)

        with torch.no_grad():
            targets = wrapper(padded).argmax(dim=-1)

        final_prefix = prefix_length == inputs.shape[1]
        if final_prefix:
            signed, diagnostics = explainer.attribute(
                padded,
                targets=targets,
                baseline=baseline,
                n_steps=int(n_steps),
                path_batch_size=int(path_batch),
                enforce_exact_endpoints=True,
                return_diagnostics=True,
            )
            final_signed = signed
            final_output_delta = diagnostics["output_delta"].reshape(-1)

            explicit_residual = (
                signed.reshape(signed.shape[0], -1).sum(dim=1)
                - final_output_delta
            )
            native_residual = diagnostics["convergence_delta"].reshape(-1)
            if not torch.allclose(
                explicit_residual,
                native_residual,
                atol=1e-5,
                rtol=1e-5,
            ):
                maximum_error = float(
                    (explicit_residual - native_residual).abs().max().item()
                )
                raise RuntimeError(
                    "IRON Completeness diagnostic mismatch: "
                    f"max_abs={maximum_error:.3e}"
                )
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

        # This exactly mirrors TIMING's synthetic prefix accumulation.
        accumulated[:, :prefix_length, :] += signed[
            :, :prefix_length, :
        ].abs()

    if final_signed is None or final_output_delta is None:
        raise RuntimeError("Full-sequence IRON attribution was not produced")

    return accumulated, final_signed, final_output_delta


def completeness_arrays(
    signed_attribution: Tensor,
    output_delta: Tensor,
) -> Tuple[Tensor, Tensor]:
    """Return per-sample absolute residual and DeltaF in float64."""

    attribution_sum = (
        signed_attribution.detach()
        .cpu()
        .double()
        .reshape(signed_attribution.shape[0], -1)
        .sum(dim=1)
    )
    delta = output_delta.detach().cpu().double().reshape(-1)

    if attribution_sum.shape != delta.shape:
        raise ValueError(
            f"Completeness shape mismatch: sum_attr={tuple(attribution_sum.shape)}, "
            f"output_delta={tuple(delta.shape)}"
        )
    if not torch.isfinite(attribution_sum).all() or not torch.isfinite(delta).all():
        raise FloatingPointError("Non-finite IRON Completeness values")

    absolute_error = (attribution_sum - delta).abs()
    return absolute_error, delta


def summarize_completeness(
    absolute_error: Tensor,
    output_delta: Tensor,
) -> Dict[str, float | int | str]:
    error = absolute_error.detach().cpu().double().reshape(-1)
    delta = output_delta.detach().cpu().double().reshape(-1)

    if error.numel() == 0 or error.numel() != delta.numel():
        raise RuntimeError("Invalid Completeness accumulator")

    return {
        "definition": (
            "residual=sum(signed attribution)-"
            "[F_target(x)-F_target(0)]"
        ),
        "ce": float(error.mean().item()),
        "nce": float(
            error.sum().item()
            / (delta.abs().sum().item() + float(COMPLETENESS_EPS))
        ),
        "median_abs_error": float(error.median().item()),
        "p95_abs_error": float(torch.quantile(error, 0.95).item()),
        "max_abs_error": float(error.max().item()),
        "n_samples": int(error.numel()),
    }


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
        "evaluation_revision": EVALUATION_REVISION,
        "format_version": FORMAT_VERSION,
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
        "aup_aur_attribution": "absolute prefix-accumulated saliency",
        "completeness_attribution": "signed final full-sequence IRON",
    }
    output = result_path(Path(args.output_root), dataset, args.fold, args.seed)
    if output.is_file() and not args.force:
        previous = json.loads(output.read_text(encoding="utf-8"))
        if (
            previous.get("status") == "complete"
            and previous.get("protocol") == protocol
            and isinstance(previous.get("completeness"), dict)
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
    absolute_error_parts: List[Tensor] = []
    output_delta_parts: List[Tensor] = []
    signed_parts: List[Tensor] = []

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
            saliency, signed_full, output_delta = timing_style_iron_attribution(
                inputs=x,
                classifier=classifier,
                flow=flow,
                n_steps=args.n_steps,
                path_batch=path_batch,
            )
            absolute_error, delta = completeness_arrays(
                signed_full,
                output_delta,
            )

            saliency_parts.append(saliency.detach().cpu())
            absolute_error_parts.append(absolute_error)
            output_delta_parts.append(delta)
            if args.save_attributions:
                signed_parts.append(signed_full.detach().cpu())

            del saliency, signed_full, output_delta, absolute_error, delta
    finally:
        if device.type == "cuda":
            torch.backends.cudnn.enabled = old_cudnn

    attributions = torch.cat(saliency_parts, dim=0)
    absolute_errors = torch.cat(absolute_error_parts, dim=0)
    output_deltas = torch.cat(output_delta_parts, dim=0)

    if tuple(attributions.shape) != tuple(true_saliency.shape):
        raise RuntimeError(
            f"Attribution shape={tuple(attributions.shape)} does not match "
            f"true saliency={tuple(true_saliency.shape)}"
        )
    if not torch.isfinite(attributions).all():
        raise FloatingPointError("Non-finite synthetic IRON attribution")

    completeness = summarize_completeness(
        absolute_errors,
        output_deltas,
    )

    metric_attr = attributions.to(device)
    metric_true = true_saliency.to(device)
    aup_value = _to_float(aup(metric_attr, metric_true))
    aur_value = _to_float(aur(metric_attr, metric_true))

    attribution_path = None
    signed_attribution_path = None
    if args.save_attributions:
        attribution_path = output.with_name("iron_attributions.npy")
        signed_attribution_path = output.with_name(
            "iron_signed_full_sequence.npy"
        )
        attribution_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(attribution_path, attributions.numpy())
        np.save(
            signed_attribution_path,
            torch.cat(signed_parts, dim=0).numpy(),
        )

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
        "signed_attribution_path": (
            None
            if signed_attribution_path is None
            else str(signed_attribution_path)
        ),
    }
    atomic_write_json(output, result)
    print(
        f"[RESULT] data={dataset} fold={args.fold} "
        f"AUP={aup_value:.6f} AUR={aur_value:.6f} "
        f"CE={completeness['ce']:.6f} NCE={completeness['nce']:.6f}",
        flush=True,
    )
    print(f"[SAVE] {output}", flush=True)

    del (
        metric_attr,
        metric_true,
        attributions,
        absolute_errors,
        output_deltas,
        flow,
        classifier,
    )
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


if __name__ == "__main__":
    run(parse_args())
