#!/usr/bin/env python3
"""Metrics used by the synthetic IRON hyperparameter search."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset

from attribution.nf_ig import NFIntegratedGradients


COMPLETENESS_EPS = 1e-8


class PrefixClassifier(nn.Module):
    """Expose a fixed-size flow input while evaluating one temporal prefix."""

    def __init__(self, classifier: nn.Module) -> None:
        super().__init__()
        self.classifier = classifier
        self.prefix_length = 1

    def set_prefix_length(self, value: int) -> None:
        value = int(value)
        if value < 1:
            raise ValueError("prefix_length must be positive")
        self.prefix_length = value

    def forward(self, inputs: Tensor) -> Tensor:
        prefix = inputs[:, : self.prefix_length, :]
        return self.classifier(
            prefix,
            mask=None,
            timesteps=None,
            return_all=False,
        )


def _as_float(value: Any) -> float:
    if isinstance(value, Tensor):
        value = value.detach().cpu().item()
    return float(value)


def timing_style_iron_attribution(
    *,
    inputs: Tensor,
    classifier: nn.Module,
    flow: nn.Module,
    n_steps: int,
    path_batch: int,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Mirror TIMING's synthetic prefix accumulation with IRON.

    Returns:
        accumulated_absolute: [B,T,D] saliency used by AUP/AUR and CPD.
        final_signed: signed full-sequence IRON attribution.
        final_output_delta: predicted-class F(x)-F(0) for the final prefix.
    """

    if inputs.ndim != 3:
        raise ValueError(f"Expected [B,T,D], got {tuple(inputs.shape)}")

    wrapper = PrefixClassifier(classifier)
    explainer = NFIntegratedGradients(wrapper, flow)

    padded = torch.zeros_like(inputs)
    baseline = torch.zeros_like(inputs)
    accumulated = torch.zeros_like(inputs)
    final_signed: Optional[Tensor] = None
    final_output_delta: Optional[Tensor] = None

    for prefix_length in range(1, int(inputs.shape[1]) + 1):
        padded[:, prefix_length - 1 : prefix_length, :] = inputs[
            :, prefix_length - 1 : prefix_length, :
        ]
        wrapper.set_prefix_length(prefix_length)

        with torch.no_grad():
            targets = wrapper(padded).argmax(dim=-1)

        final_prefix = prefix_length == int(inputs.shape[1])
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
            output_delta = diagnostics["output_delta"].reshape(-1)
            explicit_residual = (
                signed.reshape(signed.shape[0], -1).sum(dim=1) - output_delta
            )
            native_residual = diagnostics["convergence_delta"].reshape(-1)
            if not torch.allclose(
                explicit_residual,
                native_residual,
                atol=1e-5,
                rtol=1e-5,
            ):
                raise RuntimeError("IRON completeness diagnostic mismatch")
            final_signed = signed
            final_output_delta = output_delta
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

        accumulated[:, :prefix_length, :] += signed[:, :prefix_length, :].abs()

    if final_signed is None or final_output_delta is None:
        raise RuntimeError("The full-sequence prefix was not evaluated")

    return accumulated, final_signed, final_output_delta


@torch.no_grad()
def _encode_batches(
    flow: nn.Module,
    inputs: Tensor,
    *,
    device: torch.device,
    batch_size: int,
) -> Tensor:
    parts: List[Tensor] = []
    actual_batch = max(1, min(int(batch_size), int(inputs.shape[0])))
    for start in range(0, int(inputs.shape[0]), actual_batch):
        batch = inputs[start : start + actual_batch].to(
            device=device,
            dtype=torch.float32,
            non_blocking=True,
        )
        parts.append(flow.encode(batch).detach().cpu().float())
    return torch.cat(parts, dim=0)


def latent_gaussianity_metrics(
    *,
    flow: nn.Module,
    validation_inputs: Tensor,
    device: torch.device,
    batch_size: int,
    seed: int,
    num_projections: int = 64,
) -> Dict[str, float]:
    """Match the real-data search diagnostics on the synthetic validation set."""

    flow.eval()
    z = _encode_batches(
        flow,
        validation_inputs,
        device=device,
        batch_size=batch_size,
    )
    generator = torch.Generator().manual_seed(int(seed))
    normal = torch.randn(z.shape, generator=generator, dtype=z.dtype)

    dimensions = int(z.shape[1])
    sample_count = int(z.shape[0])
    if sample_count < 4:
        raise ValueError("At least four validation samples are required")

    projection_generator = torch.Generator().manual_seed(int(seed) + 17)
    directions = torch.randn(
        dimensions,
        int(num_projections),
        generator=projection_generator,
        dtype=z.dtype,
    )
    directions = directions / directions.norm(dim=0, keepdim=True).clamp_min(1e-12)
    projected_z = z @ directions
    projected_normal = normal @ directions
    sliced_w1 = (
        projected_z.sort(dim=0).values
        - projected_normal.sort(dim=0).values
    ).abs().mean()

    scale = math.sqrt(float(dimensions))
    radius_z = z.norm(p=2, dim=1) / scale
    radius_normal = normal.norm(p=2, dim=1) / scale
    radius_w1 = (
        radius_z.sort().values - radius_normal.sort().values
    ).abs().mean()

    two_sample_auc = _two_sample_auc(
        z.numpy(),
        normal.numpy(),
        seed=int(seed) + 31,
    )

    return {
        "two_sample_auc": float(two_sample_auc),
        "two_sample_auc_gap": float(abs(two_sample_auc - 0.5)),
        "sliced_w1": float(sliced_w1.item()),
        "radius_w1_per_dim": float(radius_w1.item()),
        "latent_mean_abs": float(z.mean(dim=0).abs().mean().item()),
        "latent_std_abs_error": float(
            (z.std(dim=0, unbiased=False) - 1.0).abs().mean().item()
        ),
        "n_validation": sample_count,
        "latent_dimensions": dimensions,
    }


def _two_sample_auc(z: np.ndarray, normal: np.ndarray, seed: int) -> float:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    features = np.concatenate([z, normal], axis=0).astype(np.float64, copy=False)
    labels = np.concatenate(
        [np.zeros(len(z), dtype=np.int64), np.ones(len(normal), dtype=np.int64)]
    )
    folds = min(5, len(z), len(normal))
    if folds < 2:
        return float("nan")

    estimator = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=0.1,
            max_iter=2000,
            solver="liblinear",
            random_state=int(seed),
        ),
    )
    splitter = StratifiedKFold(
        n_splits=int(folds),
        shuffle=True,
        random_state=int(seed),
    )
    probabilities = cross_val_predict(
        estimator,
        features,
        labels,
        cv=splitter,
        method="predict_proba",
        n_jobs=1,
    )[:, 1]
    auc_value = float(roc_auc_score(labels, probabilities))
    # Direction is arbitrary in a two-sample test; report distinguishability.
    return max(auc_value, 1.0 - auc_value)


def evaluate_attribution_metrics(
    *,
    classifier: nn.Module,
    flow: nn.Module,
    test_inputs: Tensor,
    true_saliency: Tensor,
    device: torch.device,
    n_steps: int,
    attribution_batch: int,
    path_batch: int,
    cpd_batch: int,
    cpd_topk: float,
    cpd_top: int,
) -> Dict[str, Any]:
    """Compute AUP, AUR, CE/NCE, and 10% CPD for one candidate."""

    from synthetic.switchstate.cumulative_difference import cumulative_difference
    from tint.metrics.white_box import aup, aur

    if tuple(test_inputs.shape) != tuple(true_saliency.shape):
        raise ValueError(
            f"Input shape {tuple(test_inputs.shape)} does not match true "
            f"saliency {tuple(true_saliency.shape)}"
        )

    classifier.eval()
    flow.eval()
    actual_batch = max(1, min(int(attribution_batch), int(test_inputs.shape[0])))
    loader = DataLoader(
        TensorDataset(test_inputs),
        batch_size=actual_batch,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    saliency_parts: List[Tensor] = []
    signed_parts: List[Tensor] = []
    delta_parts: List[Tensor] = []

    previous_cudnn = torch.backends.cudnn.enabled
    if device.type == "cuda":
        torch.backends.cudnn.enabled = False

    try:
        for (inputs_cpu,) in loader:
            inputs = inputs_cpu.to(
                device=device,
                dtype=torch.float32,
                non_blocking=True,
            )
            saliency, signed, output_delta = timing_style_iron_attribution(
                inputs=inputs,
                classifier=classifier,
                flow=flow,
                n_steps=n_steps,
                path_batch=path_batch,
            )
            saliency_parts.append(saliency.detach().cpu())
            signed_parts.append(signed.detach().cpu())
            delta_parts.append(output_delta.detach().cpu())
            del saliency, signed, output_delta, inputs
    finally:
        if device.type == "cuda":
            torch.backends.cudnn.enabled = previous_cudnn

    saliency = torch.cat(saliency_parts, dim=0)
    signed = torch.cat(signed_parts, dim=0)
    output_delta = torch.cat(delta_parts, dim=0).reshape(-1)

    if not torch.isfinite(saliency).all() or not torch.isfinite(signed).all():
        raise FloatingPointError("Non-finite synthetic IRON attribution")

    sum_attribution = signed.double().reshape(signed.shape[0], -1).sum(dim=1)
    delta_double = output_delta.double()
    absolute_error = (sum_attribution - delta_double).abs()
    ce_value = float(absolute_error.mean().item())
    nce_value = float(
        absolute_error.sum().item()
        / (delta_double.abs().sum().item() + COMPLETENESS_EPS)
    )

    metric_saliency = saliency.to(device=device, dtype=torch.float32)
    metric_true = true_saliency.to(device=device)
    aup_value = _as_float(aup(metric_saliency, metric_true))
    aur_value = _as_float(aur(metric_saliency, metric_true))

    x_device = test_inputs.to(device=device, dtype=torch.float32)
    average_baseline = x_device.mean(dim=1, keepdim=True).repeat(
        1, x_device.shape[1], 1
    )
    cpd_zero = _compute_cpd(
        classifier=classifier,
        inputs=x_device,
        attributions=saliency,
        baseline=0.0,
        topk=cpd_topk,
        top=cpd_top,
        batch_size=cpd_batch,
    )
    cpd_average = _compute_cpd(
        classifier=classifier,
        inputs=x_device,
        attributions=saliency,
        baseline=average_baseline,
        topk=cpd_topk,
        top=cpd_top,
        batch_size=cpd_batch,
    )

    return {
        "aup": aup_value,
        "aur": aur_value,
        "ce": ce_value,
        "nce": nce_value,
        "median_abs_completeness": float(absolute_error.median().item()),
        "cpd_zero": cpd_zero["cpd"],
        "cpd_average": cpd_average["cpd"],
        "cpd_zero_details": cpd_zero,
        "cpd_average_details": cpd_average,
        "n_test": int(test_inputs.shape[0]),
    }


def _compute_cpd(
    *,
    classifier: nn.Module,
    inputs: Tensor,
    attributions: Tensor,
    baseline: Any,
    topk: float,
    top: int,
    batch_size: int,
) -> Dict[str, Any]:
    from synthetic.switchstate.cumulative_difference import cumulative_difference

    actual_batch = max(1, min(int(batch_size), int(inputs.shape[0])))
    cpd, aucc, cumulative_50, curve = cumulative_difference(
        classifier,
        inputs,
        attributions=attributions.cpu(),
        baselines=baseline,
        topk=float(topk),
        top=int(top),
        testbs=actual_batch,
        additional_forward_args=(None, None, True),
    )
    if isinstance(curve, Tensor):
        curve_values: Sequence[Any] = curve.detach().cpu().reshape(-1).tolist()
    else:
        curve_values = list(curve)
    return {
        "cpd": _as_float(cpd),
        "aucc": _as_float(aucc),
        "cumulative_50": _as_float(cumulative_50),
        "curve": [_as_float(value) for value in curve_values],
        "topk": float(topk),
        "top": int(top),
    }
