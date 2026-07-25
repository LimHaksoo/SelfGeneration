#!/usr/bin/env python3
"""Train and evaluate one synthetic IRON hyperparameter candidate."""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from attribution.official_ffjord import (
    OfficialFFJORDTimeSeries,
    load_checkpoint,
    save_checkpoint,
)
from attribution.official_ffjord_epoch import (
    EpochTrainingConfig,
    train_official_ffjord_epoch,
)
from configs.official_cnf_fullfold_by_dataset import split_candidate
from configs.synthetic_iron_search_space import (
    CHECKPOINT_ROOT,
    CPD_TOP,
    CPD_TOPK,
    DEFAULT_MAX_SAMPLES,
    FFJORD_ROOT,
    FORMAT_VERSION,
    N_STEPS,
    RESULT_ROOT,
    SEARCH_PROTOCOL,
    canonical_dataset,
    filesystem_name,
    get_candidate,
    get_compute,
)
from synthetic.iron_search_metrics import (
    evaluate_attribution_metrics,
    latent_gaussianity_metrics,
)
from synthetic.iron_whitebox_common import (
    load_bundle,
    load_classifier,
    split_train_validation,
)


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def checkpoint_path(data: str, fold: int, seed: int, candidate: str) -> Path:
    return (
        PROJECT_ROOT
        / CHECKPOINT_ROOT
        / filesystem_name(data)
        / f"fold{int(fold)}_seed{int(seed)}"
        / f"{candidate}.pt"
    )


def result_path(data: str, fold: int, seed: int, candidate: str) -> Path:
    return (
        PROJECT_ROOT
        / RESULT_ROOT
        / filesystem_name(data)
        / f"fold{int(fold)}_seed{int(seed)}"
        / f"{candidate}.json"
    )


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--fold", type=int, required=True, choices=range(5))
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--epochs",
        type=int,
        default=int(SEARCH_PROTOCOL["epochs"]),
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=int(DEFAULT_MAX_SAMPLES),
        help="Fixed first-N screening subset; 0 uses the complete test split.",
    )
    parser.add_argument("--force-retrain", action="store_true")
    parser.add_argument("--force-eval", action="store_true")
    args = parser.parse_args(argv)
    if args.epochs < 1:
        parser.error("--epochs must be positive")
    if args.max_samples < 0:
        parser.error("--max-samples must be non-negative")
    return args


def _training_identity(
    *,
    full_candidate: Mapping[str, Any],
    epochs: int,
    fold: int,
    seed: int,
) -> Dict[str, Any]:
    return {
        "full_candidate": dict(full_candidate),
        "epochs": int(epochs),
        "fold": int(fold),
        "seed": int(seed),
        "early_stopping_epochs": int(
            SEARCH_PROTOCOL["early_stopping_epochs"]
        ),
        "validate_every_epochs": int(
            SEARCH_PROTOCOL["validate_every_epochs"]
        ),
    }


def train_or_load(
    *,
    data: str,
    fold: int,
    seed: int,
    epochs: int,
    candidate: Mapping[str, Any],
    train_inputs: torch.Tensor,
    validation_inputs: torch.Tensor,
    device: torch.device,
    force_retrain: bool,
) -> Tuple[OfficialFFJORDTimeSeries, Dict[str, Any], Path]:
    candidate_name = str(candidate["name"])
    destination = checkpoint_path(data, fold, seed, candidate_name)
    model_candidate, optimizer_config = split_candidate(candidate)
    identity = _training_identity(
        full_candidate=candidate,
        epochs=epochs,
        fold=fold,
        seed=seed,
    )

    if destination.is_file() and not force_retrain:
        model, metadata = load_checkpoint(
            destination,
            ffjord_root=PROJECT_ROOT / FFJORD_ROOT,
            device=device,
            expected_input_shape=train_inputs.shape[1:],
        )
        if dict(model.candidate) != model_candidate:
            raise RuntimeError(
                f"Checkpoint model candidate mismatch: {destination}"
            )
        recorded_identity = metadata.get("synthetic_search_identity")
        if recorded_identity != identity:
            raise RuntimeError(
                f"Checkpoint training identity changed: {destination}. "
                "Use --force-retrain."
            )
        print(
            f"[LOAD] data={data} fold={fold} candidate={candidate_name} "
            f"best_epoch={metadata.get('best_epoch')}",
            flush=True,
        )
        return model, metadata, destination

    compute = get_compute(data)
    model = OfficialFFJORDTimeSeries(
        input_shape=train_inputs.shape[1:],
        candidate=model_candidate,
        ffjord_root=PROJECT_ROOT / FFJORD_ROOT,
    )
    training_config = EpochTrainingConfig(
        epochs=int(epochs),
        batch_size=int(compute["train_batch"]),
        eval_batch_size=int(compute["nll_eval_batch"]),
        learning_rate=float(optimizer_config["learning_rate"]),
        weight_decay=float(optimizer_config["weight_decay"]),
        early_stopping_epochs=int(
            SEARCH_PROTOCOL["early_stopping_epochs"]
        ),
        validate_every_epochs=int(
            SEARCH_PROTOCOL["validate_every_epochs"]
        ),
        log_every_steps=int(SEARCH_PROTOCOL["log_every_steps"]),
        seed=int(seed) + 1009 * int(fold),
    )

    print(
        f"[TRAIN] data={data} fold={fold} candidate={candidate_name} "
        f"shape={tuple(train_inputs.shape[1:])} "
        f"train={len(train_inputs)} val={len(validation_inputs)} "
        f"epochs={epochs} batch={compute['train_batch']}",
        flush=True,
    )
    metadata = train_official_ffjord_epoch(
        model=model,
        train_inputs=train_inputs,
        val_inputs=validation_inputs,
        device=device,
        config=training_config,
        verbose=True,
    )
    metadata.update(
        {
            "dataset": data,
            "fold": int(fold),
            "seed": int(seed),
            "full_candidate": dict(candidate),
            "synthetic_search_identity": identity,
            "train_samples": int(len(train_inputs)),
            "validation_samples": int(len(validation_inputs)),
        }
    )
    save_checkpoint(destination, model, metadata)
    print(f"[SAVE] {destination}", flush=True)
    return model, metadata, destination


def run(args: argparse.Namespace) -> Dict[str, Any]:
    data = canonical_dataset(args.data)
    seed_all(args.seed)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    candidate = get_candidate(data, args.candidate)
    candidate_name = str(candidate["name"])
    compute = get_compute(data)
    bundle = load_bundle(data, args.fold, args.seed)
    train_inputs, validation_inputs = split_train_validation(
        bundle.train_x,
        bundle.train_y,
        fold=args.fold,
        seed=args.seed,
    )

    flow, training_metadata, checkpoint = train_or_load(
        data=data,
        fold=args.fold,
        seed=args.seed,
        epochs=args.epochs,
        candidate=candidate,
        train_inputs=train_inputs,
        validation_inputs=validation_inputs,
        device=device,
        force_retrain=bool(args.force_retrain),
    )
    flow.eval()
    for parameter in flow.parameters():
        parameter.requires_grad_(False)

    test_inputs = bundle.test_x
    true_saliency = bundle.true_saliency
    if args.max_samples > 0:
        count = min(int(args.max_samples), int(test_inputs.shape[0]))
        test_inputs = test_inputs[:count].contiguous()
        true_saliency = true_saliency[:count].contiguous()

    protocol = {
        "format_version": FORMAT_VERSION,
        "data": data,
        "fold": int(args.fold),
        "seed": int(args.seed),
        "candidate": dict(candidate),
        "epochs": int(args.epochs),
        "max_samples": int(args.max_samples),
        "actual_test_samples": int(test_inputs.shape[0]),
        "n_steps": int(N_STEPS),
        "cpd_topk": float(CPD_TOPK),
        "cpd_top": int(CPD_TOP),
        "compute": dict(compute),
        "attribution_protocol": "TIMING-style prefix accumulation",
        "target": "prefix predicted class",
        "baseline": "zero",
    }
    output = result_path(data, args.fold, args.seed, candidate_name)
    if output.is_file() and not args.force_eval:
        previous = json.loads(output.read_text(encoding="utf-8"))
        if previous.get("status") == "complete" and previous.get(
            "protocol"
        ) == protocol:
            print(f"[SKIP] current result: {output}", flush=True)
            return previous

    gaussianity = latent_gaussianity_metrics(
        flow=flow,
        validation_inputs=validation_inputs,
        device=device,
        batch_size=int(compute["gaussianity_batch"]),
        seed=int(args.seed) + 7919 * int(args.fold),
    )

    classifier = load_classifier(data, args.fold, args.seed, device)
    attribution = evaluate_attribution_metrics(
        classifier=classifier,
        flow=flow,
        test_inputs=test_inputs,
        true_saliency=true_saliency,
        device=device,
        n_steps=int(N_STEPS),
        attribution_batch=int(compute["attribution_batch"]),
        path_batch=int(compute["path_batch"]),
        cpd_batch=int(compute["cpd_batch"]),
        cpd_topk=float(CPD_TOPK),
        cpd_top=int(CPD_TOP),
    )

    result = {
        "status": "complete",
        "protocol": protocol,
        "data": data,
        "fold": int(args.fold),
        "seed": int(args.seed),
        "candidate": candidate_name,
        "checkpoint": str(checkpoint),
        "best_epoch": training_metadata.get("best_epoch"),
        "best_val_nll_per_dim": training_metadata.get(
            "best_val_nll_per_dim"
        ),
        "training_seconds": training_metadata.get("training_seconds"),
        "gaussianity": gaussianity,
        "metrics": attribution,
    }
    atomic_write_json(output, result)
    print(
        f"[RESULT] data={data} fold={args.fold} candidate={candidate_name} "
        f"AUP={attribution['aup']:.6f} "
        f"AUR={attribution['aur']:.6f} "
        f"CPD0={attribution['cpd_zero']:.6f} "
        f"NCE={attribution['nce']:.6f} "
        f"AUC={gaussianity['two_sample_auc']:.6f} "
        f"SW1={gaussianity['sliced_w1']:.6f}",
        flush=True,
    )
    print(f"[SAVE] {output}", flush=True)

    del classifier, flow
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


if __name__ == "__main__":
    run(parse_args())
