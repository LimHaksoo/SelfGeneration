"""Shared loaders for synthetic IRON training and white-box evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from torch import Tensor

from configs.synthetic_iron_whitebox import (
    CHECKPOINT_ROOT,
    FFJORD_ROOT,
    canonical_dataset,
    filesystem_name,
    get_candidate,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class SyntheticBundle:
    train_x: Tensor
    train_y: Tensor
    test_x: Tensor
    test_y: Tensor
    true_saliency: Tensor


def _preprocess_or_prepare(datamodule, split: str) -> Dict[str, Tensor]:
    """Load one split and invoke ``prepare_data`` once when required."""

    try:
        return datamodule.preprocess(split=split)
    except (FileNotFoundError, RuntimeError) as first_error:
        prepare = getattr(datamodule, "prepare_data", None)
        if not callable(prepare):
            raise

        prepare()

        try:
            return datamodule.preprocess(split=split)
        except FileNotFoundError as second_error:
            resolved = getattr(datamodule, "data_dir", "<unknown>")
            raise FileNotFoundError(
                f"Could not prepare synthetic split={split!r}. "
                f"Resolved DataModule data_dir={resolved!r}. "
                f"First error: {first_error}; second error: {second_error}"
            ) from second_error


def load_bundle(data: str, fold: int, seed: int) -> SyntheticBundle:
    dataset = canonical_dataset(data)

    if dataset == "state":
        from tint.datasets import HMM

        # Tint HMM appends ``data/hmm`` to the supplied root.  Passing
        # ``data/hmm`` therefore creates the invalid nested path
        # ``data/hmm/data/hmm``.  The repository root resolves to the intended
        # ``<repo>/data/hmm`` location.
        datamodule = HMM(
            n_folds=5,
            fold=int(fold),
            seed=int(seed),
            data_dir=str(PROJECT_ROOT),
        )
    elif dataset == "switch-feature":
        from synthetic.switchstate.switchloader import Switch

        datamodule = Switch(
            n_folds=5,
            fold=int(fold),
            seed=int(seed),
            data_dir=str(PROJECT_ROOT / "data" / "switchstate"),
        )
    else:  # pragma: no cover
        raise AssertionError(dataset)

    train = _preprocess_or_prepare(datamodule, "train")
    test = _preprocess_or_prepare(datamodule, "test")
    true_saliency = datamodule.true_saliency(split="test")

    train_x = train["x"].detach().cpu().float()
    train_y = train["y"].detach().cpu().long()
    test_x = test["x"].detach().cpu().float()
    test_y = test["y"].detach().cpu().long()
    true_saliency = true_saliency.detach().cpu().long()

    for name, tensor in (
        ("train_x", train_x),
        ("test_x", test_x),
        ("true_saliency", true_saliency),
    ):
        if tensor.ndim != 3:
            raise ValueError(
                f"{dataset} {name} must be [N,T,D], got {tuple(tensor.shape)}"
            )

    if tuple(test_x.shape) != tuple(true_saliency.shape):
        raise ValueError(
            f"{dataset} true-saliency shape {tuple(true_saliency.shape)} "
            f"does not match test shape {tuple(test_x.shape)}"
        )

    return SyntheticBundle(
        train_x=train_x,
        train_y=train_y,
        test_x=test_x,
        test_y=test_y,
        true_saliency=true_saliency,
    )


def _stratification_labels(y: Tensor) -> np.ndarray:
    labels = y.detach().cpu()
    if labels.ndim == 1:
        selected = labels
    else:
        selected = labels.reshape(labels.shape[0], -1)[:, -1]
    return selected.long().numpy()


def split_train_validation(
    x: Tensor,
    y: Tensor,
    *,
    fold: int,
    seed: int,
) -> Tuple[Tensor, Tensor]:
    labels = _stratification_labels(y)
    unique, counts = np.unique(labels, return_counts=True)

    if unique.size >= 2 and int(counts.min()) >= 5:
        splitter = StratifiedKFold(
            n_splits=5,
            shuffle=True,
            random_state=int(seed),
        )
        train_index, val_index = list(
            splitter.split(np.zeros(len(labels), dtype=np.float32), labels)
        )[int(fold)]
    else:
        generator = torch.Generator().manual_seed(int(seed) + int(fold))
        permutation = torch.randperm(len(x), generator=generator)
        val_size = max(1, int(round(0.2 * len(x))))
        val_index = permutation[:val_size].numpy()
        train_index = permutation[val_size:].numpy()

    return (
        x[torch.from_numpy(np.asarray(train_index))].contiguous(),
        x[torch.from_numpy(np.asarray(val_index))].contiguous(),
    )


def classifier_checkpoint(data: str, fold: int, seed: int) -> Path:
    dataset = canonical_dataset(data)
    if dataset == "state":
        return PROJECT_ROOT / "model" / "hmm" / f"classifier_{fold}_{seed}"
    return (
        PROJECT_ROOT
        / "model"
        / "switch_feature"
        / f"classifier_{fold}_{seed}"
    )


def load_classifier(
    data: str,
    fold: int,
    seed: int,
    device: torch.device,
):
    dataset = canonical_dataset(data)

    if dataset == "state":
        from synthetic.hmm.classifier import StateClassifierNet

        classifier = StateClassifierNet(
            feature_size=3,
            n_state=2,
            hidden_size=200,
            regres=True,
            loss="cross_entropy",
            lr=1e-4,
            l2=1e-3,
        )
    else:
        from synthetic.switchstate.classifier import SpikeClassifierNet

        classifier = SpikeClassifierNet(
            feature_size=3,
            n_state=2,
            hidden_size=200,
            regres=True,
            loss="cross_entropy",
            lr=1e-4,
            l2=1e-3,
        )

    checkpoint = classifier_checkpoint(dataset, fold, seed)
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"Synthetic classifier checkpoint not found: {checkpoint}. "
            "Train it first with the repository's original synthetic script."
        )

    payload = torch.load(checkpoint, map_location="cpu")
    state_dict = (
        payload.get("state_dict", payload)
        if isinstance(payload, dict)
        else payload
    )
    classifier.load_state_dict(state_dict, strict=True)
    classifier.to(device).eval()
    for parameter in classifier.parameters():
        parameter.requires_grad_(False)
    return classifier


def flow_checkpoint(data: str, fold: int, seed: int) -> Path:
    dataset = canonical_dataset(data)
    candidate = get_candidate(dataset)
    return (
        PROJECT_ROOT
        / CHECKPOINT_ROOT
        / filesystem_name(dataset)
        / f"fold{int(fold)}_seed{int(seed)}"
        / f"{candidate['name']}.pt"
    )


def ffjord_root() -> Path:
    return PROJECT_ROOT / FFJORD_ROOT
