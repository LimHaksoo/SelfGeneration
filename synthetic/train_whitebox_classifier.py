#!/usr/bin/env python3
"""Train the original TIMING synthetic classifier for one dataset/fold.

The IRON flow and the task classifier are separate models.  This script
reproduces the classifier architecture and 50-epoch training protocol used by
``synthetic/hmm/main.py`` and ``synthetic/switchstate/main.py``, while using
absolute data paths so the custom synthetic loaders do not duplicate relative
paths.
"""

from __future__ import annotations

import argparse
import gc
import os
import random
import sys
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
import torch
from pytorch_lightning import Trainer, seed_everything

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.synthetic_iron_whitebox import canonical_dataset
from synthetic.iron_whitebox_common import (
    classifier_checkpoint,
    load_classifier,
)


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    seed_everything(seed, workers=True)


def build_training_objects(data: str, fold: int, seed: int):
    dataset = canonical_dataset(data)

    if dataset == "state":
        from tint.datasets import HMM
        from synthetic.hmm.classifier import StateClassifierNet

        # Tint's HMM loader appends data/hmm internally.
        datamodule = HMM(
            n_folds=5,
            fold=int(fold),
            seed=int(seed),
            data_dir=str(PROJECT_ROOT),
        )
        classifier = StateClassifierNet(
            feature_size=3,
            n_state=2,
            hidden_size=200,
            regres=True,
            loss="cross_entropy",
            lr=1e-4,
            l2=1e-3,
        )
    elif dataset == "switch-feature":
        from synthetic.switchstate.classifier import SpikeClassifierNet
        from synthetic.switchstate.switchloader import Switch

        data_dir = (PROJECT_ROOT / "data" / "switchstate").resolve()
        data_dir.mkdir(parents=True, exist_ok=True)
        datamodule = Switch(
            n_folds=5,
            fold=int(fold),
            seed=int(seed),
            data_dir=str(data_dir),
        )
        classifier = SpikeClassifierNet(
            feature_size=3,
            n_state=2,
            hidden_size=200,
            regres=True,
            loss="cross_entropy",
            lr=1e-4,
            l2=1e-3,
        )
    else:  # pragma: no cover
        raise AssertionError(dataset)

    return dataset, datamodule, classifier


def trainer_device(device: str) -> Tuple[str, object]:
    value = str(device).strip()
    if value.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        index = int(value.split(":", 1)[1]) if ":" in value else 0
        return "cuda", [index]
    return "cpu", 1


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--fold", type=int, required=True, choices=range(5))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--deterministic", action="store_true")
    args = parser.parse_args(argv)
    if args.epochs < 1:
        parser.error("--epochs must be positive")
    return args


def run(args: argparse.Namespace) -> Path:
    os.chdir(PROJECT_ROOT)
    set_seed(args.seed)

    dataset = canonical_dataset(args.data)
    checkpoint = classifier_checkpoint(dataset, args.fold, args.seed)
    if checkpoint.is_file() and not args.force:
        print(f"[SKIP] classifier exists: {checkpoint}", flush=True)
        return checkpoint

    dataset, datamodule, classifier = build_training_objects(
        dataset,
        args.fold,
        args.seed,
    )
    accelerator, devices = trainer_device(args.device)

    # This is the classifier-training protocol in the original synthetic code.
    trainer = Trainer(
        max_epochs=int(args.epochs),
        accelerator=accelerator,
        devices=devices,
        deterministic=bool(args.deterministic),
        logger=False,
        enable_checkpointing=False,
    )

    print(
        f"[TRAIN] data={dataset} fold={args.fold} epochs={args.epochs} "
        f"checkpoint={checkpoint}",
        flush=True,
    )
    trainer.fit(classifier, datamodule=datamodule)

    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    temporary = checkpoint.with_name(checkpoint.name + f".{os.getpid()}.tmp")
    torch.save(classifier.state_dict(), temporary)
    os.replace(temporary, checkpoint)

    # Strict reload verifies that eval_iron_whitebox.py can consume the file.
    reload_device = torch.device(args.device)
    reloaded = load_classifier(
        dataset,
        args.fold,
        args.seed,
        reload_device,
    )
    del reloaded

    print(f"[SAVE] {checkpoint}", flush=True)
    del classifier, trainer, datamodule
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return checkpoint


if __name__ == "__main__":
    run(parse_args())
