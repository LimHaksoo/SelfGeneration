#!/usr/bin/env python3
"""Dataset-specific search space for synthetic IRON experiments.

The candidates deliberately cover the FFJORD configurations that were useful
in the real-data experiments: the 64-width one-block RK4 model, a smaller RK4
step, the 128-width Epilepsy branch, the two-block Boiler/PAM branch, lower
learning rates, and adaptive Dopri5 variants.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Mapping, Tuple

from configs.official_cnf_fullfold_by_dataset import cnf_candidate


FORMAT_VERSION = 1
DATASETS: Tuple[str, ...] = ("state", "switch-feature")
FOLDS: Tuple[int, ...] = (0, 1, 2, 3, 4)
SEED = 42

N_STEPS = 100
CPD_TOPK = 0.10
CPD_TOP = 0
DEFAULT_MAX_SAMPLES = 50

CHECKPOINT_ROOT = "model/synthetic_iron_search"
RESULT_ROOT = "results_synthetic_iron_search"
FFJORD_ROOT = "third_party/ffjord_official"

SEARCH_PROTOCOL: Dict[str, int] = {
    "epochs": 500,
    "early_stopping_epochs": 50,
    "validate_every_epochs": 1,
    "log_every_steps": 10,
}

COMPUTE_BY_DATASET: Dict[str, Dict[str, int]] = {
    # A full synthetic test split is normally 200 examples.  The attribution
    # batch therefore evaluates the search subset in one outer batch while the
    # path dimension remains safely chunked.
    "state": {
        "train_batch": 512,
        "nll_eval_batch": 512,
        "gaussianity_batch": 512,
        "attribution_batch": 200,
        "path_batch": 8,
        "cpd_batch": 200,
    },
    "switch-feature": {
        "train_batch": 512,
        "nll_eval_batch": 512,
        "gaussianity_batch": 512,
        "attribution_batch": 200,
        "path_batch": 8,
        "cpd_batch": 200,
    },
}


def _candidate(
    *,
    name: str,
    dims: str,
    num_blocks: int,
    solver: str,
    step_size: float | None,
    learning_rate: float,
) -> Dict[str, Any]:
    return cnf_candidate(
        name=name,
        dims=dims,
        num_blocks=num_blocks,
        solver=solver,
        step_size=step_size,
        learning_rate=learning_rate,
    )


CANDIDATES_BY_DATASET: Dict[str, List[Dict[str, Any]]] = {
    "state": [
        # Existing synthetic default and the real-data Wafer/Freezer winner.
        _candidate(
            name="rk4_64x2_b1_s010_lr1e-3",
            dims="64-64",
            num_blocks=1,
            solver="rk4",
            step_size=0.10,
            learning_rate=1e-3,
        ),
        # Learning-rate ablation around the default architecture.
        _candidate(
            name="rk4_64x2_b1_s010_lr5e-4",
            dims="64-64",
            num_blocks=1,
            solver="rk4",
            step_size=0.10,
            learning_rate=5e-4,
        ),
        # Finer fixed-step solver.
        _candidate(
            name="rk4_64x2_b1_s005_lr1e-3",
            dims="64-64",
            num_blocks=1,
            solver="rk4",
            step_size=0.05,
            learning_rate=1e-3,
        ),
        # Wider Epilepsy-style branch.
        _candidate(
            name="rk4_128x2_b1_s005_lr1e-3",
            dims="128-128",
            num_blocks=1,
            solver="rk4",
            step_size=0.05,
            learning_rate=1e-3,
        ),
        # Deeper Boiler-style branch.
        _candidate(
            name="rk4_64x2_b2_s010_lr5e-4",
            dims="64-64",
            num_blocks=2,
            solver="rk4",
            step_size=0.10,
            learning_rate=5e-4,
        ),
        # Lower-LR deep branch, motivated by the PAM search.
        _candidate(
            name="rk4_64x2_b2_s010_lr1e-4",
            dims="64-64",
            num_blocks=2,
            solver="rk4",
            step_size=0.10,
            learning_rate=1e-4,
        ),
        # Adaptive-solver branches.
        _candidate(
            name="dopri5_64x2_b1_lr1e-3",
            dims="64-64",
            num_blocks=1,
            solver="dopri5",
            step_size=None,
            learning_rate=1e-3,
        ),
        _candidate(
            name="dopri5_64x2_b2_lr5e-4",
            dims="64-64",
            num_blocks=2,
            solver="dopri5",
            step_size=None,
            learning_rate=5e-4,
        ),
    ],
    "switch-feature": [
        _candidate(
            name="rk4_64x2_b1_s010_lr1e-3",
            dims="64-64",
            num_blocks=1,
            solver="rk4",
            step_size=0.10,
            learning_rate=1e-3,
        ),
        _candidate(
            name="rk4_64x2_b1_s010_lr5e-4",
            dims="64-64",
            num_blocks=1,
            solver="rk4",
            step_size=0.10,
            learning_rate=5e-4,
        ),
        # Switch-feature was less stable in the initial runs, so include a
        # lower-LR one-block branch explicitly.
        _candidate(
            name="rk4_64x2_b1_s010_lr1e-4",
            dims="64-64",
            num_blocks=1,
            solver="rk4",
            step_size=0.10,
            learning_rate=1e-4,
        ),
        _candidate(
            name="rk4_64x2_b1_s005_lr5e-4",
            dims="64-64",
            num_blocks=1,
            solver="rk4",
            step_size=0.05,
            learning_rate=5e-4,
        ),
        _candidate(
            name="rk4_128x2_b1_s005_lr5e-4",
            dims="128-128",
            num_blocks=1,
            solver="rk4",
            step_size=0.05,
            learning_rate=5e-4,
        ),
        _candidate(
            name="rk4_64x2_b2_s010_lr5e-4",
            dims="64-64",
            num_blocks=2,
            solver="rk4",
            step_size=0.10,
            learning_rate=5e-4,
        ),
        _candidate(
            name="rk4_64x2_b2_s010_lr1e-4",
            dims="64-64",
            num_blocks=2,
            solver="rk4",
            step_size=0.10,
            learning_rate=1e-4,
        ),
        _candidate(
            name="dopri5_64x2_b1_lr5e-4",
            dims="64-64",
            num_blocks=1,
            solver="dopri5",
            step_size=None,
            learning_rate=5e-4,
        ),
    ],
}

# Search ranking.  Attribution quality is primary; Gaussianity is a secondary
# regularizer used to break ties between comparably faithful paths.
DOWNSTREAM_METRICS: Dict[str, str] = {
    "aup": "max",
    "aur": "max",
    "cpd_zero": "max",
    "nce": "min",
}
GAUSSIANITY_METRICS: Dict[str, str] = {
    "two_sample_auc_gap": "min",
    "sliced_w1": "min",
    "radius_w1_per_dim": "min",
}
DOWNSTREAM_WEIGHT = 0.75
GAUSSIANITY_WEIGHT = 0.25


def canonical_dataset(value: str) -> str:
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


def filesystem_name(data: str) -> str:
    return canonical_dataset(data).replace("-", "_")


def get_candidates(data: str) -> List[Dict[str, Any]]:
    dataset = canonical_dataset(data)
    candidates = deepcopy(CANDIDATES_BY_DATASET[dataset])
    names = [str(item["name"]) for item in candidates]
    if len(names) != len(set(names)):
        raise ValueError(f"Duplicate candidate names for {dataset}: {names}")
    return candidates


def get_candidate(data: str, name: str) -> Dict[str, Any]:
    matches = [item for item in get_candidates(data) if item["name"] == name]
    if len(matches) != 1:
        raise KeyError(
            f"Candidate {name!r} was not resolved exactly for {data}; "
            f"available={[item['name'] for item in get_candidates(data)]}"
        )
    return matches[0]


def get_compute(data: str) -> Dict[str, int]:
    return deepcopy(COMPUTE_BY_DATASET[canonical_dataset(data)])


def validate() -> None:
    if not np_is_close(DOWNSTREAM_WEIGHT + GAUSSIANITY_WEIGHT, 1.0):
        raise ValueError("Selection weights must sum to one")
    for dataset in DATASETS:
        get_candidates(dataset)
        compute = COMPUTE_BY_DATASET[dataset]
        for key in (
            "train_batch",
            "nll_eval_batch",
            "gaussianity_batch",
            "attribution_batch",
            "path_batch",
            "cpd_batch",
        ):
            if int(compute.get(key, 0)) < 1:
                raise ValueError(f"Invalid {key} for {dataset}")


def np_is_close(left: float, right: float, tolerance: float = 1e-12) -> bool:
    return abs(float(left) - float(right)) <= float(tolerance)


validate()
