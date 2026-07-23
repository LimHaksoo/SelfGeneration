from __future__ import annotations

from typing import Dict, Tuple

FORMAT_VERSION = 7
DATASETS: Tuple[str, ...] = ("wafer", "freezer")
METHODS: Tuple[str, ...] = ("IG", "IRON")
FOLDS: Tuple[int, ...] = (0, 1, 2, 3, 4)
SEED = 42

K_VALUES: Tuple[int, ...] = (5, 10, 20, 50, 100, 200)
REFERENCE_K = 1000

OUTPUT_ROOT = "results_finite_step_convergence"
CHECKPOINT_ROOT = "model"
FFJORD_ROOT = "third_party/ffjord_official"
MAIN_TABLE_COMP_ROOT = "results_our/comp"
MAIN_TABLE_IRON_METRIC_ROOT = "results_cnf_checkpoint_metrics"

# These are the exact IRON candidates used for the reported main-table rows.
# Candidate selection is name-based so that editing candidate-list order cannot
# silently change the RQ3 model.
MAIN_TABLE_IRON_CANDIDATE: Dict[str, str] = {
    "wafer": "rk4_64x2_b1_s010",
    "freezer": "rk4_64x2_b1_s010_lr1e-3",
}

# Main-table baseline generation used TESTBS=128.  Keeping the same outer batch
# also minimizes irrelevant floating-point differences in the K=100 parity test.
ATTRIBUTION_BATCH: Dict[str, int] = {
    "wafer": 128,
    "freezer": 128,
}

# NFIntegratedGradients decodes path points in this many-point chunks.
PATH_POINT_BATCH: Dict[str, int] = {
    "wafer": 24,
    "freezer": 24,
}

# Captum K=100 is intentionally evaluated without internal batching, exactly as
# in main_td.py.  Higher-resolution references use this cap to avoid OOM.
IG_INTERNAL_BATCH_SIZE: Dict[str, int] = {
    "wafer": 2048,
    "freezer": 2048,
}

COMPLETENESS_EPS = 1e-8
ATTRIBUTION_NORM_EPS = 1e-8
