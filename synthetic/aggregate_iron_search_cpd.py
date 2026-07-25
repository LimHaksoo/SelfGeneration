#!/usr/bin/env python3
"""Aggregate and rank synthetic IRON CPD@10% candidate results."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.synthetic_iron_search_space import (
    DATASETS,
    canonical_dataset,
    filesystem_name,
    get_candidates,
)


DEFAULT_RESULTS_ROOT = "results_synthetic_iron_search_cpd"


def project_path(value: str | os.PathLike[str]) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS))
    parser.add_argument("--folds", nargs="+", type=int, default=[0])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--results-root", default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--state-candidates", nargs="*", type=int, default=None)
    parser.add_argument("--switch-candidates", nargs="*", type=int, default=None)
    parser.add_argument(
        "--primary-baseline",
        choices=("zero", "average"),
        default="zero",
    )
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args(argv)


def selected_indices(
    data: str,
    total: int,
    state_candidates: Optional[Sequence[int]],
    switch_candidates: Optional[Sequence[int]],
) -> List[int]:
    selected = state_candidates if data == "state" else switch_candidates
    if selected is None or len(selected) == 0:
        return list(range(total))

    result = [int(value) for value in selected]
    invalid = [value for value in result if not 0 <= value < total]
    if invalid:
        raise IndexError(f"Invalid candidate indices for {data}: {invalid}")
    return result


def mean_std(values: pd.Series) -> tuple[float, float]:
    array = values.to_numpy(dtype=np.float64)
    return float(array.mean()), float(array.std(ddof=0))


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    root = project_path(args.results_root)
    datasets = [canonical_dataset(value) for value in args.datasets]

    fold_rows: List[Dict[str, Any]] = []
    missing: List[str] = []

    for data in datasets:
        candidates = get_candidates(data)
        indices = selected_indices(
            data,
            len(candidates),
            args.state_candidates,
            args.switch_candidates,
        )

        for candidate_index in indices:
            candidate = candidates[candidate_index]
            candidate_name = str(candidate["name"])

            for fold in args.folds:
                path = (
                    root
                    / filesystem_name(data)
                    / f"fold{fold}_seed{args.seed}"
                    / f"{candidate_name}.json"
                )
                if not path.is_file():
                    missing.append(str(path))
                    continue

                payload = json.loads(path.read_text(encoding="utf-8"))
                if payload.get("status") != "complete":
                    missing.append(f"{path}:status={payload.get('status')}")
                    continue

                metrics = payload["metrics"]
                row: Dict[str, Any] = {
                    "dataset": data,
                    "fold": int(fold),
                    "candidate_index": int(candidate_index),
                    "candidate": candidate_name,
                    "dims": str(candidate["dims"]),
                    "num_blocks": int(candidate["num_blocks"]),
                    "solver": str(candidate["solver"]),
                    "step_size": candidate["step_size"],
                    "learning_rate": float(candidate["learning_rate"]),
                    "n_steps": int(payload["protocol"]["n_steps"]),
                    "n_samples": int(
                        payload["protocol"]["actual_eval_samples"]
                    ),
                }

                for baseline in ("zero", "average"):
                    if baseline not in metrics:
                        continue
                    block = metrics[baseline]
                    row[f"cpd_{baseline}"] = float(block["cpd_10pct"])
                    row[f"aucc_{baseline}"] = float(block["aucc_10pct"])

                fold_rows.append(row)

    if args.strict and missing:
        preview = "\n".join(missing[:20])
        raise RuntimeError(
            f"Missing {len(missing)} CPD results. First entries:\n{preview}"
        )
    if not fold_rows:
        raise RuntimeError("No completed synthetic IRON CPD results found")

    fold_frame = pd.DataFrame(fold_rows).sort_values(
        ["dataset", "candidate_index", "fold"],
        kind="stable",
    )
    root.mkdir(parents=True, exist_ok=True)
    fold_frame.to_csv(root / "cpd_fold_results.csv", index=False)

    metadata_columns = [
        "dataset",
        "candidate_index",
        "candidate",
        "dims",
        "num_blocks",
        "solver",
        "step_size",
        "learning_rate",
        "n_steps",
        "n_samples",
    ]
    metric_columns = [
        column
        for column in (
            "cpd_zero",
            "aucc_zero",
            "cpd_average",
            "aucc_average",
        )
        if column in fold_frame.columns
    ]

    summary_rows: List[Dict[str, Any]] = []
    for _, group in fold_frame.groupby(
        ["dataset", "candidate_index"],
        sort=False,
    ):
        first = group.iloc[0]
        row: Dict[str, Any] = {
            column: first[column] for column in metadata_columns
        }
        row["n_folds"] = int(len(group))

        for metric in metric_columns:
            mean, std = mean_std(group[metric])
            row[f"{metric}_mean"] = mean
            row[f"{metric}_std"] = std

        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(root / "cpd_summary.csv", index=False)

    primary = f"cpd_{args.primary_baseline}_mean"
    if primary not in summary.columns:
        raise RuntimeError(
            f"Primary baseline {args.primary_baseline!r} was not evaluated"
        )

    ranked_parts: List[pd.DataFrame] = []
    best_payload: Dict[str, Any] = {
        "primary_metric": primary,
        "higher_is_better": True,
        "datasets": {},
    }

    for data in datasets:
        part = summary[summary["dataset"] == data].copy()
        if part.empty:
            continue

        part = part.sort_values(
            [primary, "candidate_index"],
            ascending=[False, True],
            kind="stable",
        )
        part["dataset_rank"] = np.arange(1, len(part) + 1)
        ranked_parts.append(part)

        best = part.iloc[0]
        best_payload["datasets"][data] = {
            "candidate_index": int(best["candidate_index"]),
            "candidate": str(best["candidate"]),
            "cpd_zero_mean": (
                None
                if "cpd_zero_mean" not in part.columns
                else float(best["cpd_zero_mean"])
            ),
            "cpd_average_mean": (
                None
                if "cpd_average_mean" not in part.columns
                else float(best["cpd_average_mean"])
            ),
        }

    ranked = pd.concat(ranked_parts, ignore_index=True)
    ranked.to_csv(root / "cpd_ranked_candidates.csv", index=False)
    atomic_json(root / "cpd_best_by_dataset.json", best_payload)

    visible = [
        column
        for column in (
            "dataset",
            "dataset_rank",
            "candidate_index",
            "candidate",
            "cpd_zero_mean",
            "cpd_zero_std",
            "cpd_average_mean",
            "cpd_average_std",
        )
        if column in ranked.columns
    ]
    print(ranked[visible].to_string(index=False))
    print(f"[SAVE] {root / 'cpd_fold_results.csv'}")
    print(f"[SAVE] {root / 'cpd_summary.csv'}")
    print(f"[SAVE] {root / 'cpd_ranked_candidates.csv'}")
    print(f"[SAVE] {root / 'cpd_best_by_dataset.json'}")
    if missing:
        print(f"[WARN] missing results: {len(missing)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
