#!/usr/bin/env python3
"""Aggregate and rank synthetic IRON hyperparameter-search results."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.synthetic_iron_search_space import (
    DATASETS,
    DOWNSTREAM_METRICS,
    DOWNSTREAM_WEIGHT,
    FOLDS,
    GAUSSIANITY_METRICS,
    GAUSSIANITY_WEIGHT,
    RESULT_ROOT,
    SEED,
    canonical_dataset,
    filesystem_name,
    get_candidates,
)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS))
    parser.add_argument("--folds", nargs="+", type=int, default=[0])
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--results-root", default=RESULT_ROOT)
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args(argv)


def _result_path(
    root: Path,
    data: str,
    fold: int,
    seed: int,
    candidate: str,
) -> Path:
    return (
        root
        / filesystem_name(data)
        / f"fold{int(fold)}_seed{int(seed)}"
        / f"{candidate}.json"
    )


def _atomic_dataframe(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def _load_fold_rows(args: argparse.Namespace) -> tuple[pd.DataFrame, List[str]]:
    root = Path(args.results_root)
    rows: List[Dict[str, Any]] = []
    missing: List[str] = []

    for raw_data in args.datasets:
        data = canonical_dataset(raw_data)
        for candidate in get_candidates(data):
            name = str(candidate["name"])
            for fold in args.folds:
                path = _result_path(root, data, fold, args.seed, name)
                if not path.is_file():
                    missing.append(str(path))
                    continue
                payload = json.loads(path.read_text(encoding="utf-8"))
                if payload.get("status") != "complete":
                    missing.append(f"{path}: status={payload.get('status')}")
                    continue

                metrics = payload["metrics"]
                gaussianity = payload["gaussianity"]
                rows.append(
                    {
                        "dataset": data,
                        "candidate": name,
                        "fold": int(fold),
                        "seed": int(args.seed),
                        "n_test": int(metrics["n_test"]),
                        "best_epoch": payload.get("best_epoch"),
                        "best_val_nll_per_dim": payload.get(
                            "best_val_nll_per_dim"
                        ),
                        "training_seconds": payload.get("training_seconds"),
                        "aup": float(metrics["aup"]),
                        "aur": float(metrics["aur"]),
                        "ce": float(metrics["ce"]),
                        "nce": float(metrics["nce"]),
                        "cpd_zero": float(metrics["cpd_zero"]),
                        "cpd_average": float(metrics["cpd_average"]),
                        "two_sample_auc": float(
                            gaussianity["two_sample_auc"]
                        ),
                        "two_sample_auc_gap": float(
                            gaussianity["two_sample_auc_gap"]
                        ),
                        "sliced_w1": float(gaussianity["sliced_w1"]),
                        "radius_w1_per_dim": float(
                            gaussianity["radius_w1_per_dim"]
                        ),
                        "latent_mean_abs": float(
                            gaussianity["latent_mean_abs"]
                        ),
                        "latent_std_abs_error": float(
                            gaussianity["latent_std_abs_error"]
                        ),
                        "result_path": str(path),
                    }
                )

    return pd.DataFrame(rows), missing


def _mean_std_summary(folds: pd.DataFrame) -> pd.DataFrame:
    numeric = [
        "n_test",
        "best_epoch",
        "best_val_nll_per_dim",
        "training_seconds",
        "aup",
        "aur",
        "ce",
        "nce",
        "cpd_zero",
        "cpd_average",
        "two_sample_auc",
        "two_sample_auc_gap",
        "sliced_w1",
        "radius_w1_per_dim",
        "latent_mean_abs",
        "latent_std_abs_error",
    ]
    records: List[Dict[str, Any]] = []
    for (dataset, candidate), group in folds.groupby(
        ["dataset", "candidate"], sort=False
    ):
        row: Dict[str, Any] = {
            "dataset": dataset,
            "candidate": candidate,
            "n_folds": int(len(group)),
        }
        for column in numeric:
            values = pd.to_numeric(group[column], errors="coerce").dropna().to_numpy(
                dtype=np.float64
            )
            if values.size:
                row[f"{column}_mean"] = float(values.mean())
                row[f"{column}_std"] = float(values.std(ddof=0))
            else:
                row[f"{column}_mean"] = float("nan")
                row[f"{column}_std"] = float("nan")
        records.append(row)
    return pd.DataFrame(records)


def _rank_candidates(summary: pd.DataFrame) -> pd.DataFrame:
    ranked_parts: List[pd.DataFrame] = []
    for dataset, group in summary.groupby("dataset", sort=False):
        current = group.copy()

        downstream_rank_columns: List[str] = []
        for metric, direction in DOWNSTREAM_METRICS.items():
            source = f"{metric}_mean"
            destination = f"rank_{metric}"
            current[destination] = current[source].rank(
                ascending=direction == "min",
                method="average",
                na_option="bottom",
            )
            downstream_rank_columns.append(destination)

        gaussianity_rank_columns: List[str] = []
        for metric, direction in GAUSSIANITY_METRICS.items():
            source = f"{metric}_mean"
            destination = f"rank_{metric}"
            current[destination] = current[source].rank(
                ascending=direction == "min",
                method="average",
                na_option="bottom",
            )
            gaussianity_rank_columns.append(destination)

        current["downstream_rank"] = current[
            downstream_rank_columns
        ].mean(axis=1)
        current["gaussianity_rank"] = current[
            gaussianity_rank_columns
        ].mean(axis=1)
        current["selection_rank"] = (
            float(DOWNSTREAM_WEIGHT) * current["downstream_rank"]
            + float(GAUSSIANITY_WEIGHT) * current["gaussianity_rank"]
        )
        current = current.sort_values(
            ["selection_rank", "downstream_rank", "best_val_nll_per_dim_mean"],
            ascending=[True, True, True],
            kind="stable",
        ).reset_index(drop=True)
        current["position"] = np.arange(1, len(current) + 1)
        ranked_parts.append(current)

    if not ranked_parts:
        return pd.DataFrame()
    return pd.concat(ranked_parts, ignore_index=True)


def _write_best_candidates(root: Path, ranking: pd.DataFrame) -> None:
    payload: Dict[str, Any] = {
        "selection": {
            "downstream_metrics": DOWNSTREAM_METRICS,
            "gaussianity_metrics": GAUSSIANITY_METRICS,
            "downstream_weight": DOWNSTREAM_WEIGHT,
            "gaussianity_weight": GAUSSIANITY_WEIGHT,
        },
        "datasets": {},
    }
    snippet_lines = ["BEST_SYNTHETIC_IRON_CANDIDATE = {"]

    for dataset, group in ranking.groupby("dataset", sort=False):
        best = group.iloc[0].to_dict()
        serializable = {
            key: (None if pd.isna(value) else value.item() if hasattr(value, "item") else value)
            for key, value in best.items()
        }
        payload["datasets"][dataset] = serializable
        snippet_lines.append(
            f"    {dataset!r}: {str(best['candidate'])!r},"
        )

        per_dataset = root / filesystem_name(dataset) / "best_candidate.json"
        per_dataset.parent.mkdir(parents=True, exist_ok=True)
        per_dataset.write_text(
            json.dumps(serializable, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    snippet_lines.append("}")
    (root / "best_candidates.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (root / "best_candidates_snippet.py").write_text(
        "\n".join(snippet_lines) + "\n",
        encoding="utf-8",
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    folds, missing = _load_fold_rows(args)
    if args.strict and missing:
        preview = "\n".join(missing[:20])
        raise RuntimeError(
            f"Missing {len(missing)} search results. First entries:\n{preview}"
        )
    if folds.empty:
        raise RuntimeError("No complete synthetic IRON search results were found")

    summary = _mean_std_summary(folds)
    ranking = _rank_candidates(summary)
    root = Path(args.results_root)
    root.mkdir(parents=True, exist_ok=True)
    _atomic_dataframe(root / "fold_results.csv", folds)
    _atomic_dataframe(root / "candidate_summary.csv", summary)
    _atomic_dataframe(root / "ranking.csv", ranking)
    _write_best_candidates(root, ranking)

    display_columns = [
        "dataset",
        "position",
        "candidate",
        "selection_rank",
        "downstream_rank",
        "gaussianity_rank",
        "aup_mean",
        "aur_mean",
        "cpd_zero_mean",
        "nce_mean",
        "two_sample_auc_mean",
        "sliced_w1_mean",
        "best_val_nll_per_dim_mean",
    ]
    print(ranking[display_columns].to_string(index=False))
    print(f"[SAVE] {root / 'ranking.csv'}")
    print(f"[SAVE] {root / 'best_candidates.json'}")
    if missing:
        print(f"[WARN] missing results: {len(missing)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
