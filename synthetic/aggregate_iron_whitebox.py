#!/usr/bin/env python3
"""Aggregate synthetic IRON AUP/AUR/Completeness across five folds."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.synthetic_iron_whitebox import (
    DATASETS,
    FOLDS,
    RESULT_ROOT,
    SEED,
    canonical_dataset,
    filesystem_name,
)


def atomic_write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError("Cannot write an empty CSV")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS))
    parser.add_argument("--folds", nargs="+", type=int, default=list(FOLDS))
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--results-root", default=RESULT_ROOT)
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args(argv)


def _mean_std(values: Sequence[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return float("nan"), float("nan")
    # Existing synthetic result aggregation uses np.std(ddof=0).
    return float(array.mean()), float(array.std(ddof=0))


def _latex(mean: float, std: float) -> str:
    return f"${mean:.4f} \\pm {std:.4f}$"


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    root = Path(args.results_root)
    fold_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []
    missing: List[str] = []

    for raw_data in args.datasets:
        data = canonical_dataset(raw_data)
        aup_values: List[float] = []
        aur_values: List[float] = []
        ce_values: List[float] = []
        nce_values: List[float] = []

        for fold in args.folds:
            path = (
                root
                / filesystem_name(data)
                / f"fold{fold}_seed{args.seed}"
                / "iron.json"
            )
            if not path.is_file():
                missing.append(str(path))
                continue

            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("status") != "complete":
                missing.append(str(path))
                continue

            completeness = payload.get("completeness")
            if not isinstance(completeness, dict):
                missing.append(f"{path}: completeness missing")
                continue

            aup_value = float(payload["aup"])
            aur_value = float(payload["aur"])
            ce_value = float(completeness["ce"])
            nce_value = float(completeness["nce"])

            aup_values.append(aup_value)
            aur_values.append(aur_value)
            ce_values.append(ce_value)
            nce_values.append(nce_value)

            fold_rows.append(
                {
                    "dataset": data,
                    "method": "IRON",
                    "fold": int(fold),
                    "seed": int(args.seed),
                    "n_samples": int(payload["protocol"]["n_samples"]),
                    "AUP": aup_value,
                    "AUR": aur_value,
                    "CE": ce_value,
                    "NCE": nce_value,
                    "median_abs_error": float(
                        completeness["median_abs_error"]
                    ),
                    "candidate": payload["protocol"]["candidate"],
                }
            )

        if args.strict and len(aup_values) != len(args.folds):
            raise RuntimeError(
                f"Missing folds for {data}: got {len(aup_values)}, "
                f"expected {len(args.folds)}"
            )
        if not aup_values:
            continue

        aup_mean, aup_std = _mean_std(aup_values)
        aur_mean, aur_std = _mean_std(aur_values)
        ce_mean, ce_std = _mean_std(ce_values)
        nce_mean, nce_std = _mean_std(nce_values)

        summary_rows.append(
            {
                "dataset": data,
                "method": "IRON",
                "n_folds": int(len(aup_values)),
                "AUP_mean": aup_mean,
                "AUP_std": aup_std,
                "AUR_mean": aur_mean,
                "AUR_std": aur_std,
                "CE_mean": ce_mean,
                "CE_std": ce_std,
                "NCE_mean": nce_mean,
                "NCE_std": nce_std,
                "AUP_latex": _latex(aup_mean, aup_std),
                "AUR_latex": _latex(aur_mean, aur_std),
                "CE_latex": _latex(ce_mean, ce_std),
                "NCE_latex": _latex(nce_mean, nce_std),
            }
        )

    root.mkdir(parents=True, exist_ok=True)
    if fold_rows:
        atomic_write_csv(root / "iron_fold_metrics.csv", fold_rows)
    if summary_rows:
        atomic_write_csv(root / "iron_summary.csv", summary_rows)

    # The paper now uses NCE as the headline Completeness metric.
    tex_lines = [
        (
            f"{row['dataset']} & IRON & {row['AUP_latex']} "
            f"& {row['AUR_latex']} & {row['NCE_latex']} \\\\"
        )
        for row in summary_rows
    ]
    (root / "iron_table_rows.tex").write_text(
        "\n".join(tex_lines) + ("\n" if tex_lines else ""),
        encoding="utf-8",
    )

    for row in summary_rows:
        print(
            f"{row['dataset']:14s} IRON "
            f"AUP={row['AUP_mean']:.4f}±{row['AUP_std']:.4f} "
            f"AUR={row['AUR_mean']:.4f}±{row['AUR_std']:.4f} "
            f"CE={row['CE_mean']:.4f}±{row['CE_std']:.4f} "
            f"NCE={row['NCE_mean']:.4f}±{row['NCE_std']:.4f}"
        )
    if missing:
        print(f"[WARN] missing results: {len(missing)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
