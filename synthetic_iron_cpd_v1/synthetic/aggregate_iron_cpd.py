#!/usr/bin/env python3
"""Aggregate standalone synthetic IRON CPD@10% results across folds."""

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
    SEED,
    canonical_dataset,
    filesystem_name,
)


DEFAULT_RESULTS_ROOT = "results_synthetic_iron_cpd"
BASELINE_ORDER = ("Average", "Zeros")


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
    parser.add_argument("--results-root", default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    root = Path(args.results_root)

    fold_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []
    missing: List[str] = []

    for raw_data in args.datasets:
        data = canonical_dataset(raw_data)
        values_by_baseline: Dict[str, List[float]] = {
            baseline: [] for baseline in BASELINE_ORDER
        }

        for fold in args.folds:
            path = (
                root
                / filesystem_name(data)
                / f"fold{fold}_seed{args.seed}"
                / "iron_cpd.json"
            )
            if not path.is_file():
                missing.append(str(path))
                continue

            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("status") != "complete":
                missing.append(str(path))
                continue

            results = payload.get("results")
            if not isinstance(results, Mapping):
                missing.append(f"{path}:results")
                continue

            for raw_name, item in results.items():
                if not isinstance(item, Mapping):
                    continue
                display_name = str(item.get("baseline", raw_name))
                if display_name not in values_by_baseline:
                    continue
                cpd = float(item["cpd"])
                values_by_baseline[display_name].append(cpd)
                fold_rows.append(
                    {
                        "dataset": data,
                        "method": "IRON",
                        "baseline": display_name,
                        "fold": int(fold),
                        "seed": int(args.seed),
                        "topk": float(payload["protocol"]["topk"]),
                        "CPD": cpd,
                        "saliency_path": payload.get("saliency_path"),
                        "saliency_cache_status": payload.get(
                            "saliency_cache_status"
                        ),
                    }
                )

        for baseline in BASELINE_ORDER:
            values = values_by_baseline[baseline]
            if args.strict and len(values) != len(args.folds):
                raise RuntimeError(
                    f"Missing folds for {data}/{baseline}: got {len(values)}, "
                    f"expected {len(args.folds)}"
                )
            if not values:
                continue

            array = np.asarray(values, dtype=np.float64)
            mean = float(array.mean())
            # Existing synthetic result aggregation uses population std.
            std = float(array.std(ddof=0))
            summary_rows.append(
                {
                    "dataset": data,
                    "method": "IRON",
                    "baseline": baseline,
                    "n_folds": int(array.size),
                    "CPD_mean": mean,
                    "CPD_std": std,
                    "CPD_latex": f"${mean:.4f} \\pm {std:.4f}$",
                }
            )

    root.mkdir(parents=True, exist_ok=True)
    if fold_rows:
        atomic_write_csv(root / "iron_cpd_fold.csv", fold_rows)
    if summary_rows:
        atomic_write_csv(root / "iron_cpd_summary.csv", summary_rows)

    tex_lines = [
        (
            f"{row['dataset']} & IRON & {row['baseline']} "
            f"& {row['CPD_latex']} " + r"\\"
        )
        for row in summary_rows
    ]
    (root / "iron_cpd_table_rows.tex").write_text(
        "\n".join(tex_lines) + ("\n" if tex_lines else ""),
        encoding="utf-8",
    )

    for row in summary_rows:
        print(
            f"{row['dataset']:14s} IRON {row['baseline']:7s} "
            f"CPD={row['CPD_mean']:.4f}±{row['CPD_std']:.4f}"
        )
    if missing:
        print(f"[WARN] missing results: {len(missing)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
