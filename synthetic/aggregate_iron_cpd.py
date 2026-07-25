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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.synthetic_iron_whitebox import (
    DATASETS,
    FOLDS,
    SEED,
    canonical_dataset,
    filesystem_name,
)

DEFAULT_RESULTS_ROOT = "results_synthetic_iron_cpd"


def canonical_baseline(value: str) -> str:
    key = str(value).strip().casefold()
    mapping = {
        "average": "Average",
        "avg": "Average",
        "zero": "Zeros",
        "zeros": "Zeros",
    }
    if key not in mapping:
        raise ValueError(f"Unsupported baseline: {value!r}")
    return mapping[key]


def atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
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
    parser.add_argument("--baselines", nargs="+", default=["average", "zero"])
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--results-root", default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    root = Path(args.results_root)
    selected_baselines = [canonical_baseline(value) for value in args.baselines]
    if len(selected_baselines) != len(set(selected_baselines)):
        raise ValueError(f"Duplicate baselines: {args.baselines}")

    fold_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []
    missing: List[str] = []

    for raw_data in args.datasets:
        data = canonical_dataset(raw_data)
        values: Dict[str, List[float]] = {
            baseline: [] for baseline in selected_baselines
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

            result_map = payload.get("results", {})
            for item in result_map.values():
                if not isinstance(item, Mapping):
                    continue
                baseline = str(item["baseline"])
                if baseline not in values:
                    continue
                cpd = float(item["cpd"])
                values[baseline].append(cpd)
                fold_rows.append(
                    {
                        "dataset": data,
                        "method": "IRON",
                        "baseline": baseline,
                        "fold": fold,
                        "seed": args.seed,
                        "topk": float(payload["protocol"]["topk"]),
                        "CPD": cpd,
                        "saliency_path": payload.get("saliency_path"),
                        "saliency_cache_status": payload.get(
                            "saliency_cache_status"
                        ),
                    }
                )

        for baseline in selected_baselines:
            current = values[baseline]
            if args.strict and len(current) != len(args.folds):
                raise RuntimeError(
                    f"Missing folds for {data}/{baseline}: "
                    f"got {len(current)}, expected {len(args.folds)}"
                )
            if not current:
                continue

            array = np.asarray(current, dtype=np.float64)
            mean = float(array.mean())
            # Existing synthetic aggregation uses np.std(ddof=0).
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
    atomic_csv(root / "iron_cpd_fold.csv", fold_rows)
    atomic_csv(root / "iron_cpd_summary.csv", summary_rows)

    tex_rows = [
        (
            f"{row['dataset']} & IRON & {row['baseline']} "
            f"& {row['CPD_latex']} " + r"\\"
        )
        for row in summary_rows
    ]
    (root / "iron_cpd_table_rows.tex").write_text(
        "\n".join(tex_rows) + ("\n" if tex_rows else ""),
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
