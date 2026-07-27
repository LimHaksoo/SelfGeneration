#!/usr/bin/env python3
"""Aggregate fold-level CNF round-trip reconstruction measurements."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

METRICS = (
    "mae_per_element",
    "rmse_per_element",
    "mean_sample_mae",
    "median_sample_mae",
    "mean_sample_rmse",
    "median_sample_rmse",
    "mean_sample_max_abs",
    "median_sample_max_abs",
    "p95_sample_max_abs",
    "global_max_abs",
    "mean_relative_l2",
    "median_relative_l2",
    "p95_relative_l2",
)
ZERO_METRICS = (
    "mae_per_element",
    "rmse_per_element",
    "max_abs",
)
DATASET_ORDER = {
    "state": 0,
    "switch-feature": 1,
    "PAM": 2,
    "boiler": 3,
    "epilepsy": 4,
    "wafer": 5,
    "freezer": 6,
}
DISPLAY_NAME = {
    "state": "State",
    "switch-feature": "Switch-feature",
    "PAM": "PAM",
    "boiler": "Boiler",
    "epilepsy": "Epilepsy",
    "wafer": "Wafer",
    "freezer": "Freezer",
}


def resolve_path(value: str | os.PathLike[str]) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def read_completed_results(root: Path) -> list[Dict[str, Any]]:
    results: list[Dict[str, Any]] = []
    for path in sorted(root.rglob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if payload.get("status") != "complete":
            continue
        if "reconstruction" not in payload or "protocol" not in payload:
            continue
        payload["_result_path"] = str(path)
        results.append(payload)
    return results


def flatten_result(payload: Mapping[str, Any]) -> Dict[str, Any]:
    protocol = payload["protocol"]
    candidate = payload.get("candidate", {})
    training = payload.get("training", {})
    runtime = payload.get("runtime", {})
    reconstruction = payload["reconstruction"]
    zero = payload.get("zero_baseline_roundtrip", {})

    row: Dict[str, Any] = {
        "suite": payload["suite"],
        "synthetic_source": payload.get("synthetic_source"),
        "dataset": payload["data"],
        "fold": int(payload["fold"]),
        "seed": int(payload["seed"]),
        "split": payload["split"],
        "candidate": protocol["candidate_name"],
        "ambient_dimension": int(payload["ambient_dimension"]),
        "input_shape": "x".join(str(value) for value in payload["input_shape"]),
        "dims": candidate.get("dims"),
        "hidden_width_max": candidate.get("hidden_width_max"),
        "hidden_layers": candidate.get("hidden_layers"),
        "num_blocks": candidate.get("num_blocks"),
        "solver": candidate.get("solver"),
        "step_size": candidate.get("step_size"),
        "test_solver": candidate.get("test_solver"),
        "test_atol": candidate.get("test_atol"),
        "test_rtol": candidate.get("test_rtol"),
        "parameter_count": candidate.get("parameter_count"),
        "best_epoch": training.get("best_epoch"),
        "best_val_nll_per_dim": training.get("best_val_nll_per_dim"),
        "n_samples": reconstruction.get("n_samples"),
        "evaluation_seconds": runtime.get("evaluation_seconds"),
        "ms_per_sample": runtime.get("ms_per_sample"),
        "peak_memory_mib": runtime.get("peak_memory_mib"),
        "checkpoint": payload.get("checkpoint"),
        "result_path": payload.get("_result_path"),
    }
    for metric in METRICS:
        row[metric] = float(reconstruction[metric])
    for metric in ZERO_METRICS:
        value = zero.get(metric)
        row[f"zero_{metric}"] = None if value is None else float(value)
    return row


def group_key(row: Mapping[str, Any]) -> Tuple[Any, ...]:
    return (
        row["suite"],
        row.get("synthetic_source"),
        row["dataset"],
        row["split"],
        row["candidate"],
        row["ambient_dimension"],
        row["input_shape"],
        row["dims"],
        row["num_blocks"],
        row["solver"],
        row["step_size"],
        row["test_solver"],
        row["test_atol"],
        row["test_rtol"],
        row["parameter_count"],
    )


def mean_std(values: Iterable[Any]) -> Tuple[float, float]:
    array = np.asarray(
        [float(value) for value in values if value is not None],
        dtype=np.float64,
    )
    array = array[np.isfinite(array)]
    if array.size == 0:
        return float("nan"), float("nan")
    std = float(array.std(ddof=1)) if array.size > 1 else 0.0
    return float(array.mean()), std


def aggregate_rows(
    fold_rows: Sequence[Mapping[str, Any]],
    expected_folds: Sequence[int],
    strict: bool,
) -> list[Dict[str, Any]]:
    grouped: Dict[Tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in fold_rows:
        grouped[group_key(row)].append(row)

    expected = set(int(value) for value in expected_folds)
    summary_rows: list[Dict[str, Any]] = []

    for _key, rows in grouped.items():
        rows = sorted(rows, key=lambda item: int(item["fold"]))
        first = rows[0]
        observed = {int(item["fold"]) for item in rows}
        if strict and observed != expected:
            raise RuntimeError(
                f"Missing folds for {first['suite']}/{first['dataset']}/"
                f"{first['candidate']}: expected={sorted(expected)}, "
                f"observed={sorted(observed)}"
            )

        summary: Dict[str, Any] = {
            "suite": first["suite"],
            "synthetic_source": first.get("synthetic_source"),
            "dataset": first["dataset"],
            "split": first["split"],
            "candidate": first["candidate"],
            "ambient_dimension": first["ambient_dimension"],
            "input_shape": first["input_shape"],
            "dims": first["dims"],
            "hidden_width_max": first["hidden_width_max"],
            "hidden_layers": first["hidden_layers"],
            "num_blocks": first["num_blocks"],
            "solver": first["solver"],
            "step_size": first["step_size"],
            "test_solver": first["test_solver"],
            "test_atol": first["test_atol"],
            "test_rtol": first["test_rtol"],
            "parameter_count": first["parameter_count"],
            "n_folds": len(rows),
            "folds": " ".join(str(value) for value in sorted(observed)),
            "total_samples": int(sum(int(item["n_samples"]) for item in rows)),
        }

        for metric in METRICS:
            mean, std = mean_std(item[metric] for item in rows)
            summary[f"{metric}_mean"] = mean
            summary[f"{metric}_std"] = std
        for metric in ZERO_METRICS:
            key = f"zero_{metric}"
            mean, std = mean_std(item[key] for item in rows)
            summary[f"{key}_mean"] = mean
            summary[f"{key}_std"] = std

        for metric in (
            "best_val_nll_per_dim",
            "evaluation_seconds",
            "ms_per_sample",
            "peak_memory_mib",
        ):
            mean, std = mean_std(item[metric] for item in rows)
            summary[f"{metric}_mean"] = mean
            summary[f"{metric}_std"] = std

        summary_rows.append(summary)

    return sorted(
        summary_rows,
        key=lambda row: (
            0 if row["suite"] == "synthetic" else 1,
            DATASET_ORDER.get(str(row["dataset"]), 999),
            str(row["candidate"]),
        ),
    )


def latex_sci(value: float) -> str:
    if not np.isfinite(value):
        return "--"
    if value == 0.0:
        return "0"
    exponent = int(math.floor(math.log10(abs(value))))
    mantissa = value / (10 ** exponent)
    return f"{mantissa:.2f}\\times10^{{{exponent}}}"


def latex_mean_std(mean: float, std: float) -> str:
    if not np.isfinite(mean):
        return "--"
    return f"${latex_sci(mean)} \\pm {latex_sci(std)}$"


def latex_rows(summary_rows: Sequence[Mapping[str, Any]]) -> str:
    lines: list[str] = []
    for row in summary_rows:
        dataset = DISPLAY_NAME.get(str(row["dataset"]), str(row["dataset"]))
        dims = str(row["dims"] or "--").replace("_", "\\_")
        blocks = row["num_blocks"] if row["num_blocks"] is not None else "--"
        mae = latex_mean_std(
            float(row["mae_per_element_mean"]),
            float(row["mae_per_element_std"]),
        )
        rmse = latex_mean_std(
            float(row["rmse_per_element_mean"]),
            float(row["rmse_per_element_std"]),
        )
        mean_max = latex_mean_std(
            float(row["mean_sample_max_abs_mean"]),
            float(row["mean_sample_max_abs_std"]),
        )
        lines.append(
            f"{dataset} & {int(row['ambient_dimension']):,} & "
            f"\\texttt{{{dims}}} & {blocks} & {mae} & {rmse} & {mean_max} \\\\"
        )
    return "\n".join(lines) + ("\n" if lines else "")


def full_latex_table(summary_rows: Sequence[Mapping[str, Any]]) -> str:
    rows = latex_rows(summary_rows).rstrip()
    return (
        "\\begin{table}[t]\n"
        "\\centering\n"
        "\\small\n"
        "\\begin{tabular}{l r c c c c c}\n"
        "\\toprule\n"
        "Dataset & $d$ & Hidden dims & Blocks & MAE & RMSE & Mean max $|e|$ \\\\n"
        "\\midrule\n"
        f"{rows}\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\caption{Numerical encode--decode round-trip error of the trained "
        "CNFs on the held-out test split. Values are mean $\\pm$ standard "
        "deviation across folds. No endpoint clamping is applied.}\n"
        "\\label{tab:cnf_reconstruction}\n"
        "\\end{table}\n"
    )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root",
        default="results_cnf_reconstruction",
    )
    parser.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    root = resolve_path(args.results_root)
    payloads = read_completed_results(root)
    if not payloads:
        raise RuntimeError(f"No completed reconstruction JSON files under {root}")

    fold_rows = [flatten_result(payload) for payload in payloads]
    fold_rows = sorted(
        fold_rows,
        key=lambda row: (
            0 if row["suite"] == "synthetic" else 1,
            DATASET_ORDER.get(str(row["dataset"]), 999),
            str(row["candidate"]),
            int(row["fold"]),
        ),
    )
    summary_rows = aggregate_rows(
        fold_rows=fold_rows,
        expected_folds=args.folds,
        strict=args.strict,
    )

    write_csv(root / "fold_reconstruction.csv", fold_rows)
    write_csv(root / "reconstruction_summary.csv", summary_rows)
    atomic_write_text(root / "reconstruction_table_rows.tex", latex_rows(summary_rows))
    atomic_write_text(root / "reconstruction_table.tex", full_latex_table(summary_rows))

    print(
        "dataset          d      dims       blocks   MAE(mean±std)             "
        "RMSE(mean±std)"
    )
    for row in summary_rows:
        print(
            f"{str(row['dataset']):14s} "
            f"{int(row['ambient_dimension']):6d} "
            f"{str(row['dims']):10s} "
            f"{str(row['num_blocks']):>6s}   "
            f"{row['mae_per_element_mean']:.3e}±"
            f"{row['mae_per_element_std']:.3e}   "
            f"{row['rmse_per_element_mean']:.3e}±"
            f"{row['rmse_per_element_std']:.3e}"
        )

    print(f"[SAVE] {root / 'fold_reconstruction.csv'}")
    print(f"[SAVE] {root / 'reconstruction_summary.csv'}")
    print(f"[SAVE] {root / 'reconstruction_table_rows.tex'}")
    print(f"[SAVE] {root / 'reconstruction_table.tex'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
