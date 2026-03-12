"""
embryo_time_polish.py

Polishes time-quantised embryo one-hot CSVs so that:
  1. No "empty" time bin: rows with all zeros are filled by interpolation
     using the patient's previous/next non-empty class, or (if no neighbor)
     the population mode at that time_hours.
  2. No "multi-class" bin: rows with 2+ classes get a single class chosen
     as the most probable at that time_hours according to population counts.

Input:  directory of *_time_quantised.csv (output of embryo_time_quantize.py)
Output: directory of *_time_polished.csv with same columns; every row has
        exactly one class (one-hot sum == 1).

Usage:

    python embryo_time_polish.py \
        --input /path/to/embryo_time_quantised \
        --output /path/to/embryo_time_polished \
        [--verbose]
"""

from __future__ import annotations

import argparse
import os
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


STAGE_COLUMNS: List[str] = [
    "tPB2", "tPNa", "tPNf",
    "t2", "t3", "t4", "t5", "t6", "t7", "t8", "t9+",
    "tM", "tSB", "tB", "tEB", "tHB",
]

CLASS_TO_IDX: Dict[str, int] = {c: i for i, c in enumerate(STAGE_COLUMNS)}
IDX_TO_CLASS: Dict[int, str] = {i: c for i, c in enumerate(STAGE_COLUMNS)}


def _row_onehot_sum(df: pd.DataFrame, row_idx: int) -> float:
    return df[STAGE_COLUMNS].iloc[row_idx].astype(float).sum()


def _row_to_class_index(df: pd.DataFrame, row_idx: int) -> Optional[int]:
    """Return the single class index if exactly one class is set; else None."""
    row = df[STAGE_COLUMNS].iloc[row_idx].astype(float)
    ones = np.where(row > 0.5)[0]
    if len(ones) == 1:
        return int(ones[0])
    return None


def _row_to_class_indices(df: pd.DataFrame, row_idx: int) -> List[int]:
    """Return list of class indices that are set (for multi-class rows)."""
    row = df[STAGE_COLUMNS].iloc[row_idx].astype(float)
    return list(np.where(row > 0.5)[0].astype(int))


def build_population_stats(input_dir: str) -> Dict[float, Dict[str, int]]:
    """
    For each time_hours that appears in any CSV, count how many patients
    have each class (only from rows with exactly one class).
    Returns: time_hours -> { class_name: count }
    """
    population: Dict[float, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    csv_files = [f for f in os.listdir(input_dir) if f.endswith("_time_quantised.csv")]
    for fname in csv_files:
        path = os.path.join(input_dir, fname)
        df = pd.read_csv(path)
        if "time_hours" not in df.columns:
            continue
        for i in range(len(df)):
            idx = _row_to_class_index(df, i)
            if idx is None:
                continue
            t = float(df["time_hours"].iloc[i])
            c = IDX_TO_CLASS[idx]
            population[t][c] += 1
    return dict(population)


def interpolate_empty_bin(
    df: pd.DataFrame,
    row_idx: int,
    population: Dict[float, Dict[str, int]],
) -> int:
    """
    Choose a single class index for a row that currently has sum 0.
    Uses previous/next non-empty row in same patient; if both exist and differ,
    use the stage between them; if no neighbor, use population mode at this time.
    """
    t = float(df["time_hours"].iloc[row_idx])
    n = len(df)

    # Previous non-empty
    prev_idx: Optional[int] = None
    for j in range(row_idx - 1, -1, -1):
        if _row_onehot_sum(df, j) >= 1:
            prev_idx = _row_to_class_index(df, j)
            if prev_idx is None:
                prev_idx = _row_to_class_indices(df, j)[0]  # take first if multi
            break

    # Next non-empty
    next_idx: Optional[int] = None
    for j in range(row_idx + 1, n):
        if _row_onehot_sum(df, j) >= 1:
            next_idx = _row_to_class_index(df, j)
            if next_idx is None:
                next_idx = _row_to_class_indices(df, j)[0]
            break

    if prev_idx is not None and next_idx is not None:
        if prev_idx == next_idx:
            return prev_idx
        # "Between": use middle stage index so trajectory is monotonic
        mid = (prev_idx + next_idx) // 2
        return mid
    if prev_idx is not None:
        return prev_idx
    if next_idx is not None:
        return next_idx

    # No neighbor: use population mode at this time_hours
    by_time = population.get(t)
    if by_time:
        best_class = max(by_time.keys(), key=lambda c: by_time[c])
        return CLASS_TO_IDX[best_class]
    # Fallback: first stage
    return 0


def pick_most_probable_class(
    df: pd.DataFrame,
    row_idx: int,
    population: Dict[float, Dict[str, int]],
) -> int:
    """
    For a row with 2+ classes, return the class index that has the highest
    population count at this time_hours. Tie-break: earlier stage (lower index).
    """
    t = float(df["time_hours"].iloc[row_idx])
    candidates = _row_to_class_indices(df, row_idx)
    by_time = population.get(t) or {}
    def count_for_idx(idx: int) -> Tuple[int, int]:
        c = IDX_TO_CLASS[idx]
        return (by_time.get(c, 0), -idx)  # tie-break: higher index last
    best_idx = max(candidates, key=lambda idx: count_for_idx(idx))
    return best_idx


def polish_patient(
    patient_id: str,
    input_dir: str,
    output_dir: str,
    population: Dict[float, Dict[str, int]],
    verbose: bool,
) -> Tuple[bool, str]:
    """
    Polish one patient's quantised CSV.

    - Only bins BETWEEN the first and last non-empty bin (per patient) are
      interpolated if empty.
    - Multi-class bins are always collapsed to a single class.
    - Output always has 18 stage columns:
        starting, tPB2, ..., tHB, ending
    """
    in_path = os.path.join(input_dir, f"{patient_id}_time_quantised.csv")
    if not os.path.exists(in_path):
        return False, f"[SKIP] {patient_id} — file not found"

    df = pd.read_csv(in_path)
    missing = [c for c in STAGE_COLUMNS + ["time_hours"] if c not in df.columns]
    if missing:
        return False, f"[ERROR] {patient_id} — missing columns: {missing}"

    # Copy; we will overwrite class and one-hot columns where needed
    out = df.copy()
    out[STAGE_COLUMNS] = out[STAGE_COLUMNS].astype(float)
    # Prepare starting/ending columns (we will fill them per-row below)
    out["starting"] = out.get("starting", 0.0)
    out["ending"] = out.get("ending", 0.0)

    # Locate first and last non-empty bins for this patient
    non_empty_indices = [i for i in range(len(out)) if _row_onehot_sum(out, i) > 0]
    if non_empty_indices:
        first_idx = min(non_empty_indices)
        last_idx = max(non_empty_indices)
    else:
        first_idx = None
        last_idx = None

    n_empty = 0
    n_multi = 0

    for i in range(len(out)):
        s = _row_onehot_sum(out, i)

        # Multi-class bins: always collapse, even at the edges
        if s > 1:
            chosen = pick_most_probable_class(out, i, population)
            out.loc[out.index[i], "class"] = IDX_TO_CLASS[chosen]
            for j, col in enumerate(STAGE_COLUMNS):
                out.loc[out.index[i], col] = 1.0 if j == chosen else 0.0
            # Make sure starting/ending are zero for true embryo stages
            out.loc[out.index[i], "starting"] = 0.0
            out.loc[out.index[i], "ending"] = 0.0
            n_multi += 1
            continue

        # Empty bins
        if s == 0 and first_idx is not None and last_idx is not None:
            # Before first embryo frame  → mark as starting
            if i < first_idx:
                out.loc[out.index[i], "class"] = "starting"
                out.loc[out.index[i], "starting"] = 1.0
                out.loc[out.index[i], "ending"] = 0.0
                for col in STAGE_COLUMNS:
                    out.loc[out.index[i], col] = 0.0
                continue
            # After last embryo frame  → mark as ending
            if i > last_idx:
                out.loc[out.index[i], "class"] = "ending"
                out.loc[out.index[i], "starting"] = 0.0
                out.loc[out.index[i], "ending"] = 1.0
                for col in STAGE_COLUMNS:
                    out.loc[out.index[i], col] = 0.0
                continue
            # Strictly inside [first_idx, last_idx] → interpolate stage
            if first_idx < i < last_idx:
                chosen = interpolate_empty_bin(out, i, population)
                out.loc[out.index[i], "class"] = IDX_TO_CLASS[chosen]
                out.loc[out.index[i], "starting"] = 0.0
                out.loc[out.index[i], "ending"] = 0.0
                for j, col in enumerate(STAGE_COLUMNS):
                    out.loc[out.index[i], col] = 1.0 if j == chosen else 0.0
                n_empty += 1

        # Single-stage, non-empty interior rows: ensure starting/ending are zero
        if s == 1:
            out.loc[out.index[i], "starting"] = 0.0
            out.loc[out.index[i], "ending"] = 0.0

    # Reorder to 18 stages
    cols = ["time_hours", "class", "starting"] + STAGE_COLUMNS + ["ending"]
    out = out[cols]

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{patient_id}_time_polished.csv")
    out.to_csv(out_path, index=False)

    msg = f"[OK] {patient_id} → {len(out)} rows"
    if n_empty or n_multi:
        msg += f" (filled {n_empty} empty, resolved {n_multi} multi)"
    if verbose:
        print(msg)
    return True, msg


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Polish time-quantised CSVs: no empty bins, no multi-class bins."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Directory containing *_time_quantised.csv files.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Directory where *_time_polished.csv files will be written.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print one line per processed patient.",
    )
    args = parser.parse_args()

    input_dir = args.input
    output_dir = args.output
    verbose = args.verbose

    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    # Build population from all quantised files (only single-class rows count)
    if verbose:
        print("Building population statistics from quantised CSVs...")
    population = build_population_stats(input_dir)
    if verbose:
        print(f"  Time bins in population: {len(population)}")

    csv_files = [f for f in os.listdir(input_dir) if f.endswith("_time_quantised.csv")]
    patient_ids = [f.replace("_time_quantised.csv", "") for f in csv_files]
    ok = 0
    err = 0
    for pid in patient_ids:
        try:
            success, _ = polish_patient(pid, input_dir, output_dir, population, verbose)
            if success:
                ok += 1
            else:
                err += 1
        except Exception as e:
            print(f"[ERROR] {pid}: {e}")
            err += 1

    print("\n── Polish summary ──")
    print(f"  Processed : {ok}")
    print(f"  Errors    : {err}")
    print(f"  Output dir: {output_dir}")


if __name__ == "__main__":
    main()
