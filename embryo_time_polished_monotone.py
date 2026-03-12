"""
embryo_time_polished_monotone.py

Convert time-polished one-hot stage matrices into monotone-encoded matrices:

  - For embryo stages (16 internal stages tPB2..tHB) at each time bin:
        previous stages  → -1
        current stage    →  1
        upcoming stages  →  0

  - For padding rows:
        class == 'starting' → starting = 1, others = 0
        class == 'ending'   → ending   = 1, others = 0

Input per patient (from embryo_time_polish.py):
    time_hours, class,
    starting, tPB2, tPNa, tPNf, t2, t3, t4, t5, t6, t7, t8, t9+, tM, tSB, tB, tEB, tHB, ending

Output per patient:
    Same columns, but stage columns are monotone-encoded as above.

Usage:

    python embryo_time_polished_monotone.py \\
        --input  /path/to/embryo_time_polished \\
        --output /path/to/embryo_time_polished_monotone \\
        [--verbose]
"""

from __future__ import annotations

import argparse
import os
from typing import List

import numpy as np
import pandas as pd


STAGE_COLUMNS: List[str] = [
    "tPB2",
    "tPNa",
    "tPNf",
    "t2",
    "t3",
    "t4",
    "t5",
    "t6",
    "t7",
    "t8",
    "t9+",
    "tM",
    "tSB",
    "tB",
    "tEB",
    "tHB",
]

ALL_STAGE_COLS: List[str] = ["starting"] + STAGE_COLUMNS + ["ending"]


def one_hot_row_to_monotone(values: np.ndarray) -> np.ndarray:
    """
    Given a 1D array of length len(STAGE_COLUMNS) with exactly one 1.0 and 0.0 elsewhere,
    return a monotone encoding:
        -1 for stages before the active index,
         1 for the active stage,
         0 for stages after.

    If no active stage is found, returns zeros.
    """
    active = np.where(values == 1.0)[0]
    if active.size == 0:
        return np.zeros_like(values, dtype=float)
    k = int(active[-1])
    out = np.zeros_like(values, dtype=float)
    out[:k] = -1.0
    out[k] = 1.0
    # out[k+1:] already zero
    return out


def process_patient(path_in: str) -> pd.DataFrame:
    """
    Load one *_time_polished.csv and return a monotone-encoded DataFrame
    with the same columns.
    """
    df = pd.read_csv(path_in)

    missing = [c for c in ["time_hours", "class"] + ALL_STAGE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{os.path.basename(path_in)} missing columns: {missing}")

    # Prepare result
    result = df.copy()

    stage_mat = df[STAGE_COLUMNS].to_numpy(dtype=float)
    classes = df["class"].astype(str).to_numpy()

    for i in range(len(df)):
        cls = classes[i]
        if cls == "starting":
            # starting padding: starting=1, others=0
            result.loc[result.index[i], "starting"] = 1.0
            for col in STAGE_COLUMNS:
                result.loc[result.index[i], col] = 0.0
            result.loc[result.index[i], "ending"] = 0.0
        elif cls == "ending":
            # ending padding: ending=1, others=0
            result.loc[result.index[i], "starting"] = 0.0
            for col in STAGE_COLUMNS:
                result.loc[result.index[i], col] = 0.0
            result.loc[result.index[i], "ending"] = 1.0
        else:
            # Embryo stage row: apply monotone encoding on the 16 stage columns
            row_vals = stage_mat[i]
            mono = one_hot_row_to_monotone(row_vals)
            for j, col in enumerate(STAGE_COLUMNS):
                result.loc[result.index[i], col] = mono[j]
            # starting/ending should be 0 here
            result.loc[result.index[i], "starting"] = 0.0
            result.loc[result.index[i], "ending"] = 0.0

    return result[["time_hours", "class"] + ALL_STAGE_COLS]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert time-polished one-hot matrices into monotone encoding (-1,1,0) per time bin."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Directory containing *_time_polished.csv files.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Directory where *_time_polished_monotone.csv files will be written.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print each processed patient.",
    )
    args = parser.parse_args()

    input_dir = args.input
    output_dir = args.output
    verbose = args.verbose

    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    os.makedirs(output_dir, exist_ok=True)

    files = [f for f in os.listdir(input_dir) if f.endswith("_time_polished.csv")]
    files.sort()

    ok = 0
    err = 0

    for fname in files:
        pid = fname.replace("_time_polished.csv", "")
        path_in = os.path.join(input_dir, fname)
        try:
            df_out = process_patient(path_in)
            out_path = os.path.join(output_dir, f"{pid}_time_polished_monotone.csv")
            df_out.to_csv(out_path, index=False)
            ok += 1
            if verbose:
                print(f"[OK] {pid} → {df_out.shape[0]} rows")
        except Exception as e:
            err += 1
            print(f"[ERROR] {pid}: {e}")

    print("\n── Monotone conversion summary ──")
    print(f"  Processed : {ok}")
    print(f"  Errors    : {err}")
    print(f"  Output dir: {output_dir}")


if __name__ == "__main__":
    main()

