"""
embryo_monotone_encode.py

Converts per-patient one-hot embryo stage matrices into monotone-encoded CSVs.

Monotone encoding:
  -1  →  stages already passed
   1  →  current active stage
   0  →  upcoming stages not yet reached

Output columns (21 total):
  frame_number, time_hours, class,
  starting, tPB2, tPNa, tPNf, t2, t3, t4, t5, t6, t7, t8, t9+,
  tM, tSB, tB, tEB, tHB, ending

Usage:
    python embryo_monotone_encode.py \
        --root /path/to/root \
        --output /path/to/output_dir \
        [--pad]          # enable starting/ending padding to 550 rows
        [--verbose]      # print progress per patient
"""

import argparse
import json
import os

import pandas as pd
import numpy as np


# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────

TOTAL_FRAMES = 550

STAGE_COLUMNS = [
    "tPB2", "tPNa", "tPNf",
    "t2", "t3", "t4", "t5", "t6", "t7", "t8", "t9+",
    "tM", "tSB", "tB", "tEB", "tHB",
]

META_COLUMNS = ["frame_number", "time_hours", "class"]

OUTPUT_STAGE_COLUMNS = ["starting"] + STAGE_COLUMNS + ["ending"]

OUTPUT_COLUMNS = META_COLUMNS + OUTPUT_STAGE_COLUMNS  # 21 columns total


# ──────────────────────────────────────────────
# Core logic
# ──────────────────────────────────────────────

def one_hot_to_monotone(row: pd.Series) -> pd.Series:
    """
    Given a row of one-hot stage values, return monotone encoding:
      -1 for past stages, 1 for current stage, 0 for future stages.
    If no stage is active (all zeros), returns all zeros.
    """
    values = row.values.astype(float)
    active_indices = np.where(values == 1)[0]

    if len(active_indices) == 0:
        # No active stage — return zeros (should not happen per spec)
        return pd.Series(np.zeros(len(row), dtype=float), index=row.index)

    current_idx = active_indices[-1]  # last active stage (monotone assumption)
    result = np.zeros(len(row), dtype=float)
    result[:current_idx] = -1.0
    result[current_idx] = 1.0
    # result[current_idx+1:] stays 0

    return pd.Series(result, index=row.index)


def process_patient(patient_name: str, onehot_path: str, pad: bool) -> pd.DataFrame:
    """
    Load a patient's onehot_matrix_with_metadata.csv, apply monotone encoding,
    add starting/ending columns, and optionally pad to TOTAL_FRAMES rows.

    Returns a DataFrame with OUTPUT_COLUMNS.
    """
    df = pd.read_csv(onehot_path)

    # ── Validate expected columns exist ──
    missing = [c for c in META_COLUMNS + STAGE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"[{patient_name}] Missing columns: {missing}")

    # ── Apply monotone encoding to stage columns ──
    stage_df = df[STAGE_COLUMNS].copy()
    monotone_df = stage_df.apply(one_hot_to_monotone, axis=1)

    # ── Build result with metadata ──
    result = df[META_COLUMNS].copy().reset_index(drop=True)
    monotone_df = monotone_df.reset_index(drop=True)

    # Add starting column (all 0 initially — filled during padding or left as 0)
    result["starting"] = 0.0
    for col in STAGE_COLUMNS:
        result[col] = monotone_df[col]
    result["ending"] = 0.0

    # ── Ensure frame_number is integer ──
    result["frame_number"] = result["frame_number"].astype(int)

    if not pad:
        return result[OUTPUT_COLUMNS]

    # ──────────────────────────────────────────
    # Padding to TOTAL_FRAMES rows (pad=True)
    # ──────────────────────────────────────────

    # Determine actual frame range in the data
    existing_frames = set(result["frame_number"].tolist())
    first_frame = result["frame_number"].min()
    last_frame = result["frame_number"].max()

    all_frames = list(range(1, TOTAL_FRAMES + 1))

    padded_rows = []

    for frame in all_frames:
        if frame in existing_frames:
            row = result[result["frame_number"] == frame].iloc[0].to_dict()
            padded_rows.append(row)

        elif frame < first_frame:
            # "starting" padding — before embryo data begins
            row = {
                "frame_number": frame,
                "time_hours": np.nan,
                "class":       "starting",
                "starting":    1.0,
                "ending":      0.0,
            }
            for col in STAGE_COLUMNS:
                row[col] = 0.0
            padded_rows.append(row)

        else:
            # frame > last_frame — "ending" padding
            row = {
                "frame_number": frame,
                "time_hours": np.nan,
                "class":       "ending",
                "starting":    -1.0,
                "ending":      1.0,
            }
            for col in STAGE_COLUMNS:
                row[col] = -1.0
            padded_rows.append(row)

    padded_df = pd.DataFrame(padded_rows, columns=OUTPUT_COLUMNS)
    return padded_df


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Convert embryo one-hot matrices to monotone encoding."
    )
    parser.add_argument(
        "--root",
        required=True,
        help="Path to the dataset root directory.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Directory where output CSV files will be saved.",
    )
    parser.add_argument(
        "--pad",
        action="store_true",
        default=False,
        help=(
            "If set, pad each patient's data to exactly 550 rows using "
            "'starting' (before first frame) and 'ending' (after last frame) classes."
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Print progress for each patient.",
    )
    args = parser.parse_args()

    root = args.root
    output_dir = args.output
    pad = args.pad
    verbose = args.verbose

    # ── Load selected patients ──
    json_path = os.path.join(root, "selected_patients.json")
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"selected_patients.json not found at: {json_path}")

    with open(json_path, "r") as f:
        selected = json.load(f)

    patient_list = selected["patients"]
    print(f"Found {len(patient_list)} selected patients.")

    # ── Resolve one_hot_matrix directory ──
    onehot_dir = os.path.join(root, "embryo_onehot_matrix")
    if not os.path.isdir(onehot_dir):
        raise FileNotFoundError(f"embryo_onehot_matrix directory not found at: {onehot_dir}")

    # ── Create output directory ──
    os.makedirs(output_dir, exist_ok=True)

    # ── Process each patient ──
    success_count = 0
    skip_count = 0
    error_count = 0

    for patient in patient_list:
        onehot_path = os.path.join(onehot_dir, f"{patient}_onehot_with_metadata.csv")

        if not os.path.exists(onehot_path):
            if verbose:
                print(f"  [SKIP] {patient} — file not found: {onehot_path}")
            skip_count += 1
            continue

        try:
            result_df = process_patient(patient, onehot_path, pad=pad)

            out_path = os.path.join(output_dir, f"{patient}_monotone.csv")
            result_df.to_csv(out_path, index=False)

            if verbose:
                print(f"  [OK]   {patient} → {result_df.shape[0]} rows saved.")
            success_count += 1

        except Exception as e:
            print(f"  [ERROR] {patient}: {e}")
            error_count += 1

    # ── Summary ──
    print("\n── Summary ──")
    print(f"  Processed : {success_count}")
    print(f"  Skipped   : {skip_count}  (file not found)")
    print(f"  Errors    : {error_count}")
    print(f"  Output dir: {output_dir}")
    print(f"  Padding   : {'enabled (550 rows)' if pad else 'disabled (raw frames only)'}")


if __name__ == "__main__":
    main()