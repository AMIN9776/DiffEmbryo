"""
embryo_time_quantize.py

Re-index per-patient one-hot embryo stage matrices onto a global
time_hours grid with fixed resolution (default 0.2 hours).

Input per patient (one row per acquired frame):
    frame_number, time_hours, class,
    tPB2, tPNa, tPNf, t2, t3, t4, t5, t6, t7, t8, t9+, tM, tSB, tB, tEB, tHB

Output per patient (one row per global time bin):
    time_hours, class, tPB2, ..., tHB

Each row corresponds to a quantised time bin; the stage one-hot
is placed into the bin whose centre is nearest to the original
time_hours. Bins without any original frame keep all-zero
stage columns and class=NaN.

Usage example:

    python embryo_time_quantize.py \
        --root /path/to/Embryo_data_cropped_applied_on_all \
        --output /path/to/output_dir \
        --bin_size 0.2 \
        --verbose
"""

import argparse
import json
import os
from typing import List, Tuple

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

META_COLUMNS: List[str] = ["frame_number", "time_hours", "class"]


def discover_global_time_range(
    root: str,
    patients: List[str],
    bin_size: float,
    verbose: bool = False,
) -> Tuple[float, float]:
    """
    Scan all selected patients and compute global [t_min, t_max] over time_hours.
    Returns start and end aligned to the bin_size grid.
    """
    onehot_dir = os.path.join(root, "embryo_onehot_matrix")
    if not os.path.isdir(onehot_dir):
        raise FileNotFoundError(f"embryo_onehot_matrix not found at: {onehot_dir}")

    global_min = float("inf")
    global_max = float("-inf")

    for patient in patients:
        path = os.path.join(onehot_dir, f"{patient}_onehot_with_metadata.csv")
        if not os.path.exists(path):
            if verbose:
                print(f"  [RANGE-SKIP] {patient} — file not found")
            continue

        df = pd.read_csv(path, usecols=["time_hours"])
        if df.empty:
            if verbose:
                print(f"  [RANGE-SKIP] {patient} — empty CSV")
            continue

        # Drop NaNs just in case
        times = df["time_hours"].dropna().to_numpy()
        if times.size == 0:
            if verbose:
                print(f"  [RANGE-SKIP] {patient} — no valid time_hours")
            continue

        local_min = float(times.min())
        local_max = float(times.max())
        global_min = min(global_min, local_min)
        global_max = max(global_max, local_max)

    if not np.isfinite(global_min) or not np.isfinite(global_max):
        raise RuntimeError("Could not determine global time range (no valid time_hours found).")

    # Align to bin grid
    start = np.floor(global_min / bin_size) * bin_size
    end = np.ceil(global_max / bin_size) * bin_size

    # Round to a sensible precision to avoid floating noise in CSV
    start = float(np.round(start, 4))
    end = float(np.round(end, 4))

    if verbose:
        print(f"Global time_hours range: [{global_min:.3f}, {global_max:.3f}]")
        print(f"Aligned to bin_size={bin_size}: start={start:.3f}, end={end:.3f}")

    return start, end


def quantize_patient(
    patient: str,
    root: str,
    output_dir: str,
    time_start: float,
    time_end: float,
    bin_size: float,
) -> Tuple[bool, str]:
    """
    Quantise a single patient's onehot matrix onto the global time grid.

    Returns (success, message).
    """
    onehot_dir = os.path.join(root, "embryo_onehot_matrix")
    in_path = os.path.join(onehot_dir, f"{patient}_onehot_with_metadata.csv")
    if not os.path.exists(in_path):
        return False, f"[SKIP] {patient} — file not found"

    df = pd.read_csv(in_path)

    # Validate columns
    missing = [c for c in META_COLUMNS + STAGE_COLUMNS if c not in df.columns]
    if missing:
        return False, f"[ERROR] {patient} — missing columns: {missing}"

    # Drop rows without valid time
    df = df.dropna(subset=["time_hours"])
    if df.empty:
        return False, f"[SKIP] {patient} — no valid time_hours rows"

    times = df["time_hours"].to_numpy(dtype=float)

    # Construct global time grid
    num_bins = int(np.round((time_end - time_start) / bin_size)) + 1
    grid = time_start + np.arange(num_bins) * bin_size
    grid = np.round(grid, 4)  # keep grid numerically stable for CSV

    # Map each original time to nearest grid index
    rel = (times - time_start) / bin_size
    idx = np.round(rel).astype(int)
    idx = np.clip(idx, 0, num_bins - 1)

    # Prepare output arrays
    out_stage = np.zeros((num_bins, len(STAGE_COLUMNS)), dtype=float)
    out_class = np.array([np.nan] * num_bins, dtype=object)

    # For each frame, copy its one-hot + class into the target bin.
    # If multiple frames map to the same bin, the last one wins.
    stage_values = df[STAGE_COLUMNS].to_numpy(dtype=float)
    class_values = df["class"].to_numpy(dtype=object)

    for i, bin_idx in enumerate(idx):
        out_stage[bin_idx] = stage_values[i]
        out_class[bin_idx] = class_values[i]

    out_df = pd.DataFrame(
        {
            "time_hours": grid,
            "class": out_class,
        }
    )
    for j, col in enumerate(STAGE_COLUMNS):
        out_df[col] = out_stage[:, j]

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{patient}_time_quantised.csv")
    out_df.to_csv(out_path, index=False)

    return True, f"[OK]   {patient} → {out_df.shape[0]} time bins saved"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Quantise embryo one-hot matrices onto a fixed time_hours grid."
    )
    parser.add_argument(
        "--root",
        required=True,
        help="Path to the dataset root directory (contains embryo_onehot_matrix and selected_patients.json).",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Directory where output CSV files will be saved.",
    )
    parser.add_argument(
        "--bin_size",
        type=float,
        default=0.2,
        help="Time bin size in hours (default: 0.2).",
    )
    parser.add_argument(
        "--patients_json",
        default="selected_patients.json",
        help="Name of the JSON file listing patients (default: selected_patients.json).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Print progress details.",
    )

    args = parser.parse_args()

    root = args.root
    output_dir = args.output
    bin_size = float(args.bin_size)
    patients_json = args.patients_json
    verbose = args.verbose

    if bin_size <= 0:
        raise ValueError("bin_size must be positive.")

    # Load patients from JSON
    json_path = os.path.join(root, patients_json)
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"{patients_json} not found at: {json_path}")

    with open(json_path, "r") as f:
        data = json.load(f)

    if isinstance(data, dict) and "patients" in data:
        patients = list(data["patients"])
    elif isinstance(data, list):
        patients = list(data)
    else:
        raise ValueError(
            f"Unexpected structure in {patients_json}; expected dict with 'patients' or simple list."
        )

    if verbose:
        print(f"Loaded {len(patients)} patients from {patients_json}")

    # First pass: discover global time range
    time_start, time_end = discover_global_time_range(
        root=root,
        patients=patients,
        bin_size=bin_size,
        verbose=verbose,
    )

    # Second pass: quantise each patient onto the global grid
    ok_count = 0
    skip_count = 0
    error_count = 0

    for patient in patients:
        try:
            success, msg = quantize_patient(
                patient=patient,
                root=root,
                output_dir=output_dir,
                time_start=time_start,
                time_end=time_end,
                bin_size=bin_size,
            )
            if success:
                ok_count += 1
            else:
                skip_count += 1
            if verbose:
                print(msg)
        except Exception as e:
            error_count += 1
            print(f"[ERROR] {patient}: {e}")

    print("\n── Time quantisation summary ──")
    print(f"  Bin size    : {bin_size} hours")
    print(f"  Time grid   : start={time_start:.3f}, end={time_end:.3f}")
    print(f"  Processed   : {ok_count}")
    print(f"  Skipped     : {skip_count}")
    print(f"  Errors      : {error_count}")
    print(f"  Output dir  : {output_dir}")


if __name__ == "__main__":
    main()

