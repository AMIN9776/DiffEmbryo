"""
split_patients.py

Splits selected_patients.json into training_set.json and validation_set.json.
Reads only from selected_patients.json — never overwrites it.

Usage:
    python split_patients.py \
        --selected_json /path/to/selected_patients.json \
        --output_dir    /path/to/splits \
        [--val_ratio 0.2] \
        [--seed 42]
"""

import argparse
import json
import os
import random


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--selected_json", required=True,
                        help="Path to selected_patients.json")
    parser.add_argument("--output_dir", required=True,
                        help="Directory to save training_set.json and validation_set.json")
    parser.add_argument("--val_ratio", type=float, default=0.2,
                        help="Fraction of patients for validation (default: 0.2)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    args = parser.parse_args()

    # ── Load selected patients ──────────────────────────────────────────────
    with open(args.selected_json, "r") as f:
        data = json.load(f)

    patients = data["patients"]
    print(f"Total selected patients: {len(patients)}")

    # ── Shuffle and split ───────────────────────────────────────────────────
    random.seed(args.seed)
    shuffled = patients.copy()
    random.shuffle(shuffled)

    n_val   = int(len(shuffled) * args.val_ratio)
    val     = shuffled[:n_val]
    train   = shuffled[n_val:]

    print(f"  Train : {len(train)}")
    print(f"  Val   : {len(val)}")

    # ── Save ────────────────────────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)

    train_path = os.path.join(args.output_dir, "training_set.json")
    val_path   = os.path.join(args.output_dir, "validation_set.json")

    with open(train_path, "w") as f:
        json.dump({"num_patients": len(train), "patients": train}, f, indent=2)

    with open(val_path, "w") as f:
        json.dump({"num_patients": len(val), "patients": val}, f, indent=2)

    print(f"  Saved: {train_path}")
    print(f"  Saved: {val_path}")


if __name__ == "__main__":
    main()