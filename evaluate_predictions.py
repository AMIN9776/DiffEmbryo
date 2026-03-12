"""
evaluate_predictions.py

Compute per-class accuracy, precision, recall, F1, and macro F1 for diffusion
predictions saved by inference.py.

Assumes each CSV has columns:
    frame_number, gt_class, is_valid, predicted_class, starting, tPB2, ..., ending

Only rows with is_valid == 1 are evaluated (starting/ending padding are ignored).
"""

from __future__ import annotations

import argparse
import os
from typing import List

import numpy as np
import pandas as pd


CLASS_NAMES: List[str] = [
    "starting",
    "tPB2", "tPNa", "tPNf",
    "t2", "t3", "t4", "t5", "t6", "t7", "t8", "t9+",
    "tM", "tSB", "tB", "tEB", "tHB",
    "ending",
]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_NAMES)}


def collect_labels(pred_dir: str):
    files = [f for f in os.listdir(pred_dir) if f.endswith("_prediction.csv")]
    if not files:
        raise SystemExit(f"No *_prediction.csv files found in {pred_dir}")
    files.sort()

    y_true = []
    y_pred = []
    for fname in files:
        path = os.path.join(pred_dir, fname)
        df = pd.read_csv(path)
        if "gt_class" not in df.columns or "predicted_class" not in df.columns:
            continue
        if "is_valid" in df.columns:
            df = df[df["is_valid"] == 1]
        gt = df["gt_class"].astype(str).to_numpy()
        pr = df["predicted_class"].astype(str).to_numpy()

        for g, p in zip(gt, pr):
            if g not in CLASS_TO_IDX or p not in CLASS_TO_IDX:
                continue
            y_true.append(CLASS_TO_IDX[g])
            y_pred.append(CLASS_TO_IDX[p])

    if not y_true:
        raise SystemExit("No valid (is_valid==1) frames found to evaluate.")

    return np.array(y_true, dtype=np.int64), np.array(y_pred, dtype=np.int64)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    num_classes = len(CLASS_NAMES)
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    support = cm.sum(axis=1)           # TP + FN per class
    tp = np.diag(cm)
    pred_count = cm.sum(axis=0)        # TP + FP per class

    per_class = []
    for c in range(num_classes):
        if support[c] == 0:
            acc_c = np.nan
            prec_c = np.nan
            rec_c = np.nan
            f1_c = np.nan
        else:
            acc_c = tp[c] / support[c]
            prec_c = tp[c] / pred_count[c] if pred_count[c] > 0 else 0.0
            rec_c = tp[c] / support[c]
            if prec_c + rec_c > 0:
                f1_c = 2 * prec_c * rec_c / (prec_c + rec_c)
            else:
                f1_c = 0.0
        per_class.append((acc_c, prec_c, rec_c, f1_c, support[c]))

    overall_acc = float((y_true == y_pred).mean())

    # Macro F1 over classes with support > 0
    f1_vals = [pc[3] for pc in per_class if not np.isnan(pc[3])]
    macro_f1 = float(np.mean(f1_vals)) if f1_vals else float("nan")

    return cm, per_class, overall_acc, macro_f1


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute per-class accuracy and macro F1 from *_prediction.csv files."
    )
    parser.add_argument(
        "--pred_dir",
        required=True,
        help="Directory containing *_prediction.csv files from inference.py.",
    )
    args = parser.parse_args()

    y_true, y_pred = collect_labels(args.pred_dir)
    cm, per_class, overall_acc, macro_f1 = compute_metrics(y_true, y_pred)

    total_frames = len(y_true)
    print(f"Evaluated {total_frames} valid frames across all patients.")
    print(f"Overall accuracy: {overall_acc:.4f}")
    print(f"Macro F1 (over classes with support > 0): {macro_f1:.4f}\n")

    print("Per-class metrics (valid frames only):")
    print(f"{'Class':<8s} {'Acc':>8s} {'Prec':>8s} {'Rec':>8s} {'F1':>8s} {'Support':>10s}")
    for name, (acc_c, prec_c, rec_c, f1_c, sup_c) in zip(CLASS_NAMES, per_class):
        acc_str = "nan" if np.isnan(acc_c) else f"{acc_c:.3f}"
        prec_str = "nan" if np.isnan(prec_c) else f"{prec_c:.3f}"
        rec_str = "nan" if np.isnan(rec_c) else f"{rec_c:.3f}"
        f1_str = "nan" if np.isnan(f1_c) else f"{f1_c:.3f}"
        print(f"{name:<8s} {acc_str:>8s} {prec_str:>8s} {rec_str:>8s} {f1_str:>8s} {sup_c:10d}")


if __name__ == "__main__":
    main()

