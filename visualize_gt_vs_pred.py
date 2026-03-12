"""
visualize_gt_vs_pred.py

Visualize ground-truth vs model prediction for time-series stage labels.
For each of N random patients:
  - Row 2k     : GT classes over time
  - Row 2k + 1 : Predicted classes over time
"""

from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


CLASS_NAMES = [
    "starting",
    "tPB2", "tPNa", "tPNf",
    "t2", "t3", "t4", "t5", "t6", "t7", "t8", "t9+",
    "tM", "tSB", "tB", "tEB", "tHB",
    "ending",
]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_NAMES)}
SPACER_IDX = len(CLASS_NAMES)  # extra index for blank spacer rows


def build_colormap():
    red_dark = (0.8, 0.2, 0.2)
    black_dark = (0.1, 0.1, 0.1)
    stage_colors = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
        "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
        "#c49c94",
    ]
    colors = [red_dark] + stage_colors + [black_dark]
    # Add one extra color (white) for spacer rows
    colors.append((1.0, 1.0, 1.0))
    cmap = mcolors.ListedColormap(colors)
    norm = mcolors.BoundaryNorm(
        np.arange(-0.5, len(CLASS_NAMES) + 1.5, 1.0),
        cmap.N,
        clip=True,
    )
    return cmap, norm


def load_seq_from_prediction_csv(path: str):
    df = pd.read_csv(path)
    gt = df["gt_class"].astype(str).to_numpy()
    pred = df["predicted_class"].astype(str).to_numpy()
    time = df["frame_number"].to_numpy()

    def to_idx(arr):
        idx = np.zeros_like(arr, dtype=np.int32)
        for i, name in enumerate(arr):
            idx[i] = CLASS_TO_IDX.get(name, 0)
        return idx

    noisy_idx = None
    inference_step = None
    if "noisy_input_class" in df.columns:
        noisy = df["noisy_input_class"].astype(str).to_numpy()
        noisy_idx = to_idx(noisy)
    if "inference_step" in df.columns:
        inference_step = int(df["inference_step"].iloc[0])

    return time, to_idx(gt), to_idx(pred), noisy_idx, inference_step


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize GT (top) vs Pred (bottom) for N random patients."
    )
    parser.add_argument(
        "--pred_dir",
        required=True,
        help="Directory containing *_prediction.csv files from inference.py.",
    )
    parser.add_argument(
        "-n",
        "--num_patients",
        type=int,
        default=10,
        help="Number of random patients to plot (default 10).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for patient sampling.",
    )
    parser.add_argument(
        "--output",
        default="gt_vs_pred.png",
        help="Output image path (default gt_vs_pred.png).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="Figure DPI (default 150).",
    )
    args = parser.parse_args()

    pred_dir = args.pred_dir
    files = [f for f in os.listdir(pred_dir) if f.endswith("_prediction.csv")]
    if not files:
        raise SystemExit(f"No *_prediction.csv files in {pred_dir}")
    files.sort()
    patient_ids = [f.replace("_prediction.csv", "") for f in files]

    rng = np.random.default_rng(args.seed)
    n = min(args.num_patients, len(patient_ids))
    chosen_idx = rng.choice(len(patient_ids), size=n, replace=False)
    chosen_idx.sort()
    chosen_ids = [patient_ids[i] for i in chosen_idx]

    times = None
    gt_rows = []
    pred_rows = []
    noisy_rows = []   # list of (noisy_idx, step) or None per patient
    for pid in chosen_ids:
        path = os.path.join(pred_dir, f"{pid}_prediction.csv")
        t, gt_idx, pred_idx, noisy_idx, inference_step = load_seq_from_prediction_csv(path)
        if times is None:
            times = t
        gt_rows.append(gt_idx)
        pred_rows.append(pred_idx)
        noisy_rows.append((noisy_idx, inference_step) if noisy_idx is not None else None)

    gt_mat = np.stack(gt_rows, axis=0)
    pred_mat = np.stack(pred_rows, axis=0)
    has_noisy = any(x is not None for x in noisy_rows)

    # Layout per patient:
    #   With noisy: GT, spacer, Noisy (s=X), spacer, Pred, spacer  (6 rows)
    #   Without:    GT, spacer, Pred, spacer  (4 rows)
    rows_per_patient = 6 if has_noisy else 4
    num_rows = rows_per_patient * n
    mat = np.full((num_rows, gt_mat.shape[1]), SPACER_IDX, dtype=np.int32)
    labels = []
    label_positions = []

    r = 0
    for i, pid in enumerate(chosen_ids):
        # GT row
        mat[r] = gt_mat[i]
        labels.append(f"{pid} GT")
        label_positions.append(r)
        r += 1
        # spacer
        mat[r] = SPACER_IDX
        labels.append("")
        label_positions.append(r)
        r += 1
        # Noisy input row (if we're including noisy at all)
        if has_noisy:
            if noisy_rows[i] is not None:
                noisy_idx, step = noisy_rows[i]
                mat[r] = noisy_idx
                labels.append(f"{pid} Noisy (s={step})")
            else:
                labels.append("")
            label_positions.append(r)
            r += 1
            mat[r] = SPACER_IDX
            labels.append("")
            label_positions.append(r)
            r += 1
        # Pred row
        mat[r] = pred_mat[i]
        labels.append(f"{pid} Pred")
        label_positions.append(r)
        r += 1
        # larger spacer between patients
        mat[r] = SPACER_IDX
        labels.append("")
        label_positions.append(r)
        r += 1

    cmap, norm = build_colormap()

    fig, ax = plt.subplots(figsize=(14, max(4, num_rows * 0.3)))
    im = ax.imshow(
        mat,
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        norm=norm,
        origin="upper",
        extent=[times[0] - 0.5, times[-1] + 0.5, num_rows - 0.5, -0.5],
    )
    ax.set_xlabel("Frame index")
    ax.set_yticks(label_positions)
    ax.set_yticklabels(labels, fontsize=6)
    title = "GT vs Noisy input vs Pred" if has_noisy else "GT (top) vs Pred (bottom)"
    ax.set_title(f"{title}, n={n}")

    # Draw thin vertical lines where the class changes, to form "boxes"
    n_cols = mat.shape[1]
    for r_idx, label in zip(range(num_rows), labels):
        if not label:  # skip spacer rows
            continue
        row_vals = mat[r_idx]
        for c_idx in range(1, n_cols):
            if row_vals[c_idx] != row_vals[c_idx - 1]:
                x_mid = (times[c_idx - 1] + times[c_idx]) / 2.0
                ax.vlines(
                    x_mid,
                    r_idx - 0.5,
                    r_idx + 0.5,
                    colors="black",
                    linewidth=0.3,
                    alpha=0.7,
                )

    # Legend
    legend_handles = [
        Patch(facecolor=cmap(norm(0)), label="starting"),
        Patch(facecolor=cmap(norm(len(CLASS_NAMES) - 1)), label="ending"),
    ]
    for i, name in enumerate(CLASS_NAMES[1:-1], start=1):
        legend_handles.append(Patch(facecolor=cmap(norm(i)), label=name))

    ax.legend(
        handles=legend_handles,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        fontsize=6,
        ncol=1,
    )

    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
    plt.close()
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()

