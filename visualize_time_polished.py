"""
visualize_time_polished.py

Visualize time-polished embryo stage matrices: each row = one patient (random sample),
x-axis = time_hours, color = class (18 classes: starting=shaded red, 16 stages=distinct
colors, ending=shaded black; empty bins = light gray).

Usage:

    python visualize_time_polished.py \
        --input /path/to/embryo_time_polished \
        --output plot.png \
        --n 30 \
        [--seed 42]
"""

from __future__ import annotations

import argparse
import os
from typing import List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd


# 18 classes in display order: starting, 16 stages, ending
CLASS_COLUMNS: List[str] = [
    "starting",
    "tPB2", "tPNa", "tPNf",
    "t2", "t3", "t4", "t5", "t6", "t7", "t8", "t9+",
    "tM", "tSB", "tB", "tEB", "tHB",
    "ending",
]

# Sentinel for "no class" (all zeros) for colormap
EMPTY_IDX = 18


def load_patient_class_sequence(csv_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load one patient CSV. Returns (time_hours, class_indices).
    class_indices: int array of length T with values 0..17 or EMPTY_IDX.
    """
    df = pd.read_csv(csv_path)
    missing = [c for c in CLASS_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    T = len(df)
    time_hours = df["time_hours"].to_numpy(dtype=float)
    out = np.full(T, EMPTY_IDX, dtype=np.int32)
    stage_arr = df[CLASS_COLUMNS].astype(float).to_numpy()
    for t in range(T):
        row = stage_arr[t]
        ones = np.where(row > 0.5)[0]
        if len(ones) == 1:
            out[t] = int(ones[0])
        elif len(ones) > 1:
            out[t] = int(ones[0])  # fallback first
    return time_hours, out


def build_colormap():
    """Returns (cmap, norm) for 0..17 + EMPTY_IDX. Starting=shaded red, ending=shaded black."""
    # Shaded red for starting, distinct colors for 16 stages, shaded black for ending
    red_light = (1.0, 0.7, 0.7)
    red_dark = (0.8, 0.2, 0.2)
    black_light = (0.4, 0.4, 0.4)
    black_dark = (0.15, 0.15, 0.15)
    # 16 stages: use a readable colormap (tab20 or distinct hues)
    stage_colors = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
        "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
        "#c49c94",
    ]
    colors = [red_dark] + stage_colors + [black_dark]  # 0=starting, 1..16=stages, 17=ending
    colors.append((0.92, 0.92, 0.92))  # EMPTY_IDX = light gray
    cmap = mcolors.ListedColormap(colors)
    norm = mcolors.BoundaryNorm(
        np.arange(-0.5, EMPTY_IDX + 1.5, 1.0),
        cmap.N,
        clip=True,
    )
    return cmap, norm


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize polished time-stage matrices (n random patients, time vs class color)."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Directory containing *_time_polished.csv files.",
    )
    parser.add_argument(
        "--output",
        default="embryo_time_stages.png",
        help="Output image path (default: embryo_time_stages.png).",
    )
    parser.add_argument(
        "-n",
        "--num_patients",
        type=int,
        default=25,
        help="Number of randomly chosen patients to show (default: 25).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for patient sampling (default: 42).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="Figure DPI (default: 150).",
    )
    args = parser.parse_args()

    input_dir = args.input
    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    csv_files = [f for f in os.listdir(input_dir) if f.endswith("_time_polished.csv")]
    patient_ids = sorted([f.replace("_time_polished.csv", "") for f in csv_files])
    if not patient_ids:
        raise FileNotFoundError(f"No *_time_polished.csv files in {input_dir}")

    rng = np.random.default_rng(args.seed)
    n = min(args.num_patients, len(patient_ids))
    chosen = rng.choice(len(patient_ids), size=n, replace=False)
    chosen_ids = [patient_ids[i] for i in sorted(chosen)]

    # Load all chosen patients; assume same time grid
    time_hours = None
    rows = []
    for pid in chosen_ids:
        path = os.path.join(input_dir, f"{pid}_time_polished.csv")
        if not os.path.exists(path):
            continue
        th, seq = load_patient_class_sequence(path)
        if time_hours is None:
            time_hours = th
        rows.append(seq)
    if not rows:
        raise RuntimeError("No patient data loaded.")

    matrix = np.stack(rows, axis=0)  # (n_patients, n_time_bins)
    cmap, norm = build_colormap()

    fig, ax = plt.subplots(figsize=(14, max(4, n * 0.25)))
    im = ax.imshow(
        matrix,
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        norm=norm,
        origin="upper",
        extent=[time_hours[0], time_hours[-1], n - 0.5, -0.5],
    )
    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("Patient (random sample)")
    ax.set_yticks(np.arange(n))
    ax.set_yticklabels(chosen_ids, fontsize=6)
    ax.set_title(f"Embryo stage over time (n={n} patients, seed={args.seed})")

    # Legend: starting, 16 stages, ending, empty
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=(0.8, 0.2, 0.2), label="starting"),
    ]
    for i in range(1, 17):  # tPB2 .. tHB
        legend_elements.append(
            Patch(facecolor=cmap(norm(i)), label=CLASS_COLUMNS[i])
        )
    legend_elements.append(Patch(facecolor=(0.15, 0.15, 0.15), label="ending"))
    legend_elements.append(Patch(facecolor=(0.92, 0.92, 0.92), label="empty"))
    ax.legend(
        handles=legend_elements,
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
        fontsize=6,
        ncol=1,
    )
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
    plt.close()
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
