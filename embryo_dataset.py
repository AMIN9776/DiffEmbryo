"""
embryo_dataset.py

PyTorch Dataset for the monotone-encoded embryo stage matrices.

Each sample contains:
  y0          (550, 18)  one-hot ground truth  — the clean signal the model must recover
  time_feat   (550, 1)   normalised time_hours — patient-specific temporal signal
  valid_mask  (550,)     True for frames with real embryo stages (not starting/ending)
  patient     str        patient name

Why one-hot and not the monotone matrix?
  The diffusion process adds Gaussian noise to a continuous signal and asks the
  network to recover it.  One-hot vectors (values in {0, 1}) are a clean,
  normalised representation.  The monotone values (-1, 0, 1) encode history,
  but the network's job here is purely to classify each frame — history is
  implicitly enforced by the Viterbi loss.  The one-hot target also plugs
  directly into cross-entropy and the boundary / Viterbi losses.

Why time_hours?
  Without any per-patient input the model can only predict population-average
  stage durations.  time_hours breaks this ceiling — stage transitions happen
  at biologically meaningful times, and two patients with different time axes
  tell very different stories even if their stage sequences look similar.
  Padding frames (starting/ending) have NaN time → replaced with 0 after
  normalisation so they carry no misleading temporal information.
"""

import os
import json

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


# Ordered list of all 18 class columns (matches the monotone CSV)
ALL_STAGE_COLS = [
    "starting",
    "tPB2", "tPNa", "tPNf",
    "t2", "t3", "t4", "t5", "t6", "t7", "t8", "t9+",
    "tM", "tSB", "tB", "tEB", "tHB",
    "ending",
]

NUM_CLASSES  = 18   # starting + 16 embryo stages + ending
TOTAL_FRAMES = 550

# Time-polished CSVs have a fixed global time grid (e.g. 492 rows for bin_size=0.3)
TIME_POLISHED_LEN = 492


class EmbryoTimePolishedDataset(Dataset):
    """
    Loads per-patient *_time_polished.csv files (output of embryo_time_polish.py).

    Same 18 classes as monotone; sequence length is the global time grid (e.g. 492).
    valid_mask: True only for rows where class is one of the 16 embryo stages
    (tPB2..tHB). Starting/ending rows are False — they get sigma_pad noise only,
    not full diffusion noise; we keep them only for dimension consistency.
    """

    def __init__(self, json_path: str, polished_dir: str):
        with open(json_path, "r") as f:
            data = json.load(f)
        all_patients = data["patients"]
        self.patients = [
            p for p in all_patients
            if os.path.exists(os.path.join(polished_dir, f"{p}_time_polished.csv"))
        ]
        missing = len(all_patients) - len(self.patients)
        if missing:
            print(f"[EmbryoTimePolishedDataset] WARNING: {missing} patients have no "
                  f"time_polished CSV and will be skipped.")
        self.polished_dir = polished_dir

    def __len__(self) -> int:
        return len(self.patients)

    def __getitem__(self, idx: int) -> dict:
        patient = self.patients[idx]
        path = os.path.join(self.polished_dir, f"{patient}_time_polished.csv")
        df = pd.read_csv(path)

        # Valid mask: only the 16 embryo stages are "valid" (full noise)
        # starting / ending → False (sigma_pad only)
        valid_mask = ~df["class"].isin(["starting", "ending"]).values

        stage_matrix = df[ALL_STAGE_COLS].values.astype(np.float32)
        y0 = (stage_matrix == 1.0).astype(np.float32)

        time_raw = df["time_hours"].values.astype(np.float32)
        valid_times = time_raw[valid_mask]
        if len(valid_times) > 0 and not np.all(np.isnan(valid_times)):
            t_min = np.nanmin(valid_times)
            t_max = np.nanmax(valid_times)
            denom = (t_max - t_min) if (t_max - t_min) > 0 else 1.0
            time_norm = (time_raw - t_min) / denom
        else:
            time_norm = np.zeros(len(df), dtype=np.float32)
        time_norm = np.nan_to_num(time_norm, nan=0.0).astype(np.float32)
        time_feat = time_norm[:, np.newaxis]

        return {
            "patient": patient,
            "y0": torch.from_numpy(y0),
            "time_feat": torch.from_numpy(time_feat),
            "valid_mask": torch.from_numpy(valid_mask),
        }


class EmbryoTimePolishedMonotoneDataset(Dataset):
    """
    Loads per-patient *_time_polished_monotone.csv files.

    Stage columns are monotone-encoded (-1, 1, 0) per time bin:
      - previous stages  → -1
      - current stage    →  1
      - upcoming stages  →  0

    We feed this full matrix as y0 into the diffusion schedule and model.
    Class labels for losses are still obtained via argmax over the 18
    columns, so the active stage index is the same as for one-hot.
    """

    def __init__(self, json_path: str, polished_mono_dir: str):
        with open(json_path, "r") as f:
            data = json.load(f)
        all_patients = data["patients"]
        self.patients = [
            p for p in all_patients
            if os.path.exists(os.path.join(polished_mono_dir, f"{p}_time_polished_monotone.csv"))
        ]
        missing = len(all_patients) - len(self.patients)
        if missing:
            print(
                f"[EmbryoTimePolishedMonotoneDataset] WARNING: {missing} patients have no "
                f"time_polished_monotone CSV and will be skipped."
            )
        self.polished_mono_dir = polished_mono_dir

    def __len__(self) -> int:
        return len(self.patients)

    def __getitem__(self, idx: int) -> dict:
        patient = self.patients[idx]
        path = os.path.join(self.polished_mono_dir, f"{patient}_time_polished_monotone.csv")
        df = pd.read_csv(path)

        # Valid mask: only embryo stages are valid; starting/ending are padding
        valid_mask = ~df["class"].isin(["starting", "ending"]).values

        # y0 is the full monotone matrix (starting + 16 stages + ending)
        y0 = df[ALL_STAGE_COLS].values.astype(np.float32)

        time_raw = df["time_hours"].values.astype(np.float32)
        valid_times = time_raw[valid_mask]
        if len(valid_times) > 0 and not np.all(np.isnan(valid_times)):
            t_min = np.nanmin(valid_times)
            t_max = np.nanmax(valid_times)
            denom = (t_max - t_min) if (t_max - t_min) > 0 else 1.0
            time_norm = (time_raw - t_min) / denom
        else:
            time_norm = np.zeros(len(df), dtype=np.float32)
        time_norm = np.nan_to_num(time_norm, nan=0.0).astype(np.float32)
        time_feat = time_norm[:, np.newaxis]

        return {
            "patient": patient,
            "y0": torch.from_numpy(y0),
            "time_feat": torch.from_numpy(time_feat),
            "valid_mask": torch.from_numpy(valid_mask),
        }


class EmbryoDataset(Dataset):
    """
    Loads per-patient *_monotone.csv files produced by embryo_monotone_encode.py.

    The monotone CSV has 21 columns:
        frame_number, time_hours, class,
        starting, tPB2, ..., tHB, ending

    This dataset returns one-hot Y_0, normalised time features, and a boolean
    valid_mask so the diffusion model knows which frames to noise and which to
    leave frozen.
    """

    def __init__(self, json_path: str, monotone_dir: str):
        """
        Args:
            json_path    : Path to training_set.json or validation_set.json
            monotone_dir : Directory containing <patient>_monotone.csv files
        """
        with open(json_path, "r") as f:
            data = json.load(f)

        all_patients = data["patients"]

        # Keep only patients whose monotone CSV exists
        self.patients = [
            p for p in all_patients
            if os.path.exists(os.path.join(monotone_dir, f"{p}_monotone.csv"))
        ]

        missing = len(all_patients) - len(self.patients)
        if missing:
            print(f"[EmbryoDataset] WARNING: {missing} patients have no monotone CSV "
                  f"and will be skipped.")

        self.monotone_dir = monotone_dir

    def __len__(self) -> int:
        return len(self.patients)

    def __getitem__(self, idx: int) -> dict:
        patient = self.patients[idx]
        path    = os.path.join(self.monotone_dir, f"{patient}_monotone.csv")

        df = pd.read_csv(path)

        # ── Sanity check ────────────────────────────────────────────────────
        assert len(df) == TOTAL_FRAMES, (
            f"[{patient}] Expected {TOTAL_FRAMES} rows, got {len(df)}.  "
            f"Re-run embryo_monotone_encode.py with --pad."
        )

        # ── Valid mask ───────────────────────────────────────────────────────
        # True  → real embryo stage frame  (will be noised during training)
        # False → starting / ending padding (frozen, tiny noise only)
        valid_mask = ~df["class"].isin(["starting", "ending"]).values   # (550,)

        # ── One-hot ground truth Y_0 ─────────────────────────────────────────
        # The monotone matrix has exactly one column equal to 1.0 per row
        # (the active stage).  We extract that as the one-hot target.
        stage_matrix = df[ALL_STAGE_COLS].values.astype(np.float32)     # (550, 18)
        y0 = (stage_matrix == 1.0).astype(np.float32)                   # (550, 18)

        # ── Time feature ─────────────────────────────────────────────────────
        # Normalise time_hours to [0, 1] using only valid frames so padding
        # NaNs don't affect the scale.  Padding frames → 0 after normalisation.
        #
        # Why normalise per-patient?
        #   Different patients have different total recording durations.
        #   Per-patient [0,1] normalisation makes the time signal comparable
        #   across patients while preserving the relative timing of stages.
        time_raw = df["time_hours"].values.astype(np.float32)           # (550,)
        valid_times = time_raw[valid_mask]

        if len(valid_times) > 0 and not np.all(np.isnan(valid_times)):
            t_min = np.nanmin(valid_times)
            t_max = np.nanmax(valid_times)
            denom = (t_max - t_min) if (t_max - t_min) > 0 else 1.0
            time_norm = (time_raw - t_min) / denom                      # (550,)
        else:
            time_norm = np.zeros(TOTAL_FRAMES, dtype=np.float32)

        # Padding frames have NaN time → set to 0 (model ignores them anyway)
        time_norm = np.nan_to_num(time_norm, nan=0.0).astype(np.float32)
        time_feat = time_norm[:, np.newaxis]                            # (550, 1)

        return {
            "patient"   : patient,
            "y0"        : torch.from_numpy(y0),                         # (550, 18)
            "time_feat" : torch.from_numpy(time_feat),                  # (550, 1)
            "valid_mask": torch.from_numpy(valid_mask),                 # (550,)  bool
        }