"""
inference.py

Runs the full DDPM reverse chain on a set of patients.

What happens
────────────
  1. Load a patient's Y_0  (used only to initialise the correct shape & mask)
  2. Corrupt valid frames to Y_S  (pure noise at step S)
  3. Iteratively denoise:  Y_S → Y_{S-1} → … → Y_1 → Ŷ_0
     At each step s the model predicts the clean Ŷ_0, then we perform a
     deterministic DDIM-style update for an x0-predicting model:
         infer ε from (Y_s, Ŷ_0) and reuse it to compute Y_{s-1}.
     This avoids injecting fresh random noise each step (which can collapse
     to a single-class prediction).
  4. Save per-patient prediction CSVs with columns:
         frame_number, predicted_class, starting, tPB2, …, tHB, ending

Usage
─────
    python inference.py \
        --checkpoint  /path/to/best_model.pt \
        --monotone_dir /path/to/monotone \
        --split_json   /path/to/validation_set.json \
        --output_dir   /path/to/predictions \
        [--device cuda]
"""

import argparse
import os

import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F

from embryo_dataset import (
    EmbryoDataset,
    EmbryoTimePolishedMonotoneDataset,
    ALL_STAGE_COLS,
    TIME_POLISHED_LEN,
)
from diffusion_model import DiffusionSchedule, DiffusionTransformer


# ─────────────────────────────────────────────────────────────────
# Single patient inference
# ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def denoise_patient(
    model      : DiffusionTransformer,
    schedule   : DiffusionSchedule,
    y0_shape   : tuple,              # (1, L, 18) — used only for tensor init
    time_feat  : torch.Tensor,       # (1, L, 1)  normalised time feature
    valid_mask : torch.Tensor,       # (1, L)
    device     : torch.device,
    num_steps  : int,
) -> torch.Tensor:
    """
    Full reverse-diffusion loop for one patient.

    Returns
    -------
    pred_probs : (1, L, 18)  softmax probabilities at each frame
    """
    # ── Initialise Y_S: pure noise on valid frames, near-clean on padding ──
    y_s = torch.zeros(*y0_shape, device=device)

    # Padding frames stay as a one-hot-like zero vector (model ignores them).
    # Valid frames start as pure Gaussian noise.
    valid_3d = valid_mask.unsqueeze(-1).float().to(device)           # (1,L,1)
    y_s      = valid_3d * torch.randn_like(y_s) + (1 - valid_3d) * y_s
    initial_y_s = y_s.clone()   # for visualization (how corrupted the input was)

    pad_mask  = (~valid_mask).to(device)                             # (1,L)
    valid_dev = valid_mask.to(device)
    time_feat = time_feat.to(device)                                 # (1,550,1)

    # ── Reverse chain: S → 1 ───────────────────────────────────────────────
    for s in range(num_steps, 0, -1):
        step_t = torch.tensor([s], device=device)                    # (1,)

        # Predict clean Y_0 from current Y_s (same signature as training)
        logits      = model(y_s, time_feat, step_t, pad_mask)        # (1,L,18)
        pred_y0     = F.softmax(logits, dim=-1)                      # (1,L,18)

        if s == 1:
            # Final step: return the predicted distribution and initial noisy input
            return pred_y0, initial_y_s

        # ── Deterministic reverse update (DDIM-style for x0-prediction) ─────
        # The forward process is:
        #   y_s = sqrt(ab_s) * y0 + sqrt(1-ab_s) * eps
        # Given current y_s and predicted y0, infer eps and reuse it to step
        # to s-1. This avoids injecting fresh random noise at every step,
        # which can collapse to a single class distribution.
        idx_t   = (step_t - 1).long()                                # (1,)
        idx_tm1 = (step_t - 2).long()                                # (1,)  (since s>=2 here)

        sqrt_ab_t    = schedule._sqrt_ab[idx_t][:, None, None]       # (1,1,1)
        sqrt_1mab_t  = schedule._sqrt_1m_ab[idx_t][:, None, None]    # (1,1,1)
        sqrt_ab_tm1  = schedule._sqrt_ab[idx_tm1][:, None, None]     # (1,1,1)
        sqrt_1mab_tm1= schedule._sqrt_1m_ab[idx_tm1][:, None, None]  # (1,1,1)

        # Infer eps from current y_s and predicted y0
        denom = sqrt_1mab_t.clamp(min=1e-8)
        eps   = (y_s - sqrt_ab_t * pred_y0) / denom                  # (1,550,18)

        # Step to y_{s-1} using the same eps (eta=0)
        y_prev = sqrt_ab_tm1 * pred_y0 + sqrt_1mab_tm1 * eps         # (1,550,18)

        # Only update valid frames; keep padding frames unchanged
        y_s = valid_3d * y_prev + (1 - valid_3d) * y_s

    return pred_y0, initial_y_s   # fallback (should not reach here)


# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",   required=True,
                        help="Path to best_model.pt or latest_model.pt")
    parser.add_argument(
        "--monotone_dir",
        required=True,
        help=(
            "Directory with per-patient CSV files. "
            "For 550-frame monotone use <patient>_monotone.csv; "
            "for time-polished monotone use <patient>_time_polished_monotone.csv."
        ),
    )
    parser.add_argument("--split_json",   required=True,
                        help="Path to validation_set.json (or training_set.json)")
    parser.add_argument("--output_dir",   required=True,
                        help="Directory to save prediction CSVs")
    parser.add_argument(
        "--mode",
        choices=["reverse", "single"],
        default="reverse",
        help=(
            "'reverse' = full DDPM reverse chain from pure noise at step S; "
            "'single' = single denoising step from q_sample(Y0, s_eval)."
        ),
    )
    parser.add_argument(
        "--eval_step",
        type=int,
        default=10,
        help="Diffusion step s to use for single-step inference (default 10).",
    )
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Load checkpoint ──────────────────────────────────────────────────────
    ckpt      = torch.load(args.checkpoint, map_location=device)
    train_args = ckpt["args"]

    # Detect whether this checkpoint was trained on time-polished monotone data
    use_polished_mono = bool(train_args.get("polished_monotone_dir"))
    if use_polished_mono:
        max_len = TIME_POLISHED_LEN
        print(f"Using time-polished MONOTONE checkpoint (max_len={max_len}).")
    else:
        max_len = train_args.get("max_len", 550)
        print(f"Using frame-based checkpoint (max_len={max_len}).")

    model = DiffusionTransformer(
        num_classes     = 18,
        d_model         = train_args["d_model"],
        nhead           = train_args["nhead"],
        num_layers      = train_args["num_layers"],
        dim_feedforward = train_args.get("dim_feedforward", 1024),
        dropout         = train_args.get("dropout", 0.1),
        max_len         = max_len,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded model from epoch {ckpt['epoch']}")

    schedule = DiffusionSchedule(
        num_steps  = train_args["diffusion_steps"],
        beta_start = train_args.get("beta_start", 1e-4),
        beta_end   = train_args.get("beta_end",   0.02),
        sigma_pad  = train_args.get("sigma_pad",  0.01),
    ).to(str(device))

    # ── Dataset ──────────────────────────────────────────────────────────────
    if use_polished_mono:
        dataset = EmbryoTimePolishedMonotoneDataset(args.split_json, args.monotone_dir)
    else:
        dataset = EmbryoDataset(args.split_json, args.monotone_dir)
    print(f"Patients to process: {len(dataset)}")

    # ── Inference loop ───────────────────────────────────────────────────────
    for i in range(len(dataset)):
        item       = dataset[i]
        patient    = item["patient"]
        y0         = item["y0"].unsqueeze(0).to(device)           # (1,L,18)
        valid_mask = item["valid_mask"].unsqueeze(0)              # (1,L)
        time_feat  = item["time_feat"].unsqueeze(0).to(device)

        if args.mode == "reverse":
            # Full reverse chain from pure noise at step S
            pred_probs, initial_y_s = denoise_patient(
                model, schedule,
                y0_shape   = tuple(y0.shape),
                time_feat  = time_feat,
                valid_mask = valid_mask,
                device     = device,
                num_steps  = train_args["diffusion_steps"],
            )   # (1, L, 18), (1, L, 18)
            noisy_input = initial_y_s.argmax(dim=-1).squeeze(0).cpu().numpy()
            inference_step = train_args["diffusion_steps"]
        else:
            # Single-step denoising: simulate training condition at a chosen step s
            s = int(args.eval_step)
            s = max(1, min(s, train_args["diffusion_steps"]))
            step_t = torch.tensor([s], device=device)
            y_s = schedule.q_sample(y0, step_t, valid_mask.to(device))  # (1,L,18)
            pad_mask = (~valid_mask).to(device)
            logits = model(y_s, time_feat, step_t, pad_mask)            # (1,L,18)
            pred_probs = F.softmax(logits, dim=-1).detach()
            noisy_input = y_s.argmax(dim=-1).squeeze(0).cpu().numpy()
            inference_step = s

        # ── Convert to output dataframe ──────────────────────────────────────
        prob_np = pred_probs.squeeze(0).detach().cpu().numpy()          # (L, 18)
        pred_idx = prob_np.argmax(axis=-1)                     # (L,)
        pred_labels = np.array([ALL_STAGE_COLS[c] for c in pred_idx])
        noisy_labels = np.array([ALL_STAGE_COLS[c] for c in noisy_input])

        # Load ground-truth class labels to identify padding vs valid frames
        if use_polished_mono:
            gt_path = os.path.join(args.monotone_dir, f"{patient}_time_polished_monotone.csv")
        else:
            gt_path = os.path.join(args.monotone_dir, f"{patient}_monotone.csv")
        gt_df = pd.read_csv(gt_path)
        gt_classes = gt_df["class"].astype(str).values         # (L,)

        is_padding = np.isin(gt_classes, ["starting", "ending"])
        is_valid = ~is_padding

        # For readability: keep GT labels on padding frames, model labels on valid frames
        final_labels = np.where(is_padding, gt_classes, pred_labels)

        df = pd.DataFrame(prob_np, columns=ALL_STAGE_COLS)
        df.insert(0, "frame_number", range(1, prob_np.shape[0] + 1))
        df.insert(1, "gt_class", gt_classes)
        df.insert(2, "is_valid", is_valid.astype(int))
        df.insert(3, "predicted_class", final_labels)
        df.insert(4, "noisy_input_class", noisy_labels)
        df.insert(5, "inference_step", inference_step)

        out_path = os.path.join(args.output_dir, f"{patient}_prediction.csv")
        df.to_csv(out_path, index=False)

        print(f"  [{i+1:>4}/{len(dataset)}]  {patient}  →  {out_path}")

    print(f"\nDone.  Predictions saved to: {args.output_dir}")


if __name__ == "__main__":
    main()