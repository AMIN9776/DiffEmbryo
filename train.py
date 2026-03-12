"""
train.py

Training loop for the embryo diffusion action segmentation model.

What happens each iteration
────────────────────────────
  1. Sample a random diffusion step s ~ Uniform{1, …, S} per patient
  2. Corrupt Y_0 → Y_s  (valid frames: full noise; padding: tiny noise)
  3. Mask padding positions in the transformer  (they are invisible to attention)
  4. Decode Y_s → logits Ŷ_0
  5. Compute L_sum = L_ce + λ_vit · L_viterbi + λ_bd · L_boundary
  6. Backpropagate, clip gradients, step optimiser

Checkpointing
─────────────
  best_model.pt is saved whenever validation loss improves.
  latest_model.pt is overwritten every epoch so training can be resumed.

Usage
─────
    python train.py \
        --monotone_dir  /path/to/monotone \
        --split_dir     /path/to/splits \
        --output_dir    /path/to/checkpoints \
        [--epochs 100] \
        [--batch_size 8] \
        [--lr 1e-4] \
        [--d_model 256] \
        [--nhead 8] \
        [--num_layers 6] \
        [--diffusion_steps 500] \
        [--lambda_viterbi 1.0] \
        [--lambda_boundary 1.0] \
        [--resume /path/to/latest_model.pt] \
        [--device cuda]
"""

import argparse
import math
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader

from embryo_dataset import (
    EmbryoDataset,
    EmbryoTimePolishedDataset,
    EmbryoTimePolishedMonotoneDataset,
    ALL_STAGE_COLS,
    TIME_POLISHED_LEN,
)
from diffusion_model import DiffusionSchedule, DiffusionTransformer
from losses import combined_loss, compute_class_weights


# ─────────────────────────────────────────────────────────────────
# Sample printer  — shows one patient's input vs target
# ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def print_sample(model, schedule, dataset, device, epoch):
    """
    Pick the first validation patient and show:
      1. GT vs Pred sequence (transitions only, compressed)
      2. Top-3 class probabilities at 5 evenly-spaced valid frames
    Uses step=10 (low noise) so the model has a fair chance.
    """
    model.eval()
    item       = dataset[0]
    patient    = item["patient"]
    y0         = item["y0"].unsqueeze(0).to(device)
    time_feat  = item["time_feat"].unsqueeze(0).to(device)
    valid_mask = item["valid_mask"].unsqueeze(0).to(device)

    # Low noise step — input is still mostly clean, model has real signal
    step = torch.tensor([10], device=device)
    y_s  = schedule.q_sample(y0, step, valid_mask)

    pad_mask = ~valid_mask
    logits   = model(y_s, time_feat, step, pad_mask)
    probs    = F.softmax(logits, dim=-1)

    vm        = valid_mask[0].cpu()
    gt_idx    = y0[0].argmax(-1).cpu()[vm]
    pred_idx  = probs[0].argmax(-1).cpu()[vm]

    y_s_masked = y_s[0].cpu().clone()
    for blocked in [0, 16, 17]:
        y_s_masked[:, blocked] = -1e9
    noisy_idx = y_s_masked.argmax(-1)[vm]

    gt_labels    = [ALL_STAGE_COLS[i] for i in gt_idx.tolist()]
    pred_labels  = [ALL_STAGE_COLS[i] for i in pred_idx.tolist()]
    noisy_labels = [ALL_STAGE_COLS[i] for i in noisy_idx.tolist()]

    def compress(labels):
        out, prev = [], None
        for i, l in enumerate(labels):
            if l != prev:
                out.append(f"f{i}:{l}")
                prev = l
        return " → ".join(out)

    correct = (pred_idx == gt_idx).float().mean().item()
    print(f"\n  ── Sample [{patient}]  epoch {epoch}  step=10  acc={correct:.2%} ──")
    print(f"  GT   : {compress(gt_labels)}")
    print(f"  Pred : {compress(pred_labels)}")
    print(f"  Noisy: {compress(noisy_labels)}")

    # Top-3 at 5 evenly-spaced valid frames
    valid_positions = vm.nonzero(as_tuple=True)[0]
    T = len(valid_positions)
    print(f"  Top-3 probs at sampled frames (GT → top predictions):")
    for r in [0.1, 0.3, 0.5, 0.7, 0.9]:
        pos  = valid_positions[int(T * r)].item()
        p    = probs[0, pos].cpu()
        gt_c = ALL_STAGE_COLS[y0[0, pos].argmax().item()]
        top3 = p.topk(3)
        top3_str = "  ".join(
            f"{ALL_STAGE_COLS[idx]}={val:.2f}"
            for val, idx in zip(top3.values.tolist(), top3.indices.tolist())
        )
        print(f"    frame {pos:>3}  GT={gt_c:<6}  {top3_str}")
    print()




# ─────────────────────────────────────────────────────────────────
# One training epoch
# ─────────────────────────────────────────────────────────────────

def train_epoch(model, schedule, loader, optimizer, device,
                lambda_ce, lambda_vit, lambda_bd, lambda_smooth=0.0, class_weights=None):
    model.train()
    sums = {"total": 0.0, "ce": 0.0, "viterbi": 0.0, "boundary": 0.0, "smooth": 0.0}
    S = schedule.num_steps
    step_counts = torch.zeros(S, dtype=torch.long, device=device)

    for batch in loader:
        y0         = batch["y0"].to(device)           # (B, 550, 18)
        time_feat  = batch["time_feat"].to(device)    # (B, 550, 1)
        valid_mask = batch["valid_mask"].to(device)   # (B, 550)  bool
        B          = y0.shape[0]

        # Sample a diffusion step per patient
        steps = schedule.sample_steps(B, device)      # (B,) 1-indexed
        step_counts += torch.bincount(
            (steps - 1).clamp(0, S - 1),
            minlength=S,
        )

        # Corrupt Y_0 → Y_s  (only valid frames get full noise)
        y_s = schedule.q_sample(y0, steps, valid_mask)   # (B, 550, 18)

        # Padding mask for transformer: True = ignore this position as a key
        pad_mask = ~valid_mask                            # (B, 550)

        # Denoise — model now also receives time_feat
        logits = model(y_s, time_feat, steps, pad_mask)  # (B, 550, 18)

        # Losses
        losses = combined_loss(logits, y0, valid_mask,
                               lambda_ce=lambda_ce,
                               lambda_vit=lambda_vit,
                               lambda_bd=lambda_bd,
                               lambda_smooth=lambda_smooth,
                               class_weights=class_weights)

        optimizer.zero_grad()
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        for k in sums:
            sums[k] += losses[k] if isinstance(losses[k], float) else losses[k].item()

    n = max(len(loader), 1)
    losses_out = {k: v / n for k, v in sums.items()}
    losses_out["step_counts"] = step_counts.cpu().numpy()
    return losses_out


# ─────────────────────────────────────────────────────────────────
# Validation epoch
# ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def val_epoch(model, schedule, loader, device, lambda_ce, lambda_vit, lambda_bd, lambda_smooth=0.0, class_weights=None):
    model.eval()
    sums = {"total": 0.0, "ce": 0.0, "viterbi": 0.0, "boundary": 0.0, "smooth": 0.0}

    for batch in loader:
        y0         = batch["y0"].to(device)
        time_feat  = batch["time_feat"].to(device)
        valid_mask = batch["valid_mask"].to(device)
        B          = y0.shape[0]

        steps    = schedule.sample_steps(B, device)
        y_s      = schedule.q_sample(y0, steps, valid_mask)
        pad_mask = ~valid_mask

        logits = model(y_s, time_feat, steps, pad_mask)
        losses = combined_loss(logits, y0, valid_mask,
                               lambda_ce=lambda_ce,
                               lambda_vit=lambda_vit,
                               lambda_bd=lambda_bd,
                               lambda_smooth=lambda_smooth,
                               class_weights=class_weights)

        for k in sums:
            sums[k] += losses[k] if isinstance(losses[k], float) else losses[k].item()

    n = max(len(loader), 1)
    return {k: v / n for k, v in sums.items()}


# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()

    # Paths
    parser.add_argument("--monotone_dir", default=None,
                        help="Directory with <patient>_monotone.csv (frame-based); unused if --polished_dir / --polished_monotone_dir are set")
    parser.add_argument("--polished_dir", default=None,
                        help="Directory with <patient>_time_polished.csv (time-based, one-hot targets).")
    parser.add_argument("--polished_monotone_dir", default=None,
                        help="Directory with <patient>_time_polished_monotone.csv (time-based, monotone -1/1/0 targets).")
    parser.add_argument("--split_dir", required=True,
                        help="Directory with training_set.json and validation_set.json (create via split_patients.py)")
    parser.add_argument("--output_dir", required=True,
                        help="Where to save checkpoints and logs")
    parser.add_argument("--resume", default=None,
                        help="Path to latest_model.pt to resume training")

    # Training hyper-parameters
    parser.add_argument("--epochs",     type=int,   default=100)
    parser.add_argument("--batch_size", type=int,   default=8)
    parser.add_argument("--num_workers",type=int,   default=4)
    parser.add_argument("--lr",         type=float, default=3e-5,
                        help="Peak learning rate after warmup (default: 3e-5 for large models)")
    parser.add_argument("--warmup_epochs", type=int, default=10,
                        help="Number of linear LR warmup epochs before cosine decay")

    # Loss weights
    parser.add_argument("--lambda_ce",       type=float, default=1.0,
                        help="Weight for cross-entropy (default 1.0). Increase to push correct class predictions harder (e.g. escape tSB collapse).")
    parser.add_argument("--lambda_viterbi",  type=float, default=0.7,
                        help="Weight for Viterbi monotonicity loss (default 0.7). Lower than 1.0 gives CE more room to fix late-stage confusions like t8/t9+ vs tSB.")
    parser.add_argument("--lambda_boundary", type=float, default=1.0)
    parser.add_argument("--lambda_smooth", type=float, default=0.0,
                        help="Weight for smoothness loss (MSE of log-probs between adjacent valid frames). 0 = disabled.")
    parser.add_argument("--late_stage_boost", type=float, default=1.0,
                        help="If > 1, multiply class weights for t8, t9+, tM only (default 1.0).")
    parser.add_argument("--max_weight_ratio", type=float, default=2.0,
                        help="Cap class weight at median(weights)*this so one short stage (e.g. t3) does not dominate (default 2.0). Use 0 to disable.")
    parser.add_argument("--min_weight_ratio", type=float, default=2.0,
                        help="Floor class weight at median(weights)/this so long stages (e.g. tPNa, t9+) are not ignored (default 2.0). Use 0 to disable.")
    parser.add_argument("--no_class_weights", action="store_true",
                        help="If set, do not use per-class weights in CE (uniform weights).")

    # Model architecture
    parser.add_argument("--d_model",         type=int,   default=256)
    parser.add_argument("--nhead",           type=int,   default=8)
    parser.add_argument("--num_layers",      type=int,   default=6)
    parser.add_argument("--dim_feedforward", type=int,   default=1024)
    parser.add_argument("--dropout",         type=float, default=0.1)

    # Diffusion
    parser.add_argument("--diffusion_steps", type=int,   default=100)
    parser.add_argument("--beta_start",      type=float, default=1e-4)
    parser.add_argument("--beta_end",        type=float, default=0.02)
    parser.add_argument("--sigma_pad",       type=float, default=0.01,
                        help="Noise std applied to starting/ending padding frames")
    parser.add_argument("--step_sample",     type=str,   default="sqrt_low",
                        choices=["sqrt_low", "uniform", "sqrt_high"],
                        help="Training step sampling: sqrt_low (more low noise), uniform (all steps equal), sqrt_high (more high noise, better for denoising from pure noise)")

    parser.add_argument("--sample_every", type=int, default=5,
                        help="Print an input/target sample every N epochs (0 = never)")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    use_polished_mono = args.polished_monotone_dir is not None
    use_polished = (not use_polished_mono) and (args.polished_dir is not None)

    if use_polished_mono:
        data_dir = args.polished_monotone_dir
        max_len = TIME_POLISHED_LEN
        print("Using time-polished MONOTONE data (y0 in {-1,0,1}, valid = embryo stages; starting/ending = padding)")
    elif use_polished:
        data_dir = args.polished_dir
        max_len = TIME_POLISHED_LEN
        print("Using time-polished data (one-hot targets, valid = embryo stages; starting/ending = padding)")
    else:
        if args.monotone_dir is None:
            raise SystemExit("Provide one of --polished_monotone_dir, --polished_dir or --monotone_dir")
        data_dir = args.monotone_dir
        max_len = 550

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Datasets ────────────────────────────────────────────────────────────
    if use_polished_mono:
        train_set = EmbryoTimePolishedMonotoneDataset(
            os.path.join(args.split_dir, "training_set.json"),
            args.polished_monotone_dir,
        )
        val_set = EmbryoTimePolishedMonotoneDataset(
            os.path.join(args.split_dir, "validation_set.json"),
            args.polished_monotone_dir,
        )
    elif use_polished:
        train_set = EmbryoTimePolishedDataset(
            os.path.join(args.split_dir, "training_set.json"),
            args.polished_dir,
        )
        val_set = EmbryoTimePolishedDataset(
            os.path.join(args.split_dir, "validation_set.json"),
            args.polished_dir,
        )
    else:
        train_set = EmbryoDataset(
            os.path.join(args.split_dir, "training_set.json"),
            args.monotone_dir,
        )
        val_set = EmbryoDataset(
            os.path.join(args.split_dir, "validation_set.json"),
            args.monotone_dir,
        )

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size,
        shuffle=True, num_workers=args.num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers, pin_memory=True,
    )

    print(f"Train: {len(train_set)} patients  |  Val: {len(val_set)} patients")

    # ── Class weights (computed from training set only, unless disabled) ─────
    if args.no_class_weights:
        print("\n[Class weights] Disabled (using uniform CE weights).\n")
        class_weights = None
    else:
        print("\nComputing class weights from training set...")
        if use_polished_mono:
            class_weights = compute_class_weights(
                os.path.join(args.split_dir, "training_set.json"),
                data_dir,
                device=device,
                file_suffix="_time_polished_monotone.csv",
                late_stage_boost=args.late_stage_boost,
                max_weight_ratio=args.max_weight_ratio,
                min_weight_ratio=args.min_weight_ratio,
            )
        else:
            class_weights = compute_class_weights(
                os.path.join(args.split_dir, "training_set.json"),
                data_dir,
                device=device,
                file_suffix="_time_polished.csv" if use_polished else "_monotone.csv",
                late_stage_boost=args.late_stage_boost,
                max_weight_ratio=args.max_weight_ratio,
                min_weight_ratio=args.min_weight_ratio,
            )
        print()

    # ── Model ────────────────────────────────────────────────────────────────
    model = DiffusionTransformer(
        num_classes        = 18,
        num_input_features = 19,   # 18 noisy classes + 1 time_hours feature
        d_model            = args.d_model,
        nhead              = args.nhead,
        num_layers         = args.num_layers,
        dim_feedforward    = args.dim_feedforward,
        dropout            = args.dropout,
        max_len            = max_len,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    # ── Diffusion schedule ───────────────────────────────────────────────────
    schedule = DiffusionSchedule(
        num_steps   = args.diffusion_steps,
        beta_start  = args.beta_start,
        beta_end    = args.beta_end,
        sigma_pad   = args.sigma_pad,
        step_sample = args.step_sample,
    ).to(str(device))
    print(f"Step sampling: {args.step_sample}")

    # ── Optimiser & scheduler ────────────────────────────────────────────────
    # Lower base LR (3e-5) + linear warmup prevents mode collapse in large models.
    # Without warmup a 512d transformer immediately fires large gradients that
    # push the model into a degenerate single-class prediction.
    #
    # Schedule:
    #   epochs 1 … warmup_epochs : LR linearly ramps 0 → args.lr
    #   epochs warmup+1 … epochs : cosine decay args.lr → 1e-6
    #
    # Cosine decay keeps later training gentle so the model can refine without
    # over-committing to the top prediction, giving the second-best class a chance.
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    def lr_lambda(epoch):
        # epoch is 0-indexed here (scheduler calls after each step)
        if epoch < args.warmup_epochs:
            return float(epoch + 1) / float(args.warmup_epochs)
        progress = (epoch - args.warmup_epochs) / max(args.epochs - args.warmup_epochs, 1)
        return max(1e-6 / args.lr, 0.5 * (1.0 + math.cos(math.pi * progress)))

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ── Resume ───────────────────────────────────────────────────────────────
    start_epoch   = 1
    best_val_loss = float("inf")

    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        lr_scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch   = ckpt["epoch"] + 1
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        print(f"Resumed from epoch {ckpt['epoch']}  (best val loss: {best_val_loss:.4f})")

    # ── Training loop ────────────────────────────────────────────────────────
    log_path = os.path.join(args.output_dir, "train_log.csv")
    if start_epoch == 1:
        with open(log_path, "w") as f:
            f.write("epoch,train_total,train_ce,train_viterbi,train_boundary,train_smooth,"
                    "val_total,val_ce,val_viterbi,val_boundary,val_smooth\n")

    for epoch in range(start_epoch, args.epochs + 1):

        t_losses = train_epoch(
            model, schedule, train_loader, optimizer, device,
            lambda_ce=args.lambda_ce,
            lambda_vit=args.lambda_viterbi,
            lambda_bd=args.lambda_boundary,
            lambda_smooth=args.lambda_smooth,
            class_weights=class_weights,
        )
        step_counts = t_losses.pop("step_counts", None)   # (S,) counts for steps 1..S
        v_losses = val_epoch(model, schedule, val_loader, device,
                             lambda_ce=args.lambda_ce,
                             lambda_vit=args.lambda_viterbi,
                             lambda_bd=args.lambda_boundary,
                             lambda_smooth=args.lambda_smooth,
                             class_weights=class_weights)
        lr_scheduler.step()

        # ── Optional sample print ─────────────────────────────────────────────
        if args.sample_every > 0 and epoch % args.sample_every == 0:
            print_sample(model, schedule, val_set, device, epoch)

        current_lr = optimizer.param_groups[0]["lr"]

        # ── Console log ──────────────────────────────────────────────────────
        print(
            f"Epoch {epoch:03d}/{args.epochs}  lr={current_lr:.2e}  "
            f"| Train  total={t_losses['total']:.4f}  "
            f"ce={t_losses['ce']:.4f}  vit={t_losses['viterbi']:.4f}  "
            f"bd={t_losses['boundary']:.4f}  sm={t_losses['smooth']:.4f}  "
            f"| Val  total={v_losses['total']:.4f}  "
            f"ce={v_losses['ce']:.4f}  vit={v_losses['viterbi']:.4f}  "
            f"bd={v_losses['boundary']:.4f}  sm={v_losses['smooth']:.4f}"
        )

        # ── Noise step counts (per epoch) ─────────────────────────────────────
        if step_counts is not None:
            c = step_counts
            total_seen = int(c.sum())
            min_c, max_c = int(c.min()), int(c.max())
            mean_c = total_seen / len(c)
            unseen = (np.where(c == 0)[0] + 1).tolist()   # 1-indexed step numbers
            unseen_str = f", unseen: {unseen}" if len(unseen) <= 20 else f", unseen: {len(unseen)} steps"
            # 10 bins: 1–10, 11–20, …, 91–100
            bins = [int(c[i*10:(i+1)*10].sum()) for i in range(10)]
            print(
                f"  Noise steps this epoch: total={total_seen}  "
                f"min={min_c} max={max_c} mean={mean_c:.1f}{unseen_str}"
            )
            print(f"  Step bins [1–10],[11–20],…,[91–100]: {bins}")

        # ── CSV log ──────────────────────────────────────────────────────────
        with open(log_path, "a") as f:
            f.write(
                f"{epoch},"
                f"{t_losses['total']:.6f},{t_losses['ce']:.6f},"
                f"{t_losses['viterbi']:.6f},{t_losses['boundary']:.6f},{t_losses['smooth']:.6f},"
                f"{v_losses['total']:.6f},{v_losses['ce']:.6f},"
                f"{v_losses['viterbi']:.6f},{v_losses['boundary']:.6f},{v_losses['smooth']:.6f}\n"
            )

        # ── Checkpointing ─────────────────────────────────────────────────────
        state = {
            "epoch"               : epoch,
            "model_state_dict"    : model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": lr_scheduler.state_dict(),
            "best_val_loss"       : best_val_loss,
            "args"                : vars(args),
        }

        # Always save latest
        torch.save(state, os.path.join(args.output_dir, "latest_model.pt"))

        # Save best
        if v_losses["total"] < best_val_loss:
            best_val_loss = v_losses["total"]
            state["best_val_loss"] = best_val_loss
            torch.save(state, os.path.join(args.output_dir, "best_model.pt"))
            print(f"  ✓ New best model saved  (val loss: {best_val_loss:.4f})")

    print(f"\nTraining complete.  Best val loss: {best_val_loss:.4f}")
    print(f"Checkpoints: {args.output_dir}")


if __name__ == "__main__":
    main()