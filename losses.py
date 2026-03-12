"""
losses.py

Three training losses for the embryo diffusion action segmentation model.

  L_ce      Cross-entropy on valid frames
  L_viterbi  Differentiable Viterbi (forward / sum-product algorithm in log-space)
             Penalises non-monotone predictions
  L_bd      Boundary alignment loss (BCE between smoothed GT boundaries and
             predicted boundary probabilities)

All losses are computed only on valid frames (starting/ending → zero weight).

─────────────────────────────────────────────────────────────────────────────
Background: Differentiable Viterbi via the Forward Algorithm
─────────────────────────────────────────────────────────────────────────────
The standard Viterbi algorithm uses max-product to find the single best path.
To make it differentiable we replace max with logsumexp (sum-product / forward
algorithm), which computes the log-likelihood of *all* valid monotone paths.

Monotone constraint: from class c at time t, only two transitions are legal:
    stay:    c → c          (embryo remains in same stage)
    advance: c → c+1        (embryo moves to next stage)

Given log-probabilities lp[t, c] = log P(class=c | frame t), the forward
recurrence is:

    log α[0, c] = lp[0, c]

    log α[t, c] = lp[t, c] + logsumexp(
                      log α[t-1, c],        ← stayed at c
                      log α[t-1, c-1]       ← advanced from c-1
                  )

The log-likelihood of all monotone paths = logsumexp_c log α[T-1, c].

Loss = −log_likelihood  (maximise the probability mass on monotone paths).

This is fully differentiable via autograd because logsumexp is smooth.
"""

import os
import json
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

from embryo_dataset import ALL_STAGE_COLS


# ─────────────────────────────────────────────────────────────────
# Class weight computation  (call once before training)
# ─────────────────────────────────────────────────────────────────

# Indices of late stages (optional boost): t8, t9+, tM
LATE_STAGE_INDICES = [ALL_STAGE_COLS.index(c) for c in ("t8", "t9+", "tM")]


def compute_class_weights(split_json: str,
                          monotone_dir: str,
                          device: torch.device,
                          *,
                          file_suffix: str = "_monotone.csv",
                          late_stage_boost: float = 1.0,
                          max_weight_ratio: float = 2.0,
                          min_weight_ratio: float = 2.0) -> torch.Tensor:
    """
    Compute class weights by inverse duration (frame count) in the training set.

    Shorter stages (fewer frames in time) get higher weight so the model pays
    equal attention to brief stages (e.g. t3, tPNf) and long ones (e.g. tPNa, t9+):

        w_c = total_valid_frames / (num_classes * count_c)

    Optional: late_stage_boost > 1 multiplies weights for t8, t9+, tM only.
    Optional: max_weight_ratio caps each weight at median(weights) * max_weight_ratio
              so one very short stage does not dominate (e.g. avoid over-predicting t3).
    Optional: min_weight_ratio raises very small weights up to
              median(weights) / min_weight_ratio so the longest stages
              (e.g. tPNa, t9+) are not almost ignored.

    Classes with too few samples get weight 0. starting/ending are 0.

    Args:
        split_json: path to training_set.json
        monotone_dir: directory containing per-patient CSVs
        device: where to put the weight tensor
        file_suffix: e.g. "_monotone.csv" or "_time_polished.csv"
        late_stage_boost: if > 1, multiply weights for t8, t9+, tM by this (default 1.0)
        max_weight_ratio: cap weight at median * this (default 2.0); use 0 to disable
        min_weight_ratio: floor weight at median / this (default 2.0); use 0 to disable

    Returns (18,) float tensor on `device`.
    """
    with open(split_json) as f:
        patients = json.load(f)["patients"]

    counts = np.zeros(len(ALL_STAGE_COLS), dtype=np.float64)

    for patient in patients:
        path = os.path.join(monotone_dir, f"{patient}{file_suffix}")
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        valid_df = df[~df["class"].isin(["starting", "ending"])]
        if len(valid_df) == 0:
            continue
        stage_vals = valid_df[ALL_STAGE_COLS].values    # (T, 18)
        class_idx  = stage_vals.argmax(axis=1)          # (T,)
        for c in class_idx:
            counts[c] += 1

    total = counts.sum()
    n_cls = len(ALL_STAGE_COLS)

    weights = np.zeros(n_cls, dtype=np.float32)
    for c in range(n_cls):
        if counts[c] > 0:
            weights[c] = total / (n_cls * counts[c])

    # Zero out classes with too few samples to learn from (< min_count).
    # tHB for example has only 3 frames across all patients — not a learnable signal.
    min_count = 50
    for c in range(n_cls):
        if counts[c] < min_count:
            print(f"  [ignoring {ALL_STAGE_COLS[c]}: only {int(counts[c])} frames]")
            weights[c] = 0.0

    # Optional: boost only late stages (t8, t9+, tM).
    if late_stage_boost > 1.0:
        for c in LATE_STAGE_INDICES:
            if weights[c] > 0:
                weights[c] *= late_stage_boost
        print(f"  [late_stage_boost={late_stage_boost}] boosted t8, t9+, tM")

    # Cap and floor weights so no single class dominates, and long stages
    # (with tiny inverse-duration weights) are not almost ignored.
    nonzero = weights[weights > 0]
    if len(nonzero) > 0:
        med = float(np.median(nonzero))

        if max_weight_ratio > 0:
            cap = med * max_weight_ratio
            for c in range(len(weights)):
                if weights[c] > cap:
                    weights[c] = cap
            print(f"  [max_weight_ratio={max_weight_ratio}] capped weights at {cap:.3f} (median={med:.3f})")

        if min_weight_ratio > 0:
            floor = med / min_weight_ratio
            for c in range(len(weights)):
                if 0 < weights[c] < floor:
                    weights[c] = floor
            print(f"  [min_weight_ratio={min_weight_ratio}] floored weights at {floor:.3f} (median={med:.3f})")

    print("Class weights:")
    for name, cnt, w in zip(ALL_STAGE_COLS, counts, weights):
        print(f"  {name:<8s}  count={int(cnt):>8,}  weight={w:.4f}")

    return torch.tensor(weights, dtype=torch.float32, device=device)


# ─────────────────────────────────────────────────────────────────
# 1.  Cross-Entropy Loss  (weighted)
# ─────────────────────────────────────────────────────────────────

def ce_loss(logits        : torch.Tensor,
            y0            : torch.Tensor,
            valid_mask    : torch.Tensor,
            class_weights : torch.Tensor = None) -> torch.Tensor:
    """
    Weighted cross-entropy between model logits and ground-truth class indices.

    Args:
        logits        : (B, L, C)   raw network output
        y0            : (B, L, C)   one-hot ground truth
        valid_mask    : (B, L)      True for valid frames
        class_weights : (C,)        per-class weights; None = uniform

    Why weighted CE?
    ────────────────
    Embryo stages have wildly different durations — e.g. tHB can span 200+
    frames while tPNa lasts ~10.  Without weighting the model predicts the
    dominant stage everywhere.  Inverse-frequency weighting forces equal
    attention to every stage regardless of how rarely it appears.
    """
    B, L, C = logits.shape

    targets      = y0.argmax(dim=-1)
    logits_flat  = logits.reshape(B * L, C)
    targets_flat = targets.reshape(B * L)
    valid_flat   = valid_mask.reshape(B * L).float()

    per_frame_loss = F.cross_entropy(
        logits_flat, targets_flat,
        weight=class_weights,
        reduction="none",
    )                                                   # (B*L,)

    return (per_frame_loss * valid_flat).sum() / valid_flat.sum().clamp(min=1.0)


# ─────────────────────────────────────────────────────────────────
# 2.  Viterbi Loss  (differentiable forward algorithm)
# ─────────────────────────────────────────────────────────────────

def viterbi_loss(log_probs : torch.Tensor,
                 valid_mask: torch.Tensor,
                 per_frame : bool = True) -> torch.Tensor:
    """
    Differentiable monotone-path loss via the log-domain forward algorithm.

    Args:
        log_probs  : (B, L, C)   log_softmax probabilities
        valid_mask : (B, L)      True for valid frames
        per_frame  : if True (default), return mean negative log-likelihood
                     per frame so the scale matches CE (~2–4). This prevents
                     Viterbi from dominating and avoids collapse to a single
                     class (e.g. predict tB everywhere to minimize path NLL).

    Returns scalar loss (mean over patients; if per_frame, normalized by T).

    Notes:
        • Only valid frames participate; variable frame counts are handled
          by extracting each patient's valid sub-sequence in a Python loop.
        • per_frame=True: L_vit ≈ -log_likelihood / T per sample → same
          order of magnitude as CE, so the model must get classes right
          while Viterbi regularizes monotonicity.
    """
    NEG_INF = -1e9
    B = log_probs.shape[0]
    total, count = 0.0, 0

    for b in range(B):
        valid_idx = valid_mask[b].nonzero(as_tuple=True)[0]   # positions of valid frames
        if len(valid_idx) == 0:
            continue

        lp = log_probs[b][valid_idx]    # (T, C)
        T, C = lp.shape

        # Initialise forward variable at t=0
        log_alpha = lp[0]               # (C,)

        for t in range(1, T):
            # Stay at same class
            stay    = log_alpha                                             # (C,)

            # Advance from previous class (class c-1 → c)
            # For c=0 there is no predecessor → −∞
            advance = torch.cat(
                [log_alpha.new_full((1,), NEG_INF), log_alpha[:-1]], dim=0  # (C,)
            )

            # New forward variable
            log_alpha = lp[t] + torch.logaddexp(stay, advance)             # (C,)

        # Log-likelihood = logsumexp over all reachable final states
        log_likelihood = torch.logsumexp(log_alpha, dim=0)                 # scalar
        neg_ll = -log_likelihood
        if per_frame:
            neg_ll = neg_ll / T
        total += neg_ll
        count += 1

    return total / max(count, 1)


# ─────────────────────────────────────────────────────────────────
# 3.  Boundary Loss
# ─────────────────────────────────────────────────────────────────

def boundary_loss(probs     : torch.Tensor,
                  y0        : torch.Tensor,
                  valid_mask: torch.Tensor,
                  sigma     : float = 2.0) -> torch.Tensor:
    """
    Aligns predicted stage-transition boundaries with smoothed GT boundaries.

    GT boundaries (hard):
        B_i = 1  if  class(frame i) ≠ class(frame i+1),  else 0

    GT boundaries (soft):
        B̄ = GaussianFilter(B, σ)          ← smooth so nearby frames matter too

    Predicted boundary probability at frame i:
        p_boundary_i = 1 − P_i · P_{i+1}  ← dot product of neighbouring probs
                                              (high when distributions differ)

    Loss (BCE):
        L_bd = −B̄_i · log(1 − P_i·P_{i+1}) − (1−B̄_i) · log(P_i·P_{i+1})

    Args:
        probs      : (B, L, C)   softmax probabilities
        y0         : (B, L, C)   one-hot ground truth
        valid_mask : (B, L)      True for valid frames
        sigma      : Gaussian smoothing std (default 2.0 frames)

    Returns scalar loss (mean over patients).
    """
    B = probs.shape[0]
    total, count = 0.0, 0

    for b in range(B):
        valid_idx = valid_mask[b].nonzero(as_tuple=True)[0]
        if len(valid_idx) < 2:
            continue

        p  = probs[b][valid_idx]    # (T, C)  predicted probs on valid frames
        g  = y0[b][valid_idx]       # (T, C)  one-hot GT on valid frames
        T  = p.shape[0]

        # ── GT hard boundaries ──────────────────────────────────────────────
        gt_class = g.argmax(dim=-1).cpu().numpy()           # (T,) integer classes
        hard_b   = (gt_class[:-1] != gt_class[1:]).astype(np.float32)  # (T-1,)

        # ── Smooth GT boundaries ────────────────────────────────────────────
        soft_b = gaussian_filter1d(hard_b, sigma=sigma)
        soft_b = torch.tensor(soft_b, dtype=torch.float32, device=probs.device)   # (T-1,)

        # ── Predicted boundary probabilities ────────────────────────────────
        # dot_product[i] = Σ_c  p[i,c] · p[i+1,c]   (high → same stage, low → boundary)
        dot_prod = (p[:-1] * p[1:]).sum(dim=-1)             # (T-1,)
        pred_b   = 1.0 - dot_prod                           # (T-1,)  boundary prob

        # Clamp to keep log() finite
        dot_prod = dot_prod.clamp(1e-6, 1.0 - 1e-6)
        pred_b   = pred_b.clamp(1e-6, 1.0 - 1e-6)

        # ── BCE loss ────────────────────────────────────────────────────────
        bce = (
            -soft_b       * torch.log(pred_b)
            -(1 - soft_b) * torch.log(dot_prod)
        )                                                   # (T-1,)

        total += bce.mean()
        count += 1

    return total / max(count, 1)


# ─────────────────────────────────────────────────────────────────
# 4.  Smoothness Loss  (MSE of log-probs between adjacent valid frames)
# ─────────────────────────────────────────────────────────────────

def smoothness_loss(log_probs : torch.Tensor,
                   valid_mask: torch.Tensor) -> torch.Tensor:
    """
    Penalize large changes in predicted distribution between adjacent frames.
    Encourages smooth segmentation even under high noise.

    For each valid adjacent pair (i, i+1), compute MSE between log P_i and
    log P_{i+1} over the C classes; return the mean over all such pairs.

    Args:
        log_probs  : (B, L, C)  log_softmax probabilities
        valid_mask : (B, L)     True for valid frames

    Returns:
        Scalar: mean over valid adjacent pairs of (1/C) * sum_c (log p_i(c) - log p_{i+1}(c))^2
    """
    B, L, C = log_probs.shape
    # Adjacent pairs: (i, i+1) for i in 0..L-2
    log_p_cur = log_probs[:, :-1, :]   # (B, L-1, C)
    log_p_nxt = log_probs[:, 1:, :]   # (B, L-1, C)
    diff = log_p_cur - log_p_nxt       # (B, L-1, C)
    pair_mse = (diff ** 2).mean(dim=-1)   # (B, L-1)  MSE over C per pair

    # Both frame i and i+1 must be valid
    valid_pair = valid_mask[:, :-1] & valid_mask[:, 1:]   # (B, L-1)
    n_pairs = valid_pair.sum().float().clamp(min=1.0)
    return (pair_mse * valid_pair.float()).sum() / n_pairs


# ─────────────────────────────────────────────────────────────────
# Combined Loss
# ─────────────────────────────────────────────────────────────────

def combined_loss(logits        : torch.Tensor,
                  y0            : torch.Tensor,
                  valid_mask    : torch.Tensor,
                  lambda_ce     : float = 1.0,
                  lambda_vit    : float = 1.0,
                  lambda_bd     : float = 1.0,
                  lambda_smooth : float = 0.0,
                  class_weights : torch.Tensor = None) -> dict:
    """
    Convenience wrapper that computes all losses and their weighted sum.

        total = lambda_ce*L_ce + lambda_vit*L_vit + lambda_bd*L_bd + lambda_smooth*L_smooth

    Returns a dict with keys: total, ce, viterbi, boundary, smooth
    """
    log_p = F.log_softmax(logits, dim=-1)   # (B, L, C)
    p     = log_p.exp()                     # (B, L, C)

    l_ce  = ce_loss(logits, y0, valid_mask, class_weights=class_weights)
    l_vit = viterbi_loss(log_p, valid_mask)
    l_bd  = boundary_loss(p, y0, valid_mask)
    l_smooth = smoothness_loss(log_p, valid_mask)

    total = lambda_ce * l_ce + lambda_vit * l_vit + lambda_bd * l_bd + lambda_smooth * l_smooth

    return {
        "total"   : total,
        "ce"      : l_ce.item(),
        "viterbi" : l_vit.item(),
        "boundary": l_bd.item(),
        "smooth"  : l_smooth.item(),
    }