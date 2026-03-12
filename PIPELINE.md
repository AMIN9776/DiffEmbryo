# Diffusion Decoder — Action Segmentation Pipeline (Stages Only)

This document confirms the pipeline step-by-step as requested.

---

## 1. Scope

- **Task**: Action segmentation on **stages only** (no visual embryo dataset as baseline).
- **Target**: Recover the ground-truth matrix **Y₀** of shape **550×18** (one-hot over 18 stage classes per frame).
- **Noising**: Apply diffusion noise **only to valid frames** (frames that are real embryo stages). **Starting and ending** frames are **not** noised (only tiny σ_pad noise so the model can identify and ignore them). So we **only and only** noise the valid (stage) frames.

---

## 2. Training (per batch)

1. **Sample diffusion step**  
   \( s \sim \{1, 2, \ldots, S\} \) (in practice biased toward lower steps, e.g. \( p(s) \propto 1/\sqrt{s} \)).

2. **Corrupt Y₀ → Y_s**  
   - **Valid frames**: full forward diffusion  
     \( Y_s = \sqrt{\bar{\alpha}_s} Y_0 + \sqrt{1-\bar{\alpha}_s} \varepsilon \), \( \varepsilon \sim \mathcal{N}(0, I) \).  
   - **Padding (starting/ending)**: \( Y_s = Y_0 + \sigma_{\text{pad}} \varepsilon_{\text{pad}} \) (tiny noise).

3. **Decoder**  
   Input: noisy matrix **Y_s** (B, 550, 18) + time feature (B, 550, 1) + step **s** + padding mask.  
   Output: **logits** for Ŷ₀ (B, 550, 18).

4. **Losses** (at the same randomly selected step **s**):
   - **L_ce**: Cross-entropy on **valid frames only**, optionally class-weighted.
   - **L_smo (Viterbi)**: Differentiable monotone-path loss so predictions are **monotonic** (stay or advance: c → c or c → c+1). Implemented as **per-frame** negative log-likelihood (÷ T) so its scale matches CE; this avoids the model collapsing to a single class (e.g. tB everywhere) just to minimize path NLL.
   - **L_bd (boundary)**:
     - GT boundaries: \( B_i = \mathbf{1}(Y_{0,i} \neq Y_{0,i-1}) \), then smooth with Gaussian \( \bar{B} = \lambda(B) \).
     - Predicted boundary prob at frame i: \( 1 - P_{s,i} \cdot P_{s,i-1} \) (dot product of neighbouring frame probs).
     - Align with BCE:  
       \( \mathcal{L}_{bd} = \frac{1}{L-1} \sum_{i=1}^{L-1} \bigl[ -\bar{B}_i \log(1 - P_{s,i}\cdot P_{s,i-1}) - (1-\bar{B}_i) \log(P_{s,i}\cdot P_{s,i-1}) \bigr] \).
   - **Total**: \( \mathcal{L}_{\text{sum}} = \mathcal{L}_{ce} + \mathcal{L}_{smo} + \mathcal{L}_{bd} \) (with optional λ_vit, λ_bd).

5. **Backprop** on \(\mathcal{L}_{\text{sum}}\), then optimizer step.

---

## 3. Inference

1. **Input**: Corrupted matrix **Y_S** (full noise on valid frames, near-clean on starting/ending).
2. **Process**: Feed **Y_S** (and time feature, step, padding mask) into the **decoder** and run the **reverse diffusion** chain:  
   Y_S → Y_{S-1} → … → Y_1 → **Ŷ₀**.
3. **Output**: Recovered **Ŷ₀** (550×18) as the predicted stage matrix (e.g. argmax per frame for class labels).

---

## 4. Train / validation split

- Patient names are stored in:
  - **training_set.json**: `{"patients": ["...", ...], "num_patients": N}`
  - **validation_set.json**: `{"patients": ["...", ...], "num_patients": M}`
- **Data loading**: `EmbryoDataset` reads from these JSONs (one for train, one for val). No mixing of train/val patients.
- **Creating splits (do this first)**:
  ```bash
  python split_patients.py \
    --selected_json /path/to/selected_patients.json \
    --output_dir    /path/to/split_dir \
    --val_ratio 0.2 --seed 42
  ```
  This writes `split_dir/training_set.json` and `split_dir/validation_set.json`. Then run training with `--split_dir <split_dir>`.

---

## 5. Summary

| Item | Implementation |
|------|----------------|
| Noising | Only valid frames get full diffusion noise; starting/ending get σ_pad |
| Decoder input | Y_s (550×18) + time_feat + step s + pad_mask |
| Decoder output | Logits for Ŷ₀ (550×18) |
| L_ce | On valid frames only |
| L_smo | Viterbi (monotonic paths) |
| L_bd | BCE between smoothed GT boundaries and \( 1 - P_i \cdot P_{i-1} \) |
| Inference | Corrupted Y_S → decoder in reverse chain → Ŷ₀ |
