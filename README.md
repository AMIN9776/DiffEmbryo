# Embryo Diffusion — Stage Segmentation

Diffusion-based action segmentation for embryo developmental stages. The model denoises a noisy stage matrix over a global time grid and recovers **Ŷ₀** (one-hot over 18 stage classes). Training uses cross-entropy, a differentiable Viterbi (monotonicity) loss, and a boundary-alignment loss.

**Best results** are obtained with **time-polished monotone data**, **uniform CE weights** (`--no_class_weights`), and **single-step inference** at a low diffusion step (e.g. `--mode single --eval_step 10`), yielding high per-frame accuracy and good late-stage (t8, t9+) predictions.

---

## Pipeline overview

```
One-hot stage matrix (per frame)
        ↓
  embryo_time_quantize.py   →  global time grid (e.g. 492 bins)
        ↓
  embryo_time_polish.py     →  filled bins, starting/ending padding
        ↓
  embryo_time_polished_monotone.py  →  monotone (-1/0/1) per bin
        ↓
  train.py (--polished_monotone_dir)  →  checkpoints
        ↓
  inference.py (--mode single --eval_step 10)  →  *_prediction.csv
        ↓
  evaluate_predictions.py   →  per-class metrics, macro F1
  visualize_gt_vs_pred.py   →  GT vs pred plots
```

See **PIPELINE.md** for the formal description of noising, decoder, and losses.

---

## Setup

- **Python 3** with **PyTorch**, **pandas**, **numpy**. No extra heavy dependencies.
- Create train/val splits first (see below).

---

## 1. Splits

```bash
python split_patients.py \
  --selected_json /path/to/selected_patients.json \
  --output_dir   /path/to/split_dir \
  --val_ratio 0.2 --seed 42
```

This writes `training_set.json` and `validation_set.json` under `--output_dir`. Use `--split_dir` pointing to that directory in all downstream steps.

---

## 2. Data preparation (time-polished monotone, recommended)

From per-patient one-hot stage CSVs (frame-based):

```bash
# 1) Time quantisation (global grid, e.g. bin_size 0.3h)
python embryo_time_quantize.py \
  --input_dir  /path/to/embryo_onehot_matrix \
  --output_dir /path/to/time_quantised \
  --bin_size 0.3

# 2) Polish (fill empty bins, set starting/ending)
python embryo_time_polish.py \
  --input_dir  /path/to/time_quantised \
  --output_dir /path/to/time_polished

# 3) Monotone encoding (-1/0/1 per time bin)
python embryo_time_polished_monotone.py \
  --input_dir  /path/to/time_polished \
  --output_dir /path/to/embryo_time_polished_monotone
```

Use the final directory as `--polished_monotone_dir` for training and inference.

---

## 3. Training (best settings)

Train on time-polished monotone data with **uniform CE weights** (no per-class weighting):

```bash
python train.py \
  --polished_monotone_dir /path/to/embryo_time_polished_monotone \
  --split_dir             /path/to/split_dir \
  --output_dir            /path/to/checkpoints \
  --epochs 100 \
  --batch_size 16 \
  --diffusion_steps 100 \
  --lambda_ce 1.0 \
  --lambda_viterbi 0.7 \
  --lambda_boundary 1.0 \
  --no_class_weights \
  --device cuda
```

- **Checkpoints**: `best_model.pt` (best validation loss), `latest_model.pt` (for resuming).
- **Resume**: add `--resume /path/to/checkpoints/latest_model.pt`.

**Note:** Reported *train* total loss is on a different scale (sum over batches) than *validation* total loss (mean). Focus on validation loss and sample accuracy; both train and val use the same per-batch mean loss internally.

---

## 4. Inference

Two modes:

- **Full reverse chain** (DDIM-style from pure noise):
  ```bash
  python inference.py \
    --checkpoint  /path/to/checkpoints/best_model.pt \
    --monotone_dir /path/to/embryo_time_polished_monotone \
    --split_json   /path/to/split_dir/validation_set.json \
    --output_dir   /path/to/predictions \
    --mode reverse \
    --device cuda
  ```

- **Single-step denoising** (recommended for best accuracy): one denoising step from `q_sample(y0, s)` at a fixed step `s`:
  ```bash
  python inference.py \
    --checkpoint  /path/to/checkpoints/best_model.pt \
    --monotone_dir /path/to/embryo_time_polished_monotone \
    --split_json   /path/to/split_dir/validation_set.json \
    --output_dir   /path/to/predictions \
    --mode single \
    --eval_step 10 \
    --device cuda
  ```

Predictions are written as `*_prediction.csv` with columns: `frame_number`, `gt_class`, `is_valid`, `predicted_class`, plus the 18 stage probability columns.

---

## 5. Evaluation and visualisation

- **Metrics** (only rows with `is_valid == 1`):
  ```bash
  python evaluate_predictions.py --pred_dir /path/to/predictions
  ```
  Reports per-class accuracy, precision, recall, F1, support; overall accuracy and macro F1.

- **GT vs prediction plots** (random patients, 4 rows: GT, spacer, Pred, spacer):
  ```bash
  python visualize_gt_vs_pred.py \
    --pred_dir /path/to/predictions \
    --output_path /path/to/gt_vs_pred.png \
    --num_patients 5
  ```

---

## Main scripts

| Script | Purpose |
|--------|--------|
| `split_patients.py` | Build train/val JSONs from a selected patient list. |
| `embryo_time_quantize.py` | Map frame-based one-hot to a global time grid. |
| `embryo_time_polish.py` | Fill empty bins, assign starting/ending. |
| `embryo_time_polished_monotone.py` | Convert polished one-hot to monotone (-1/0/1). |
| `train.py` | Train diffusion transformer (CE + Viterbi + boundary). |
| `inference.py` | Run reverse chain or single-step denoising. |
| `evaluate_predictions.py` | Per-class and macro metrics on `*_prediction.csv`. |
| `visualize_gt_vs_pred.py` | Plot GT vs predicted stage sequences. |

---

## Model and data

- **Decoder**: Transformer over sequence length 492 (time-polished) or 550 (frame-based). Input: noisy matrix **Y_s** + normalised time feature + step **s** + padding mask. Output: logits for **Ŷ₀**.
- **Losses**: L_ce (valid frames only), L_viterbi (monotone paths), L_boundary (smoothed GT boundaries vs predicted boundary probabilities).
- **18 classes**: starting, tPB2, tPNa, tPNf, t2–t9+, tM, tSB, tB, tEB, tHB, ending. Only the 16 embryo stages are noised and evaluated; starting/ending are padding.

For full equations and noising details, see **PIPELINE.md**.
