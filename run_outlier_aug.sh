#!/usr/bin/env bash
set -eu

PYTHON="${PYTHON:-python3}"
LOG_DIR="${LOG_DIR:-./sci/logs/medclip}"
CKPT_DIR="${CKPT_DIR:-./sci/checkpoints}"
mkdir -p "$LOG_DIR" "$CKPT_DIR"

########################################
# Path configuration
########################################
CHEX_TRAIN_CSV="${CHEX_TRAIN_CSV:-/path/to/chexpert/train.csv}"
CHEX_VAL_CSV="${CHEX_VAL_CSV:-/path/to/chexpert/valid.csv}"
CHEX_TEST_CSV="${CHEX_TEST_CSV:-/path/to/chexpert/test.csv}"
CHEX_OUTLIER_CSV="${CHEX_OUTLIER_CSV:-/path/to/chexpert/outlier_bands.csv}"

########################################
# Shared hyperparameters
########################################
BATCH="${BATCH:-64}"
EPOCHS="${EPOCHS:-30}"
LR="${LR:-1e-4}"
WD="${WD:-1e-6}"
GPU="${GPU:-0}"
RUN_NAME="${RUN_NAME:-chexpert_MedCLIP_TFM}"

########################################
# Optional outlier-band augmentation
########################################
OUTLIER_PROB="${OUTLIER_PROB:-0.5}"
OUTLIER_SCALE="${OUTLIER_SCALE:-2.0}"
OUTLIER_TOPK="${OUTLIER_TOPK:-0}"
OUTLIER_SUBSET_K="${OUTLIER_SUBSET_K:-0}"

########################################
# CheXpert training
########################################
CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" main.py \
  --train_data_path "$CHEX_TRAIN_CSV" \
  --val_data_path "$CHEX_VAL_CSV" \
  --test_data_path "$CHEX_TEST_CSV" \
  --model_name medclip_swin_sgn_insert \
  --dataset chexpert \
  --checkpoint_dir "$CKPT_DIR" \
  --save_model "$RUN_NAME" \
  --batch_size "$BATCH" \
  --num_epochs "$EPOCHS" \
  --learning_rate "$LR" \
  --weight_decay "$WD" \
  --project_name "SCI_CheXpert_TFM" \
  --gpu "$GPU" \
  --train --test \
  --freeze_backbone OFF \
  --block-selection "4:1-5" \
  --sgn_mode grid \
  --sgn_grid_size 16x16 \
  --sgn_tau 0.25 \
  --sgn_amp 1.0 \
  --sgn_block 16x16 \
  --sgn_stride same \
  --sgn_ratio_thresh 0.02 \
  --sgn_outlier_topk 0 \
  --sgn_min_radius 1 \
  --sgn_max_radius -1 \
  > "$LOG_DIR/${RUN_NAME}.txt" 2>&1

# To enable outlier-band augmentation, add the following options above:
#   --outlier_csv "$CHEX_OUTLIER_CSV" \
#   --outlier_prob "$OUTLIER_PROB" \
#   --outlier_scale "$OUTLIER_SCALE" \
#   --outlier_topk "$OUTLIER_TOPK" \
#   --outlier_subset_k "$OUTLIER_SUBSET_K" \
#   --post_hard_mask

echo "CheXpert experiment finished: $RUN_NAME"
