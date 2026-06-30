#!/usr/bin/env bash
set -eu

PYTHON="${PYTHON:-python3}"
LOG_DIR=./sci/logs/medclip/
CKPT_DIR=./sci/checkpoints
mkdir -p "$LOG_DIR" "$CKPT_DIR"

########################################
# 경로 설정
########################################
# RDS
RDS_TRAIN_CSV="${RDS_TRAIN_CSV:-/path/to/rds/train_seg.csv}"
RDS_VAL_CSV="${RDS_VAL_CSV:-/path/to/rds/valid_seg.csv}"
RDS_TEST_CSV="${RDS_TEST_CSV:-/path/to/rds/test_seg.csv}"
RDS_OUTLIER_CSV="${RDS_OUTLIER_CSV:-/path/to/rds/outlier_bands.csv}"

# CheXpert
CHEX_TRAIN_CSV="${CHEX_TRAIN_CSV:-/path/to/chexpert/train.csv}"
CHEX_VAL_CSV="${CHEX_VAL_CSV:-/path/to/chexpert/valid.csv}"
CHEX_TEST_CSV="${CHEX_TEST_CSV:-/path/to/chexpert/test.csv}"
CHEX_OUTLIER_CSV="${CHEX_OUTLIER_CSV:-/path/to/chexpert/outlier_bands.csv}"

########################################
# 공통 하이퍼파라미터
########################################
BATCH=64
EPOCHS=30
LR=1e-4
WD=1e-6

# Outlier-band 증강 하이퍼파라미터
OUTLIER_PROB=0.5
OUTLIER_SCALE=2.0
OUTLIER_TOPK=0
OUTLIER_SUBSET_K=0

########################################
# RDS 학습 (GPU 0)
########################################
# RUN_NAME="rds_MedCLIP_outlier_SB_insert_21_grid_30epoch"
RUN_NAME="rds_MedCLIP_SB_45_H_V"
$PYTHON main.py \
  --train_data_path "$RDS_TRAIN_CSV" \
  --val_data_path   "$RDS_VAL_CSV" \
  --test_data_path  "$RDS_TEST_CSV" \
  --model_name medclip_swin_sgn_insert \
  --dataset rds \
  --checkpoint_dir "$CKPT_DIR" \
  --save_model "$RUN_NAME" \
  --batch_size $BATCH \
  --num_epochs $EPOCHS \
  --learning_rate $LR \
  --weight_decay $WD \
  --project_name "SCI_SGN" \
  --gpu "0" \
  --data_parallel \
  --train --test \
  --use_wandb \
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
  > "$LOG_DIR/${RUN_NAME}.txt" 2>&1 &
  # --guidedmix \
  # --mix_prob 0.5 \
  # --condition greedy \
  # --saliency_mode spectral \
    # --model_name medclip_swin_sgn_insert \

########################################
# CheXpert 학습 (GPU 1)
########################################
# RUN_NAME="chexpert_MedCLIP_freeze_h_off"
# $PYTHON main.py \
#   --train_data_path "$CHEX_TRAIN_CSV" \
#   --val_data_path   "$CHEX_VAL_CSV" \
#   --test_data_path  "$CHEX_TEST_CSV" \
#   --model_name medclip_vit\
#   --dataset chexpert \
#   --checkpoint_dir "$CKPT_DIR" \
#   --save_model "$RUN_NAME" \
#   --batch_size $BATCH \
#   --num_epochs $EPOCHS \
#   --learning_rate $LR \
#   --weight_decay $WD \
#   --gpu "0" \
#   --train --test \
#   --use_wandb \
#   --freeze_backbone OFF \
#   > "$LOG_DIR/${RUN_NAME}.txt" 2>&1 &
#   # --outlier_csv "$CHEX_OUTLIER_CSV" \
#   # --outlier_prob $OUTLIER_PROB \
#   # --outlier_scale $OUTLIER_SCALE \
#   # --outlier_topk $OUTLIER_TOPK \
#   # --outlier_subset_k $OUTLIER_SUBSET_K \
#   # --post_hard_mask \

wait
echo "✅ 두 실험이 모두 실행 완료"
