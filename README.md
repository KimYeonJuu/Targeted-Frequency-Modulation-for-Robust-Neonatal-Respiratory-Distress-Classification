# Targeted Frequency Modulation for Robust Neonatal RDS Classification

This repository provides the code for the manuscript:

**Targeted Frequency Modulation for Robust Neonatal Respiratory Distress Classification**

The code implements frequency outlier-band selection and targeted spectral modulation for chest radiograph classification experiments.

## Repository Contents

- `main.py`: training and evaluation entry point.
- `main_code/`: spectral gating / TFM model components and MedCLIP wrappers.
- `sci/`: dataset loaders, preprocessing utilities, augmentation modules, metrics, and backbone utilities.
- `run_outlier_aug.sh`: example training script with placeholder data paths.

## Data Availability

The neonatal RDS clinical images are not included in this repository and cannot be publicly redistributed without institutional approval. Users should provide their own images, labels, lung masks, and train/validation/test CSV files.

Expected CSV format for the RDS binary classification pipeline:

```text
Path,rds,No Finding,Finding
```

CheXpert and NIH ChestX-ray14 data must be obtained from their original providers and used under their own data-use terms.

## Environment Setup

### Original Server Environment

On the original experiment server, activate the provided environment before running experiments:

```bash
cd /workspace/experiment/RDS
source SCI/bin/activate
cd PR_Review/code_release/sci_rds
```

If the repository is used outside the original server, create a fresh environment and install the frozen dependencies:

```bash
pip install -r requirements.txt
```

The `requirements.txt` file was prepared from the SCI experiment environment. Depending on the local CUDA/PyTorch setup, users may need to install the matching PyTorch and torchvision wheels separately.

If using optional StyleMix augmentation, provide the pretrained StyleMix weights separately and set:

```bash
export STYLEMIX_MODEL_DIR=/path/to/stylemix_model
```

## Data Preparation

Prepare train/validation/test CSV files. For the RDS binary classification pipeline, each CSV should contain:

```text
Path,rds,No Finding,Finding
```

`Path` should point to the corresponding image file. Lung masks are expected to follow the naming convention used by `sci/dataset.py`, where a segmented crop path ending in `_seg_crop.png` has a corresponding mask path ending in `_seg_crop_mask.png`.

## Running Experiments

Edit `run_outlier_aug.sh` or provide paths via environment variables:

```bash
cd /workspace/experiment/RDS
source SCI/bin/activate
cd PR_Review/code_release/sci_rds

export RDS_TRAIN_CSV=/path/to/train_seg.csv
export RDS_VAL_CSV=/path/to/valid_seg.csv
export RDS_TEST_CSV=/path/to/test_seg.csv
bash run_outlier_aug.sh
```

The script writes local checkpoints and logs under:

```text
sci/checkpoints/
sci/logs/
```

These folders are ignored by Git and should not be uploaded.

### Direct Command Example

```bash
python main.py \
  --train_data_path "$RDS_TRAIN_CSV" \
  --val_data_path "$RDS_VAL_CSV" \
  --test_data_path "$RDS_TEST_CSV" \
  --model_name medclip_swin_sgn_insert \
  --dataset rds \
  --checkpoint_dir ./sci/checkpoints \
  --save_model rds_tfm_fold0 \
  --batch_size 64 \
  --num_epochs 30 \
  --learning_rate 1e-4 \
  --weight_decay 1e-6 \
  --gpu "0" \
  --train --test \
  --freeze_backbone OFF \
  --block-selection "4:1-5" \
  --sgn_mode grid \
  --sgn_grid_size 16x16 \
  --sgn_tau 0.25 \
  --sgn_amp 1.0 \
  --sgn_ratio_thresh 0.02 \
  --sgn_outlier_topk 0 \
  --sgn_min_radius 1 \
  --sgn_max_radius -1
```

The repository intentionally excludes clinical images, masks, checkpoints, prediction outputs, and reviewer annotation files.

## Regenerating Requirements

On the original server, the dependency file can be regenerated with:

```bash
cd /workspace/experiment/RDS
source SCI/bin/activate
python -m pip freeze > PR_Review/code_release/sci_rds/requirements.txt
```

If `python` is not found after activation, check whether `SCI/bin/python` points to a valid interpreter in the current container.

## Reproducibility Parameters

The manuscript experiments used:

- image size: `256 x 256`
- optimizer: AdamW
- learning rate: `1e-4`
- weight decay: `1e-6`
- epochs: `30`
- batch size: `64`
- seed: `42`
- FOBE/TFM settings: `tau=0.02`, `rho_min=1`, `rho_max=-1`, `K=0`, `epsilon=1e-12`, `delta_max=1.0`

## License

This code is released under the MIT License.
