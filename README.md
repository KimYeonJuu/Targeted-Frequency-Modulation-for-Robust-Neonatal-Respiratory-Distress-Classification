# Targeted Frequency Modulation for Chest Radiograph Classification

This repository provides code for the manuscript:

**Targeted Frequency Modulation for Robust Neonatal Respiratory Distress Classification**

Because the neonatal RDS cohort used in the manuscript cannot be publicly redistributed, the public execution guide is written around **CheXpert**, which users can obtain from the original data provider. The same training entry point and TFM/FOBE implementation are used for the private RDS experiments.

## Repository Contents

- `main.py`: training and evaluation entry point.
- `main_code/`: spectral gating / TFM model components and MedCLIP wrappers.
- `sci/`: dataset loaders, preprocessing utilities, augmentation modules, metrics, and backbone utilities.
- `run_outlier_aug.sh`: CheXpert-oriented example training script.

## Data Availability

The neonatal RDS clinical images are not included in this repository and cannot be publicly redistributed without institutional approval.

CheXpert must be obtained separately from the Stanford ML Group and used under its data-use terms. This repository expects preprocessed CheXpert CSV files and image paths supplied by the user.

Expected CSV format for the CheXpert binary classification pipeline:

```text
Path,No Finding,Finding
```

`Path` may be an absolute image path, a path relative to the current directory, or a path relative to `CHEXPERT_DATA_DIR`. Because public CheXpert releases do not include neonatal lung masks, the CheXpert loader uses an all-one mask so the full image remains available to the spectral module.

## Environment Setup

### Existing SCI Environment

If you already have the SCI environment used for these experiments, activate it before running:

```bash
cd /path/to/project/root
source SCI/bin/activate
cd PR_Review/code_release/sci_rds
```

If the repository is used outside the original server, create a fresh environment and install the frozen dependencies:

```bash
pip install -r requirements.txt
```

The `requirements.txt` file was prepared from the SCI experiment environment. Depending on the local CUDA/PyTorch setup, users may need to install matching PyTorch and torchvision wheels separately.

Optional StyleMix augmentation requires pretrained StyleMix weights. If used, set:

```bash
export STYLEMIX_MODEL_DIR=/path/to/stylemix_model
```

## CheXpert Data Preparation

Prepare train/validation/test CSV files with the CheXpert binary columns:

```text
Path,No Finding,Finding
```

Example:

```text
Path,No Finding,Finding
/path/to/chexpert/image_000001.png,1,0
/path/to/chexpert/image_000002.png,0,1
```

Set the CSV paths before running the training script:

```bash
export CHEX_TRAIN_CSV=/path/to/chexpert/train.csv
export CHEX_VAL_CSV=/path/to/chexpert/valid.csv
export CHEX_TEST_CSV=/path/to/chexpert/test.csv
```

## Running CheXpert Experiments

The recommended public example is:

```bash
cd /path/to/project/root
source SCI/bin/activate
cd PR_Review/code_release/sci_rds

export CHEX_TRAIN_CSV=/path/to/chexpert/train.csv
export CHEX_VAL_CSV=/path/to/chexpert/valid.csv
export CHEX_TEST_CSV=/path/to/chexpert/test.csv
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
  --train_data_path "$CHEX_TRAIN_CSV" \
  --val_data_path "$CHEX_VAL_CSV" \
  --test_data_path "$CHEX_TEST_CSV" \
  --model_name medclip_swin_sgn_insert \
  --dataset chexpert \
  --checkpoint_dir ./sci/checkpoints \
  --save_model chexpert_tfm \
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
cd /path/to/project/root
source SCI/bin/activate
python -m pip freeze > /path/to/sci_rds/requirements.txt
```

If `python` is not found after activation, check whether `SCI/bin/python` points to a valid interpreter in the current container.

## Reproducibility Parameters

The public CheXpert example uses:

- image size: `256 x 256`
- optimizer: AdamW
- learning rate: `1e-4`
- weight decay: `1e-6`
- epochs: `30`
- batch size: `64`
- seed: `42`
- TFM/SGN settings: `sgn_tau=0.25`, `sgn_ratio_thresh=0.02`, `sgn_min_radius=1`, `sgn_max_radius=-1`, `sgn_outlier_topk=0`, `sgn_amp=1.0`

## License

This code is released under the MIT License.
