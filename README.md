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

## Minimal Setup

```bash
pip install -r requirements.txt
```

If using optional StyleMix augmentation, provide the pretrained StyleMix weights separately and set:

```bash
export STYLEMIX_MODEL_DIR=/path/to/stylemix_model
```

## Example Training

Edit `run_outlier_aug.sh` or provide paths via environment variables:

```bash
export RDS_TRAIN_CSV=/path/to/train_seg.csv
export RDS_VAL_CSV=/path/to/valid_seg.csv
export RDS_TEST_CSV=/path/to/test_seg.csv
bash run_outlier_aug.sh
```

The repository intentionally excludes clinical images, masks, checkpoints, prediction outputs, and reviewer annotation files.

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
