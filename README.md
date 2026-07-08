# Targeted Frequency Modulation for Chest Radiograph Classification

This repository provides code for the manuscript:

**Targeted Frequency Modulation for Robust Neonatal Respiratory Distress Classification**

Because the neonatal RDS cohort used in the manuscript cannot be publicly redistributed, the public execution guide is written around **CheXpert**, which users can obtain from the original data provider. The same training entry point and TFM/FOBE implementation are used for the private RDS experiments.

## Repository Contents

- `main.py`: training and evaluation entry point.
- `main_code/`: spectral gating / TFM model components and MedCLIP wrappers. Some modules use the SE gate implementation from [`ai-med/squeeze_and_excitation`](https://github.com/ai-med/squeeze_and_excitation). Download this code and place it inside `main_code/` before running SE-gate-based models.
- `sci/`: dataset loaders, preprocessing utilities, augmentation modules, metrics, and backbone utilities.
- `run_outlier_aug.sh`: CheXpert-oriented example training script.

## Data Availability

The neonatal RDS clinical images are not included in this repository and cannot be publicly redistributed without institutional approval.

CheXpert must be obtained separately from the Stanford ML Group and used under its data-use terms. This repository expects preprocessed CheXpert CSV files and image paths supplied by the user.

Expected CSV format for the CheXpert binary classification pipeline:

```text
Path,No Finding,Finding
