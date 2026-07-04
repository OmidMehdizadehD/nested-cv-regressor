# Nested Cross-Validation Regression Framework

Python framework for developing and evaluating regression models using patient-level train/test splitting, nested cross-validation, automated preprocessing, feature selection, hyperparameter optimization, and bootstrap confidence intervals. Designed primarily for biomedical machine learning applications to prevent common sources of data leakage and optimistic bias.

## Key Features
* **Patient-Level Splitting:** Utilizes `GroupShuffleSplit` to ensure records from the same patient are never split across train and test sets, avoiding patient leakage.
* **Leakage-Free Preprocessing:** Standardizers, feature selectors, and outlier detection (`OneClassSVM`) are strictly placed *inside* the pipeline, ensuring they are fit exclusively on training folds during cross-validation.
* **Nested Cross-Validation:** Uses a $10$-fold outer loop for unbiased evaluation and a $3$-fold inner loop for randomized hyperparameter tuning.
* **Feature Selection Methods:** Includes Recursive Feature Elimination (RFECV, RFE), Mutual Information, and LASSO. Automatically exports the selected variables to CSV.
* **Robust Evaluation Metrics:** Calculates bootstrap $95\%$ confidence intervals for $MAE$ and $R^2$ on the independent test set.

## Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/YOUR_USERNAME/nested-cv-regressor.git
cd nested-cv-regressor
pip install -r requirements.txt
```

## Quick Start: Synthetic Data

You do not need real patient data to test the framework. First, generate a synthetic clinical dataset:

```bash
python generate_synthetic_data.py
```

Then run the pipeline on the synthetic dataset:

```bash
python main.py --data synthetic_dataset.csv --target RPImax --scaling robust --selection rfecv_rf
```

## Command-Line Arguments

* `--data`: Path to your dataset in CSV or Excel format. **Required**.
* `--target`: Target column to predict. Default: `RPImax`.
* `--scaling`: Feature-scaling method. Options: `standard`, `robust`, `minmax`, `maxabs`, `power`, `quantile`. Default: `robust`.
* `--selection`: Feature-selection algorithm. Options: `mutual_info`, `rfe_rf`, `rfecv_rf`, `lasso`. Default: `rfecv_rf`.
* `--outdir`: Directory for plots and CSV metrics. Default: `results`.
* `--no-outliers`: Disables the `OneClassSVM` outlier-removal step. Use this when physiological extremes may contain meaningful predictive variance.
* `--contamination`: Expected proportion of outliers used by `OneClassSVM`. Default: `0.01`.

---
