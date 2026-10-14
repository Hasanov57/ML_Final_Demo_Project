# MLE-AI-201 Final Project — Bina.az ML

From-scratch **Decision Tree** and **Pegasos SVM** on the Bina.az sale-listings dataset.

## Tasks
- **Task A — Regression:** predict sale `price` (tree is trained on `log1p(price)` and transformed back for price-scale metrics).
- **Task B — Classification:** predict **premium vs standard** using the median price learned from the training split only.

## Core rule
The main Decision Tree and SVM are implemented from scratch with NumPy. scikit-learn is used only for comparison baselines.

## Project structure
```text
.
├── data/
│   └── README.md              # put bina_az_sale.csv here locally
├── src/
│   ├── data_prep.py           # cleaning, features, split, scaling
│   ├── decision_tree.py       # from-scratch Gini/entropy/MSE tree
│   ├── svm.py                 # from-scratch Pegasos + RFF
│   ├── evaluate.py            # NumPy metrics
│   └── run_all.py             # reproduces experiments/results/figures
├── report/
│   ├── report.tex
│   ├── report.pdf             # current generated draft
│   ├── results.csv
│   ├── cv_summary.csv
│   ├── experiment_results.csv
│   └── figures/
├── presentation/
├── contribution_report.md
├── requirements.txt
└── PROJECT_STATEMENT.pdf
```

## Dataset
Download the Kaggle dataset **Bina.az Sale Project** by `sehriyarmemmedli` and place the CSV at:

```text
data/bina_az_sale.csv
```

The raw dataset is intentionally excluded from Git.

## Reproduce everything
Create/activate a virtual environment, install requirements, then run from the repository root:

```bash
pip install -r requirements.txt
python -m src.run_all
```

The command performs:
1. raw-data loading and cleaning,
2. duplicate listing removal (latest snapshot kept),
3. leakage removal (`unit_price`, `total_price`),
4. feature creation and leakage-safe 70/15/15 split,
5. validation-only hyperparameter selection,
6. from-scratch regression/classification trees,
7. from-scratch linear SVM with lambda / epoch / learning-rate-schedule tuning,
8. Random Fourier Feature non-linear extension,
9. scikit-learn baselines (SVM baseline uses `StandardScaler + SGDClassifier` in a Pipeline),
10. fit/predict timing + runtime scaling study,
11. test metrics, figures, and 3-fold uncertainty on a fixed-seed development subset.

Generated outputs are written under `report/`.

## Current reproducible results
With seed `42` on the supplied dataset version:

### Data
- Raw rows: **100,775**
- Clean unique listings: **64,128**
- Features: **29**
- Train / validation / test: **44,890 / 9,619 / 9,619**
- Training-only premium threshold: **210,000 AZN**

### Regression test results
| Model | RMSE (AZN) | MAE (AZN) | R² |
|---|---:|---:|---:|
| Our Decision Tree | 310,716 | 87,181 | 0.5921 |
| sklearn Decision Tree | 299,329 | 86,475 | 0.6215 |

### Classification test results
| Model | F1 | ROC-AUC |
|---|---:|---:|
| Our Decision Tree | **0.9099** | **0.9630** |
| sklearn Decision Tree | 0.9103 | 0.9629 |
| Our Linear Pegasos | 0.8228 | 0.9083 |
| Our RFF Pegasos | 0.8101 | 0.8936 |
| sklearn SGD hinge | 0.8259 | 0.9077 |

## Important preprocessing decisions
- `unit_price` and `total_price` are target leakage and are never used as features.
- Repeated scrape snapshots are deduplicated by listing URL, keeping the latest snapshot.
- Obvious impossible/extreme scrape values are removed using fixed plausibility bounds instead of test-derived quantiles.
- Missing room/floor/land fields use a fixed zero representation plus explicit missing indicators.
- Our SVM standardization uses **training mean/std only**. The sklearn hinge baseline uses `StandardScaler()` inside a sklearn `Pipeline`.
- Price-tier threshold is learned from **training prices only**.

## Validation-selected SVM configuration
- `lambda = 0.001`
- `epochs = 3`
- learning-rate schedule = `pegasos` (`eta_t = 1 / (lambda * t)`)
- RFF: `gamma = 0.1`, `120` components

The experiment CSV also contains epoch and schedule sweeps, margin widths, objective values, and empirical fit/predict scaling.

## Tests
```bash
python -m pytest -q
```
Current core suite: **6 tests**.

## Before final submission
- Replace `Team Member 1/2/3` in `report/report.tex` with the real names and rebuild the PDF.
- Complete `contribution_report.md` according to actual work/commits.
- Add final slides PDF.
- Run `python -m src.run_all` from a clean environment.
- Tag the final Git commit as `v1.0-final`.
