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
│   ├── report_revised.html    # optional HTML preview of the report
│   ├── report.pdf             # final two-column PDF compiled from report.tex
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
Use Python 3.12 for the pinned dependencies. On Windows PowerShell, create the
virtual environment once if `.venv` does not already exist:

```powershell
py -3.12 -m venv .venv
```

Then install and run from the repository root using that environment's Python:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m src.run_all
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

The submitted two-column report PDF was compiled from `report/report.tex`, which
uses the official self-contained IEEE-style template. After generating figures,
run `tectonic report.tex` from the `report/` directory to rebuild it (or use
`pdflatex` twice). `report/report_revised.html` is an optional HTML preview.
The three PNG figures referenced by `report.tex` are kept in Git; the other
generated figures can be recreated with `python -m src.run_all`.

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
Current core suite: **7 tests**. Verified in a fresh Python 3.12.10 environment on 16 October 2026; the full pipeline completed and reproduced the headline metrics. Fit/predict timings vary with machine load.

## Optional visual demo
Using the same Python 3.12 virtual environment, install the separate UI requirement
and launch Streamlit from the repository root:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt -r requirements-ui.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

The dashboard reads the local CSV and generated files in `report/`. Run
`python -m src.run_all` to refresh result tables and figures. The example prediction
trains the project's own final models on the training split when requested; it does not
change the core pipeline or its results.

## Before final submission
- Confirm that all three members understand their own code, the shared pipeline, the reported metrics, and the limitations for the live demo.
- Merge each member's work into the main submission repository, commit the final report and slides, and create the `v1.0-final` tag on that final commit.
- Submit the main repository link, `report/report.pdf`, and `presentation/bina_az_presentation.pdf` on Moodle before the deadline. One member submits for the whole team.
