"""Optional Streamlit presentation layer for the Bina.az ML project.

Run from the repository root with: streamlit run app.py
The experiment pipeline and from-scratch model implementations live in src/.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from src import data_prep
from src.decision_tree import DecisionTree
from src.svm import PegasosSVM, RandomFourierFeatures


ROOT = Path(__file__).resolve().parent
REPORT = ROOT / "report"
FIGURES = REPORT / "figures"
DATA_FILE = Path(data_prep.DATA_PATH)
if not DATA_FILE.is_absolute():
    DATA_FILE = ROOT / DATA_FILE


@st.cache_data(show_spinner="Loading and preparing the local dataset...")
def prepared_data(data_mtime: int):
    """Use exactly the core project's cleaning, feature and split functions."""
    raw = data_prep.load_raw(str(DATA_FILE))
    clean = data_prep.clean(raw)
    raw_count = len(raw)
    del raw
    X, prices, feature_names = data_prep.make_features(clean)
    train, validation, test, threshold = data_prep.split_indices(prices)
    return raw_count, clean, X, prices, feature_names, train, validation, test, threshold


@st.cache_data(show_spinner=False)
def read_csv(filename: str, mtime: int):
    return pd.read_csv(REPORT / filename, encoding="utf-8")


def report_csv(filename: str):
    path = REPORT / filename
    return read_csv(filename, path.stat().st_mtime_ns) if path.is_file() else None


def selected_settings():
    """Read the validation-selected final settings written by src.run_all."""
    text = (REPORT / "results_summary.txt").read_text(encoding="utf-8")
    patterns = {
        "reg": r"Regression tree: depth=(\d+), min_samples_leaf=(\d+)",
        "cls": r"Classification tree: criterion=(gini|entropy), depth=(\d+), min_samples_leaf=(\d+)",
        "svm": r"Linear SVM: lambda=([\d.eE+-]+), epochs=(\d+), schedule=(pegasos|inverse_sqrt|constant), eta0=([\d.eE+-]+)",
        "rff": r"RFF: gamma=([\d.eE+-]+), components=(\d+)",
    }
    found = {key: re.search(pattern, text) for key, pattern in patterns.items()}
    if any(match is None for match in found.values()):
        return None
    reg, cls, svm, rff = (found[key].groups() for key in patterns)
    return {
        "reg_depth": int(reg[0]), "reg_leaf": int(reg[1]),
        "cls_criterion": cls[0], "cls_depth": int(cls[1]), "cls_leaf": int(cls[2]),
        "lambda": float(svm[0]), "epochs": int(svm[1]),
        "schedule": svm[2], "eta0": float(svm[3]),
        "gamma": float(rff[0]), "components": int(rff[1]),
    }


@st.cache_resource(show_spinner="Training the project's four from-scratch demo models...")
def demo_models(data_mtime: int, summary_mtime: int):
    bundle = prepared_data(data_mtime)
    _, _, X, prices, _, train, _, test, threshold = bundle
    settings = selected_settings()
    X_train, X_test = X[train], X[test]
    y_train, _ = data_prep.make_tier_label(prices[train], threshold)
    X_train_scaled, X_test_scaled = data_prep.standardize(X_train, X_test)

    reg = DecisionTree(
        task="regression", criterion="mse", max_depth=settings["reg_depth"],
        min_samples_split=max(2, 2 * settings["reg_leaf"]),
        min_samples_leaf=settings["reg_leaf"],
    ).fit(X_train, np.log1p(prices[train]))
    cls = DecisionTree(
        task="classification", criterion=settings["cls_criterion"],
        max_depth=settings["cls_depth"],
        min_samples_split=max(2, 2 * settings["cls_leaf"]),
        min_samples_leaf=settings["cls_leaf"],
    ).fit(X_train, y_train)
    svm_args = dict(
        lambda_=settings["lambda"], n_epochs=settings["epochs"],
        learning_rate=settings["schedule"], eta0=settings["eta0"],
        random_state=data_prep.SEED,
    )
    linear = PegasosSVM(**svm_args, record_points=12).fit(X_train_scaled, y_train)
    rff = RandomFourierFeatures(
        n_components=settings["components"], gamma=settings["gamma"],
        random_state=data_prep.SEED,
    )
    rff_svm = PegasosSVM(**svm_args, record_points=6).fit(
        rff.fit_transform(X_train_scaled), y_train
    )
    return reg, cls, linear, rff, rff_svm, X_test_scaled


def figure(filename: str, caption: str):
    path = FIGURES / filename
    if path.is_file():
        st.image(str(path), caption=caption, width="stretch")
    else:
        st.info(f"Figure unavailable: {caption}. Run `python -m src.run_all` to generate it.")


def figure_grid(items):
    for start in range(0, len(items), 2):
        columns = st.columns(2)
        for column, (filename, caption) in zip(columns, items[start:start + 2]):
            with column:
                figure(filename, caption)


def missing_data_message():
    st.info(
        f"Local dataset not found at `{DATA_FILE.relative_to(ROOT) if DATA_FILE.is_relative_to(ROOT) else DATA_FILE}`. "
        "Place the CSV at the path described in `data/README.md`, then run "
        "`python -m src.run_all` to generate the report outputs."
    )


def overview():
    st.title("Bina.az machine learning project")
    st.write("A simple visual guide to the university project's two tasks, algorithms, and results.")
    a, b = st.columns(2)
    a.subheader("Task A · Regression")
    a.write("Predict a sale listing's price in AZN.")
    b.subheader("Task B · Classification")
    b.write("Classify a listing as Standard or Premium using a training-only price threshold.")
    st.subheader("Project pipeline")
    st.markdown(
        "**Raw Bina.az sale listings**  ↓  **Data cleaning**  ↓  **Feature engineering**  "
        "↓  **Train / validation / test**  ↓  **Decision Tree + Pegasos SVM**  ↓  **Evaluation**"
    )
    st.write(
        "The Decision Tree and Pegasos SVM are implemented from scratch with NumPy. "
        "Random Fourier Features (RFF) provide the nonlinear SVM extension. "
        "scikit-learn models appear only as comparison baselines."
    )


def dataset_page():
    st.title("Dataset and preprocessing")
    if not DATA_FILE.is_file():
        missing_data_message()
        return
    raw_count, clean, X, prices, names, train, validation, test, threshold = prepared_data(DATA_FILE.stat().st_mtime_ns)
    top = st.columns(4)
    for column, label, value in zip(
        top,
        ["Raw rows", "Clean listings", "Features", "Premium threshold"],
        [f"{raw_count:,}", f"{len(clean):,}", str(len(names)), f"{threshold:,.0f} AZN"],
    ):
        column.metric(label, value)
    st.write(f"Split sizes: **{len(train):,} training / {len(validation):,} validation / {len(test):,} test**")
    st.caption("The premium threshold is the median of training prices, then reused for validation and test.")
    st.subheader("Cleaned listing preview")
    st.dataframe(
        clean[["price", "area_m2", "rooms", "current_floor", "total_floors", "category", "city_clean"]].head(10),
        hide_index=True, width="stretch",
    )
    st.subheader("What happens before modelling")
    st.markdown(
        "- Repeated scrape snapshots are deduplicated by listing URL; the latest is kept.\n"
        "- Prices, areas, rooms, floors, land size, and coordinates are converted to numbers; fixed plausibility bounds remove obvious errors.\n"
        "- Missing structural values become zero with separate missing-value indicators.\n"
        "- Category and city use fixed one-hot levels; repair, deed, and mortgage become binary features.\n"
        "- `unit_price` and `total_price` are excluded because they reveal the target price. Identifiers and raw text are also excluded.\n"
        "- The price-tier threshold and SVM scaling mean/std are learned from training rows only."
    )


def eda_page():
    st.title("Exploratory data analysis")
    st.write("These are the figures generated by the main project pipeline from the cleaned listings.")
    figure_grid([
        ("eda_price_distribution.png", "Sale price distribution"),
        ("eda_log_price_distribution.png", "Log-price distribution"),
        ("eda_class_balance.png", "Standard / Premium balance"),
        ("eda_area_vs_price.png", "Area versus price"),
        ("eda_rooms_price.png", "Room count and median price"),
        ("eda_locations.png", "Listing locations"),
    ])


def tree_page():
    st.title("Decision Tree")
    st.write("**Regression:** property features → repeated splits → leaf → predicted price")
    st.write("**Classification:** property features → repeated splits → leaf → Standard or Premium")
    st.markdown(
        "At each node, the tree tries feature thresholds and keeps the **best split**: "
        "the one with the largest reduction in impurity. It then grows the left and right branches recursively."
    )
    st.markdown(
        "- **Gini** and **entropy** measure how mixed the classes are in a node.\n"
        "- **MSE** measures price variation for regression; leaves store the mean training log-price.\n"
        "- **max_depth** limits levels; **min_samples_split** limits when a node may split.\n"
        "- **min_samples_leaf** requires enough rows in each child; **min_impurity_decrease** requires a minimum improvement.\n"
        "- **Feature importance** sums each feature's weighted impurity reduction across the tree."
    )
    results = report_csv("results.csv")
    if results is not None and {"task", "model", "RMSE", "MAE", "R2", "Precision", "Recall", "F1", "ROC_AUC"}.issubset(results):
        st.subheader("Our trees versus sklearn baselines · test set")
        tree_rows = results[results["model"].astype(str).str.contains("Tree")]
        st.dataframe(tree_rows[["task", "model", "RMSE", "MAE", "R2", "Precision", "Recall", "F1", "ROC_AUC"]], hide_index=True, width="stretch")
    else:
        st.info("Tree result table unavailable. Run `python -m src.run_all`.")
    figure_grid([
        ("tree_feature_importance.png", "Our tree: feature importance"),
        ("tree_regression_depth_sweep.png", "Regression depth sweep"),
        ("tree_classification_depth_sweep.png", "Classification depth sweep"),
        ("tree_criterion_comparison.png", "Gini versus entropy"),
    ])


def svm_page():
    st.title("Pegasos SVM")
    st.markdown("**Listing features**  ↓  **Training-only scaling**  ↓  **Linear decision function**  ↓  **Margin**  ↓  **Standard / Premium**")
    st.markdown(
        "- The **hyperplane** is the boundary between the two predicted classes. The **margin** is the space around it.\n"
        "- **Hinge loss** penalizes points on the wrong side of the margin. **Support vectors** are training points on or inside that margin.\n"
        "- **Lambda** controls regularization and the margin. **Epochs** count full passes through the training data.\n"
        "- The **learning-rate schedule** controls update size. Pegasos repeatedly shrinks the weights and, for a margin violation, moves them toward the correct class.\n"
        "- **Random Fourier Features** first map scaled features into a new representation. The same from-scratch linear Pegasos solver then learns on that representation."
    )
    figure_grid([
        ("svm_convergence.png", "Pegasos objective during training"),
        ("svm_weights.png", "Largest standardized linear SVM weights"),
        ("svm_lambda_sweep.png", "Regularization sweep"),
        ("svm_margin_sweep.png", "Regularization and margin"),
        ("svm_epoch_sweep.png", "Epoch sweep"),
        ("svm_epoch_margin.png", "Epochs and margin"),
        ("svm_schedule_comparison.png", "Learning-rate schedules"),
        ("svm_schedule_margin.png", "Learning-rate schedule and margin"),
        ("rff_gamma_sweep.png", "RFF bandwidth sweep"),
    ])


def results_page():
    st.title("Model results")
    st.caption("Tables read the latest saved pipeline outputs. Rerun `python -m src.run_all` to refresh them.")
    results = report_csv("results.csv")
    if results is None or not {"task", "model"}.issubset(results):
        st.info("Results are unavailable. Run `python -m src.run_all`.")
    else:
        for task, label, columns in [
            ("regression", "Regression · test set", ["model", "RMSE", "MAE", "R2"]),
            ("classification", "Classification · test set", ["model", "Precision", "Recall", "F1", "ROC_AUC"]),
        ]:
            st.subheader(label)
            available = [column for column in columns if column in results]
            subset = results.loc[results["task"] == task, available]
            if not subset.empty:
                st.dataframe(subset, hide_index=True, width="stretch")
                chart_metric = "RMSE" if task == "regression" else "F1"
                if chart_metric in subset:
                    st.bar_chart(subset.set_index("model")[[chart_metric]])
            else:
                st.info(f"No saved {task} rows are available.")
        timing = [c for c in ["model", "fit_seconds", "predict_seconds", "predict_ms_per_1000"] if c in results]
        if len(timing) > 1:
            st.subheader("Runtime")
            st.dataframe(results[timing], hide_index=True, width="stretch")

    cv = report_csv("cv_summary.csv")
    if cv is not None:
        st.subheader("Three-fold uncertainty")
        st.dataframe(cv, hide_index=True, width="stretch")

    experiments = report_csv("experiment_results.csv")
    if experiments is not None and "experiment" in experiments:
        st.subheader("Validation experiments")
        choices = experiments["experiment"].dropna().unique().tolist()
        choice = st.selectbox("Experiment", choices)
        subset = experiments.loc[experiments["experiment"] == choice].dropna(axis=1, how="all")
        st.dataframe(subset, hide_index=True, width="stretch")
    figure_grid([
        ("confusion_tree.png", "Our classification tree confusion matrix"),
        ("confusion_svm.png", "Our linear Pegasos confusion matrix"),
        ("runtime_scaling.png", "Training-time scaling"),
    ])


def example_page():
    st.title("Example prediction")
    st.write("Choose a held out test listing. The app trains the same final from-scratch model configurations as the pipeline, using training rows only.")
    if not DATA_FILE.is_file():
        missing_data_message()
        return
    summary = REPORT / "results_summary.txt"
    if not summary.is_file() or selected_settings() is None:
        st.info("Final model settings are unavailable. Run `python -m src.run_all` first.")
        return
    data_mtime = DATA_FILE.stat().st_mtime_ns
    _, clean, X, prices, _, _, _, test, threshold = prepared_data(data_mtime)
    row_number = st.number_input("Test listing number", min_value=1, max_value=len(test), value=1, step=1)
    if st.button("Train demo models and show prediction"):
        st.session_state["demo_ready"] = True
    if not st.session_state.get("demo_ready", False):
        st.caption("Models are trained once and cached after you press the button.")
        return

    reg, cls, linear, rff, rff_svm, X_test_scaled = demo_models(data_mtime, summary.stat().st_mtime_ns)
    i = int(row_number) - 1
    source_row = int(test[i])
    x = X[source_row:source_row + 1]
    x_scaled = X_test_scaled[i:i + 1]
    actual_tier, _ = data_prep.make_tier_label(prices[source_row:source_row + 1], threshold)
    label = lambda value: "Premium" if int(value) == 1 else "Standard"
    st.subheader("Listing")
    display = clean.loc[source_row, ["area_m2", "rooms", "current_floor", "total_floors", "lat", "lng", "category", "city_clean"]]
    st.dataframe(display.astype(str).rename("Value").to_frame(), width="stretch")
    a, b = st.columns(2)
    a.metric("Actual sale price", f"{prices[source_row]:,.0f} AZN")
    a.metric("Our Tree predicted price", f"{max(float(np.expm1(reg.predict(x)[0])), 0.0):,.0f} AZN")
    b.metric("Actual tier", label(actual_tier[0]))
    st.write("**Predicted tier**")
    st.dataframe(pd.DataFrame({
        "Our model": ["Decision Tree", "Linear Pegasos SVM", "RFF Pegasos SVM"],
        "Prediction": [
            label(cls.predict(x)[0]),
            label(linear.predict(x_scaled)[0]),
            label(rff_svm.predict(rff.transform(x_scaled))[0]),
        ],
    }), hide_index=True, width="stretch")
    st.caption("The classification label is derived from the training-only price threshold; predictions use no price feature.")


def about_page():
    st.title("About the from-scratch requirement")
    st.write("The Decision Tree algorithm and Pegasos SVM optimization are implemented manually with NumPy in `src/decision_tree.py` and `src/svm.py`.")
    st.write("The project also implements its evaluation functions in `src/evaluate.py`. scikit-learn is used for reference baselines in the experiment pipeline.")
    st.write("This Streamlit app is only a visualization and demo layer. It does not replace the project's core algorithms or `python -m src.run_all`.")


st.set_page_config(page_title="Bina.az ML demo", page_icon="🏠", layout="wide")
st.sidebar.title("Bina.az ML demo")
page = st.sidebar.radio("Explore", [
    "Project Overview", "Dataset / Preprocessing", "Exploratory Data Analysis",
    "Decision Tree", "Pegasos SVM", "Model Results", "Example Prediction",
    "About the From-Scratch Requirement",
])
pages = {
    "Project Overview": overview,
    "Dataset / Preprocessing": dataset_page,
    "Exploratory Data Analysis": eda_page,
    "Decision Tree": tree_page,
    "Pegasos SVM": svm_page,
    "Model Results": results_page,
    "Example Prediction": example_page,
    "About the From-Scratch Requirement": about_page,
}
pages[page]()
