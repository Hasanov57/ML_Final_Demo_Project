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
    available = [(filename, caption) for filename, caption in items if (FIGURES / filename).is_file()]
    if not available:
        st.info("These extra charts are generated locally by `python -m src.run_all` after the dataset is downloaded.")
        return
    for start in range(0, len(available), 2):
        columns = st.columns(2)
        for column, (filename, caption) in zip(columns, available[start:start + 2]):
            with column:
                figure(filename, caption)


def missing_data_message():
    st.info(
        f"Local dataset not found at `{DATA_FILE.relative_to(ROOT) if DATA_FILE.is_relative_to(ROOT) else DATA_FILE}`. "
        "Place the CSV at the path described in `data/README.md`, then run "
        "`python -m src.run_all` to generate the report outputs."
    )


def overview():
    st.title("Bina.az ML project: start here")
    st.write("An AI Academy project using property listings to answer two simple questions.")
    a, b = st.columns(2)
    a.subheader("1. How much is a property?")
    a.write("The regression tree predicts a listing price in AZN. We compare the guess with the actual price.")
    b.subheader("2. Standard or Premium?")
    b.write("The classification models predict whether the price is above 210,000 AZN, the median learned from training listings. A listing exactly at the threshold is Standard.")
    st.subheader("The result in plain language")
    st.write(
        "Our classification tree distinguishes Standard from Premium listings well: its F1 score is about **0.91 out of 1**. "
        "Our linear SVM scores about **0.82**. The tree's price predictions have an average absolute error "
        "of about **87,000 AZN**; expensive properties cause particularly large errors. "
        "These numbers describe a held-out test set, not guaranteed accuracy for a new listing."
    )
    st.subheader("How to use this dashboard")
    st.markdown(
        "1. Open **Model Results** for the answer to 'which model worked best?'.\n"
        "2. Open **Dataset / Preprocessing** to see what the models learned from.\n"
        "3. Open **Decision Tree** and **Pegasos SVM** for one key chart each.\n"
        "4. Open **Example Prediction** to see one held-out listing; one example does not measure overall quality."
    )
    st.subheader("Project pipeline")
    st.markdown(
        "**Raw Bina.az sale listings**  ↓  **Data cleaning**  ↓  **Feature engineering**  "
        "↓  **Train / validation / test**  ↓  **Decision Tree + Pegasos SVM**  ↓  **Evaluation**"
    )
    with st.expander("What is special about the code?"):
        st.write(
            "We implemented the Decision Tree and Pegasos SVM algorithms ourselves with NumPy. "
            "Random Fourier Features (RFF) add a nonlinear SVM experiment. "
            "scikit-learn models are used only for comparison."
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
    st.write("These charts describe the cleaned listings before model training.")
    st.subheader("What to look for")
    st.markdown(
        "- **Prices:** most listings are in the lower price range; a few expensive listings stretch the chart to the right.\n"
        "- **Area versus price:** larger homes tend to cost more, but area alone cannot determine price.\n"
        "- **Class balance:** Standard and Premium are close to equal because the threshold uses the training median."
    )
    figure_grid([
        ("eda_price_distribution.png", "Sale price distribution"),
        ("eda_area_vs_price.png", "Area versus price"),
        ("eda_class_balance.png", "Standard / Premium balance"),
    ])
    with st.expander("More data exploration charts"):
        figure_grid([
            ("eda_log_price_distribution.png", "Log-price distribution: compresses the long right tail"),
            ("eda_rooms_price.png", "Room count and median price"),
            ("eda_locations.png", "Listing locations"),
        ])


def tree_page():
    st.title("Decision Tree")
    st.write("A tree repeatedly asks simple yes/no questions about a listing, such as its area or location.")
    st.subheader("Which information did the tree use most?")
    st.write(
        "Longer bars mean the fitted tree used that feature more often to make useful splits. "
        "Area is the strongest feature here; longitude and latitude describe location. "
        "This is a model summary, not proof that a feature causes the price tier."
    )
    figure("tree_feature_importance.png", "Tree feature importance; longer means more useful splits in this model")
    st.subheader("Does making the tree deeper help?")
    st.write(
        "The chart compares training F1 with validation F1. **Higher is better.** "
        "The validation line rises over the tested depths, while the gap between the lines grows slightly. "
        "We choose settings using validation results, then use the test set once for the final score."
    )
    figure("tree_classification_depth_sweep.png", "Tree depth versus F1 on training and validation data")
    with st.expander("Technical details and extra tree charts"):
        st.markdown(
            "- For **price regression**, leaves predict the mean training log-price; predictions are converted back to AZN.\n"
            "- For **classification**, leaves predict Standard or Premium and class probabilities.\n"
            "- **Gini** and **entropy** measure how mixed classes are. **MSE** measures price variation.\n"
            "- Depth and minimum sample settings limit how complex the tree becomes."
        )
        results = report_csv("results.csv")
        if results is not None and {"task", "model", "RMSE", "MAE", "R2", "Precision", "Recall", "F1", "ROC_AUC"}.issubset(results):
            tree_rows = results[results["model"].astype(str).str.contains("Tree")]
            st.dataframe(tree_rows[["task", "model", "RMSE", "MAE", "R2", "Precision", "Recall", "F1", "ROC_AUC"]].round(3), hide_index=True, width="stretch")
        figure_grid([
            ("tree_regression_depth_sweep.png", "Regression depth sweep"),
            ("tree_criterion_comparison.png", "Gini versus entropy"),
        ])


def svm_page():
    st.title("Pegasos SVM")
    st.write(
        "The SVM draws a boundary between Standard and Premium listings. Pegasos is the method "
        "we coded to adjust that boundary during training."
    )
    st.subheader("Is training working?")
    st.write(
        "The line shows the training objective, a measure of model error plus a complexity penalty. "
        "**Lower is better.** It drops overall as training proceeds, though individual points need not fall smoothly. "
        "This chart checks training behaviour; the separate test F1 score tells us how well predictions work."
    )
    figure("svm_convergence.png", "Pegasos training objective over updates; look for an overall downward trend")
    st.info(
        "On the held-out test set, linear Pegasos reaches F1 ≈ 0.82, below the tree's ≈ 0.91. "
        "The RFF extension reaches ≈ 0.81, so added complexity did not improve this dataset's test result."
    )
    with st.expander("Technical terms and extra SVM charts"):
        st.markdown(
            "- The **margin** is the space around the class boundary. **Support vectors** lie on or inside it.\n"
            "- **Lambda** controls regularisation; **epochs** count passes through the training data.\n"
            "- The **learning-rate schedule** sets how large each update is.\n"
            "- **Random Fourier Features (RFF)** make a nonlinear feature map before the same linear solver."
        )
        figure_grid([
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
    st.write("These scores come from listings kept out of training and model selection.")
    st.caption("Tables read saved pipeline outputs. Rerun `python -m src.run_all` to refresh them.")
    results = report_csv("results.csv")
    if results is None or not {"task", "model"}.issubset(results):
        st.info("Results are unavailable. Run `python -m src.run_all`.")
    else:
        for task, label, columns, explanation in [
            ("regression", "Price prediction · test set", ["model", "RMSE", "MAE", "R2"],
             "**MAE** is the typical absolute price error in AZN; **RMSE** penalises large mistakes more. Lower is better for both. **R²** is higher when the model explains more price variation."),
            ("classification", "Standard versus Premium · test set", ["model", "Precision", "Recall", "F1", "ROC_AUC"],
             "**F1** balances correct Premium predictions with Premium listings found. **ROC-AUC** measures ranking across thresholds. Both run from 0 to 1; higher is better."),
        ]:
            st.subheader(label)
            st.write(explanation)
            available = [column for column in columns if column in results]
            subset = results.loc[results["task"] == task, available]
            if not subset.empty:
                display = subset.copy()
                number_columns = [column for column in available if column != "model"]
                if task == "classification":
                    display[number_columns] = display[number_columns].round(3)
                else:
                    display[[c for c in number_columns if c != "R2"]] = display[[c for c in number_columns if c != "R2"]].round(0)
                    if "R2" in display:
                        display["R2"] = display["R2"].round(3)
                st.dataframe(display, hide_index=True, width="stretch")
                chart_metric = "RMSE" if task == "regression" else "F1"
                if chart_metric in subset:
                    st.bar_chart(subset.set_index("model")[[chart_metric]])
                    st.caption("Read the bars as: lower RMSE is better for price; higher F1 is better for the price tier.")
            else:
                st.info(f"No saved {task} rows are available.")
        st.info(
            "Main takeaway: our tree nearly matches scikit-learn on classification. "
            "Price prediction is less reliable, especially for expensive listings. "
            "The nonlinear RFF experiment did not beat the simpler linear SVM."
        )
        with st.expander("Runtime and advanced result tables"):
            timing = [c for c in ["model", "fit_seconds", "predict_seconds", "predict_ms_per_1000"] if c in results]
            if len(timing) > 1:
                st.subheader("Runtime; smaller is faster")
                st.caption("Fit and predict totals are in seconds. The final column is milliseconds per 1,000 listings.")
                st.dataframe(results[timing].round(3), hide_index=True, width="stretch")

    with st.expander("Uncertainty, validation experiments, and other charts"):
        cv = report_csv("cv_summary.csv")
        if cv is not None:
            st.subheader("Three-fold uncertainty")
            st.write("We repeated evaluation on three development-data folds. The spread shows how much scores changed across those folds.")
            st.dataframe(cv, hide_index=True, width="stretch")
        experiments = report_csv("experiment_results.csv")
        if experiments is not None and "experiment" in experiments:
            st.subheader("Validation experiments")
            st.write("These settings were compared on validation data before the final test evaluation.")
            choices = experiments["experiment"].dropna().unique().tolist()
            choice = st.selectbox("Experiment", choices)
            subset = experiments.loc[experiments["experiment"] == choice].dropna(axis=1, how="all")
            st.dataframe(subset, hide_index=True, width="stretch")
        st.write("In each confusion matrix, the diagonal cells are correct predictions; the other cells are mistakes.")
        figure_grid([
            ("confusion_tree.png", "Our classification tree confusion matrix"),
            ("confusion_svm.png", "Our linear Pegasos confusion matrix"),
            ("runtime_scaling.png", "Training-time scaling"),
        ])


def example_page():
    st.title("Example prediction")
    st.write("Choose a held out test listing. The app trains the same final from-scratch model configurations as the pipeline, using training rows only.")
    st.info("One example can be right or wrong by chance. Use Model Results for overall model quality.")
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
