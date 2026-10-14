"""Run the complete, reproducible ML final-project pipeline.

From the repository root:
    python -m src.run_all
"""

from __future__ import annotations

import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from . import data_prep, evaluate
from .decision_tree import DecisionTree
from .svm import PegasosSVM, RandomFourierFeatures

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "report" / "figures"
RESULTS_DIR = ROOT / "report"
SEED = data_prep.SEED
CV_MAX_SAMPLES = 18_000  # fixed-seed development subset keeps 3-fold CV reproducible and practical


def _savefig(name: str):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(FIG_DIR / name, dpi=120)
    plt.close("all")


def _sanity_checks():
    """Tiny deterministic checks against sklearn/reference behaviour."""
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    yc = np.array([0, 0, 1, 1])
    ours = DecisionTree(task="classification", criterion="gini", max_depth=1).fit(X, yc)
    sk = DecisionTreeClassifier(criterion="gini", max_depth=1, random_state=SEED).fit(X, yc)
    assert ours.root.feature == int(sk.tree_.feature[0]) == 0
    assert abs(ours.root.threshold - float(sk.tree_.threshold[0])) < 1e-12
    assert np.array_equal(ours.predict(X), sk.predict(X))

    yr = np.array([1.0, 1.0, 8.0, 8.0])
    ours_r = DecisionTree(task="regression", criterion="mse", max_depth=1).fit(X, yr)
    sk_r = DecisionTreeRegressor(criterion="squared_error", max_depth=1, random_state=SEED).fit(X, yr)
    assert abs(ours_r.root.threshold - float(sk_r.tree_.threshold[0])) < 1e-12

    # SVM sanity check: both our implementation and sklearn's hinge-loss baseline
    # should perfectly separate this tiny, fully linearly-separable dataset.
    Xs = np.array([[-2.0, -1.0], [-1.0, -2.0], [2.0, 1.0], [1.0, 2.0]])
    ys = np.array([0, 0, 1, 1])
    svm = PegasosSVM(lambda_=1e-2, n_epochs=20, learning_rate="pegasos", random_state=SEED, record_points=5).fit(Xs, ys)
    sk_svm = make_pipeline(
        StandardScaler(),
        SGDClassifier(loss="hinge", alpha=1e-2, max_iter=5000, tol=1e-7, random_state=SEED),
    ).fit(Xs, ys)
    assert evaluate.accuracy(ys, svm.predict(Xs)) == 1.0
    assert evaluate.accuracy(ys, sk_svm.predict(Xs)) == 1.0
    assert np.array_equal(svm.predict(Xs), sk_svm.predict(Xs))
    assert svm.objective_history_[-1][1] < svm.objective_history_[0][1]
    print("[OK] Sanity checks passed (tree split matches sklearn; our SVM and sklearn hinge separate toy data).")


def _plot_eda(clean_df: pd.DataFrame, train_threshold: float):
    price = clean_df["price"].to_numpy()

    plt.figure(figsize=(7, 4))
    plt.hist(price, bins=60)
    plt.xlabel("Price (AZN)")
    plt.ylabel("Listings")
    plt.title("Cleaned sale-price distribution")
    _savefig("eda_price_distribution.png")

    plt.figure(figsize=(7, 4))
    plt.hist(np.log1p(price), bins=60)
    plt.xlabel("log(1 + price)")
    plt.ylabel("Listings")
    plt.title("Log-price distribution")
    _savefig("eda_log_price_distribution.png")

    sample = clean_df.sample(min(5000, len(clean_df)), random_state=SEED)
    plt.figure(figsize=(7, 4))
    plt.scatter(sample["area_m2"], sample["price"], s=7, alpha=0.25)
    plt.yscale("log")
    plt.xscale("log")
    plt.xlabel("Area (m², log scale)")
    plt.ylabel("Price (AZN, log scale)")
    plt.title("Area vs price")
    _savefig("eda_area_vs_price.png")

    rooms = clean_df.loc[clean_df["rooms"] > 0].groupby("rooms")["price"].median()
    rooms = rooms[rooms.index <= 10]
    plt.figure(figsize=(7, 4))
    plt.bar(rooms.index.astype(str), rooms.values)
    plt.xlabel("Rooms")
    plt.ylabel("Median price (AZN)")
    plt.title("Room count and median price")
    _savefig("eda_rooms_price.png")

    tier = (clean_df["price"].to_numpy() > train_threshold).astype(int)
    counts = np.bincount(tier, minlength=2)
    plt.figure(figsize=(5, 4))
    plt.bar(["Standard", "Premium"], counts)
    plt.ylabel("Listings")
    plt.title("Price-tier class balance (training threshold)")
    _savefig("eda_class_balance.png")

    plt.figure(figsize=(6, 5))
    geo = sample[(sample["lat"] > 39.5) & (sample["lat"] < 41.0) & (sample["lng"] > 49.0) & (sample["lng"] < 50.7)]
    plt.scatter(geo["lng"], geo["lat"], s=7, alpha=0.3)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Listing locations (sample)")
    _savefig("eda_locations.png")


def _tree_regression_experiment(X_tr, y_tr, X_val, y_val):
    log_tr = np.log1p(y_tr)
    log_val = np.log1p(y_val)
    depths = [4, 6, 8, 10, 12]
    rows = []
    for depth in depths:
        model = DecisionTree(
            task="regression", criterion="mse", max_depth=depth,
            min_samples_split=40, min_samples_leaf=20,
        ).fit(X_tr, log_tr)
        pred_tr = model.predict(X_tr)
        pred_val = model.predict(X_val)
        rows.append({
            "depth": depth,
            "train_log_rmse": evaluate.rmse(log_tr, pred_tr),
            "val_log_rmse": evaluate.rmse(log_val, pred_val),
        })

    exp = pd.DataFrame(rows)
    best_depth = int(exp.loc[exp["val_log_rmse"].idxmin(), "depth"])

    leaf_rows = []
    for leaf in [5, 20, 50, 100]:
        model = DecisionTree(
            task="regression", criterion="mse", max_depth=best_depth,
            min_samples_split=max(2, 2 * leaf), min_samples_leaf=leaf,
        ).fit(X_tr, log_tr)
        pred_val = model.predict(X_val)
        leaf_rows.append({"min_samples_leaf": leaf, "val_log_rmse": evaluate.rmse(log_val, pred_val)})
    leaf_exp = pd.DataFrame(leaf_rows)
    best_leaf = int(leaf_exp.loc[leaf_exp["val_log_rmse"].idxmin(), "min_samples_leaf"])

    plt.figure(figsize=(7, 4))
    plt.plot(exp["depth"], exp["train_log_rmse"], marker="o", label="Train")
    plt.plot(exp["depth"], exp["val_log_rmse"], marker="o", label="Validation")
    plt.xlabel("max_depth")
    plt.ylabel("RMSE on log(1+price)")
    plt.title("Regression tree: depth / bias-variance trade-off")
    plt.legend()
    _savefig("tree_regression_depth_sweep.png")

    return best_depth, best_leaf, exp, leaf_exp


def _tree_classification_experiment(X_tr, y_tr, X_val, y_val):
    criterion_rows = []
    for criterion in ["gini", "entropy"]:
        model = DecisionTree(
            task="classification", criterion=criterion, max_depth=7,
            min_samples_split=40, min_samples_leaf=20,
        ).fit(X_tr, y_tr)
        pred = model.predict(X_val)
        score = model.predict_proba(X_val)[:, 1]
        p, r, f1 = evaluate.precision_recall_f1(y_val, pred)
        criterion_rows.append({
            "criterion": criterion, "precision": p, "recall": r, "f1": f1,
            "auc": evaluate.roc_auc(y_val, score),
        })
    criterion_exp = pd.DataFrame(criterion_rows)
    best_criterion = str(criterion_exp.sort_values(["f1", "auc"], ascending=False).iloc[0]["criterion"])

    rows = []
    for depth in [3, 5, 7, 9, 11]:
        model = DecisionTree(
            task="classification", criterion=best_criterion, max_depth=depth,
            min_samples_split=40, min_samples_leaf=20,
        ).fit(X_tr, y_tr)
        tr_pred = model.predict(X_tr)
        val_pred = model.predict(X_val)
        rows.append({
            "depth": depth,
            "train_f1": evaluate.precision_recall_f1(y_tr, tr_pred)[2],
            "val_f1": evaluate.precision_recall_f1(y_val, val_pred)[2],
            "val_auc": evaluate.roc_auc(y_val, model.predict_proba(X_val)[:, 1]),
        })
    depth_exp = pd.DataFrame(rows)
    best_depth = int(depth_exp.sort_values(["val_f1", "val_auc"], ascending=False).iloc[0]["depth"])

    leaf_rows = []
    for leaf in [5, 20, 50, 100]:
        model = DecisionTree(
            task="classification", criterion=best_criterion, max_depth=best_depth,
            min_samples_split=max(2, 2 * leaf), min_samples_leaf=leaf,
        ).fit(X_tr, y_tr)
        pred = model.predict(X_val)
        score = model.predict_proba(X_val)[:, 1]
        leaf_rows.append({
            "min_samples_leaf": leaf,
            "val_f1": evaluate.precision_recall_f1(y_val, pred)[2],
            "val_auc": evaluate.roc_auc(y_val, score),
        })
    leaf_exp = pd.DataFrame(leaf_rows)
    best_leaf = int(leaf_exp.sort_values(["val_f1", "val_auc"], ascending=False).iloc[0]["min_samples_leaf"])

    plt.figure(figsize=(7, 4))
    plt.plot(depth_exp["depth"], depth_exp["train_f1"], marker="o", label="Train F1")
    plt.plot(depth_exp["depth"], depth_exp["val_f1"], marker="o", label="Validation F1")
    plt.xlabel("max_depth")
    plt.ylabel("F1")
    plt.title("Classification tree: depth / bias-variance trade-off")
    plt.legend()
    _savefig("tree_classification_depth_sweep.png")

    plt.figure(figsize=(5, 4))
    plt.bar(criterion_exp["criterion"], criterion_exp["f1"])
    plt.ylabel("Validation F1")
    plt.title("Gini vs entropy")
    _savefig("tree_criterion_comparison.png")

    return best_criterion, best_depth, best_leaf, criterion_exp, depth_exp, leaf_exp


def _svm_experiment(X_tr_s, y_tr, X_val_s, y_val):
    """Tune lambda, epochs and learning-rate schedule using validation data only."""
    # 1) Regularisation sweep with a short, fixed training budget.
    lambda_rows = []
    for lambda_ in [1e-2, 3e-3, 1e-3, 3e-4]:
        model = PegasosSVM(
            lambda_=lambda_, n_epochs=2, learning_rate="pegasos",
            random_state=SEED, record_points=8,
        ).fit(X_tr_s, y_tr)
        pred = model.predict(X_val_s)
        score = model.decision_function(X_val_s)
        p, r, f1 = evaluate.precision_recall_f1(y_val, pred)
        lambda_rows.append({
            "lambda": lambda_, "precision": p, "recall": r, "f1": f1,
            "auc": evaluate.roc_auc(y_val, score), "margin_width": model.margin_width(),
        })
    lambda_exp = pd.DataFrame(lambda_rows)
    best_lambda = float(lambda_exp.sort_values(["f1", "auc"], ascending=False).iloc[0]["lambda"])

    # 2) Number-of-epochs sweep.
    epoch_rows = []
    for epochs in [1, 2, 3, 5]:
        model = PegasosSVM(
            lambda_=best_lambda, n_epochs=epochs, learning_rate="pegasos",
            random_state=SEED, record_points=8,
        ).fit(X_tr_s, y_tr)
        pred = model.predict(X_val_s)
        score = model.decision_function(X_val_s)
        epoch_rows.append({
            "epochs": epochs,
            "f1": evaluate.precision_recall_f1(y_val, pred)[2],
            "auc": evaluate.roc_auc(y_val, score),
            "margin_width": model.margin_width(),
            "final_objective": model.objective_history_[-1][1],
        })
    epoch_exp = pd.DataFrame(epoch_rows)
    best_epochs = int(epoch_exp.sort_values(["f1", "auc"], ascending=False).iloc[0]["epochs"])

    # 3) Learning-rate schedule sweep.  eta0 is ignored by the Pegasos schedule.
    schedules = [
        ("pegasos", 0.01),
        ("inverse_sqrt", 0.05),
        ("constant", 0.002),
    ]
    schedule_rows = []
    for schedule, eta0 in schedules:
        model = PegasosSVM(
            lambda_=best_lambda, n_epochs=best_epochs, learning_rate=schedule, eta0=eta0,
            random_state=SEED, record_points=8,
        ).fit(X_tr_s, y_tr)
        pred = model.predict(X_val_s)
        score = model.decision_function(X_val_s)
        schedule_rows.append({
            "schedule": schedule, "eta0": eta0,
            "f1": evaluate.precision_recall_f1(y_val, pred)[2],
            "auc": evaluate.roc_auc(y_val, score),
            "margin_width": model.margin_width(),
            "final_objective": model.objective_history_[-1][1],
        })
    schedule_exp = pd.DataFrame(schedule_rows)
    best_row = schedule_exp.sort_values(["f1", "auc"], ascending=False).iloc[0]
    best_schedule = str(best_row["schedule"])
    best_eta0 = float(best_row["eta0"])

    plt.figure(figsize=(7, 4))
    plt.semilogx(lambda_exp["lambda"], lambda_exp["f1"], marker="o")
    plt.xlabel("lambda")
    plt.ylabel("Validation F1")
    plt.title("Linear SVM: regularisation sweep")
    _savefig("svm_lambda_sweep.png")

    plt.figure(figsize=(7, 4))
    plt.semilogx(lambda_exp["lambda"], lambda_exp["margin_width"], marker="o")
    plt.xlabel("lambda")
    plt.ylabel("Margin width (2 / ||w||)")
    plt.title("Regularisation and SVM margin")
    _savefig("svm_margin_sweep.png")

    plt.figure(figsize=(7, 4))
    plt.plot(epoch_exp["epochs"], epoch_exp["f1"], marker="o")
    plt.xlabel("Training epochs")
    plt.ylabel("Validation F1")
    plt.title("SVM: epoch sweep")
    _savefig("svm_epoch_sweep.png")

    plt.figure(figsize=(7, 4))
    plt.plot(epoch_exp["epochs"], epoch_exp["margin_width"], marker="o")
    plt.xlabel("Training epochs")
    plt.ylabel("Margin width (2 / ||w||)")
    plt.title("Epochs and SVM margin")
    _savefig("svm_epoch_margin.png")

    plt.figure(figsize=(7, 4))
    plt.bar(schedule_exp["schedule"], schedule_exp["f1"])
    plt.ylabel("Validation F1")
    plt.title("Learning-rate schedule comparison")
    _savefig("svm_schedule_comparison.png")

    plt.figure(figsize=(7, 4))
    plt.bar(schedule_exp["schedule"], schedule_exp["margin_width"])
    plt.ylabel("Margin width (2 / ||w||)")
    plt.title("Learning-rate schedule and margin")
    _savefig("svm_schedule_margin.png")

    return (
        best_lambda, best_epochs, best_schedule, best_eta0,
        lambda_exp, epoch_exp, schedule_exp,
    )


def _rff_experiment(
    X_tr_s, y_tr, X_val_s, y_val,
    best_lambda, best_epochs, best_schedule, best_eta0,
):
    rows = []
    for gamma in [0.03, 0.1, 0.3]:
        rff = RandomFourierFeatures(n_components=120, gamma=gamma, random_state=SEED)
        Ztr = rff.fit_transform(X_tr_s)
        Zval = rff.transform(X_val_s)
        model = PegasosSVM(
            lambda_=best_lambda, n_epochs=best_epochs, learning_rate=best_schedule,
            eta0=best_eta0, random_state=SEED, record_points=6,
        ).fit(Ztr, y_tr)
        pred = model.predict(Zval)
        score = model.decision_function(Zval)
        rows.append({
            "gamma": gamma,
            "f1": evaluate.precision_recall_f1(y_val, pred)[2],
            "auc": evaluate.roc_auc(y_val, score),
            "margin_width": model.margin_width(),
        })
    exp = pd.DataFrame(rows)
    best_gamma = float(exp.sort_values(["f1", "auc"], ascending=False).iloc[0]["gamma"])

    plt.figure(figsize=(7, 4))
    plt.semilogx(exp["gamma"], exp["f1"], marker="o")
    plt.xlabel("RBF gamma (RFF approximation)")
    plt.ylabel("Validation F1")
    plt.title("Random Fourier Features: bandwidth sweep")
    _savefig("rff_gamma_sweep.png")
    return best_gamma, exp


def _plot_importance(names, values, filename, title, top_n=12):
    values = np.asarray(values, dtype=float)
    order = np.argsort(np.abs(values))[-top_n:]
    plt.figure(figsize=(8, 5))
    plt.barh(np.asarray(names)[order], np.abs(values[order]))
    plt.xlabel("Importance" if "importance" in filename else "|weight|")
    plt.title(title)
    _savefig(filename)


def _plot_confusion(cm, filename, title):
    plt.figure(figsize=(4.5, 4))
    plt.imshow(cm)
    plt.xticks([0, 1], ["Standard", "Premium"])
    plt.yticks([0, 1], ["Standard", "Premium"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(title)
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(int(cm[i, j])), ha="center", va="center")
    _savefig(filename)


def _runtime_scaling_experiment(
    X_tr, y_tr, X_tr_s, cls_criterion, cls_depth, cls_leaf,
    best_lambda, best_epochs, best_schedule, best_eta0,
):
    """Small fixed-seed timing study showing how fit/predict cost grows with n."""
    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(y_tr))
    rows = []
    for n_samples in [2_000, 5_000, 10_000]:
        n = min(n_samples, len(y_tr))
        idx = order[:n]

        tree = DecisionTree(
            task="classification", criterion=cls_criterion, max_depth=cls_depth,
            min_samples_split=max(2, 2 * cls_leaf), min_samples_leaf=cls_leaf,
        )
        t0 = time.perf_counter()
        tree.fit(X_tr[idx], y_tr[idx])
        fit_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        tree.predict(X_tr[idx])
        pred_s = time.perf_counter() - t0
        rows.append({"model": "Our Tree Classifier", "n_samples": n, "fit_seconds": fit_s, "predict_seconds": pred_s})

        svm = PegasosSVM(
            lambda_=best_lambda, n_epochs=best_epochs, learning_rate=best_schedule, eta0=best_eta0,
            random_state=SEED, record_points=4,
        )
        t0 = time.perf_counter()
        svm.fit(X_tr_s[idx], y_tr[idx])
        fit_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        svm.predict(X_tr_s[idx])
        pred_s = time.perf_counter() - t0
        rows.append({"model": "Our Linear Pegasos", "n_samples": n, "fit_seconds": fit_s, "predict_seconds": pred_s})

    exp = pd.DataFrame(rows)
    plt.figure(figsize=(7, 4))
    for model_name, group in exp.groupby("model"):
        plt.plot(group["n_samples"], group["fit_seconds"], marker="o", label=model_name)
    plt.xlabel("Training samples")
    plt.ylabel("Fit time (seconds)")
    plt.title("Empirical training-time scaling")
    plt.legend()
    _savefig("runtime_scaling.png")
    return exp


def _write_experiments(frames):
    path = RESULTS_DIR / "experiment_results.csv"
    rows = []
    for name, frame in frames.items():
        tmp = frame.copy()
        tmp.insert(0, "experiment", name)
        rows.append(tmp)
    pd.concat(rows, ignore_index=True, sort=False).to_csv(path, index=False)



def _three_fold_indices(n, seed=SEED):
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    return np.array_split(idx, 3)


def _cross_validate_uncertainty(
    X_dev, y_price_dev, feature_names, reg_depth, reg_leaf,
    cls_criterion, cls_depth, cls_leaf, best_lambda, best_gamma,
    best_epochs, best_schedule, best_eta0,
):
    """3-fold CV on a fixed-seed development subset; the test set remains untouched.

    The subset keeps the from-scratch tree benchmark practical while still giving
    a reproducible fold-to-fold uncertainty estimate.
    """
    if len(y_price_dev) > CV_MAX_SAMPLES:
        rng = np.random.default_rng(SEED)
        keep = np.sort(rng.choice(len(y_price_dev), size=CV_MAX_SAMPLES, replace=False))
        X_dev = X_dev[keep]
        y_price_dev = y_price_dev[keep]
    folds = _three_fold_indices(len(y_price_dev))
    rows = []

    for fold_id in range(3):
        va = folds[fold_id]
        tr = np.concatenate([folds[j] for j in range(3) if j != fold_id])
        Xtr, Xva = X_dev[tr], X_dev[va]
        ytr_p, yva_p = y_price_dev[tr], y_price_dev[va]

        # Regression models on log-price.
        ours_reg = DecisionTree(
            task="regression", criterion="mse", max_depth=reg_depth,
            min_samples_split=max(2, 2 * reg_leaf), min_samples_leaf=reg_leaf,
        ).fit(Xtr, np.log1p(ytr_p))
        pred = np.maximum(np.expm1(ours_reg.predict(Xva)), 0.0)
        rows.append({"fold": fold_id + 1, "model": "Our Tree Regressor", "metric": "RMSE", "value": evaluate.rmse(yva_p, pred)})

        sk_reg = DecisionTreeRegressor(
            criterion="squared_error", max_depth=reg_depth,
            min_samples_split=max(2, 2 * reg_leaf), min_samples_leaf=reg_leaf,
            random_state=SEED,
        ).fit(Xtr, np.log1p(ytr_p))
        pred = np.maximum(np.expm1(sk_reg.predict(Xva)), 0.0)
        rows.append({"fold": fold_id + 1, "model": "sklearn Tree Regressor", "metric": "RMSE", "value": evaluate.rmse(yva_p, pred)})

        # Classification threshold/scaling are re-learned inside each fold.
        threshold = float(np.median(ytr_p))
        ytr, _ = data_prep.make_tier_label(ytr_p, threshold)
        yva, _ = data_prep.make_tier_label(yva_p, threshold)
        Xtr_s, Xva_s = data_prep.standardize(Xtr, Xva)

        ours_cls = DecisionTree(
            task="classification", criterion=cls_criterion, max_depth=cls_depth,
            min_samples_split=max(2, 2 * cls_leaf), min_samples_leaf=cls_leaf,
        ).fit(Xtr, ytr)
        pred = ours_cls.predict(Xva)
        rows.append({"fold": fold_id + 1, "model": "Our Tree Classifier", "metric": "F1", "value": evaluate.precision_recall_f1(yva, pred)[2]})

        sk_cls = DecisionTreeClassifier(
            criterion=cls_criterion, max_depth=cls_depth,
            min_samples_split=max(2, 2 * cls_leaf), min_samples_leaf=cls_leaf,
            random_state=SEED,
        ).fit(Xtr, ytr)
        pred = sk_cls.predict(Xva)
        rows.append({"fold": fold_id + 1, "model": "sklearn Tree Classifier", "metric": "F1", "value": evaluate.precision_recall_f1(yva, pred)[2]})

        svm = PegasosSVM(
            lambda_=best_lambda, n_epochs=best_epochs, learning_rate=best_schedule, eta0=best_eta0,
            random_state=SEED + fold_id, record_points=4,
        ).fit(Xtr_s, ytr)
        pred = svm.predict(Xva_s)
        rows.append({"fold": fold_id + 1, "model": "Our Linear Pegasos", "metric": "F1", "value": evaluate.precision_recall_f1(yva, pred)[2]})

        rff = RandomFourierFeatures(n_components=120, gamma=best_gamma, random_state=SEED)
        Ztr = rff.fit_transform(Xtr_s)
        Zva = rff.transform(Xva_s)
        rff_svm = PegasosSVM(
            lambda_=best_lambda, n_epochs=best_epochs, learning_rate=best_schedule, eta0=best_eta0,
            random_state=SEED + fold_id, record_points=4,
        ).fit(Ztr, ytr)
        pred = rff_svm.predict(Zva)
        rows.append({"fold": fold_id + 1, "model": "Our RFF Pegasos", "metric": "F1", "value": evaluate.precision_recall_f1(yva, pred)[2]})

        sk_svm = make_pipeline(
            StandardScaler(),
            SGDClassifier(
                loss="hinge", alpha=best_lambda, max_iter=3000, tol=1e-5,
                random_state=SEED, shuffle=True,
            ),
        ).fit(Xtr, ytr)
        pred = sk_svm.predict(Xva)
        rows.append({"fold": fold_id + 1, "model": "sklearn SGD hinge", "metric": "F1", "value": evaluate.precision_recall_f1(yva, pred)[2]})

    raw = pd.DataFrame(rows)
    summary = raw.groupby(["model", "metric"])["value"].agg(["mean", "std"]).reset_index()
    raw.to_csv(RESULTS_DIR / "cv_folds.csv", index=False)
    summary.to_csv(RESULTS_DIR / "cv_summary.csv", index=False)
    return summary

def main() -> None:
    total_start = time.perf_counter()
    np.random.seed(SEED)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    print("\n=== 0. Sanity checks ===")
    _sanity_checks()

    print("\n=== 1. Data preparation ===")
    raw = data_prep.load_raw(str(ROOT / "data" / "bina_az_sale.csv"))
    clean = data_prep.clean(raw)
    X, y_price, feature_names = data_prep.make_features(clean)
    tr_idx, va_idx, te_idx, tier_threshold = data_prep.split_indices(y_price)

    X_tr, X_val, X_te = X[tr_idx], X[va_idx], X[te_idx]
    y_tr_price, y_val_price, y_te_price = y_price[tr_idx], y_price[va_idx], y_price[te_idx]
    y_tr_tier, _ = data_prep.make_tier_label(y_tr_price, tier_threshold)
    y_val_tier, _ = data_prep.make_tier_label(y_val_price, tier_threshold)
    y_te_tier, _ = data_prep.make_tier_label(y_te_price, tier_threshold)
    X_tr_s, X_val_s, X_te_s = data_prep.standardize(X_tr, X_val, X_te)

    print(f"Raw rows: {len(raw):,} | Clean unique listings: {len(clean):,} | Features: {X.shape[1]}")
    print(f"Split: train={len(tr_idx):,}, val={len(va_idx):,}, test={len(te_idx):,}")
    print(f"Training-only premium threshold: {tier_threshold:,.0f} AZN")
    print(f"Class balance train/val/test: {y_tr_tier.mean():.3f} / {y_val_tier.mean():.3f} / {y_te_tier.mean():.3f}")
    _plot_eda(clean, tier_threshold)
    print(f"  [elapsed {time.perf_counter() - total_start:.1f}s]")

    print("\n=== 2. Validation experiments: trees ===")
    reg_depth, reg_leaf, reg_depth_exp, reg_leaf_exp = _tree_regression_experiment(X_tr, y_tr_price, X_val, y_val_price)
    cls_criterion, cls_depth, cls_leaf, crit_exp, cls_depth_exp, cls_leaf_exp = _tree_classification_experiment(
        X_tr, y_tr_tier, X_val, y_val_tier
    )
    print(f"Selected regression tree: depth={reg_depth}, min_samples_leaf={reg_leaf}")
    print(f"Selected classification tree: criterion={cls_criterion}, depth={cls_depth}, min_samples_leaf={cls_leaf}")
    print(f"  [elapsed {time.perf_counter() - total_start:.1f}s]")

    print("\n=== 3. Validation experiments: SVM ===")
    (
        best_lambda, best_epochs, best_schedule, best_eta0,
        svm_exp, svm_epoch_exp, svm_schedule_exp,
    ) = _svm_experiment(X_tr_s, y_tr_tier, X_val_s, y_val_tier)
    best_gamma, rff_exp = _rff_experiment(
        X_tr_s, y_tr_tier, X_val_s, y_val_tier,
        best_lambda, best_epochs, best_schedule, best_eta0,
    )
    print(
        f"Selected linear SVM: lambda={best_lambda:g}, epochs={best_epochs}, "
        f"schedule={best_schedule}, eta0={best_eta0:g}"
    )
    print(f"Selected RFF gamma={best_gamma:g}, components=120")
    print(f"  [elapsed {time.perf_counter() - total_start:.1f}s]")

    experiments = {
        "reg_tree_depth": reg_depth_exp,
        "reg_tree_leaf": reg_leaf_exp,
        "cls_tree_criterion": crit_exp,
        "cls_tree_depth": cls_depth_exp,
        "cls_tree_leaf": cls_leaf_exp,
        "svm_lambda": svm_exp,
        "svm_epochs": svm_epoch_exp,
        "svm_schedule": svm_schedule_exp,
        "rff_gamma": rff_exp,
    }
    runtime_exp = _runtime_scaling_experiment(
        X_tr, y_tr_tier, X_tr_s, cls_criterion, cls_depth, cls_leaf,
        best_lambda, best_epochs, best_schedule, best_eta0,
    )
    experiments["runtime_scaling"] = runtime_exp
    _write_experiments(experiments)
    print(f"  Runtime scaling + experiment files done [elapsed {time.perf_counter() - total_start:.1f}s]")

    print(f"\n=== 4. Three-fold uncertainty (fixed-seed development subset, max {CV_MAX_SAMPLES:,}) ===")
    dev_idx = np.concatenate([tr_idx, va_idx])
    cv_summary = _cross_validate_uncertainty(
        X[dev_idx], y_price[dev_idx], feature_names,
        reg_depth, reg_leaf, cls_criterion, cls_depth, cls_leaf, best_lambda, best_gamma,
        best_epochs, best_schedule, best_eta0,
    )
    for _, row in cv_summary.iterrows():
        print(f"{row['model']}: {row['metric']} = {row['mean']:.4f} ± {row['std']:.4f}")
    print(f"  [elapsed {time.perf_counter() - total_start:.1f}s]")

    print("\n=== 5. Final models (test set is touched only now) ===")
    # Regression: model log(price), then invert for price-scale metrics.
    fit_times = {}
    predict_times = {}

    print("  Fitting our regression tree...")
    t0 = time.perf_counter()
    reg_tree = DecisionTree(
        task="regression", criterion="mse", max_depth=reg_depth,
        min_samples_split=max(2, 2 * reg_leaf), min_samples_leaf=reg_leaf,
    ).fit(X_tr, np.log1p(y_tr_price))
    fit_times["Our Tree Regressor"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    reg_pred = np.maximum(np.expm1(reg_tree.predict(X_te)), 0.0)
    predict_times["Our Tree Regressor"] = time.perf_counter() - t0

    print("  Fitting sklearn regression tree...")
    t0 = time.perf_counter()
    sk_reg = DecisionTreeRegressor(
        criterion="squared_error", max_depth=reg_depth,
        min_samples_split=max(2, 2 * reg_leaf), min_samples_leaf=reg_leaf,
        random_state=SEED,
    ).fit(X_tr, np.log1p(y_tr_price))
    fit_times["sklearn Tree Regressor"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    sk_reg_pred = np.maximum(np.expm1(sk_reg.predict(X_te)), 0.0)
    predict_times["sklearn Tree Regressor"] = time.perf_counter() - t0

    # Classification trees.
    print("  Fitting our classification tree...")
    t0 = time.perf_counter()
    cls_tree = DecisionTree(
        task="classification", criterion=cls_criterion, max_depth=cls_depth,
        min_samples_split=max(2, 2 * cls_leaf), min_samples_leaf=cls_leaf,
    ).fit(X_tr, y_tr_tier)
    fit_times["Our Tree Classifier"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    cls_pred = cls_tree.predict(X_te).astype(int)
    cls_score = cls_tree.predict_proba(X_te)[:, 1]
    predict_times["Our Tree Classifier"] = time.perf_counter() - t0

    print("  Fitting sklearn classification tree...")
    t0 = time.perf_counter()
    sk_cls = DecisionTreeClassifier(
        criterion=cls_criterion, max_depth=cls_depth,
        min_samples_split=max(2, 2 * cls_leaf), min_samples_leaf=cls_leaf,
        random_state=SEED,
    ).fit(X_tr, y_tr_tier)
    fit_times["sklearn Tree Classifier"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    sk_cls_pred = sk_cls.predict(X_te)
    sk_cls_score = sk_cls.predict_proba(X_te)[:, 1]
    predict_times["sklearn Tree Classifier"] = time.perf_counter() - t0

    # Linear from-scratch SVM using the validation-selected training controls.
    print("  Fitting our linear SVM...")
    t0 = time.perf_counter()
    svm = PegasosSVM(
        lambda_=best_lambda, n_epochs=best_epochs, learning_rate=best_schedule, eta0=best_eta0,
        random_state=SEED, record_points=12,
    ).fit(X_tr_s, y_tr_tier)
    fit_times["Our Linear Pegasos"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    svm_pred = svm.predict(X_te_s).astype(int)
    svm_score = svm.decision_function(X_te_s)
    predict_times["Our Linear Pegasos"] = time.perf_counter() - t0

    # Non-linear extension: RFF -> the same from-scratch linear SVM solver.
    print("  Fitting RFF + our SVM...")
    t0 = time.perf_counter()
    rff = RandomFourierFeatures(n_components=120, gamma=best_gamma, random_state=SEED)
    Ztr = rff.fit_transform(X_tr_s)
    rff_svm = PegasosSVM(
        lambda_=best_lambda, n_epochs=best_epochs, learning_rate=best_schedule, eta0=best_eta0,
        random_state=SEED, record_points=6,
    ).fit(Ztr, y_tr_tier)
    fit_times["Our RFF Pegasos"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    Zte = rff.transform(X_te_s)
    rff_pred = rff_svm.predict(Zte).astype(int)
    rff_score = rff_svm.decision_function(Zte)
    predict_times["Our RFF Pegasos"] = time.perf_counter() - t0

    # sklearn linear hinge baseline on the exact standardized features.
    print("  Fitting sklearn hinge baseline...")
    t0 = time.perf_counter()
    sk_svm = make_pipeline(
        StandardScaler(),
        SGDClassifier(
            loss="hinge", alpha=best_lambda, max_iter=3000, tol=1e-5,
            random_state=SEED, shuffle=True,
        ),
    ).fit(X_tr, y_tr_tier)
    fit_times["sklearn SGD hinge"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    sk_svm_pred = sk_svm.predict(X_te)
    sk_svm_score = sk_svm.decision_function(X_te)
    predict_times["sklearn SGD hinge"] = time.perf_counter() - t0

    reg_results = {
        "Our Tree Regressor": evaluate.regression_metrics(y_te_price, reg_pred),
        "sklearn Tree Regressor": evaluate.regression_metrics(y_te_price, sk_reg_pred),
    }
    cls_results = {
        "Our Tree Classifier": evaluate.classification_metrics(y_te_tier, cls_pred, cls_score),
        "sklearn Tree Classifier": evaluate.classification_metrics(y_te_tier, sk_cls_pred, sk_cls_score),
        "Our Linear Pegasos": evaluate.classification_metrics(y_te_tier, svm_pred, svm_score),
        "Our RFF Pegasos": evaluate.classification_metrics(y_te_tier, rff_pred, rff_score),
        "sklearn SGD hinge": evaluate.classification_metrics(y_te_tier, sk_svm_pred, sk_svm_score),
    }

    print("\nRegression — TEST")
    evaluate.compare(reg_results)
    print("\nClassification — TEST")
    evaluate.compare(cls_results)

    # Diagnostics / figures.
    print("  Saving tree feature-importance figure...")
    _plot_importance(feature_names, cls_tree.feature_importances_, "tree_feature_importance.png", "Our tree: impurity-decrease feature importance")
    print("  Saving SVM weight figure...")
    _plot_importance(feature_names, svm.w, "svm_weights.png", "Linear Pegasos: largest standardized weights")

    print("  Saving SVM convergence figure...")
    steps, objectives = zip(*svm.objective_history_)
    plt.figure(figsize=(7, 4))
    plt.plot(steps, objectives)
    plt.xlabel("Pegasos update step")
    plt.ylabel("Regularised hinge objective")
    plt.title("Linear Pegasos convergence")
    _savefig("svm_convergence.png")

    print("  Saving confusion matrices...")
    _plot_confusion(evaluate.confusion_matrix(y_te_tier, cls_pred), "confusion_tree.png", "Our decision tree")
    _plot_confusion(evaluate.confusion_matrix(y_te_tier, svm_pred), "confusion_svm.png", "Our linear Pegasos SVM")

    print("  Computing SVM margin/support-vector diagnostics...")
    support_count = int(svm.support_vector_mask(X_tr_s, y_tr_tier).sum())
    w_norm = float(np.linalg.norm(svm.w))
    margin = svm.margin_width()

    print("  Writing final result tables...")
    # Save machine-readable headline tables for the report.
    reg_df = pd.DataFrame(reg_results).T.reset_index(names="model")
    reg_df.insert(0, "task", "regression")
    cls_df = pd.DataFrame(cls_results).T.reset_index(names="model")
    cls_df.insert(0, "task", "classification")
    result_table = pd.concat([reg_df, cls_df], ignore_index=True, sort=False)
    result_table["fit_seconds"] = result_table["model"].map(fit_times)
    result_table["predict_seconds"] = result_table["model"].map(predict_times)
    result_table["predict_ms_per_1000"] = result_table["predict_seconds"] / len(y_te_price) * 1_000_000.0
    result_table.to_csv(RESULTS_DIR / "results.csv", index=False)

    with open(RESULTS_DIR / "results_summary.txt", "w", encoding="utf-8") as f:
        f.write(f"Raw rows: {len(raw)}\n")
        f.write(f"Clean rows: {len(clean)}\n")
        f.write(f"Features: {len(feature_names)}\n")
        f.write(f"Split sizes: train={len(tr_idx)}, val={len(va_idx)}, test={len(te_idx)}\n")
        f.write(f"Training-only tier threshold: {tier_threshold:.2f} AZN\n")
        f.write(f"Regression tree: depth={reg_depth}, min_samples_leaf={reg_leaf}\n")
        f.write(f"Classification tree: criterion={cls_criterion}, depth={cls_depth}, min_samples_leaf={cls_leaf}\n")
        f.write(
            f"Linear SVM: lambda={best_lambda:g}, epochs={best_epochs}, schedule={best_schedule}, eta0={best_eta0:g}, "
            f"||w||={w_norm:.6f}, margin={margin:.6f}, support_vectors={support_count}/{len(y_tr_tier)}\n"
        )
        f.write(f"RFF: gamma={best_gamma:g}, components=120\n\n")
        f.write(result_table.to_string(index=False))
        f.write(f"\n\n3-fold CV on fixed-seed development subset (n<= {CV_MAX_SAMPLES}, mean ± std):\n")
        for _, row in cv_summary.iterrows():
            f.write(f"{row['model']} | {row['metric']}: {row['mean']:.6f} ± {row['std']:.6f}\n")


    print("\nDiagnostics")
    print(f"Linear SVM ||w||={w_norm:.4f}, margin width={margin:.4f}, support vectors={support_count:,}/{len(y_tr_tier):,}")
    print(f"Figures saved to: {FIG_DIR}")
    print(f"Results saved to: {RESULTS_DIR / 'results.csv'}")
    print(f"Total runtime: {time.perf_counter() - total_start:.1f} s")


if __name__ == "__main__":
    main()
