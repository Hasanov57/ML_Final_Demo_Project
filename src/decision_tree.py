"""decision_tree.py — a NumPy decision tree for classification and regression."""

from __future__ import annotations

import numpy as np


class Node:
    """A single decision-tree node."""

    def __init__(self):
        self.feature: int | None = None
        self.threshold: float | None = None
        self.left: Node | None = None
        self.right: Node | None = None
        self.value = None
        self.proba: np.ndarray | None = None
        self.n_samples: int = 0
        self.impurity: float = 0.0
        self.gain: float = 0.0

    def is_leaf(self) -> bool:
        return self.left is None and self.right is None


class DecisionTree:
    """
    CART-style binary decision tree implemented from scratch.

    Classification: criterion='gini' or 'entropy'.
    Regression: criterion='mse'.
    """

    def __init__(
        self,
        task: str = "classification",
        criterion: str = "gini",
        max_depth: int | None = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        min_impurity_decrease: float = 0.0,
        random_state: int = 42,
    ):
        self.task = task
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease
        self.random_state = random_state
        self.root: Node | None = None
        self.classes_: np.ndarray | None = None
        self.n_features_in_: int | None = None
        self.feature_importances_: np.ndarray | None = None
        self._n_fit_samples = 0
        self._validate_params()

    def _validate_params(self):
        if self.task not in {"classification", "regression"}:
            raise ValueError("task must be 'classification' or 'regression'")
        allowed = {"classification": {"gini", "entropy"}, "regression": {"mse"}}
        if self.criterion not in allowed[self.task]:
            raise ValueError(f"criterion {self.criterion!r} is invalid for task={self.task!r}")
        if self.min_samples_split < 2:
            raise ValueError("min_samples_split must be >= 2")
        if self.min_samples_leaf < 1:
            raise ValueError("min_samples_leaf must be >= 1")

    # ------------------------------------------------------------------ impurity
    def _impurity(self, y) -> float:
        y = np.asarray(y)
        if y.size == 0:
            return 0.0

        if self.task == "regression":
            yy = y.astype(float)
            return float(np.mean((yy - yy.mean()) ** 2))

        _, counts = np.unique(y, return_counts=True)
        p = counts / counts.sum()
        if self.criterion == "gini":
            return float(1.0 - np.sum(p * p))
        p = p[p > 0]
        return float(-np.sum(p * np.log2(p)))

    def _classification_split_impurity(self, y_sorted, valid_positions):
        """Vectorised weighted Gini/entropy for all valid split positions."""
        n = len(y_sorted)
        class_ids = np.searchsorted(self.classes_, y_sorted)
        k = len(self.classes_)
        one_hot = np.eye(k, dtype=float)[class_ids]
        cumsum = np.cumsum(one_hot, axis=0)

        left_counts = cumsum[valid_positions]
        total_counts = cumsum[-1]
        right_counts = total_counts - left_counts
        n_left = valid_positions + 1
        n_right = n - n_left

        p_left = left_counts / n_left[:, None]
        p_right = right_counts / n_right[:, None]

        if self.criterion == "gini":
            imp_left = 1.0 - np.sum(p_left**2, axis=1)
            imp_right = 1.0 - np.sum(p_right**2, axis=1)
        else:
            with np.errstate(divide="ignore", invalid="ignore"):
                log_left = np.where(p_left > 0, np.log2(p_left), 0.0)
                log_right = np.where(p_right > 0, np.log2(p_right), 0.0)
            imp_left = -np.sum(p_left * log_left, axis=1)
            imp_right = -np.sum(p_right * log_right, axis=1)

        return (n_left / n) * imp_left + (n_right / n) * imp_right

    @staticmethod
    def _regression_split_impurity(y_sorted, valid_positions):
        """Vectorised weighted variance for all valid split positions."""
        y = y_sorted.astype(float)
        n = len(y)
        csum = np.cumsum(y)
        csq = np.cumsum(y * y)
        total_sum = csum[-1]
        total_sq = csq[-1]

        n_left = valid_positions + 1
        n_right = n - n_left
        sum_left = csum[valid_positions]
        sq_left = csq[valid_positions]
        sum_right = total_sum - sum_left
        sq_right = total_sq - sq_left

        var_left = sq_left / n_left - (sum_left / n_left) ** 2
        var_right = sq_right / n_right - (sum_right / n_right) ** 2
        var_left = np.maximum(var_left, 0.0)
        var_right = np.maximum(var_right, 0.0)
        return (n_left / n) * var_left + (n_right / n) * var_right

    def _best_split(self, X, y):
        """Return (feature, threshold, impurity_decrease) for the best valid split."""
        n_samples, n_features = X.shape
        parent_impurity = self._impurity(y)
        best_feature = None
        best_threshold = None
        best_gain = -np.inf

        min_leaf = self.min_samples_leaf
        if n_samples < 2 * min_leaf:
            return None

        for feature in range(n_features):
            x = X[:, feature]
            order = np.argsort(x, kind="mergesort")
            x_sorted = x[order]
            y_sorted = y[order]

            # A split can occur only between two distinct feature values.
            positions = np.flatnonzero(x_sorted[:-1] < x_sorted[1:])
            if positions.size == 0:
                continue
            left_n = positions + 1
            right_n = n_samples - left_n
            valid = (left_n >= min_leaf) & (right_n >= min_leaf)
            positions = positions[valid]
            if positions.size == 0:
                continue

            if self.task == "classification":
                weighted_child = self._classification_split_impurity(y_sorted, positions)
            else:
                weighted_child = self._regression_split_impurity(y_sorted, positions)

            gains = parent_impurity - weighted_child
            j = int(np.argmax(gains))
            gain = float(gains[j])
            pos = int(positions[j])
            threshold = float(x_sorted[pos] + (x_sorted[pos + 1] - x_sorted[pos]) / 2.0)

            # Deterministic tie-breaking: first feature / first threshold wins.
            if gain > best_gain + 1e-15:
                best_gain = gain
                best_feature = feature
                best_threshold = threshold

        if best_feature is None or best_gain <= 0.0:
            return None
        return best_feature, best_threshold, best_gain

    # ---------------------------------------------------------------- grow/predict
    def _make_leaf(self, y, node: Node) -> Node:
        if self.task == "regression":
            node.value = float(np.mean(y))
            return node

        counts = np.array([(y == c).sum() for c in self.classes_], dtype=float)
        node.proba = counts / counts.sum()
        node.value = self.classes_[int(np.argmax(counts))]
        return node

    def _grow(self, X, y, depth: int) -> Node:
        node = Node()
        node.n_samples = len(y)
        node.impurity = self._impurity(y)

        stop = (
            len(y) < self.min_samples_split
            or len(y) < 2 * self.min_samples_leaf
            or node.impurity <= 1e-15
            or (self.max_depth is not None and depth >= self.max_depth)
        )
        if stop:
            return self._make_leaf(y, node)

        split = self._best_split(X, y)
        if split is None:
            return self._make_leaf(y, node)

        feature, threshold, gain = split
        if gain < self.min_impurity_decrease:
            return self._make_leaf(y, node)

        left_mask = X[:, feature] <= threshold
        right_mask = ~left_mask
        if left_mask.sum() < self.min_samples_leaf or right_mask.sum() < self.min_samples_leaf:
            return self._make_leaf(y, node)

        node.feature = feature
        node.threshold = threshold
        node.gain = gain
        self.feature_importances_[feature] += (len(y) / self._n_fit_samples) * gain
        node.left = self._grow(X[left_mask], y[left_mask], depth + 1)
        node.right = self._grow(X[right_mask], y[right_mask], depth + 1)
        return node

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        if X.ndim != 2 or y.ndim != 1 or len(X) != len(y):
            raise ValueError("X must be 2-D, y must be 1-D, and lengths must match")
        if len(y) == 0:
            raise ValueError("Cannot fit an empty dataset")

        self.n_features_in_ = X.shape[1]
        self._n_fit_samples = len(y)
        self.feature_importances_ = np.zeros(self.n_features_in_, dtype=float)
        if self.task == "classification":
            self.classes_ = np.unique(y)
        self.root = self._grow(X, y, depth=0)

        total = self.feature_importances_.sum()
        if total > 0:
            self.feature_importances_ /= total
        return self

    def _leaf_for_one(self, x, node: Node) -> Node:
        while not node.is_leaf():
            node = node.left if x[node.feature] <= node.threshold else node.right
        return node

    def _predict_one(self, x, node: Node):
        return self._leaf_for_one(x, node).value

    def predict(self, X):
        if self.root is None:
            raise RuntimeError("Call fit before predict")
        X = np.asarray(X, dtype=float)
        return np.asarray([self._predict_one(row, self.root) for row in X])

    def predict_proba(self, X):
        if self.task != "classification":
            raise ValueError("predict_proba is available only for classification")
        if self.root is None:
            raise RuntimeError("Call fit before predict_proba")
        X = np.asarray(X, dtype=float)
        return np.vstack([self._leaf_for_one(row, self.root).proba for row in X])

    def get_depth(self) -> int:
        """Return maximum number of edges from root to a leaf."""
        if self.root is None:
            return 0

        def walk(node):
            if node.is_leaf():
                return 0
            return 1 + max(walk(node.left), walk(node.right))

        return walk(self.root)

    def get_n_leaves(self) -> int:
        if self.root is None:
            return 0

        def walk(node):
            if node.is_leaf():
                return 1
            return walk(node.left) + walk(node.right)

        return walk(self.root)
