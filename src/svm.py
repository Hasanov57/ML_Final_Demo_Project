"""svm.py — from-scratch soft-margin SVM trained with Pegasos-style SGD.

The main model is implemented only with NumPy.  scikit-learn is intentionally not
used here; it appears only in ``run_all.py`` as a comparison baseline.
"""

from __future__ import annotations

import numpy as np


class PegasosSVM:
    """Binary linear SVM using stochastic sub-gradient descent.

    Parameters
    ----------
    lambda_:
        L2 regularisation strength.
    n_epochs:
        Number of complete passes over the training set.  This is the preferred
        training-length control used by the final project.
    learning_rate:
        ``"pegasos"`` uses eta_t = 1 / (lambda * t), ``"inverse_sqrt"`` uses
        eta0 / sqrt(t), and ``"constant"`` uses eta0.
    eta0:
        Base learning rate for ``inverse_sqrt`` and ``constant`` schedules.
    n_iters:
        Optional backwards-compatible update count.  If provided, the model
        performs exactly this many random stochastic updates instead of epochs.
    """

    VALID_SCHEDULES = {"pegasos", "inverse_sqrt", "constant"}

    def __init__(
        self,
        lambda_: float = 1e-3,
        n_epochs: int = 3,
        learning_rate: str = "pegasos",
        eta0: float = 0.01,
        random_state: int = 42,
        record_points: int = 30,
        n_iters: int | None = None,
    ):
        if lambda_ <= 0:
            raise ValueError("lambda_ must be > 0")
        if n_epochs < 1:
            raise ValueError("n_epochs must be >= 1")
        if learning_rate not in self.VALID_SCHEDULES:
            raise ValueError(f"learning_rate must be one of {sorted(self.VALID_SCHEDULES)}")
        if eta0 <= 0:
            raise ValueError("eta0 must be > 0")
        if n_iters is not None and n_iters < 1:
            raise ValueError("n_iters must be >= 1 when provided")

        self.lambda_ = float(lambda_)
        self.n_epochs = int(n_epochs)
        self.learning_rate = learning_rate
        self.eta0 = float(eta0)
        self.random_state = int(random_state)
        self.record_points = int(record_points)
        self.n_iters = None if n_iters is None else int(n_iters)

        self.w: np.ndarray | None = None
        self.b: float = 0.0
        self.classes_: np.ndarray | None = None
        self.objective_history_: list[tuple[int, float]] = []
        self.n_updates_: int = 0

    def _encode_pm1(self, y):
        y = np.asarray(y)
        classes = np.unique(y)
        if len(classes) != 2:
            raise ValueError("PegasosSVM requires exactly two classes")
        self.classes_ = classes
        return np.where(y == classes[0], -1.0, 1.0)

    @staticmethod
    def _to_pm1(y):
        """Convenience mapping for the project's usual binary labels."""
        y = np.asarray(y)
        values = np.unique(y)
        if len(values) != 2:
            raise ValueError("Expected exactly two classes")
        return np.where(y == values[0], -1.0, 1.0)

    def _eta(self, t: int) -> float:
        """Return the learning rate at stochastic update ``t`` (1-indexed)."""
        if self.learning_rate == "pegasos":
            return 1.0 / (self.lambda_ * t)
        if self.learning_rate == "inverse_sqrt":
            return self.eta0 / np.sqrt(t)
        return self.eta0  # constant

    def _objective(self, X, y_pm1):
        """Hinge loss with L2 penalty on the augmented vector (w, b).

        The Pegasos update shrinks and projects that whole vector, including
        the appended bias coordinate, so the logged objective uses the same
        regularisation as the actual training step.
        """
        scores = X @ self.w + self.b
        hinge = np.maximum(0.0, 1.0 - y_pm1 * scores)
        squared_norm = np.dot(self.w, self.w) + self.b * self.b
        return float(0.5 * self.lambda_ * squared_norm + hinge.mean())

    def _single_update(self, theta, xi, yi, t, radius):
        """One projected stochastic sub-gradient update."""
        eta = self._eta(t)
        margin = yi * float(theta @ xi)

        # L2 shrinkage.  Only when the point violates the margin do we also add
        # the hinge-loss sub-gradient term.
        theta *= 1.0 - eta * self.lambda_
        if margin < 1.0:
            theta += eta * yi * xi

        # Standard Pegasos projection onto ||theta|| <= 1/sqrt(lambda).
        norm = np.linalg.norm(theta)
        if norm > radius:
            theta *= radius / norm
        return theta

    def fit(self, X, y):
        """Fit the binary SVM and return ``self``.

        The bias is folded into the parameter vector by appending a constant 1
        feature, matching the approach allowed in the project statement.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        if X.ndim != 2 or y.ndim != 1 or len(X) != len(y):
            raise ValueError("X must be 2-D, y 1-D, and lengths must match")

        y_pm1 = self._encode_pm1(y)
        n, d = X.shape
        rng = np.random.default_rng(self.random_state)
        theta = np.zeros(d + 1, dtype=float)
        radius = 1.0 / np.sqrt(self.lambda_)
        self.objective_history_ = []

        # Legacy mode: exactly n_iters random stochastic updates.
        if self.n_iters is not None:
            total_updates = self.n_iters
            checkpoints = set(
                np.linspace(1, total_updates, max(2, self.record_points), dtype=int).tolist()
            )
            for t in range(1, total_updates + 1):
                i = int(rng.integers(0, n))
                xi = np.empty(d + 1, dtype=float)
                xi[:-1] = X[i]
                xi[-1] = 1.0
                theta = self._single_update(theta, xi, y_pm1[i], t, radius)

                if t in checkpoints:
                    self.w = theta[:-1].copy()
                    self.b = float(theta[-1])
                    self.objective_history_.append((t, self._objective(X, y_pm1)))
        else:
            # Preferred mode: complete shuffled passes over the training data.
            total_updates = self.n_epochs * n
            checkpoints = set(
                np.linspace(1, total_updates, max(2, self.record_points), dtype=int).tolist()
            )
            t = 0
            for _epoch in range(self.n_epochs):
                for i in rng.permutation(n):
                    t += 1
                    xi = np.empty(d + 1, dtype=float)
                    xi[:-1] = X[i]
                    xi[-1] = 1.0
                    theta = self._single_update(theta, xi, y_pm1[i], t, radius)

                    if t in checkpoints:
                        self.w = theta[:-1].copy()
                        self.b = float(theta[-1])
                        self.objective_history_.append((t, self._objective(X, y_pm1)))

        self.n_updates_ = total_updates
        self.w = theta[:-1].copy()
        self.b = float(theta[-1])

        # Always record the final objective.
        if not self.objective_history_ or self.objective_history_[-1][0] != total_updates:
            self.objective_history_.append((total_updates, self._objective(X, y_pm1)))
        return self

    def decision_function(self, X):
        if self.w is None:
            raise RuntimeError("Call fit before decision_function")
        X = np.asarray(X, dtype=float)
        return X @ self.w + self.b

    def predict(self, X):
        if self.classes_ is None:
            raise RuntimeError("Call fit before predict")
        scores = self.decision_function(X)
        return np.where(scores >= 0.0, self.classes_[1], self.classes_[0])

    def margin_width(self) -> float:
        if self.w is None:
            raise RuntimeError("Call fit first")
        norm = np.linalg.norm(self.w)
        return float(np.inf if norm == 0 else 2.0 / norm)

    def support_vector_mask(self, X, y, tol: float = 1e-9):
        """Return mask for points on/inside the margin: y_i f(x_i) <= 1."""
        if self.classes_ is None:
            raise RuntimeError("Call fit first")
        y = np.asarray(y)
        y_pm1 = np.where(y == self.classes_[0], -1.0, 1.0)
        margins = y_pm1 * self.decision_function(X)
        return margins <= (1.0 + tol)


class RandomFourierFeatures:
    """Approximate an RBF kernel with random Fourier features."""

    def __init__(self, n_components: int = 200, gamma: float = 0.1, random_state: int = 42):
        if n_components < 1 or gamma <= 0:
            raise ValueError("n_components and gamma must be positive")
        self.n_components = int(n_components)
        self.gamma = float(gamma)
        self.random_state = random_state
        self.W_: np.ndarray | None = None
        self.phase_: np.ndarray | None = None

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        rng = np.random.default_rng(self.random_state)
        # RBF exp(-gamma ||x-y||^2) corresponds to frequencies N(0, 2*gamma I).
        self.W_ = rng.normal(0.0, np.sqrt(2.0 * self.gamma), size=(X.shape[1], self.n_components))
        self.phase_ = rng.uniform(0.0, 2.0 * np.pi, size=self.n_components)
        return self

    def transform(self, X):
        if self.W_ is None or self.phase_ is None:
            raise RuntimeError("Call fit before transform")
        X = np.asarray(X, dtype=float)
        projection = X @ self.W_ + self.phase_
        return np.sqrt(2.0 / self.n_components) * np.cos(projection)

    def fit_transform(self, X):
        return self.fit(X).transform(X)
