import unittest
import numpy as np

from src.decision_tree import DecisionTree
from src.svm import PegasosSVM
from src import evaluate


class TestCore(unittest.TestCase):
    def test_classification_tree_toy(self):
        X = np.array([[0.0], [1.0], [2.0], [3.0]])
        y = np.array([0, 0, 1, 1])
        model = DecisionTree(task="classification", criterion="gini", max_depth=1).fit(X, y)
        self.assertAlmostEqual(model.root.threshold, 1.5)
        self.assertTrue(np.array_equal(model.predict(X), y))

    def test_regression_tree_toy(self):
        X = np.array([[0.0], [1.0], [2.0], [3.0]])
        y = np.array([1.0, 1.0, 8.0, 8.0])
        model = DecisionTree(task="regression", criterion="mse", max_depth=1).fit(X, y)
        self.assertAlmostEqual(model.root.threshold, 1.5)
        self.assertTrue(np.allclose(model.predict(X), y))

    def test_tree_stopping_parameters_change_structure(self):
        X = np.arange(20, dtype=float).reshape(-1, 1)
        y = np.array([0] * 5 + [1] * 5 + [0] * 5 + [1] * 5)
        shallow = DecisionTree(task="classification", max_depth=1).fit(X, y)
        deeper = DecisionTree(task="classification", max_depth=4).fit(X, y)
        blocked = DecisionTree(task="classification", max_depth=4, min_impurity_decrease=1.0).fit(X, y)
        self.assertLess(shallow.get_depth(), deeper.get_depth())
        self.assertEqual(blocked.get_depth(), 0)

    def test_svm_separable_toy(self):
        X = np.array([[-2.0, -1.0], [-1.0, -2.0], [2.0, 1.0], [1.0, 2.0]])
        y = np.array([0, 0, 1, 1])
        model = PegasosSVM(
            lambda_=1e-2, n_epochs=20, learning_rate="pegasos",
            random_state=42, record_points=5,
        ).fit(X, y)
        self.assertEqual(evaluate.accuracy(y, model.predict(X)), 1.0)
        self.assertLess(model.objective_history_[-1][1], model.objective_history_[0][1])

    def test_svm_learning_rate_schedules(self):
        X = np.array([[-2.0, -1.0], [-1.0, -2.0], [2.0, 1.0], [1.0, 2.0]])
        y = np.array([0, 0, 1, 1])
        for schedule, eta0 in [("pegasos", 0.01), ("inverse_sqrt", 0.05), ("constant", 0.002)]:
            model = PegasosSVM(
                lambda_=1e-2, n_epochs=5, learning_rate=schedule, eta0=eta0,
                random_state=42, record_points=3,
            ).fit(X, y)
            self.assertTrue(np.isfinite(model.margin_width()))
            self.assertEqual(len(model.predict(X)), len(y))

    def test_metrics(self):
        y = np.array([0, 0, 1, 1])
        p = np.array([0, 1, 1, 1])
        precision, recall, f1 = evaluate.precision_recall_f1(y, p)
        self.assertAlmostEqual(precision, 2/3)
        self.assertAlmostEqual(recall, 1.0)
        self.assertAlmostEqual(f1, 0.8)


if __name__ == "__main__":
    unittest.main()
