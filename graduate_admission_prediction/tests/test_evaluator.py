import math

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from graduate_admission_prediction.pipeline.evaluator import Evaluator


def make_model_with_known_predictions(y_true, y_pred):
    """Return a minimal mock model that always predicts y_pred."""

    class FixedModel:
        def __init__(self, predictions):
            self._preds = np.array(predictions)

        def predict(self, X):
            return self._preds

    return FixedModel(y_pred)


class TestEvaluator:
    def test_rmse_equals_sqrt_mse(self):
        y_true = [1.0, 2.0, 3.0, 4.0]
        y_pred = [1.1, 1.9, 3.2, 3.8]
        X = pd.DataFrame({"f": [0] * 4})
        y = pd.Series(y_true)
        model = make_model_with_known_predictions(y_true, y_pred)

        result = Evaluator().evaluate(model, X, y, "test_model")

        expected_rmse = math.sqrt(result["mse"])
        assert abs(result["rmse"] - expected_rmse) < 1e-10

    def test_meets_r2_threshold_true_at_085(self):
        # Build a model that achieves exactly r2=0.85 by controlling residuals
        # Use evaluate with a real sklearn model on synthetic data instead
        evaluator = Evaluator()
        # Directly test threshold logic via a crafted result dict
        result = {"r2": 0.85, "rmse": 0.05, "model_name": "m", "mse": 0.0025,
                  "meets_r2_threshold": 0.85 >= evaluator.R2_THRESHOLD,
                  "meets_rmse_threshold": 0.05 <= evaluator.RMSE_THRESHOLD}
        assert result["meets_r2_threshold"] is True

    def test_meets_r2_threshold_false_at_084(self):
        evaluator = Evaluator()
        meets = 0.84 >= evaluator.R2_THRESHOLD
        assert meets is False

    def test_meets_rmse_threshold_true_at_01(self):
        evaluator = Evaluator()
        meets = 0.1 <= evaluator.RMSE_THRESHOLD
        assert meets is True

    def test_meets_rmse_threshold_false_at_011(self):
        evaluator = Evaluator()
        meets = 0.11 <= evaluator.RMSE_THRESHOLD
        assert meets is False

    def test_evaluate_threshold_flags_via_model(self):
        """Integration: evaluate() sets threshold flags correctly."""
        rng = np.random.default_rng(0)
        n = 200
        X = pd.DataFrame({"f1": rng.normal(size=n), "f2": rng.normal(size=n)})
        y = pd.Series(X["f1"] * 2.0 + X["f2"] * 0.5 + rng.normal(scale=0.05, size=n))

        model = LinearRegression().fit(X, y)
        result = Evaluator().evaluate(model, X, y, "lr")

        # With very low noise the model should exceed both thresholds
        assert result["meets_r2_threshold"] == (result["r2"] >= Evaluator.R2_THRESHOLD)
        assert result["meets_rmse_threshold"] == (result["rmse"] <= Evaluator.RMSE_THRESHOLD)

    def test_best_model_returns_highest_r2(self):
        results = [
            {"model_name": "a", "r2": 0.80, "rmse": 0.12},
            {"model_name": "b", "r2": 0.92, "rmse": 0.08},
            {"model_name": "c", "r2": 0.87, "rmse": 0.09},
        ]
        best = Evaluator().best_model(results)
        assert best["model_name"] == "b"
        assert best["r2"] == 0.92
