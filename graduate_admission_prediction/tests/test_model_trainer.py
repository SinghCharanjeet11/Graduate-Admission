import math
import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from graduate_admission_prediction.pipeline.model_trainer import ModelTrainer


def make_synthetic_data(n=100, seed=42):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"f1": rng.normal(size=n), "f2": rng.normal(size=n)})
    y = pd.Series(X["f1"] * 2.0 + X["f2"] * 0.5 + rng.normal(scale=0.1, size=n))
    return X, y


class TestModelTrainer:
    def test_linear_regression_trains_without_error(self):
        X, y = make_synthetic_data()
        trainer = ModelTrainer()
        model = trainer.train_linear_regression(X, y)
        assert hasattr(model, "coef_")
        assert model.coef_.shape == (2,)

    def test_random_forest_trains_without_error(self):
        X, y = make_synthetic_data()
        trainer = ModelTrainer()
        # Override param grid to keep the test fast
        trainer.RF_PARAM_GRID = {"n_estimators": [10], "max_depth": [3]}
        model = trainer.train_random_forest(X, y)
        assert hasattr(model, "estimators_")  # fitted attribute: list of decision trees

    def test_save_load_round_trip_identical_predictions(self):
        X, y = make_synthetic_data()
        trainer = ModelTrainer()
        model = trainer.train_linear_regression(X, y)
        preds_before = model.predict(X)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = trainer.save_model(model, "lr_model", tmpdir)
            loaded = trainer.load_model(path)

        preds_after = loaded.predict(X)
        np.testing.assert_array_equal(preds_before, preds_after)

    def test_load_model_raises_file_not_found(self):
        trainer = ModelTrainer()
        with pytest.raises(FileNotFoundError):
            trainer.load_model("/nonexistent/path/model.joblib")
