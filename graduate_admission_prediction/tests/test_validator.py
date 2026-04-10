import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from graduate_admission_prediction.pipeline.validator import Validator


def make_sample_data(n=100, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"f1": rng.normal(size=n), "f2": rng.normal(size=n)})
    y = pd.Series(X["f1"] * 2.0 + X["f2"] * 0.5 + rng.normal(scale=0.1, size=n))
    return X, y


class TestValidator:
    def test_returns_correct_keys(self):
        X, y = make_sample_data()
        result = Validator().cross_validate(LinearRegression(), X, y)
        assert set(result.keys()) == {"mean_r2", "std_r2", "fold_scores"}

    def test_fold_scores_length(self):
        X, y = make_sample_data()
        result = Validator().cross_validate(LinearRegression(), X, y)
        assert len(result["fold_scores"]) == Validator.N_SPLITS

    def test_mean_r2_matches_fold_scores(self):
        X, y = make_sample_data()
        result = Validator().cross_validate(LinearRegression(), X, y)
        assert abs(result["mean_r2"] - np.mean(result["fold_scores"])) < 1e-10

    def test_std_r2_matches_fold_scores(self):
        X, y = make_sample_data()
        result = Validator().cross_validate(LinearRegression(), X, y)
        assert abs(result["std_r2"] - np.std(result["fold_scores"])) < 1e-10

    def test_reproducibility(self):
        X, y = make_sample_data()
        model = LinearRegression()
        r1 = Validator().cross_validate(model, X, y)
        r2 = Validator().cross_validate(model, X, y)
        assert r1["fold_scores"] == r2["fold_scores"]

    def test_r2_scores_are_floats(self):
        X, y = make_sample_data()
        result = Validator().cross_validate(LinearRegression(), X, y)
        assert isinstance(result["mean_r2"], float)
        assert isinstance(result["std_r2"], float)
        assert all(isinstance(s, float) for s in result["fold_scores"])
