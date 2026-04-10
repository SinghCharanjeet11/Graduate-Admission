import pandas as pd
import pytest

from graduate_admission_prediction.pipeline.splitter import Splitter


def make_df(n: int = 100) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "GRE Score": range(n),
            "CGPA": [i / 10.0 for i in range(n)],
            "Chance of Admit": [i / n for i in range(n)],
        }
    )


FEATURES = ["GRE Score", "CGPA"]
TARGET = "Chance of Admit"


class TestSplitter:
    def test_80_20_ratio(self):
        df = make_df(100)
        splitter = Splitter()
        X_train, X_test, y_train, y_test = splitter.split(df, FEATURES, TARGET)
        assert len(X_train) == 80
        assert len(X_test) == 20
        assert len(y_train) == 80
        assert len(y_test) == 20

    def test_no_duplication(self):
        df = make_df(100)
        splitter = Splitter()
        X_train, X_test, y_train, y_test = splitter.split(df, FEATURES, TARGET)
        train_idx = set(X_train.index)
        test_idx = set(X_test.index)
        assert train_idx.isdisjoint(test_idx), "Train and test sets share indices"

    def test_no_omission(self):
        df = make_df(100)
        splitter = Splitter()
        X_train, X_test, y_train, y_test = splitter.split(df, FEATURES, TARGET)
        all_idx = set(X_train.index) | set(X_test.index)
        assert all_idx == set(df.index), "Some rows are missing from train+test"

    def test_feature_columns_preserved(self):
        df = make_df(50)
        splitter = Splitter()
        X_train, X_test, _, _ = splitter.split(df, FEATURES, TARGET)
        assert list(X_train.columns) == FEATURES
        assert list(X_test.columns) == FEATURES

    def test_reproducible_with_seed(self):
        df = make_df(100)
        splitter = Splitter()
        X_train1, X_test1, _, _ = splitter.split(df, FEATURES, TARGET)
        X_train2, X_test2, _, _ = splitter.split(df, FEATURES, TARGET)
        pd.testing.assert_frame_equal(X_train1, X_train2)
        pd.testing.assert_frame_equal(X_test1, X_test2)
