import pandas as pd
import pytest

from graduate_admission_prediction.pipeline.feature_selector import FeatureSelector


def make_df() -> pd.DataFrame:
    """DataFrame where GRE Score is highly correlated with target,
    Noise is uncorrelated."""
    return pd.DataFrame(
        {
            "GRE Score": [300, 310, 320, 330, 340, 350, 360, 370, 380, 390],
            "Noise": [1, 9, 2, 8, 3, 7, 4, 6, 5, 5],
            "Chance of Admit": [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95],
        }
    )


class TestFeatureSelector:
    def test_correct_features_selected(self):
        df = make_df()
        selector = FeatureSelector()
        selected, corr = selector.select(df, target="Chance of Admit", threshold=0.5)
        # GRE Score should be selected (high correlation), Noise should not
        assert "GRE Score" in selected
        assert "Noise" not in selected

    def test_returns_correlation_series_for_all_features(self):
        df = make_df()
        selector = FeatureSelector()
        selected, corr = selector.select(df, target="Chance of Admit", threshold=0.5)
        # corr should contain all feature columns (not target)
        assert "GRE Score" in corr.index
        assert "Noise" in corr.index
        assert "Chance of Admit" not in corr.index

    def test_threshold_boundary_strict(self):
        """Features with |r| exactly equal to threshold should NOT be selected."""
        df = make_df()
        selector = FeatureSelector()
        _, corr = selector.select(df, target="Chance of Admit", threshold=0.0)
        # Use the actual correlation of Noise as threshold — it should be excluded
        noise_corr = abs(corr["Noise"])
        selected, _ = selector.select(df, target="Chance of Admit", threshold=noise_corr)
        assert "Noise" not in selected

    def test_high_threshold_selects_nothing(self):
        df = make_df()
        selector = FeatureSelector()
        # threshold=1.0 means |r| must be strictly > 1.0, which is impossible
        selected, _ = selector.select(df, target="Chance of Admit", threshold=1.0)
        assert selected == []

    def test_low_threshold_selects_all(self):
        df = make_df()
        selector = FeatureSelector()
        selected, _ = selector.select(df, target="Chance of Admit", threshold=0.0)
        assert set(selected) == {"GRE Score", "Noise"}
