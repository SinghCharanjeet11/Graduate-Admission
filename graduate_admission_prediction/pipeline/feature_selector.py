import pandas as pd


class FeatureSelector:
    def select(
        self,
        df: pd.DataFrame,
        target: str = "Chance of Admit",
        threshold: float = 0.1,
    ) -> tuple[list[str], pd.Series]:
        """Return (selected_feature_names, correlation_series) for features
        with |Pearson r| > threshold against target.

        Args:
            df: Cleaned DataFrame containing features and target column.
            target: Name of the target column.
            threshold: Minimum absolute correlation (strictly greater than).

        Returns:
            A tuple of:
                - selected_feature_names: list of feature names where |r| > threshold
                - correlation_series: Pearson correlations for ALL input features,
                  indexed by feature name
        """
        feature_cols = [col for col in df.columns if col != target]
        correlation_series = df[feature_cols].corrwith(df[target])
        selected_feature_names = [
            col for col in feature_cols if abs(correlation_series[col]) > threshold
        ]
        return selected_feature_names, correlation_series
