import logging

import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class Cleaner:
    def __init__(self):
        self.scaler: StandardScaler | None = None

    def clean(self, df: pd.DataFrame, target: str = "Chance of Admit") -> pd.DataFrame:
        """Drop serial/index cols, impute missing values, standardize numeric
        features (excluding target). Stores fitted scaler as self.scaler."""
        df = df.copy()

        # Strip whitespace from column names and drop serial number columns
        df.columns = df.columns.str.strip()
        serial_cols = [c for c in df.columns if c == "Serial No."]
        if serial_cols:
            df = df.drop(columns=serial_cols)

        # Impute missing values with median for numeric columns
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        missing_mask = df[numeric_cols].isnull().any(axis=1)
        affected_rows = int(missing_mask.sum())
        if affected_rows > 0:
            logger.info("Imputing missing values in %d row(s) using median.", affected_rows)
        for col in numeric_cols:
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].median())

        # Fit StandardScaler on numeric input features (exclude target)
        feature_cols = [c for c in numeric_cols if c != target]
        self.scaler = StandardScaler()
        df[feature_cols] = self.scaler.fit_transform(df[feature_cols])

        return df

    def save_scaler(self, path: str) -> None:
        """Persist fitted scaler via joblib."""
        if self.scaler is None:
            raise RuntimeError("Scaler has not been fitted. Call clean() first.")
        joblib.dump(self.scaler, path)

    def load_scaler(self, path: str) -> None:
        """Load a previously persisted scaler."""
        self.scaler = joblib.load(path)
