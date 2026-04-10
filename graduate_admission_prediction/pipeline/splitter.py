import pandas as pd
from sklearn.model_selection import train_test_split


class Splitter:
    RANDOM_STATE: int = 42

    def split(
        self,
        df: pd.DataFrame,
        features: list[str],
        target: str = "Chance of Admit",
        test_size: float = 0.2,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Return (X_train, X_test, y_train, y_test) with 80/20 split, seed=42."""
        X = df[features]
        y = df[target]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.RANDOM_STATE
        )
        return X_train, X_test, y_train, y_test
