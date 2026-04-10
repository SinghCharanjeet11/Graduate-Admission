import logging
import os

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV

logger = logging.getLogger(__name__)


class ModelTrainer:
    RANDOM_STATE: int = 42
    RF_PARAM_GRID: dict = {
        "n_estimators": [50, 100, 200],
        "max_depth": [None, 5, 10, 20],
    }

    def train_linear_regression(
        self, X_train: pd.DataFrame, y_train: pd.Series
    ) -> LinearRegression:
        """Fit and return a LinearRegression model."""
        model = LinearRegression()
        model.fit(X_train, y_train)
        return model

    def train_random_forest(
        self, X_train: pd.DataFrame, y_train: pd.Series
    ) -> RandomForestRegressor:
        """Fit RandomForestRegressor via GridSearchCV; return best estimator."""
        try:
            rf = RandomForestRegressor(random_state=self.RANDOM_STATE)
            grid_search = GridSearchCV(
                rf,
                self.RF_PARAM_GRID,
                cv=5,
                scoring="r2",
                n_jobs=-1,
            )
            grid_search.fit(X_train, y_train)
            return grid_search.best_estimator_
        except Exception as error:
            logger.error("Training failed for random_forest: %s", error)
            raise

    def save_model(self, model, name: str, output_dir: str) -> str:
        """Serialize model to {output_dir}/{name}.joblib; return path."""
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, f"{name}.joblib")
        joblib.dump(model, path)
        return path

    def load_model(self, path: str):
        """Deserialize and return model from joblib file."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Artifact not found: {path}")
        return joblib.load(path)
