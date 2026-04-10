import math

import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score


class Evaluator:
    R2_THRESHOLD: float = 0.85
    RMSE_THRESHOLD: float = 0.1

    def evaluate(
        self,
        model,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        model_name: str,
    ) -> dict:
        """Compute MSE, RMSE, R2. Return results dict with threshold flags."""
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        rmse = math.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        return {
            "model_name": model_name,
            "mse": float(mse),
            "rmse": float(rmse),
            "r2": float(r2),
            "meets_r2_threshold": r2 >= self.R2_THRESHOLD,
            "meets_rmse_threshold": rmse <= self.RMSE_THRESHOLD,
        }

    def best_model(self, results: list[dict]) -> dict:
        """Return the result dict with the highest R2_Score."""
        return max(results, key=lambda r: r["r2"])
