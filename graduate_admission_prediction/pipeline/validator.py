import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold


class Validator:
    RANDOM_STATE: int = 42
    N_SPLITS: int = 5

    def cross_validate(self, model, X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
        """Run 5-Fold CV; return {"mean_r2": float, "std_r2": float, "fold_scores": list[float]}."""
        kf = KFold(n_splits=self.N_SPLITS, shuffle=True, random_state=self.RANDOM_STATE)
        fold_scores = []

        for train_idx, val_idx in kf.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            fold_scores.append(r2_score(y_val, y_pred))

        return {
            "mean_r2": float(np.mean(fold_scores)),
            "std_r2": float(np.std(fold_scores)),
            "fold_scores": fold_scores,
        }
