import numpy as np
import pandas as pd
import pytest

from graduate_admission_prediction.pipeline.cleaner import Cleaner


def make_df(with_missing: bool = False) -> pd.DataFrame:
    data = {
        "GRE Score": [320.0, 310.0, 300.0, 330.0, 315.0],
        "TOEFL Score": [110.0, 105.0, 100.0, 115.0, 108.0],
        "CGPA": [9.0, 8.5, 8.0, 9.5, 8.8],
        "Chance of Admit": [0.9, 0.7, 0.6, 0.95, 0.75],
    }
    df = pd.DataFrame(data)
    if with_missing:
        df.loc[1, "GRE Score"] = np.nan
        df.loc[3, "TOEFL Score"] = np.nan
    return df


class TestCleaner:
    def test_no_missing_values_after_clean(self):
        df = make_df(with_missing=True)
        cleaner = Cleaner()
        result = cleaner.clean(df)
        assert result.isnull().sum().sum() == 0

    def test_target_preserved(self):
        df = make_df()
        original_target = df["Chance of Admit"].copy()
        cleaner = Cleaner()
        result = cleaner.clean(df)
        # Target column should still be present and values unchanged
        assert "Chance of Admit" in result.columns
        pd.testing.assert_series_equal(
            result["Chance of Admit"].reset_index(drop=True),
            original_target.reset_index(drop=True),
        )

    def test_runtime_error_on_unfitted_save_scaler(self, tmp_path):
        cleaner = Cleaner()
        with pytest.raises(RuntimeError, match="Scaler has not been fitted"):
            cleaner.save_scaler(str(tmp_path / "scaler.pkl"))

    def test_serial_col_dropped(self):
        df = make_df()
        df.insert(0, "Serial No.", range(len(df)))
        cleaner = Cleaner()
        result = cleaner.clean(df)
        assert "Serial No." not in result.columns

    def test_scaler_fitted_after_clean(self):
        df = make_df()
        cleaner = Cleaner()
        cleaner.clean(df)
        assert cleaner.scaler is not None
