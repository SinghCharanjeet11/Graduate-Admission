import os
import tempfile

import pandas as pd
import pytest

from graduate_admission_prediction.pipeline.data_loader import DataLoader


VALID_COLUMNS = [
    "GRE Score",
    "TOEFL Score",
    "University Rating",
    "SOP",
    "LOR",
    "CGPA",
    "Research",
    "Chance of Admit",
]


def make_valid_csv(path: str, rows: int = 5) -> None:
    df = pd.DataFrame(
        {col: range(rows) for col in VALID_COLUMNS}
    )
    df.to_csv(path, index=False)


class TestDataLoader:
    def test_valid_load(self):
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            path = f.name
        try:
            make_valid_csv(path)
            loader = DataLoader()
            df = loader.load(path)
            assert isinstance(df, pd.DataFrame)
            assert len(df) == 5
            for col in VALID_COLUMNS:
                assert col in df.columns
        finally:
            os.unlink(path)

    def test_file_not_found(self):
        loader = DataLoader()
        with pytest.raises(FileNotFoundError):
            loader.load("/nonexistent/path/data.csv")

    def test_missing_columns(self):
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
            path = f.name
            f.write("GRE Score,TOEFL Score\n320,110\n")
        try:
            loader = DataLoader()
            with pytest.raises(ValueError, match="Missing columns"):
                loader.load(path)
        finally:
            os.unlink(path)

    def test_empty_csv(self):
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
            path = f.name
            f.write(",".join(VALID_COLUMNS) + "\n")
        try:
            loader = DataLoader()
            with pytest.raises(ValueError, match="zero rows"):
                loader.load(path)
        finally:
            os.unlink(path)
