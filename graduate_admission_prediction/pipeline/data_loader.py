import os
import pandas as pd


class DataLoader:
    EXPECTED_COLUMNS = [
        "GRE Score",
        "TOEFL Score",
        "University Rating",
        "SOP",
        "LOR",
        "CGPA",
        "Research",
        "Chance of Admit",
    ]

    def load(self, filepath: str) -> pd.DataFrame:
        """Load CSV and validate schema.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If expected columns are missing or dataset has zero rows.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Dataset file not found: {filepath}")

        df = pd.read_csv(filepath)
        df.columns = df.columns.str.strip()

        missing_cols = [col for col in self.EXPECTED_COLUMNS if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing columns: {missing_cols}")

        if len(df) == 0:
            raise ValueError("Dataset contains zero rows")

        return df
