import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


class EDA_Module:
    def summary_statistics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return descriptive stats extended with an explicit median row.

        Covers mean, std, min, max, 50% (from describe) and median for all
        numeric columns.
        """
        stats = df.describe()
        median_row = pd.DataFrame(df.median(numeric_only=True), columns=["median"]).T
        return pd.concat([stats, median_row])

    def correlation_heatmap(self, df: pd.DataFrame, save_path: str | None = None) -> None:
        """Plot Pearson correlation heatmap; optionally save to save_path."""
        fig, ax = plt.subplots()
        sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
        ax.set_title("Pearson Correlation Heatmap")
        if save_path:
            fig.savefig(save_path)
            plt.close(fig)
        else:
            plt.show()

    def distribution_plots(self, df: pd.DataFrame, save_dir: str | None = None) -> None:
        """Plot histogram + KDE for each numeric column; optionally save to save_dir."""
        numeric_cols = df.select_dtypes(include="number").columns
        for col in numeric_cols:
            fig, ax = plt.subplots()
            sns.histplot(df[col].dropna(), kde=True, ax=ax)
            ax.set_title(f"Distribution of {col}")
            ax.set_xlabel(col)
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                fig.savefig(os.path.join(save_dir, f"{col}_distribution.png"))
                plt.close(fig)
            else:
                plt.show()

    def scatter_plots(
        self,
        df: pd.DataFrame,
        target: str = "Chance of Admit",
        save_dir: str | None = None,
    ) -> None:
        """Plot scatter of each numeric feature vs target; optionally save to save_dir."""
        numeric_cols = [c for c in df.select_dtypes(include="number").columns if c != target]
        for col in numeric_cols:
            fig, ax = plt.subplots()
            ax.scatter(df[col], df[target], alpha=0.5)
            ax.set_xlabel(col)
            ax.set_ylabel(target)
            ax.set_title(f"{col} vs {target}")
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                fig.savefig(os.path.join(save_dir, f"{col}_vs_{target}.png"))
                plt.close(fig)
            else:
                plt.show()
