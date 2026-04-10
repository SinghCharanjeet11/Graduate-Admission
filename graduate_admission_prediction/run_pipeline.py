"""CLI entry point for the Graduate Admission Prediction pipeline."""

import argparse
import json
import logging
import os

from pipeline.cleaner import Cleaner
from pipeline.data_loader import DataLoader
from pipeline.eda import EDA_Module
from pipeline.evaluator import Evaluator
from pipeline.feature_selector import FeatureSelector
from pipeline.model_trainer import ModelTrainer
from pipeline.splitter import Splitter
from pipeline.validator import Validator

BASE_DIR = os.path.dirname(__file__)
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")


def main(csv_path: str) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger(__name__)

    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    # --- Data Loading ---
    logger.info("Loading data from %s", csv_path)
    loader = DataLoader()
    df = loader.load(csv_path)
    logger.info("Loaded %d rows", len(df))

    # --- EDA ---
    logger.info("Running EDA")
    eda = EDA_Module()
    stats = eda.summary_statistics(df)
    logger.info("Summary statistics:\n%s", stats)

    heatmap_path = os.path.join(ARTIFACTS_DIR, "eda_heatmap.png")
    eda.correlation_heatmap(df, save_path=heatmap_path)
    logger.info("Saved correlation heatmap to %s", heatmap_path)

    dist_dir = os.path.join(ARTIFACTS_DIR, "eda_plots")
    eda.distribution_plots(df, save_dir=dist_dir)
    logger.info("Saved distribution plots to %s", dist_dir)

    scatter_dir = os.path.join(ARTIFACTS_DIR, "eda_scatter")
    eda.scatter_plots(df, save_dir=scatter_dir)
    logger.info("Saved scatter plots to %s", scatter_dir)

    # --- Cleaning ---
    logger.info("Cleaning data")
    cleaner = Cleaner()
    df_clean = cleaner.clean(df)
    scaler_path = os.path.join(ARTIFACTS_DIR, "scaler.joblib")
    cleaner.save_scaler(scaler_path)
    logger.info("Saved scaler to %s", scaler_path)

    # --- Feature Selection ---
    logger.info("Selecting features")
    selector = FeatureSelector()
    features, corr = selector.select(df_clean)
    logger.info("Selected features: %s", features)

    # --- Splitting ---
    logger.info("Splitting data (80/20)")
    splitter = Splitter()
    X_train, X_test, y_train, y_test = splitter.split(df_clean, features)
    logger.info("Train size: %d, Test size: %d", len(X_train), len(X_test))

    # --- Model Training ---
    trainer = ModelTrainer()

    logger.info("Training Linear Regression")
    lr_model = trainer.train_linear_regression(X_train, y_train)
    lr_path = trainer.save_model(lr_model, "linear_regression", ARTIFACTS_DIR)
    logger.info("Saved linear regression model to %s", lr_path)

    logger.info("Training Random Forest (GridSearchCV)")
    rf_model = trainer.train_random_forest(X_train, y_train)
    rf_path = trainer.save_model(rf_model, "random_forest", ARTIFACTS_DIR)
    logger.info("Saved random forest model to %s", rf_path)

    # --- Cross-Validation ---
    validator = Validator()

    logger.info("Cross-validating Linear Regression")
    lr_cv = validator.cross_validate(lr_model, X_train, y_train)
    logger.info("LR CV — mean_r2: %.4f, std_r2: %.4f", lr_cv["mean_r2"], lr_cv["std_r2"])

    logger.info("Cross-validating Random Forest")
    rf_cv = validator.cross_validate(rf_model, X_train, y_train)
    logger.info("RF CV — mean_r2: %.4f, std_r2: %.4f", rf_cv["mean_r2"], rf_cv["std_r2"])

    # --- Evaluation ---
    evaluator = Evaluator()

    logger.info("Evaluating Linear Regression on test set")
    lr_result = evaluator.evaluate(lr_model, X_test, y_test, "linear_regression")
    logger.info("LR — R2: %.4f, RMSE: %.4f", lr_result["r2"], lr_result["rmse"])

    logger.info("Evaluating Random Forest on test set")
    rf_result = evaluator.evaluate(rf_model, X_test, y_test, "random_forest")
    logger.info("RF — R2: %.4f, RMSE: %.4f", rf_result["r2"], rf_result["rmse"])

    best = evaluator.best_model([lr_result, rf_result])
    logger.info("Best model: %s (R2=%.4f)", best["model_name"], best["r2"])

    # --- Save Metrics ---
    metrics = {
        "linear_regression": lr_result,
        "random_forest": rf_result,
        "best_model": best["model_name"],
    }
    metrics_path = os.path.join(ARTIFACTS_DIR, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Saved metrics to %s", metrics_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Graduate Admission Prediction pipeline")
    parser.add_argument(
        "--data",
        default=os.path.join(BASE_DIR, "data", "Admission_Predict.csv"),
        help="Path to the input CSV file",
    )
    args = parser.parse_args()
    main(args.data)
