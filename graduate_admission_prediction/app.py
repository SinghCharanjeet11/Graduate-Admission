"""Streamlit UI for Graduate Admission Prediction."""

import json
import os

import joblib
import pandas as pd
import streamlit as st

from pipeline.cleaner import Cleaner
from pipeline.data_loader import DataLoader
from pipeline.eda import EDA_Module
from pipeline.evaluator import Evaluator
from pipeline.feature_selector import FeatureSelector
from pipeline.model_trainer import ModelTrainer
from pipeline.splitter import Splitter
from pipeline.validator import Validator

# ---------------------------------------------------------------------------
# Page config — must be the first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="🎓 Graduate Admission Predictor",
    layout="wide",
)

ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "artifacts")

FEATURE_COLUMNS = [
    "GRE Score",
    "TOEFL Score",
    "University Rating",
    "SOP",
    "LOR",
    "CGPA",
    "Research",
]


# ---------------------------------------------------------------------------
# Task 13.1 — load_artifacts
# ---------------------------------------------------------------------------
@st.cache_resource
def load_artifacts(artifacts_dir: str) -> dict:
    """Load scaler, models, and metrics from artifacts_dir."""
    try:
        scaler = joblib.load(os.path.join(artifacts_dir, "scaler.joblib"))
        lr_model = joblib.load(os.path.join(artifacts_dir, "linear_regression.joblib"))
        rf_model = joblib.load(os.path.join(artifacts_dir, "random_forest.joblib"))
        with open(os.path.join(artifacts_dir, "metrics.json")) as f:
            metrics = json.load(f)
    except FileNotFoundError:
        st.error(
            "Pipeline artifacts not found. Please run: python run_pipeline.py"
        )
        st.stop()

    return {
        "scaler": scaler,
        "models": {
            "Linear Regression": lr_model,
            "Random Forest": rf_model,
        },
        "metrics": metrics,
    }


# ---------------------------------------------------------------------------
# Task 13.2 — build_input_df
# ---------------------------------------------------------------------------
def build_input_df(
    gre: int,
    toefl: int,
    uni_rating: int,
    sop: float,
    lor: float,
    cgpa: float,
    research: int,
    scaler,
) -> pd.DataFrame:
    """Assemble a single-row DataFrame and apply the fitted scaler."""
    raw = pd.DataFrame(
        [[gre, toefl, uni_rating, sop, lor, cgpa, research]],
        columns=FEATURE_COLUMNS,
    )
    scaled = scaler.transform(raw)
    return pd.DataFrame(scaled, columns=FEATURE_COLUMNS)


# ---------------------------------------------------------------------------
# Task 13.3 — render_prediction
# ---------------------------------------------------------------------------
def render_prediction(prediction: float) -> None:
    """Display the admission probability with color-coded feedback."""
    if prediction >= 0.7:
        emoji = "✅"
        label = "High chance of admission"
    elif prediction >= 0.5:
        emoji = "⚠️"
        label = "Moderate chance of admission"
    else:
        emoji = "❌"
        label = "Low chance of admission"

    st.metric(
        label=f"Chance of Admission {emoji}",
        value=f"{prediction:.1%}",
    )
    st.progress(min(prediction, 1.0))

    if prediction >= 0.7:
        st.success(f"{emoji} {label} ({prediction:.1%})")
    elif prediction >= 0.5:
        st.warning(f"{emoji} {label} ({prediction:.1%})")
    else:
        st.error(f"{emoji} {label} ({prediction:.1%})")


# ---------------------------------------------------------------------------
# Pipeline steps metadata
# ---------------------------------------------------------------------------
PIPELINE_STEPS = [
    ("Data Loading",      "Reads raw CSV from disk, validates shape"),
    ("EDA",               "Summary stats, correlation heatmap, distribution & scatter plots"),
    ("Cleaning",          "Drops nulls, scales features, saves scaler"),
    ("Feature Selection", "Selects top correlated features via Pearson r"),
    ("Splitting",         "80 / 20 stratified train-test split"),
    ("Model Training",    "Trains Linear Regression & Random Forest (GridSearchCV)"),
    ("Cross-Validation",  "5-fold CV — mean R² and std for each model"),
    ("Evaluation",        "R² and RMSE on held-out test set, picks best model"),
]

PIPELINE_LOG = os.path.join(ARTIFACTS_DIR, "pipeline_log.json")


def load_pipeline_log() -> dict:
    try:
        with open(PIPELINE_LOG) as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def _log_step(step: str) -> None:
    log = load_pipeline_log()
    log[step] = True
    with open(PIPELINE_LOG, "w") as f:
        json.dump(log, f, indent=2)


def artifacts_ready() -> bool:
    required = ["scaler.joblib", "linear_regression.joblib", "random_forest.joblib", "metrics.json"]
    return all(os.path.exists(os.path.join(ARTIFACTS_DIR, f)) for f in required)


# ---------------------------------------------------------------------------
# Task 13.4 — Main layout
# ---------------------------------------------------------------------------
def main() -> None:
    # --- Title ---
    st.title("🎓 Graduate Admission Predictor")
    st.caption("ML pipeline dashboard — predict admission probability using trained models.")
    st.divider()

    # --- Sidebar ---
    with st.sidebar:
        st.title("⚙️ Settings")
        st.divider()
        st.subheader("About")
        st.write(
            "End-to-end ML pipeline: data loading, EDA, cleaning, "
            "feature selection, model training, and evaluation. "
        )
        st.divider()
        st.caption("Run the **Pipeline** tab to train models before predicting.")

    # --- Tabs ---
    tab_pipeline, tab_predict = st.tabs(["🚀 Pipeline", "🔮 Predict"])

    # ── Pipeline tab ──────────────────────────────────────────────────────
    with tab_pipeline:
        st.subheader("Pipeline Status")

        ready = artifacts_ready()
        if ready:
            st.success("✅ Artifacts found — pipeline has been run successfully.")
        else:
            st.warning("⚠️ Artifacts not found. Run the pipeline to train models.")

        st.divider()

        # Pipeline steps table
        st.subheader("Pipeline Steps")
        log = load_pipeline_log()
        rows = []
        for name, desc in PIPELINE_STEPS:
            status = "✅ Done" if log.get(name) else "⏳ Pending"
            rows.append({"Step": name, "Description": desc, "Status": status})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        st.divider()

        # Run Pipeline button
        st.subheader("Run Pipeline")
        csv_default = os.path.join(os.path.dirname(__file__), "data", "Admission_Predict.csv")
        if st.button("▶ Run Pipeline", type="primary", use_container_width=False):
            with open(PIPELINE_LOG, "w") as f:
                json.dump({}, f)

            with st.status("Running pipeline...", expanded=True) as status:
                try:
                    st.write("📂 Loading data...")
                    loader = DataLoader()
                    df = loader.load(csv_default)
                    _log_step("Data Loading")
                    st.write(f"✅ Data loaded — {len(df)} rows")

                    st.write("🔍 Running EDA...")
                    eda = EDA_Module()
                    eda.summary_statistics(df)
                    eda.correlation_heatmap(df, save_path=os.path.join(ARTIFACTS_DIR, "eda_heatmap.png"))
                    eda.distribution_plots(df, save_dir=os.path.join(ARTIFACTS_DIR, "eda_plots"))
                    eda.scatter_plots(df, save_dir=os.path.join(ARTIFACTS_DIR, "eda_scatter"))
                    _log_step("EDA")
                    st.write("✅ EDA complete — heatmap, distributions, scatter plots saved")

                    st.write("🧹 Cleaning data...")
                    cleaner = Cleaner()
                    df_clean = cleaner.clean(df)
                    cleaner.save_scaler(os.path.join(ARTIFACTS_DIR, "scaler.joblib"))
                    _log_step("Cleaning")
                    st.write("✅ Data cleaned — scaler saved")

                    st.write("🔎 Selecting features...")
                    selector = FeatureSelector()
                    features, _ = selector.select(df_clean)
                    _log_step("Feature Selection")
                    st.write(f"✅ Features selected — {features}")

                    st.write("✂️ Splitting data (80/20)...")
                    splitter = Splitter()
                    X_train, X_test, y_train, y_test = splitter.split(df_clean, features)
                    _log_step("Splitting")
                    st.write(f"✅ Split done — train: {len(X_train)}, test: {len(X_test)}")

                    st.write("⚙️ Training models...")
                    trainer = ModelTrainer()
                    lr_model = trainer.train_linear_regression(X_train, y_train)
                    trainer.save_model(lr_model, "linear_regression", ARTIFACTS_DIR)
                    rf_model = trainer.train_random_forest(X_train, y_train)
                    trainer.save_model(rf_model, "random_forest", ARTIFACTS_DIR)
                    _log_step("Model Training")
                    st.write("✅ Linear Regression & Random Forest trained")

                    st.write("🔁 Cross-validating...")
                    validator = Validator()
                    lr_cv = validator.cross_validate(lr_model, X_train, y_train)
                    rf_cv = validator.cross_validate(rf_model, X_train, y_train)
                    _log_step("Cross-Validation")
                    st.write(f"✅ CV done — LR R²: {lr_cv['mean_r2']:.4f} | RF R²: {rf_cv['mean_r2']:.4f}")

                    st.write("📊 Evaluating on test set...")
                    evaluator = Evaluator()
                    lr_result = evaluator.evaluate(lr_model, X_test, y_test, "linear_regression")
                    rf_result = evaluator.evaluate(rf_model, X_test, y_test, "random_forest")
                    best = evaluator.best_model([lr_result, rf_result])
                    metrics_out = {"linear_regression": lr_result, "random_forest": rf_result, "best_model": best["model_name"]}
                    with open(os.path.join(ARTIFACTS_DIR, "metrics.json"), "w") as f:
                        json.dump(metrics_out, f, indent=2)
                    _log_step("Evaluation")
                    st.write(f"✅ Evaluation done — Best model: **{best['model_name']}** (R²: {best['r2']:.4f})")

                    status.update(label="Pipeline complete!", state="complete", expanded=False)
                    st.cache_resource.clear()
                    st.rerun()

                except Exception as e:
                    status.update(label="Pipeline failed", state="error")
                    st.exception(e)

        # EDA visuals
        heatmap_path = os.path.join(ARTIFACTS_DIR, "eda_heatmap.png")
        dist_dir = os.path.join(ARTIFACTS_DIR, "eda_plots")
        scatter_dir = os.path.join(ARTIFACTS_DIR, "eda_scatter")

        if os.path.exists(heatmap_path):
            st.divider()
            st.subheader("📊 Correlation Heatmap")
            st.image(heatmap_path, use_container_width=True)

        if os.path.isdir(dist_dir):
            dist_files = sorted([
                f for f in os.listdir(dist_dir)
                if f.endswith(".png") and not f.startswith("Serial No.")
            ])
            if dist_files:
                st.divider()
                st.subheader("📈 Feature Distributions")
                cols = st.columns(3)
                for i, fname in enumerate(dist_files):
                    feature_name = fname.replace("_distribution.png", "").replace("_", " ")
                    with cols[i % 3]:
                        st.image(os.path.join(dist_dir, fname), caption=feature_name, use_container_width=True)

        if os.path.isdir(scatter_dir):
            scatter_files = sorted([
                f for f in os.listdir(scatter_dir)
                if f.endswith(".png") and not f.startswith("Serial No.")
            ])
            if scatter_files:
                st.divider()
                st.subheader("🔵 Feature vs Chance of Admit")
                cols = st.columns(3)
                for i, fname in enumerate(scatter_files):
                    feature_name = fname.replace("_vs_Chance of Admit.png", "").replace("_", " ")
                    with cols[i % 3]:
                        st.image(os.path.join(scatter_dir, fname), caption=feature_name, use_container_width=True)

    # ── Predict tab ───────────────────────────────────────────────────────
    with tab_predict:
        if not artifacts_ready():
            st.info("Run the pipeline first (Pipeline tab) to generate model artifacts.")
            st.stop()

        artifacts = load_artifacts(ARTIFACTS_DIR)
        scaler = artifacts["scaler"]
        models = artifacts["models"]
        metrics = artifacts["metrics"]

        # Model selector + metrics inline
        col_sel, col_r2, col_rmse = st.columns([2, 1, 1])
        with col_sel:
            model_name = st.radio("Select Model", options=list(models.keys()), horizontal=True)
        metrics_key = "linear_regression" if model_name == "Linear Regression" else "random_forest"
        model_metrics = metrics.get(metrics_key, {})
        col_r2.metric("R²", f"{model_metrics.get('r2', 0):.4f}")
        col_rmse.metric("RMSE", f"{model_metrics.get('rmse', 0):.4f}")

        st.divider()

        left_col, right_col = st.columns([1, 1], gap="large")

        with left_col:
            st.subheader("📋 Input Features")
            gre = st.slider("GRE Score", min_value=260, max_value=340, step=1, value=310)
            toefl = st.slider("TOEFL Score", min_value=92, max_value=120, step=1, value=107)
            uni_rating = st.slider("University Rating", min_value=1, max_value=5, step=1, value=3)
            sop = st.slider("SOP Strength", min_value=1.0, max_value=5.0, step=0.5, value=3.0)
            lor = st.slider("LOR Strength", min_value=1.0, max_value=5.0, step=0.5, value=3.0)
            cgpa = st.slider("CGPA", min_value=6.0, max_value=10.0, step=0.1, value=8.5)
            research_choice = st.selectbox("Research Experience", options=["No (0)", "Yes (1)"], index=0)
            research = 1 if research_choice == "Yes (1)" else 0

        with right_col:
            st.subheader("🔮 Prediction")
            input_summary = pd.DataFrame(
                {"Feature": FEATURE_COLUMNS, "Value": [gre, toefl, uni_rating, sop, lor, cgpa, research]}
            )
            st.dataframe(input_summary, use_container_width=True, hide_index=True)
            st.divider()
            if st.button("Predict", type="primary", use_container_width=True):
                model = models[model_name]
                input_df = build_input_df(gre, toefl, uni_rating, sop, lor, cgpa, research, scaler)
                prediction = float(model.predict(input_df)[0])
                prediction = max(0.0, min(prediction, 1.0))  # clamp — LR has no output constraint
                render_prediction(prediction)


if __name__ == "__main__":
    main()
