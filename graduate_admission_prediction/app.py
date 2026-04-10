"""Streamlit UI for Graduate Admission Prediction."""

import json
import os

import joblib
import pandas as pd
import streamlit as st

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
# Task 13.4 — Main layout
# ---------------------------------------------------------------------------
def main() -> None:
    artifacts = load_artifacts(ARTIFACTS_DIR)
    scaler = artifacts["scaler"]
    models = artifacts["models"]
    metrics = artifacts["metrics"]

    # --- Title ---
    st.title("🎓 Graduate Admission Predictor")
    st.caption("Predict your probability of graduate school admission using ML models.")
    st.divider()

    # --- Sidebar ---
    with st.sidebar:
        st.title("⚙️ Settings")
        st.divider()

        model_name = st.radio(
            "Select Model",
            options=list(models.keys()),
            index=0,
        )

        st.divider()
        st.subheader("Model Metrics")

        # Map display name → metrics.json key
        metrics_key = "linear_regression" if model_name == "Linear Regression" else "random_forest"
        model_metrics = metrics.get(metrics_key, {})

        col_r2, col_rmse = st.columns(2)
        col_r2.metric("R²", f"{model_metrics.get('r2', 'N/A'):.4f}" if isinstance(model_metrics.get('r2'), float) else "N/A")
        col_rmse.metric("RMSE", f"{model_metrics.get('rmse', 'N/A'):.4f}" if isinstance(model_metrics.get('rmse'), float) else "N/A")

        st.divider()
        st.subheader("About")
        st.write(
            "This app uses a trained ML model to estimate the probability "
            "of graduate school admission based on academic profile inputs. "
            "Run `python run_pipeline.py` to retrain the models."
        )

    # --- Main panel: two columns ---
    left_col, right_col = st.columns([1, 1], gap="large")

    with left_col:
        st.subheader("📋 Input Features")

        gre = st.slider("GRE Score", min_value=260, max_value=340, step=1, value=310)
        toefl = st.slider("TOEFL Score", min_value=92, max_value=120, step=1, value=107)
        uni_rating = st.slider("University Rating", min_value=1, max_value=5, step=1, value=3)
        sop = st.slider("SOP Strength", min_value=1.0, max_value=5.0, step=0.5, value=3.0)
        lor = st.slider("LOR Strength", min_value=1.0, max_value=5.0, step=0.5, value=3.0)
        cgpa = st.slider("CGPA", min_value=6.0, max_value=10.0, step=0.1, value=8.5)

        research_choice = st.selectbox(
            "Research Experience",
            options=["No (0)", "Yes (1)"],
            index=0,
        )
        research = 1 if research_choice == "Yes (1)" else 0

    with right_col:
        st.subheader("🔮 Prediction")

        # Summary table of inputs
        input_summary = pd.DataFrame(
            {
                "Feature": FEATURE_COLUMNS,
                "Value": [gre, toefl, uni_rating, sop, lor, cgpa, research],
            }
        )
        st.dataframe(input_summary, use_container_width=True, hide_index=True)

        st.divider()

        if st.button("Predict", type="primary", use_container_width=True):
            model = models[model_name]
            input_df = build_input_df(gre, toefl, uni_rating, sop, lor, cgpa, research, scaler)
            prediction = float(model.predict(input_df)[0])
            render_prediction(prediction)


if __name__ == "__main__":
    main()
