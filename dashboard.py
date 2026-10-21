
import streamlit as st
import pandas as pd
import numpy as np
import joblib

from pathlib import Path
import plotly.express as px

# --------------------------------------------------
# 1. Page configuration
# --------------------------------------------------
st.set_page_config(
    page_title="NetSentry | AI Security",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------
# 2. Custom styling
# --------------------------------------------------
st.markdown("""
<style>
    .stApp {
        background-color: #0B1120;
        color: #E5EDF8;
    }

    [data-testid="stSidebar"] {
        background-color: #111B2E;
        border-right: 1px solid #26364F;
    }

    [data-testid="stSidebar"] * {
        color: #DCE7F5;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    .hero {
        padding: 25px;
        border: 1px solid #263B59;
        border-radius: 18px;
        background: linear-gradient(120deg, #14243B, #102D43);
        margin-bottom: 25px;
    }

    .hero-title {
        font-size: 34px;
        font-weight: 800;
        color: #F3F8FF;
        margin-bottom: 5px;
    }

    .hero-subtitle {
        color: #9EB5D2;
        font-size: 15px;
    }

    [data-testid="stMetric"] {
        background: #121E32;
        border: 1px solid #26364F;
        padding: 18px;
        border-radius: 14px;
    }

    [data-testid="stMetricLabel"] {
        color: #A8BCD5;
    }

    [data-testid="stMetricValue"] {
        color: #F4F8FF;
    }

    div[data-testid="stFileUploader"] {
        background: #111D30;
        border: 1px dashed #3C6B9E;
        border-radius: 14px;
        padding: 12px;
    }

    div.stButton > button,
    div.stDownloadButton > button {
        border-radius: 9px;
        font-weight: 600;
    }

    h1, h2, h3 {
        color: #F1F6FF;
    }

    .small-note {
        color: #92A7C2;
        font-size: 13px;
    }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# 3. Load saved model
# --------------------------------------------------
MODEL_PATH = Path(__file__).parent / "netsentry_xgboost_bundle.pkl"

@st.cache_resource
def load_model():
    bundle = joblib.load(MODEL_PATH)
    return bundle["model"], bundle["features"]

try:
    model, feature_names = load_model()
except Exception as e:
    st.error(f"Could not load the saved model: {e}")
    st.stop()

# Store the latest prediction results for navigation between pages.
if "netsentry_results" not in st.session_state:
    st.session_state.netsentry_results = None

if "netsentry_page" not in st.session_state:
    st.session_state.netsentry_page = "Overview"


# Apply navigation before the radio widget is created
if st.session_state.pop("netsentry_go_to_results", False):
    st.session_state.netsentry_page = "Prediction Results"

# --------------------------------------------------
# 4. Sidebar navigation
# --------------------------------------------------
with st.sidebar:
    st.markdown("## 🛡️ NetSentry")
    st.caption("AI-POWERED NETWORK SECURITY")
    st.divider()

    st.radio(
        "NAVIGATION",
        [
            "Overview",
            "Analyze Traffic",
            "Prediction Results",
        ],
        key="netsentry_page"
    )

    st.divider()
    st.markdown("### Model status")
    st.success("XGBoost model loaded")
    st.caption(f"Expected features: {len(feature_names)}")
    st.caption("Mode: Uploaded CSV analysis")

# Read the currently selected page
page = st.session_state.netsentry_page

# --------------------------------------------------
# 5. Header
# --------------------------------------------------
st.markdown("""
<div class="hero">
    <div class="hero-title">NetSentry 🛡️</div>
    <div class="hero-subtitle">
        AI-Based Network Intrusion Detection System
        &nbsp; • &nbsp; CICIDS2017
    </div>
</div>
""", unsafe_allow_html=True)

# --------------------------------------------------
# 6. Overview page
# --------------------------------------------------
if page == "Overview":
    st.subheader("Security overview")
    st.write(
        "Analyze pre-extracted network-flow records and identify "
        "flows predicted as benign or potentially malicious."
    )
    st.subheader("About NetSentry")
    
    st.markdown("""
    **NetSentry** is a machine-learning prototype for binary network
    intrusion detection.
    
    **Technology stack**
    - XGBoost classifier
    - CICIDS2017 network-flow features
    - Python, Pandas and NumPy
    - Streamlit and Plotly
    
    **What it does**
    - Accepts pre-extracted network-flow records in CSV format.
    - Predicts whether each record is Benign or Attack.
    - Displays prediction probabilities and summary visualizations.
    - Exports prediction results for further analysis.
    
    **Current limitations**
    - Does not capture live network packets.
    - Does not block or mitigate attacks.
    - Predictions depend on feature compatibility and model generalization.
    - An attack prediction is a model assessment, not proof of malicious activity.
    """)

    st.warning(
        "Use the dashboard as a research prototype, not as a standalone "
        "production security control."
    )

# --------------------------------------------------
# 7. Analyze Traffic page
# --------------------------------------------------
elif page == "Analyze Traffic":
    st.subheader("Analyze network traffic")
    st.write(
        "Upload a CSV containing the features expected by the trained model."
    )

    uploaded_file = st.file_uploader(
        "Choose network-flow CSV",
        type=["csv"],
        key="traffic_uploader"
    )

    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Could not read the CSV: {e}")
            st.stop()

        if data.empty:
            st.error("The uploaded CSV contains no records.")
            st.stop()

        st.markdown("### Uploaded data")
        st.caption(f"{len(data):,} rows · {len(data.columns):,} columns")
        st.dataframe(data.head(10), use_container_width=True)

        missing_features = [
            col for col in feature_names if col not in data.columns
        ]

        if missing_features:
            st.error(
                f"Your CSV is missing {len(missing_features)} required features."
            )
            with st.expander("View missing feature names"):
                st.code(", ".join(missing_features))
            st.stop()

        if st.button("🔍 Analyze traffic",type="primary",use_container_width=True):
            with st.spinner("Analyzing network traffic..."):
                valid_input = data[feature_names].copy()
                for col in feature_names:
                    valid_input[col] = pd.to_numeric(
                        valid_input[col], errors="coerce"
                    )
                valid_input = valid_input.replace(
                    [np.inf, -np.inf], np.nan
                    )

                invalid_rows = valid_input.isna().all(axis=1)

                # Keep rows that contain at least one usable feature
                valid_input = valid_input.loc[~invalid_rows].copy()

                if valid_input.empty:
                    st.error(
                        "No usable records found. Please check your CSV data."
                    )
                else:
                    # Generate predictions
                    predictions = model.predict(valid_input)

                    # Generate attack probabilities
                    if hasattr(model, "predict_proba"):
                        probabilities = model.predict_proba(valid_input)
                        classes = list(model.classes_)
                        attack_index = classes.index(1)
                        attack_probabilities = (
                            probabilities[:, attack_index] * 100
                        )
                    else:
                        st.error(
                            "The loaded model does not support probability "
                            "predictions."
                        )
                        st.stop()

                    # Create the results DataFrame
                    results = valid_input.copy()
                    results["Prediction"] = [
                        "Attack" if prediction == 1 else "Benign"
                        for prediction in predictions
                    ]
                    results["Attack Probability (%)"] = (
                        attack_probabilities.round(2)
                    )

                    # Store results for the Prediction Results page
                    st.session_state.netsentry_results = results

                    skipped = int(invalid_rows.sum())
                    if skipped:
                        st.session_state.netsentry_notice = (
                            f"Analysis complete. {len(valid_input):,} "
                            f"records analyzed; {skipped:,} rows skipped."
                        )
                    else:
                        st.session_state.netsentry_notice = (
                            f"Analysis complete: "
                            f"{len(valid_input):,} records analyzed."
                        )
            # Request navigation on the next rerun
            st.session_state.netsentry_go_to_results = True
            st.rerun()

# --------------------------------------------------
# 8. Prediction Results page
# --------------------------------------------------
elif page == "Prediction Results":
    st.subheader("Prediction explorer")

    results = st.session_state.netsentry_results

    if results is None:
        c1, c2, c3 = st.columns(3)
        c1.metric("Detection engine", "XGBoost")
        c2.metric("Input mode", "CSV upload")
        c3.metric("Live monitoring", "Not enabled")
        st.info(
            "No analysis is available yet. Open **Analyze Traffic** "
            "in the sidebar to upload a network-flow CSV and get its prediction results."
            )
    else:
        c1, c2 = st.columns(2)
        with c1:
            prediction_filter = st.multiselect(
                "Prediction type",
                ["Attack", "Benign", "Not analyzed"],
                default=["Attack", "Benign"]
            )

        with c2:
            min_probability = st.slider(
                "Minimum attack probability (%)",
                min_value=0,
                max_value=100,
                value=0
            )

        filtered = results[
            results["Prediction"].isin(prediction_filter)
        ].copy()

        filtered = filtered[
            filtered["Attack Probability (%)"].fillna(-1) >= min_probability
        ]

        st.caption(f"Showing {len(filtered):,} matching records")
        st.dataframe(filtered, use_container_width=True, height=450)

        csv_output = results.to_csv(index=False).encode("utf-8")

        st.download_button(
            "⬇️ Download all predictions",
            data=csv_output,
            file_name="netsentry_predictions.csv",
            mime="text/csv",
            use_container_width=True
        )

        filtered_csv = filtered.to_csv(index=False).encode("utf-8")

        st.download_button(
            "⬇️ Download filtered results",
            data=filtered_csv,
            file_name="netsentry_filtered_predictions.csv",
            mime="text/csv",
            use_container_width=True
        )
        total = len(results)
        attack_count = int((results["Prediction"] == "Attack").sum())
        benign_count = int((results["Prediction"] == "Benign").sum())
        attack_rate = attack_count / total * 100 if total else 0
            
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total flows", f"{total:,}")
        c2.metric("Predicted attacks", f"{attack_count:,}")
        c3.metric("Predicted benign", f"{benign_count:,}")
        c4.metric("Attack rate", f"{attack_rate:.2f}%")
            
        left, right = st.columns(2)
            
        with left:
            st.markdown("### Prediction distribution")
            counts = results["Prediction"].value_counts().reindex(
                ["Benign", "Attack"], fill_value=0
            ).rename_axis("Prediction").reset_index(name="Records")
            
            fig = px.pie(
                counts,
                names="Prediction",
                values="Records",
                hole=0.62,
                color="Prediction",
                color_discrete_map={
                    "Benign": "#27C99A",
                    "Attack": "#FF5C72"
                }
            )
            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(t=20, b=20, l=10, r=10),
                legend_title_text=""
            )
            st.plotly_chart(fig, use_container_width=True)
            
        with right:
            st.markdown("### Attack probability distribution")
            fig = px.histogram(
                results,
                x="Attack Probability (%)",
                nbins=25,
                color_discrete_sequence=["#42BDF5"]
            )
            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                xaxis_title="Predicted attack probability (%)",
                yaxis_title="Number of flows",
                margin=dict(t=20, b=20, l=10, r=10)
            )
            st.plotly_chart(fig, use_container_width=True)
            
        st.caption(
            "Predictions are model outputs, not confirmed security incidents."
        )
            
