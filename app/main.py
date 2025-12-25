# app/main.py

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# --------------------------------------------------
# Load ONLY Logistic Regression artifacts
# --------------------------------------------------
with open('model/logreg.pkl', 'rb') as f:
    model = pickle.load(f)

with open('model/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('model/imputer.pkl', 'rb') as f:
    imputer = pickle.load(f)

with open('model/metrics.pkl', 'rb') as f:
    metrics = pickle.load(f)

with open('model/scores.pkl', 'rb') as f:
    scores = pickle.load(f)

best_params = metrics["best_params"]

# --------------------------------------------------
# Streamlit UI
# --------------------------------------------------
st.set_page_config(page_title="Breast Cancer Prediction", layout="wide")
st.title("🩺 OncoPredict – Breast Cancer Diagnosis Predictor")

# Load custom CSS (UNCHANGED)
with open("assets/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# --------------------------------------------------
# Sidebar (DESIGN KEPT)
# --------------------------------------------------
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2920/2920277.png", width=100)
    st.title("⚙️ Model Info")

    st.markdown("### 🧠 Final Selected Model")
    st.success("Logistic Regression")

    st.markdown("---")
    st.markdown("📊 **Model Performance (Test Data)**")

    st.markdown(f"- **Accuracy:** `{metrics['accuracy']*100:.2f}%`")
    st.markdown(f"- **Precision:** `{metrics['precision']*100:.2f}%`")
    st.markdown(f"- **Recall:** `{metrics['recall']*100:.2f}%`")
    st.markdown(f"- **F1 Score:** `{metrics['f1_score']*100:.2f}%`")

    st.markdown("---")
    st.markdown("⚙️ **Best Hyperparameters**")
    for param, val in best_params.items():
        st.markdown(f"- `{param}`: `{val}`")

    st.markdown("---")
    st.markdown(
        "💡 This app predicts whether a tumor is **benign or malignant** using features "
        "extracted from digitized breast biopsy images."
    )

st.markdown(
    "This application uses a **tuned Logistic Regression model** selected after "
    "comparing multiple ML algorithms."
)

st.subheader("📥 Enter Tumor Features")

def synced_input(feature, min_val, max_val):
    input_key = f"{feature}_input"
    slider_key = f"{feature}_slider"

    # Initialize session state
    if input_key not in st.session_state:
        st.session_state[input_key] = (min_val + max_val) / 2
    if slider_key not in st.session_state:
        st.session_state[slider_key] = st.session_state[input_key]

    def sync_input():
        st.session_state[input_key] = st.session_state[slider_key]

    def sync_slider():
        st.session_state[slider_key] = st.session_state[input_key]

    col1, col2 = st.columns([2, 4])

    with col1:
        st.number_input(
            feature,
            min_value=min_val,
            max_value=max_val,
            step=0.01,
            key=input_key,
            on_change=sync_slider
        )

    with col2:
        st.slider(
            feature,
            min_value=min_val,
            max_value=max_val,
            step=0.01,
            key=slider_key,
            on_change=sync_input
        )

    return st.session_state[slider_key]

# ----------------------------------------
# Default Preset Values (Approximate)
# ----------------------------------------

BENIGN_PRESET = {
    "radius_mean": 12.0, "texture_mean": 14.0, "perimeter_mean": 78.0, "area_mean": 450.0,
    "smoothness_mean": 0.09, "compactness_mean": 0.08, "concavity_mean": 0.05,
    "concave points_mean": 0.03, "symmetry_mean": 0.18, "fractal_dimension_mean": 0.06,

    "radius_se": 0.3, "texture_se": 1.2, "perimeter_se": 2.0, "area_se": 25.0,
    "smoothness_se": 0.005, "compactness_se": 0.02, "concavity_se": 0.02,
    "concave points_se": 0.01, "symmetry_se": 0.02, "fractal_dimension_se": 0.003,

    "radius_worst": 13.5, "texture_worst": 18.0, "perimeter_worst": 85.0, "area_worst": 550.0,
    "smoothness_worst": 0.12, "compactness_worst": 0.15, "concavity_worst": 0.12,
    "concave points_worst": 0.06, "symmetry_worst": 0.25, "fractal_dimension_worst": 0.07
}

MALIGNANT_PRESET = {
    "radius_mean": 18.0, "texture_mean": 22.0, "perimeter_mean": 120.0, "area_mean": 1000.0,
    "smoothness_mean": 0.11, "compactness_mean": 0.18, "concavity_mean": 0.25,
    "concave points_mean": 0.12, "symmetry_mean": 0.22, "fractal_dimension_mean": 0.07,

    "radius_se": 0.7, "texture_se": 2.5, "perimeter_se": 5.0, "area_se": 80.0,
    "smoothness_se": 0.01, "compactness_se": 0.05, "concavity_se": 0.08,
    "concave points_se": 0.04, "symmetry_se": 0.04, "fractal_dimension_se": 0.01,

    "radius_worst": 22.0, "texture_worst": 30.0, "perimeter_worst": 150.0, "area_worst": 1600.0,
    "smoothness_worst": 0.16, "compactness_worst": 0.35, "concavity_worst": 0.45,
    "concave points_worst": 0.20, "symmetry_worst": 0.35, "fractal_dimension_worst": 0.10
}

st.subheader("⚡ Quick Presets")

col1, col2 = st.columns(2)

if col1.button("🟢 Load Benign Sample"):
    for feature, value in BENIGN_PRESET.items():
        st.session_state[f"{feature}_input"] = value
        st.session_state[f"{feature}_slider"] = value

if col2.button("🔴 Load Malignant Sample"):
    for feature, value in MALIGNANT_PRESET.items():
        st.session_state[f"{feature}_input"] = value
        st.session_state[f"{feature}_slider"] = value

# --------------------------------------------------
# Feature Inputs (UNCHANGED)
# --------------------------------------------------
features = {
    "radius_mean": (5.0, 30.0), "texture_mean": (5.0, 40.0),
    "perimeter_mean": (40.0, 200.0), "area_mean": (100.0, 2500.0),
    "smoothness_mean": (0.05, 0.2), "compactness_mean": (0.01, 1.0),
    "concavity_mean": (0.0, 1.0), "concave points_mean": (0.0, 1.0),
    "symmetry_mean": (0.1, 0.5), "fractal_dimension_mean": (0.04, 0.2),

    "radius_se": (0.0, 3.0), "texture_se": (0.0, 5.0),
    "perimeter_se": (0.0, 15.0), "area_se": (0.0, 100.0),
    "smoothness_se": (0.0, 0.05), "compactness_se": (0.0, 0.2),
    "concavity_se": (0.0, 0.3), "concave points_se": (0.0, 0.2),
    "symmetry_se": (0.0, 0.1), "fractal_dimension_se": (0.0, 0.05),

    "radius_worst": (10.0, 40.0), "texture_worst": (10.0, 50.0),
    "perimeter_worst": (70.0, 250.0), "area_worst": (200.0, 4000.0),
    "smoothness_worst": (0.1, 0.3), "compactness_worst": (0.1, 1.2),
    "concavity_worst": (0.0, 1.5), "concave points_worst": (0.0, 1.5),
    "symmetry_worst": (0.2, 0.9), "fractal_dimension_worst": (0.05, 0.3)
}

user_input = []

for group_label, group in [
    ("📊 Mean Features", [f for f in features if "_mean" in f]),
    ("📉 Standard Error Features", [f for f in features if "_se" in f]),
    ("⚠️ Worst Case Features", [f for f in features if "_worst" in f]),
]:
    st.subheader(group_label)
    for feature in group:
        min_val, max_val = features.get(feature, (0.0, 1.0))
        value = synced_input(feature, float(min_val), float(max_val))
        user_input.append(value)


# --------------------------------------------------
# Prediction
# --------------------------------------------------
if st.button("Predict"):
    input_array = np.array(user_input).reshape(1, -1)
    input_imputed = imputer.transform(input_array)
    input_scaled = scaler.transform(input_imputed)

    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][prediction]

    if prediction == 1:
        st.error(
            f"🔴 The model predicts this tumor is **Malignant** "
            f"with {probability * 100:.2f}% confidence."
        )
    else:
        st.success(
            f"🟢 The model predicts this tumor is **Benign** "
            f"with {probability * 100:.2f}% confidence."
        )

    # --------------------------------------------------
    # Confusion Matrix (UNCHANGED)
    # --------------------------------------------------
    st.subheader("🧊 Confusion Matrix (Test Data)")
    cm = metrics["confusion_matrix"]

    fig, ax = plt.subplots()
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["Benign", "Malignant"],
        yticklabels=["Benign", "Malignant"],
        ax=ax
    )
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    st.pyplot(fig)

# --------------------------------------------------
# Single-Model Performance Chart (Resume Friendly)
# --------------------------------------------------
st.subheader("📊 Model Performance Summary")

df = pd.DataFrame({
    "Metric": ["Accuracy", "Precision", "Recall", "F1 Score"],
    "Score": [
        metrics["accuracy"],
        metrics["precision"],
        metrics["recall"],
        metrics["f1_score"]
    ]
})

fig = go.Figure(
    go.Bar(
        x=df["Metric"],
        y=df["Score"],
        text=[f"{v*100:.2f}%" for v in df["Score"]],
        textposition="auto"
    )
)

fig.update_layout(
    title="Logistic Regression Performance",
    yaxis_title="Score",
    height=450
)

st.plotly_chart(fig)
