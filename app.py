import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import json
import matplotlib.pyplot as plt

st.set_page_config(page_title="Tensile Strength Predictor", layout="centered")

st.title("Low-Alloy Steel Tensile Strength Predictor")
st.write(
    "Enter alloy composition (wt%) and test temperature to predict tensile strength, "
    "with a SHAP-based explanation of the prediction."
)

# ---- Load model ----
@st.cache_resource
def load_model():
    model = joblib.load("xgb.pkl")
    return model

xgb_model = load_model()

# ---- Feature order (must match training exactly) ----
FEATURES = ['C', 'Si', 'Mn', 'P', 'S', 'Ni', 'Cr', 'Mo', 'Cu', 'V', 'Al', 'N', 'Nb + Ta', 'Temperature (°C)']

# ---- Input fields ----
st.subheader("Alloy Composition (wt%)")
col1, col2, col3 = st.columns(3)

with col1:
    C = st.number_input("C", min_value=0.0, value=0.15, step=0.01, format="%.4f")
    Si = st.number_input("Si", min_value=0.0, value=0.25, step=0.01, format="%.4f")
    Mn = st.number_input("Mn", min_value=0.0, value=0.80, step=0.01, format="%.4f")
    P = st.number_input("P", min_value=0.0, value=0.01, step=0.001, format="%.4f")
    S = st.number_input("S", min_value=0.0, value=0.01, step=0.001, format="%.4f")

with col2:
    Ni = st.number_input("Ni", min_value=0.0, value=0.10, step=0.01, format="%.4f")
    Cr = st.number_input("Cr", min_value=0.0, value=0.20, step=0.01, format="%.4f")
    Mo = st.number_input("Mo", min_value=0.0, value=0.05, step=0.01, format="%.4f")
    Cu = st.number_input("Cu", min_value=0.0, value=0.05, step=0.01, format="%.4f")
    V = st.number_input("V", min_value=0.0, value=0.00, step=0.001, format="%.4f")

with col3:
    Al = st.number_input("Al", min_value=0.0, value=0.02, step=0.001, format="%.4f")
    N = st.number_input("N", min_value=0.0, value=0.01, step=0.001, format="%.4f")
    NbTa = st.number_input("Nb + Ta", min_value=0.0, value=0.00, step=0.001, format="%.4f")

st.subheader("Test Condition")
Temperature = st.number_input("Temperature (°C)", value=25, step=1)

# ---- Predict button ----
if st.button("Predict Tensile Strength"):
    input_data = pd.DataFrame([[C, Si, Mn, P, S, Ni, Cr, Mo, Cu, V, Al, N, NbTa, Temperature]],
                               columns=FEATURES)

    prediction = xgb_model.predict(input_data)[0]

    st.success(f"Predicted Tensile Strength: **{prediction:.2f} MPa**")

    # ---- SHAP explanation for this specific prediction ----
    st.subheader("Why this prediction?")

    with st.spinner("Computing SHAP explanation..."):
        # patch base_score for XGBoost 3.x + SHAP compatibility
        booster = xgb_model.get_booster()
        config = json.loads(booster.save_config())
        config['learner']['learner_model_param']['base_score'] = '0.5'
        booster.load_config(json.dumps(config))

        explainer = shap.TreeExplainer(xgb_model)
        shap_values = explainer(input_data)

    fig, ax = plt.subplots(figsize=(8, 5))
    shap.plots.waterfall(shap_values[0], show=False)
    st.pyplot(fig)

    st.caption(
        "This waterfall plot shows how each input pushed the prediction above or below "
        "the model's baseline (average) prediction."
    )

st.markdown("---")
st.caption(
    "Model: XGBoost Regressor trained on the MatNavi Low-Alloy Steels dataset. "
    "Grouped train/test split by alloy code to prevent data leakage."
)