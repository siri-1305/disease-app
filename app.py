import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os

st.set_page_config(page_title="Heart Disease Prediction", layout="wide")

# --- Load model, scaler, and features ---
MODEL_PATH = "heart_rf_model.pkl"
SCALER_PATH = "heart_scaler.pkl"
FEAT_PATH = "feature_columns.json"

if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH) or not os.path.exists(FEAT_PATH):
    st.error("Missing files! Make sure model, scaler, and feature_columns.json are in the same folder.")
    st.stop()

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
feature_cols = json.load(open(FEAT_PATH))

st.title("❤️ Heart Disease Prediction App")
st.caption("Demo app built with Streamlit · Not medical advice")

# --- UI Form ---
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=20, max_value=100, value=50)
    trestbps = st.number_input("Resting BP (trestbps)", min_value=80, max_value=220, value=120)
    chol = st.number_input("Cholesterol (chol)", min_value=100, max_value=600, value=200)
    thalach = st.number_input("Max Heart Rate (thalach)", min_value=70, max_value=220, value=150)
    ca = st.number_input("Number of major vessels (ca)", min_value=0, max_value=4, value=0)

with col2:
    oldpeak = st.number_input("ST depression (oldpeak)", min_value=0.0, max_value=6.0, value=1.0, step=0.1)
    sex = st.selectbox("Sex", ["Male", "Female"])
    cp = st.selectbox("Chest Pain Type", ["typical angina","atypical angina","non-anginal","asymptomatic"])
    exang = st.selectbox("Exercise induced angina", ["False", "True"])
    fbs = st.selectbox("Fasting blood sugar > 120 mg/dl?", ["False", "True"])
    restecg = st.selectbox("Resting ECG", ["normal","lv hypertrophy","st-t abnormality"])
    slope = st.selectbox("Slope", ["flat","upsloping","downsloping"])
    thal = st.selectbox("Thalassemia", ["normal","fixed defect","reversable defect"])
    dataset = st.selectbox("Dataset origin", ["Cleveland","Hungary","Switzerland","VA Long Beach"])

# --- Build input row ---
def build_row():
    row = {c: 0 for c in feature_cols}  # start with zeros
    
    # numeric
    row["id"] = 0
    row["age"] = age
    row["trestbps"] = trestbps
    row["chol"] = chol
    row["thalch"] = thalach
    row["oldpeak"] = oldpeak
    row["ca"] = ca

    # sex
    row["sex_Female"] = 1 if sex == "Female" else 0
    row["sex_Male"] = 1 if sex == "Male" else 0

    # dataset
    ds_map = {
        "Cleveland":"dataset_Cleveland",
        "Hungary":"dataset_Hungary",
        "Switzerland":"dataset_Switzerland",
        "VA Long Beach":"dataset_VA Long Beach"
    }
    if ds_map.get(dataset) in row:
        row[ds_map[dataset]] = 1

    # chest pain
    cp_map = {
        "typical angina":"cp_typical angina",
        "atypical angina":"cp_atypical angina",
        "non-anginal":"cp_non-anginal",
        "asymptomatic":"cp_asymptomatic"
    }
    if cp_map.get(cp) in row:
        row[cp_map[cp]] = 1

    # fbs
    row["fbs_True"] = 1 if fbs == "True" else 0
    row["fbs_False"] = 0 if fbs == "True" else 1

    # restecg
    rest_map = {
        "normal":"restecg_normal",
        "lv hypertrophy":"restecg_lv hypertrophy",
        "st-t abnormality":"restecg_st-t abnormality"
    }
    if rest_map.get(restecg) in row:
        row[rest_map[restecg]] = 1

    # exang
    row["exang_True"] = 1 if exang == "True" else 0
    row["exang_False"] = 0 if exang == "True" else 1

    # slope
    slope_map = {
        "downsloping":"slope_downsloping",
        "flat":"slope_flat",
        "upsloping":"slope_upsloping"
    }
    if slope_map.get(slope) in row:
        row[slope_map[slope]] = 1

    # thal
    thal_map = {
        "fixed defect":"thal_fixed defect",
        "normal":"thal_normal",
        "reversable defect":"thal_reversable defect"
    }
    if thal_map.get(thal) in row:
        row[thal_map[thal]] = 1

    return pd.DataFrame([row])

# --- Prediction ---
if st.button("Predict"):
    X_user = build_row()
    X_user = X_user[feature_cols]  # ensure column order
    try:
        X_scaled = scaler.transform(X_user)
        proba = model.predict_proba(X_scaled)[0,1]
        pred = int(proba >= 0.5)
    except Exception as e:
        st.error(f"Error during prediction: {e}")
    else:
        st.markdown("### Prediction Result")
        if pred == 1:
            st.error(f"⚠️ Higher risk of heart disease — probability {proba:.2f}")
        else:
            st.success(f"✅ Lower risk of heart disease — probability {proba:.2f}")
        st.caption("Note: This is only a demo model, not medical advice.")
