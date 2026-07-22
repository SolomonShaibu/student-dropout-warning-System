import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Page configuration
st.set_page_config(
    page_title="Student Dropout Risk Predictor", 
    page_icon="🎓", 
    layout="centered"
)

st.title("🎓 AI-Driven Student Dropout Risk Predictor")
st.markdown("**Project:** AI-Driven Prediction for Students Drop-Out Risk in Nigeria Universities")
st.markdown("---")

# Load model and synthetic training pipeline based on project metrics
@st.cache_resource
def load_training_model():
    np.random.seed(42)
    n = 1000
    X_train = pd.DataFrame({
        'UTME_Score': np.random.randint(150, 400, n),
        'CGPA_Y1': np.random.uniform(1.0, 5.0, n),
        'CGPA_Y2': np.random.uniform(1.0, 5.0, n),
        'Credit_Velocity': np.random.uniform(0.5, 1.0, n),
        'Tuition_Delay_Days': np.random.randint(0, 60, n),
        'Financial_Stress': np.random.uniform(1.0, 5.0, n)
    })
    # Target label logic reflecting academic and socioeconomic stress coefficients
    y_train = ((X_train['CGPA_Y2'] < 2.0) | (X_train['Financial_Stress'] > 3.5) | (X_train['Tuition_Delay_Days'] > 30)).astype(int)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

model = load_training_model()

st.subheader("Enter Student Profile Parameters")
col1, col2 = st.columns(2)

with col1:
    utme_score = st.slider("UTME Score", 150, 400, 238)
    cgpa_y1 = st.number_input("CGPA Year 1", 0.0, 5.0, 2.94, 0.01)
    cgpa_y2 = st.number_input("CGPA Year 2", 0.0, 5.0, 2.94, 0.01)

with col2:
    credit_velocity = st.slider("Credit Velocity Ratio", 0.0, 1.0, 0.88, 0.01)
    tuition_delay = st.number_input("Tuition Delay Days", 0, 90, 14)
    financial_stress = st.slider("Financial Stress Index", 1.0, 5.0, 2.18, 0.1)

st.markdown("---")

if st.button("Evaluate Dropout Risk", type="primary"):
    input_df = pd.DataFrame({
        'UTME_Score': [utme_score],
        'CGPA_Y1': [cgpa_y1],
        'CGPA_Y2': [cgpa_y2],
        'Credit_Velocity': [credit_velocity],
        'Tuition_Delay_Days': [tuition_delay],
        'Financial_Stress': [financial_stress]
    })
    
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.subheader("Prediction Analysis Results")
    if prediction == 1:
        st.error(f"⚠️ **High Risk of Dropout** (Risk Probability: {probability * 100:.2f}%)")
        st.warning("**Recommended Action:** Trigger early-warning advisory, deploy financial relief options, and assign academic tutoring.")
    else:
        st.success(f"✅ **Low Risk / Retained** (Dropout Probability: {probability * 100:.2f}%)")
        st.info("**Status:** Student is maintaining stable academic progression and integration metrics.")
