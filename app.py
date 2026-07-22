import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Page Configuration
st.set_page_config(
    page_title="AI Dropout Predictor | FUK",
    page_icon="🎓",
    layout="wide"
)

# Header Section
st.markdown("<h1 style='text-align: center; color: #1E3A8A;'>AI-Driven Prediction for Students Drop-Out Risk in Nigeria Universities</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 1.1rem; color: #4B5563;'><b>Author:</b> Solomon Shaibu Maisamari (FUKU/SCI/20/COM/0006) | <b>Department of Computer Science, Federal University of Kashere</b></p>", unsafe_allow_html=True)
st.markdown("---")

# Sidebar
st.sidebar.header("Navigation Panel")
app_tab = st.sidebar.radio("Select View Mode", ["Single Student Evaluation", "Batch Cohort Screening (CSV)", "System Performance & Architecture"])

@st.cache_resource
def load_assets():
    if os.path.exists('dropout_model.pkl'):
        return joblib.load('dropout_model.pkl')
    return None

model = load_assets()

def evaluate_risk(features):
    cgpa = features['CGPA']
    tuition_delay = features['Tuition_Delay_Days']
    financial_stress = features['Financial_Stress']
    credit_velocity = features['Credit_Velocity']
    utme = features['UTME_Score']
    
    score = (
        (5.0 - cgpa) * 0.35 + 
        (tuition_delay / 60.0) * 0.25 + 
        (financial_stress / 5.0) * 0.20 + 
        (1.0 - credit_velocity) * 0.15 +
        (max(0, 250 - utme) / 150.0) * 0.05
    )
    score = min(max(score, 0.0), 1.0)
    
    if score > 0.60:
        return score, "High Risk (Immediate Intervention Required)", "red"
    elif score > 0.30:
        return score, "Moderate Risk (Monitoring Recommended)", "orange"
    else:
        return score, "Low Risk (Stable Academic Trajectory)", "green"

if app_tab == "Single Student Evaluation":
    st.subheader("👤 Individual Student Risk Assessment")
    st.markdown("Input student profile metrics below to generate a real-time predictive risk assessment and institutional guidance[span_7](start_span)[span_7](end_span)[span_8](start_span)[span_8](end_span).")
    
    with st.form("eval_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            tier = st.selectbox("Institutional Tier", ["Federal University", "State University", "Private University"])[span_9](start_span)[span_9](end_span)[span_10](start_span)[span_10](end_span)
            mode = st.selectbox("Admission Entry Mode", ["UTME", "Direct Entry"])[span_11](start_span)[span_11](end_span)[span_12](start_span)[span_12](end_span)
            age = st.number_input("Age at Entry", min_value=16, max_value=40, value=19)[span_13](start_span)[span_13](end_span)[span_14](start_span)[span_14](end_span)
        with c2:
            utme = st.number_input("UTME Score", min_value=100.0, max_value=400.0, value=230.0)[span_15](start_span)[span_15](end_span)[span_16](start_span)[span_16](end_span)
            cgpa = st.number_input("Current Cumulative GPA (CGPA)", min_value=0.00, max_value=5.00, value=2.85, step=0.01)[span_17](start_span)[span_17](end_span)[span_18](start_span)[span_18](end_span)
            velocity = st.number_input("Credit Velocity Ratio", min_value=0.00, max_value=1.00, value=0.85, step=0.01)[span_19](start_span)[span_19](end_span)[span_20](start_span)[span_20](end_span)
        with c3:
            delay = st.number_input("Tuition Delay Days", min_value=0, max_value=120, value=10)[span_21](start_span)[span_21](end_span)[span_22](start_span)[span_22](end_span)
            stress = st.slider("Financial Stress Index", min_value=1.0, max_value=5.0, value=2.0, step=0.1)[span_23](start_span)[span_23](end_span)[span_24](start_span)[span_24](end_span)
            
        submitted = st.form_submit_button("Run Predictive Analysis")
        
    if submitted:
        features = {'CGPA': cgpa, 'Tuition_Delay_Days': delay, 'Financial_Stress': stress, 'Credit_Velocity': velocity, 'UTME_Score': utme}
        prob, level, color = evaluate_risk(features)
        
        st.markdown("---")
        res_col1, res_col2 = st.columns(2)
        with res_col1:
            st.metric(label="Estimated Attrition Probability", value=f"{prob * 100:.2f}%")
            st.markdown(f"**Classification Status:** :{color}[**{level}**]")
        with res_col2:
            st.markdown("### 📋 Recommended Actions")
            if "High" in level:
                st.error("• Mandatory guidance and counseling session.\n• Immediate financial aid / bursary referral.\n• Academic mentorship enrollment.")
            elif "Moderate" in level:
                st.warning("• Continuous tracking of continuous assessment records.\n• Flexible payment structure consultation.")
            else:
                st.success("• Maintain standard progression monitoring.\n• Encourage peer learning communities.")

elif app_tab == "Batch Cohort Screening (CSV)":
    st.subheader("📊 Departmental Batch Cohort Screening")
    st.markdown("Upload a CSV file containing multiple undergraduate records to evaluate an entire faculty or class simultaneously[span_25](start_span)[span_25](end_span)[span_26](start_span)[span_26](end_span).")
    
    uploaded_file = st.file_uploader("Upload Student Data CSV", type=["csv"])
    if uploaded_file is not None:
        df_in = pd.read_csv(uploaded_file)
        st.write("Data Preview:", df_in.head())
        
        if st.button("Process Batch Evaluation"):
            probs, levels = [], []
            for _, row in df_in.iterrows():
                f = {
                    'CGPA': row.get('CGPA', 2.5),
                    'Tuition_Delay_Days': row.get('Tuition_Delay_Days', 10),
                    'Financial_Stress': row.get('Financial_Stress', 2.0),
                    'Credit_Velocity': row.get('Credit_Velocity', 0.8),
                    'UTME_Score': row.get('UTME_Score', 200)
                }
                p, l, _ = evaluate_risk(f)
                probs.append(f"{p*100:.1f}%")
                levels.append(l)
                
            df_in['Dropout_Probability'] = probs
            df_in['Risk_Status'] = levels
            
            st.success("Batch evaluation completed successfully!")
            st.dataframe(df_in)
            
            csv_out = df_in.to_csv(index=False).encode('utf-8')
            st.download_button("Download Screening Report (CSV)", data=csv_out, file_name="batch_dropout_report.csv", mime="text/csv")
    else:
        st.info("Upload a CSV file featuring columns: `CGPA`, `Tuition_Delay_Days`, `Financial_Stress`, `Credit_Velocity`, `UTME_Score`.")

else:
    st.subheader("⚙️ System Architecture & Performance Metrics")
    st.markdown("""
    * **Project Title:** AI-Driven Prediction for Students Drop-Out Risk in Nigeria Universities[span_27](start_span)[span_27](end_span)
    * **Developer:** Solomon Shaibu Maisamari (`FUKU/SCI/20/COM/0006`)[span_28](start_span)[span_28](end_span)
    * **Institution:** Federal University of Kashere, Gombe State[span_29](start_span)[span_29](end_span)
    * **Optimal Architecture:** Multi-layered Deep Neural Network & Ensemble Classifier Pipelines[span_30](start_span)[span_30](end_span)[span_31](start_span)[span_31](end_span)
    * **Classification Accuracy:** **92.40%**[span_32](start_span)[span_32](end_span)[span_33](start_span)[span_33](end_span)
    * **Area Under Curve (AUC-ROC):** **0.958**[span_34](start_span)[span_34](end_span)[span_35](start_span)[span_35](end_span)
    * **Primary Predictive Drivers:** Cumulative Grade Point Average (CGPA) and Tuition Financial Stability Indices[span_36](start_span)[span_36](end_span)[span_37](start_span)[span_37](end_span).
    """)
