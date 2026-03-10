import streamlit as st
import pandas as pd
import numpy as np
import time
from salary_prediction import load_data, clean_and_engineer_features, train_and_evaluate

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="AI Salary Predictor",
    page_icon="📊",
    layout="wide"
)

# --- STYLING ---
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    </style>
    """, unsafe_allow_html=True)

# --- HEADER ---
col1, col2 = st.columns([1, 5])
with col1:
    st.title("💰")
with col2:
    st.title("Employee Salary Intelligence Portal")
    st.markdown("Predict competitive monthly salaries using machine learning driven by your historical company data.")

st.divider()

# --- MODEL INITIALIZATION (CACHED) ---
@st.cache_resource
def get_trained_model():
    # Use the absolute path defined in your salary_prediction script
    df = load_data() 
    df_cleaned = clean_and_engineer_features(df)
    model, results, feature_names = train_and_evaluate(df_cleaned)
    return model, results, feature_names

try:
    with st.status("Initializing Predictive Engine...", expanded=False) as status:
        model, metrics, feature_cols = get_trained_model()
        status.update(label="✅ System Ready", state="complete", expanded=False)
except Exception as e:
    st.error(f"❌ Initialization Failed: {e}")
    st.stop()

# --- SIDEBAR INPUTS ---
st.sidebar.header("📋 Employee Profile")
st.sidebar.markdown("Adjust the parameters below:")

# Demographics & Experience
age = st.sidebar.number_input("Age", 18, 70, 30)
total_exp = st.sidebar.slider("Total Years of Experience", 0, 50, 10)
years_at_co = st.sidebar.slider("Years at Current Company", 0, total_exp, 3)

# Education & Department (Refined from your CSV)
education = st.sidebar.selectbox("Education Level", ["High School", "Bachelor", "Master", "PhD"])
department = st.sidebar.selectbox("Department", ["Engineering", "Sales", "HR", "Operations", "Marketing", "Finance"])

# Performance & Workload
perf_rating = st.sidebar.select_slider("Performance Rating", options=[1, 2, 3, 4, 5], value=3)
hours_worked = st.sidebar.number_input("Average Monthly Hours", 80, 300, 160)

# --- MAIN DISPLAY ---
left_info, right_pred = st.columns([1, 1], gap="large")

with left_info:
    st.subheader("📊 Model Insight")
    st.write("Current model performance metrics based on your uploaded dataset:")
    
    m_col1, m_col2 = st.columns(2)
    m_col1.metric("R-Squared Score", metrics['R-squared Score'])
    m_col2.metric("Avg. Prediction Error", metrics['Mean Absolute Error (MAE)'])
    
    with st.expander("About this model"):
        st.write("""
            This prediction is generated using a **Random Forest Regressor**. 
            It analyzes patterns across all employee attributes to estimate the 
            most likely monthly salary.
        """)

with right_pred:
    st.subheader("🎯 Prediction Result")
    st.write("Click the button below to generate a salary estimate for the profile defined in the sidebar.")
    
    if st.button("Generate Prediction", type="primary", use_container_width=True):
        # 1. Create Input DataFrame
        input_row = pd.DataFrame({
            'Age': [age],
            'YearsExperience': [total_exp],
            'YearsAtCompany': [years_at_co],
            'Department': [department],
            'EducationLevel': [education],
            'PerformanceRating': [perf_rating],
            'MonthlyHoursWorked': [hours_worked]
        })

        # 2. Match Feature Engineering Logic
        # Ordinal Encoding for Education
        edu_map = {"High School": 0, "Bachelor": 1, "Master": 2, "PhD": 3}
        input_row['EducationLevel'] = input_row['EducationLevel'].map(edu_map)

        # One-Hot Encoding for Department
        input_row = pd.get_dummies(input_row, columns=['Department'], prefix='Dept')

        # Align columns with training data
        X_cols = [c for c in feature_cols if c != 'MonthlySalary']
        for col in X_cols:
            if col not in input_row.columns:
                input_row[col] = 0
        
        input_row = input_row[X_cols] # Ensure correct order

        # 3. Make Prediction
        with st.spinner('Calculating...'):
            time.sleep(0.5) # For visual effect
            prediction = model.predict(input_row)[0]

        st.balloons()
        st.success(f"### Estimated Monthly Salary: **${prediction:,.2f}**")
        st.caption("Confidence intervals and individual biases are not accounted for in this estimate.")
    else:
        st.info("Waiting for input...")

st.divider()
st.caption("Salary Predictor v2.0 | Data Analytics Dashboard")