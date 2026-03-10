import streamlit as st
import pandas as pd
import os
from salary_prediction import (
    load_data, clean_and_engineer_features, train_and_evaluate,
    prepare_prediction_input, EDU_MAP
)

# ──────────────────────────────────────── Page config
st.set_page_config(
    page_title="AI Salary Predictor",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ──────────────────────────────────────── Custom styling
st.markdown("""
    <style>
    .stApp { background-color: #f8f9fa; }
    .metric-box { background: white; padding: 1rem; border-radius: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }
    </style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────── Header
st.title("💰 Employee Salary Intelligence Portal")
st.markdown("Upload your company salary data and get ML-powered salary estimates for any employee profile.")

# ──────────────────────────────────────── File upload & data loading
uploaded_file = st.file_uploader(
    "Upload salary dataset (CSV)",
    type="csv",
    help="Must contain: MonthlySalary, EducationLevel, Department, YearsAtCompany, PerformanceRating, MonthlyHoursWorked"
)

sample_path = "1000_salary_dataset.csv"
use_sample = False

if uploaded_file is not None:
    try:
        df = load_data(uploaded_file)
    except Exception as e:
        st.error(f"Upload failed: {e}")
        st.stop()
else:
    if os.path.exists(sample_path):
        try:
            df = load_data(sample_path)
            use_sample = True
            st.info("Using built-in sample dataset (no file uploaded).")
        except Exception as e:
            st.warning(f"Sample dataset failed: {e}. Please upload your own CSV.")
            st.stop()
    else:
        st.warning("Please upload your salary dataset CSV to begin.")
        st.stop()

# ──────────────────────────────────────── Cached processing
@st.cache_data(show_spinner="Cleaning & preparing data...")
def cached_clean_data(df):
    return clean_and_engineer_features(df)

@st.cache_resource(show_spinner="Training Random Forest model...")
def cached_train_model(df_cleaned):
    return train_and_evaluate(df_cleaned)

with st.spinner("Preparing model (only once)..."):
    try:
        df_cleaned = cached_clean_data(df)
        model, metrics, feature_cols, model_bytes = cached_train_model(df_cleaned)
    except Exception as e:
        st.error(f"Model preparation failed: {e}")
        st.stop()

# ──────────────────────────────────────── Sidebar inputs
with st.sidebar:
    st.header("Employee Profile")
    
    age = st.number_input("Age", 18, 70, 32)
    total_exp = st.slider("Total Years of Experience", 0, 50, 10)
    years_at_company = st.slider("Years at Current Company", 0, total_exp, min(5, total_exp))
    
    education = st.selectbox("Education Level", list(EDU_MAP.keys()), index=1)
    department = st.selectbox("Department", ["Engineering", "Sales", "HR", "Operations", "Marketing", "Finance"])
    
    perf_rating = st.select_slider("Performance Rating", options=[1,2,3,4,5], value=3)
    monthly_hours = st.number_input("Average Monthly Hours Worked", 80, 300, 168)

# ──────────────────────────────────────── Main area ─ columns
left, right = st.columns([1, 1.3], gap="large")

with left:
    st.subheader("Model Performance")
    m1, m2, m3 = st.columns(3)
    m1.metric("R² Score", metrics["R-squared Score"])
    m2.metric("MAE", metrics["Mean Absolute Error (MAE)"])
    m3.metric("MSE", metrics["Mean Squared Error (MSE)"])
    
    st.download_button(
        label="Download Trained Model",
        data=model_bytes,
        file_name="salary_random_forest_model.pkl",
        mime="application/octet-stream"
    )

with right:
    st.subheader("Salary Prediction")
    
    if st.button("Generate Salary Estimate", type="primary", use_container_width=True):
        input_data = {
            'Age': age,
            'YearsExperience': total_exp,           # assuming you have this column or map it
            'YearsAtCompany': years_at_company,
            'EducationLevel': education,
            'Department': department,
            'PerformanceRating': perf_rating,
            'MonthlyHoursWorked': monthly_hours
        }
        
        try:
            X_input = prepare_prediction_input(input_data, feature_cols)
            prediction = model.predict(X_input)[0]
            
            st.balloons()
            st.success(f"**Estimated Monthly Salary:  ${prediction:,.0f}**")
            st.caption("Based on Random Forest model trained on your data.")
        except Exception as e:
            st.error(f"Prediction failed: {e}")

st.divider()
st.caption("Salary Predictor • Powered by Random Forest • v2.1")