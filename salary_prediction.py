import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
import logging
import os
import joblib
from io import StringIO, BytesIO

# ────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

np.random.seed(42)

EDUCATION_ORDER = [['High School', 'Bachelor', 'Master', 'PhD']]
EDU_MAP = {"High School": 0, "Bachelor": 1, "Master": 2, "PhD": 3}

REQUIRED_COLUMNS = [
    'MonthlySalary', 'YearsAtCompany', 'PerformanceRating',
    'MonthlyHoursWorked', 'EducationLevel', 'Department'
]

def load_data(source=None):
    """
    source can be:
    - path (str) → local file (for testing)
    - file-like object (BytesIO/StringIO) → uploaded file
    - None → raise informative error
    """
    try:
        if source is None:
            raise ValueError("No data source provided. Please upload a CSV file.")
        
        if isinstance(source, str):  # path
            if not os.path.exists(source):
                raise FileNotFoundError(f"File not found: {source}")
            df = pd.read_csv(source)
        else:  # file-like
            df = pd.read_csv(source)
        
        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"Dataset missing required columns: {', '.join(missing)}")
        
        logger.info(f"Dataset loaded successfully ({len(df)} rows).")
        return df
    except Exception as e:
        logger.error(f"Data loading failed: {e}")
        raise

def clean_and_engineer_features(df):
    try:
        df_cleaned = df.drop(columns=['EmployeeID'], errors='ignore')
        
        # Imputation
        impute_cols = ['YearsAtCompany', 'PerformanceRating', 'MonthlyHoursWorked']
        imputer = SimpleImputer(strategy='median')
        df_cleaned[impute_cols] = imputer.fit_transform(df_cleaned[impute_cols])
        
        # Ordinal encoding
        if 'EducationLevel' in df_cleaned.columns:
            enc = OrdinalEncoder(categories=EDUCATION_ORDER, handle_unknown='use_encoded_value', unknown_value=-1)
            df_cleaned['EducationLevel'] = enc.fit_transform(df_cleaned[['EducationLevel']])
        
        # One-hot encoding
        if 'Department' in df_cleaned.columns:
            df_cleaned = pd.get_dummies(df_cleaned, columns=['Department'], prefix='Dept', drop_first=False)
        
        logger.info("Feature engineering complete.")
        return df_cleaned
    except Exception as e:
        logger.error(f"Feature engineering failed: {e}")
        raise

def train_and_evaluate(df_cleaned):
    try:
        if 'MonthlySalary' not in df_cleaned.columns:
            raise ValueError("Target 'MonthlySalary' missing after cleaning.")
        
        X = df_cleaned.drop(columns=['MonthlySalary'])
        y = df_cleaned['MonthlySalary']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        model = RandomForestRegressor(n_estimators=120, random_state=42, n_jobs=-1)
        logger.info("Training model...")
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        results = {
            "R-squared Score": f"{r2_score(y_test, y_pred):.4f}",
            "Mean Absolute Error (MAE)": f"${mean_absolute_error(y_test, y_pred):,.0f}",
            "Mean Squared Error (MSE)": f"{mean_squared_error(y_test, y_pred):,.0f}"
        }
        
        for k, v in results.items():
            logger.info(f"{k}: {v}")
        
        # Save model temporarily for download
        model_bytes = BytesIO()
        joblib.dump(model, model_bytes)
        model_bytes.seek(0)
        
        return model, results, X.columns.tolist(), model_bytes
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

def prepare_prediction_input(input_dict, training_columns):
    """Convert user input to DataFrame aligned with training features"""
    df_input = pd.DataFrame([input_dict])
    
    # Encode education
    df_input['EducationLevel'] = df_input['EducationLevel'].map(EDU_MAP).fillna(-1)
    
    # One-hot department
    df_input = pd.get_dummies(df_input, columns=['Department'], prefix='Dept')
    
    # Align columns
    for col in training_columns:
        if col not in df_input.columns:
            df_input[col] = 0
    
    df_input = df_input[training_columns]
    return df_input