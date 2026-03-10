import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
import logging
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

np.random.seed(42)  # Global seed for reproducibility

def load_data(file_path=r'C:\Users\USER\Desktop\salary-prediction-model\salary-prediction-model\1000_salary_dataset.csv', url=None):
    try: 
        if url:
            df = pd.read_csv(url)
        else:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Dataset '{file_path}' not found. Check the path!")
            df = pd.read_csv(file_path)
        logger.info("Dataset loaded successfully.")
        return df
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

def clean_and_engineer_features(df):
    try:
        # Drop irrelevant column
        df_cleaned = df.drop(columns=['EmployeeID'], errors='ignore')
        
        # Impute missing values
        impute_cols = ['YearsAtCompany', 'PerformanceRating', 'MonthlyHoursWorked']
        missing_cols = [col for col in impute_cols if col not in df_cleaned.columns]
        if missing_cols:
            raise ValueError(f"Missing columns for imputation: {missing_cols}")
        imputer = SimpleImputer(strategy='median')
        df_cleaned[impute_cols] = imputer.fit_transform(df_cleaned[impute_cols])
        
        # Ordinal Encoding for EducationLevel
        education_order = [['High School', 'Bachelor', 'Master', 'PhD']]
        ord_enc = OrdinalEncoder(categories=education_order, handle_unknown='use_encoded_value', unknown_value=-1)
        if 'EducationLevel' in df_cleaned.columns:
            df_cleaned['EducationLevel'] = ord_enc.fit_transform(df_cleaned[['EducationLevel']])
        
        # One-Hot Encoding for Department
        if 'Department' in df_cleaned.columns:
            df_cleaned = pd.get_dummies(df_cleaned, columns=['Department'], prefix='Dept')
        
        logger.info("Data cleaning and feature engineering complete.")
        return df_cleaned
    except Exception as e:
        logger.error(f"Error in cleaning/engineering: {e}")
        raise

def train_and_evaluate(df_cleaned):
    try:
        # Split data
        if 'MonthlySalary' not in df_cleaned.columns:
            raise ValueError("Target column 'MonthlySalary' missing.")
        X = df_cleaned.drop(columns=['MonthlySalary'])
        y = df_cleaned['MonthlySalary']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        logger.info("Training Random Forest Regressor model...")
        rf_model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = rf_model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        results = {
            "R-squared Score": f"{r2:.4f}",
            "Mean Absolute Error (MAE)": f"${mae:,.2f}",
            "Mean Squared Error (MSE)": f"{mse:,.2f}"
        }
        logger.info("\n--- Model Evaluation Results ---")
        for key, val in results.items():
            logger.info(f"{key}: {val}")
        
        return rf_model, results, df_cleaned.head()
    except Exception as e:
        logger.error(f"Error in training/evaluation: {e}")
        raise

# Example usage
if __name__ == "__main__":
    # Now uses the absolute path provided
    df = load_data()  
    
    # We can skip the example URL load to avoid confusion
    # load_data(url="https://example.com/dataset.csv") 
    
    df_cleaned = clean_and_engineer_features(df)
    model, results, sample = train_and_evaluate(df_cleaned)
    logger.info("\nCleaned Feature Sample:")
    logger.info(sample)