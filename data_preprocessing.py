import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
import joblib


# Modify the ColumnTransformer to preserve column names
def preprocess_data():
    df = pd.read_csv('data/telco_churn.csv')
    
    # Handle TotalCharges conversion
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())
    
    # Define columns to preserve
    numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    categorical_cols = ['InternetService', 'Contract', 'PaymentMethod']
    
    # Create preprocessor that keeps numeric columns
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(), categorical_cols)
        ],
        remainder='passthrough'  # Keeps all non-transformed columns
    )
    
    # Fit and transform
    processed_data = preprocessor.fit_transform(df)
    
    # Get feature names
    feature_names = list(preprocessor.named_transformers_['cat'].get_feature_names_out())
    feature_names += numeric_cols + ['Churn']  # Add remaining columns
    
    # Save with headers
    processed_df = pd.DataFrame(processed_data, columns=feature_names)
    processed_df.to_csv('data/processed_data.csv', index=False)
    joblib.dump(preprocessor, 'models/preprocessor.joblib')

    print("Data preprocessing completed.")

if __name__ == "__main__":
    preprocess_data()
