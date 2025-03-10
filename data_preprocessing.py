import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import joblib
import os

def preprocess_data():
    # Create directories first
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # Load data
    df = pd.read_csv('data/telco_churn.csv')
    
    # Handle TotalCharges
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())
    
    # Define columns
    categorical_cols = ['InternetService', 'Contract', 'PaymentMethod']
    numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    binary_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
    
    # Convert binary columns first
    df[binary_cols] = df[binary_cols].replace({'Yes': 1, 'No': '0'}).astype(int)
    
    # Create column transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(), categorical_cols)
        ],
        remainder='passthrough'
    )
    
    # Fit and transform
    processed_data = preprocessor.fit_transform(df)
    
    # Get feature names
    feature_names = preprocessor.get_feature_names_out(
        input_features=df.columns.tolist()
    )
    
    # Create DataFrame
    processed_df = pd.DataFrame(processed_data, columns=feature_names)
    
    # Save data
    processed_df.to_csv('data/processed_data.csv', index=False)
    joblib.dump(preprocessor, 'models/preprocessor.joblib')
    
    print(f"Processed data shape: {processed_df.shape}")
    print("Data preprocessing completed.")

if __name__ == "__main__":
    preprocess_data()
