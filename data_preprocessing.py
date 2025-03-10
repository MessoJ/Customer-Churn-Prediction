import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import joblib
import os

def preprocess_data():
    # Load data
    df = pd.read_csv('data/telco_churn.csv')
    
    # Handle TotalCharges
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())
    
    # Define columns
    categorical_cols = ['InternetService', 'Contract', 'PaymentMethod']
    numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    target_col = ['Churn']
    
    # Create column transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(), categorical_cols)
        ],
        remainder='passthrough'
    )
    
    # Fit and transform
    processed_data = preprocessor.fit_transform(df)
    
    # Get correct feature names
    feature_names = preprocessor.get_feature_names_out(
        input_features=df.columns.tolist()
    )
    
    # Create DataFrame with proper columns
    processed_df = pd.DataFrame(processed_data, columns=feature_names)
    
    # Create the 'models' directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Save data
    processed_df.to_csv('data/processed_data.csv', index=False)
    joblib.dump(preprocessor, 'models/preprocessor.joblib')
    print(f"Processed data shape: {processed_df.shape}")
    print("Data preprocessing completed.")

if __name__ == "__main__":
    preprocess_data()
