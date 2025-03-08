import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
import joblib

def preprocess_data():
    # Load data
    df = pd.read_csv('data/telco_churn.csv')
    
    # Handle missing values
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)
    
    # Encode categorical variables
    binary_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn']
    label_encoder = LabelEncoder()
    for col in binary_cols:
        df[col] = label_encoder.fit_transform(df[col])
    
    # One-hot encode other categorical features
    categorical_cols = ['InternetService', 'Contract', 'PaymentMethod']
    preprocessor = ColumnTransformer(
        transformers=[('cat', OneHotEncoder(), categorical_cols)],
        remainder='passthrough'
    )
    processed_data = preprocessor.fit_transform(df)
    
    # Save processed data and preprocessor
    pd.DataFrame(processed_data).to_csv('data/processed_data.csv', index=False)
    joblib.dump(preprocessor, 'models/preprocessor.joblib')
    print("Data preprocessing completed.")

if __name__ == "__main__":
    preprocess_data()
