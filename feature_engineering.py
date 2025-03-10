import pandas as pd
import numpy as np
import os

def engineer_features():
    # Ensure directory exists
    os.makedirs('data', exist_ok=True)
    
    # Read processed data
    df = pd.read_csv('data/processed_data.csv')
    
    # Handle TotalCharges with correct column name
    df['remainder__TotalCharges'] = pd.to_numeric(
        df['remainder__TotalCharges'].replace(' ', np.nan),
        errors='coerce'
    )
    
    # Create new features with proper column names
    df['AvgMonthlyCharge'] = df['remainder__TotalCharges'] / (df['remainder__tenure'] + 1e-6)
    df['HighValueFlag'] = (df['remainder__MonthlyCharges'] > 70).astype(int)
    df['TenureToChargeRatio'] = df['remainder__tenure'] / (df['remainder__MonthlyCharges'] + 1e-6)
    
    # Fix service diversity calculation
    service_cols = [
        'remainder__OnlineSecurity',
        'remainder__TechSupport',
        'remainder__StreamingTV'
    ]
    df['ServiceDiversity'] = df[service_cols].replace({1: 1, 0: 0, 'No internet service': 0}).sum(axis=1)
    
    # Handle remaining categorical columns
    remaining_cats = [
        'remainder__MultipleLines',
        'remainder__OnlineBackup',
        'remainder__DeviceProtection',
        'remainder__StreamingMovies'
    ]
    df[remaining_cats] = df[remaining_cats].replace({'Yes': 1, 'No': 0, 'No internet service': 0})
    
    # Save engineered data
    df.to_csv('data/engineered_data.csv', index=False)
    print("Feature engineering completed.")
    print("Engineered columns:", df.columns.tolist())

if __name__ == "__main__":
    engineer_features()
