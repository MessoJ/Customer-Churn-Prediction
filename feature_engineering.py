import pandas as pd
import numpy as np
import os

def create_advanced_features(df):
    """Create more advanced features for improved model performance."""
    
    print("Creating advanced features...")
    
    # Interaction features
    df['TenureByContract'] = df['remainder__tenure'] * df['cat__Contract_Month-to-month']
    
    # Service density (services per dollar)
    service_cols = [
        'remainder__OnlineSecurity',
        'remainder__OnlineBackup',
        'remainder__DeviceProtection',
        'remainder__TechSupport',
        'remainder__StreamingTV',
        'remainder__StreamingMovies'
    ]
    
    # Convert any object columns to numeric
    for col in service_cols:
        if df[col].dtype == 'object':
            df[col] = df[col].replace({'Yes': 1, 'No': 0, 'No internet service': 0})
    
    # Count services
    df['TotalServices'] = df[service_cols].sum(axis=1)
    
    # Service density (value per service)
    df['ServiceDensity'] = df['remainder__MonthlyCharges'] / (df['TotalServices'] + 1)
    
    # Customer lifetime value
    df['PotentialLTV'] = df['remainder__MonthlyCharges'] * df['remainder__tenure']
    
    # Time-value features
    df['RevenuePerMonth'] = df['remainder__TotalCharges'] / (df['remainder__tenure'] + 1)
    
    # Churn risk score based on known risk factors
    df['ChurnRiskScore'] = 0
    
    # Higher monthly charges increase churn risk
    df.loc[df['remainder__MonthlyCharges'] > 80, 'ChurnRiskScore'] += 2
    
    # Month-to-month contracts increase churn risk
    df.loc[df['cat__Contract_Month-to-month'] == 1, 'ChurnRiskScore'] += 3
    
    # Low tenure increases churn risk
    df.loc[df['remainder__tenure'] < 12, 'ChurnRiskScore'] += 2
    
    # Fiber optic without tech support is risky
    df.loc[(df['cat__InternetService_Fiber optic'] == 1) & 
           (df['remainder__TechSupport'] == 0), 'ChurnRiskScore'] += 2
    
    # Electronic check payment method is risky
    df.loc[df['cat__PaymentMethod_Electronic check'] == 1, 'ChurnRiskScore'] += 1
    
    print(f"Created {df.shape[1] - 32} new advanced features")
    return df

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
    df[remaining_cats] = df[remaining_cats].replace({'Yes': 1, 'No': 0, 'No internet service': 0}).infer_objects(copy=False)
    
    
    # Add advanced features
    df = create_advanced_features(df)
    
    # Save engineered data
    df.to_csv('data/engineered_data.csv', index=False)
    print("Feature engineering completed.")
    print("Engineered columns:", df.columns.tolist())
if __name__ == "__main__":
    engineer_features()
