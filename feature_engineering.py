import pandas as pd
import numpy as np

def engineer_features():
    # Read the processed data file
    df = pd.read_csv('data/processed_data.csv')
    
    print(df.columns)  # Print the available columns
    
    # First replace spaces with NaN in TotalCharges - using correct column name
    df['remainder__TotalCharges'] = df['remainder__TotalCharges'].replace(' ', np.nan)
    
    # Then convert to numeric
    df['remainder__TotalCharges'] = pd.to_numeric(df['remainder__TotalCharges'], errors='coerce')
    
    # Convert MonthlyCharges to float
    df['remainder__MonthlyCharges'] = df['remainder__MonthlyCharges'].astype(float)
    
    # Create new features using correct columns
    df['AvgMonthlyCharge'] = df['remainder__TotalCharges'] / (df['remainder__tenure'] + 1e-6)
    df['HighValueFlag'] = (df['remainder__MonthlyCharges'] > 70).astype(int)
    df['TenureToChargeRatio'] = df['remainder__tenure'] / (df['remainder__MonthlyCharges'] + 1e-6)
    df['ServiceDiversity'] = df[['OnlineSecurity','TechSupport','StreamingTV']].sum(axis=1)

    remaining_cats = [
        'remainder__MultipleLines',
        'remainder__OnlineSecurity',
        'remainder__OnlineBackup',
        'remainder__DeviceProtection',
        'remainder__TechSupport',
        'remainder__StreamingTV',
        'remainder__StreamingMovies'
    ]
    df[cat_cols] = df[cat_cols].replace({'Yes': 1, 'No': 0, 'No internet service': 0})
    
    df.to_csv('data/engineered_data.csv', index=False)
    print("Feature engineering completed.")

if __name__ == "__main__":
    engineer_features()
