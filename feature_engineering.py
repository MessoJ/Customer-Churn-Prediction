import pandas as pd

def engineer_features():
    df = pd.read_csv('data/processed_data.csv')
    
    # Use correct column names from preprocessing
    df = pd.read_csv('data/telco_churn.csv')
    print(df.columns)  # Print the available columns
    df['MonthlyCharges'] = df['MonthlyCharges'].astype(float)
    df['TotalCharges'] = df['TotalCharges'].astype(float)
    df['TotalCharges'] = df['TotalCharges'].replace(' ', np.nan)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

    
    # Create new features using correct columns
    df['AvgMonthlyCharge'] = df['TotalCharges'] / (df['tenure'] + 1e-6)
    
    df.to_csv('data/engineered_data.csv', index=False)
    print("Feature engineering completed.")

if __name__ == "__main__":
    engineer_features()
