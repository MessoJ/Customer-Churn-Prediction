import pandas as pd

def engineer_features():
    df = pd.read_csv('data/processed_data.csv')
    
    # Example feature: Average monthly charges
    df['MonthlyCharges'] = df['MonthlyCharges'].astype(float)
    df['TotalCharges'] = df['TotalCharges'].astype(float)
    df['AvgMonthlyCharge'] = df['TotalCharges'] / (df['tenure'] + 1e-6)  # Avoid division by zero
    
    # Save engineered data
    df.to_csv('data/engineered_data.csv', index=False)
    print("Feature engineering completed.")

if __name__ == "__main__":
    engineer_features()
