import pandas as pd

def engineer_features():
    df = pd.read_csv('data/processed_data.csv')
    
    # Use correct column names from preprocessing
    df['remainder__MonthlyCharges'] = df['remainder__MonthlyCharges'].astype(float)
    df['remainder__TotalCharges'] = df['remainder__TotalCharges'].astype(float)
    
    # Update feature calculation
    df['AvgMonthlyCharge'] = df['remainder__TotalCharges'] / (df['remainder__tenure'] + 1e-6)
    
    # Save engineered data
    df.to_csv('data/engineered_data.csv', index=False)
    print("Feature engineering completed.")

if __name__ == "__main__":
    engineer_features()
