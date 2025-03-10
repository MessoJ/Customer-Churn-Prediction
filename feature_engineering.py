import pandas as pd
import numpy as np  # Make sure to import numpy

def engineer_features():
    # Read the processed data file
    df = pd.read_csv('data/processed_data.csv')
    
    print(df.columns)  # Print the available columns
    
    # First replace spaces with NaN in TotalCharges
    df['TotalCharges'] = df['TotalCharges'].replace(' ', np.nan)
    
    # Then convert to numeric
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    
    # Convert MonthlyCharges to float
    df['MonthlyCharges'] = df['MonthlyCharges'].astype(float)
    
    # Create new features using correct columns
    df['AvgMonthlyCharge'] = df['TotalCharges'] / (df['tenure'] + 1e-6)
    
    df.to_csv('data/engineered_data.csv', index=False)
    print("Feature engineering completed.")

if __name__ == "__main__":
    engineer_features()
