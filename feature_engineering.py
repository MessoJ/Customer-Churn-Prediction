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
    
    df.to_csv('data/engineered_data.csv', index=False)
    print("Feature engineering completed.")

if __name__ == "__main__":
    engineer_features()
