import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import numpy as np
from data_preprocessing import preprocess_data
from feature_engineering import engineer_features

def visualize_data():
    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)
    
    # Load the data
    df = pd.read_csv('data/telco_churn.csv')
    
    # Churn distribution
    plt.figure(figsize=(8, 6))
    sns.countplot(x='Churn', data=df)
    plt.title('Churn Distribution')
    plt.savefig('results/churn_distribution.png')
    plt.close()
    
    # Feature importance
    # Load the model
    model = joblib.load('models/churn_model.joblib')
    
    # Looking at the error, we need to call preprocessing and feature engineering without arguments
    processed_data = preprocess_data()  # No arguments based on error message
    X_processed = engineer_features()   # No arguments based on error message
    
    # Get features excluding the target
    if 'remainder__Churn' in X_processed.columns:
        X_features = X_processed.drop(['remainder__Churn', 'remainder__customerID'], axis=1, errors='ignore')
    else:
        X_features = X_processed.drop('remainder__customerID', axis=1, errors='ignore')
    
    # Get feature importances and handle potential shape mismatch
    if hasattr(model, 'feature_importances_'):
        if len(model.feature_importances_) == len(X_features.columns):
            feature_importance = pd.Series(model.feature_importances_, index=X_features.columns)
        else:
            # If there's a shape mismatch, create a generic feature list
            print(f"Shape mismatch: {len(model.feature_importances_)} importances vs {len(X_features.columns)} columns")
            feature_columns = [f"Feature_{i}" for i in range(len(model.feature_importances_))]
            feature_importance = pd.Series(model.feature_importances_, index=feature_columns)
    else:
        # For models that don't have feature_importances_ attribute
        print("Model doesn't provide feature importances. Using coefficients if available.")
        if hasattr(model, 'coef_'):
            # For linear models
            if len(model.coef_[0]) == len(X_features.columns):
                feature_importance = pd.Series(np.abs(model.coef_[0]), index=X_features.columns)
            else:
                feature_columns = [f"Feature_{i}" for i in range(len(model.coef_[0]))]
                feature_importance = pd.Series(np.abs(model.coef_[0]), index=feature_columns)
        else:
            print("Cannot extract feature importance from this model type.")
            return
    
    # Plot feature importance
    plt.figure(figsize=(10, 8))
    feature_importance.nlargest(10).plot(kind='barh')
    plt.title('Top 10 Feature Importance')
    plt.savefig('results/feature_importance.png')
    plt.close()
    
    # Additional visualizations
    
    # Contract Type vs Churn
    plt.figure(figsize=(10, 6))
    contract_churn = df.groupby(['Contract', 'Churn']).size().unstack()
    contract_churn.plot(kind='bar', stacked=True)
    plt.title('Contract Type vs Churn')
    plt.xlabel('Contract Type')
    plt.ylabel('Count')
    plt.savefig('results/contract_vs_churn.png')
    plt.close()
    
    # Monthly Charges vs Churn
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Churn', y='MonthlyCharges', data=df)
    plt.title('Monthly Charges vs Churn')
    plt.savefig('results/monthly_charges_vs_churn.png')
    plt.close()
    
    # Tenure vs Churn
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Churn', y='tenure', data=df)
    plt.title('Tenure vs Churn')
    plt.savefig('results/tenure_vs_churn.png')
    plt.close()

if __name__ == "__main__":
    visualize_data()
