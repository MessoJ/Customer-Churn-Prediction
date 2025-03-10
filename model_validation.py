# Add this to model_training.py or create a new file named model_validation.py

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os

def cross_validate_model():
    """
    Perform cross-validation of the model to get more robust performance metrics.
    """
    print("Performing cross-validation...")
    
    # Load data
    df = pd.read_csv('data/engineered_data.csv')
    
    # Drop the customerID column if present
    if 'remainder__customerID' in df.columns:
        df = df.drop('remainder__customerID', axis=1)
    
    # Make sure target is properly formatted
    if df['remainder__Churn'].dtype == 'object':
        df['remainder__Churn'] = df['remainder__Churn'].map({'Yes': 1, 'No': 0})
    
    # Handle remaining categorical columns
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = pd.factorize(df[col])[0]
    
    # Separate features and target
    X = df.drop('remainder__Churn', axis=1)
    y = df['remainder__Churn']
    
    # Define the feature selection if needed (using the same as in training)
    important_features = [
        'remainder__tenure',
        'remainder__MonthlyCharges',
        'cat__Contract_Month-to-month',
        'AvgMonthlyCharge'
    ]
    
    # Use only important features if defined
    if len(important_features) > 0:
        X = X[important_features]
    
    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)
    
    # Load the model
    model = joblib.load('models/churn_model.joblib')
    
    # Define cross-validation strategy (5-fold stratified CV)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Calculate cross-validated scores
    accuracy = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    precision = cross_val_score(model, X, y, cv=cv, scoring='precision')
    recall = cross_val_score(model, X, y, cv=cv, scoring='recall')
    f1 = cross_val_score(model, X, y, cv=cv, scoring='f1')
    roc_auc = cross_val_score(model, X, y, cv=cv, scoring='roc_auc')
    
    # Print and store results
    print("Cross-validation results (mean ± std):")
    print(f"Accuracy:  {accuracy.mean():.4f} ± {accuracy.std():.4f}")
    print(f"Precision: {precision.mean():.4f} ± {precision.std():.4f}")
    print(f"Recall:    {recall.mean():.4f} ± {recall.std():.4f}") 
    print(f"F1-Score:  {f1.mean():.4f} ± {f1.std():.4f}")
    print(f"AUC-ROC:   {roc_auc.mean():.4f} ± {roc_auc.std():.4f}")
    
    # Create visualization of cross-validation results
    metrics = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'AUC-ROC': roc_auc
    }
    
    # Create boxplot of CV results
    plt.figure(figsize=(12, 6))
    cv_results_df = pd.DataFrame(metrics)
    sns.boxplot(data=cv_results_df)
    plt.title('Cross-Validation Metrics Distribution')
    plt.ylim(0, 1)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('results/cv_metrics_distribution.png')
    plt.close()
    
    # Save CV results to CSV
    cv_summary = pd.DataFrame({
        'Metric': list(metrics.keys()),
        'Mean': [m.mean() for m in metrics.values()],
        'Std': [m.std() for m in metrics.values()],
        'Min': [m.min() for m in metrics.values()],
        'Max': [m.max() for m in metrics.values()]
    })
    cv_summary.to_csv('results/cross_validation_results.csv', index=False)
    
    print("Cross-validation completed.")
    
if __name__ == "__main__":
    cross_validate_model()
