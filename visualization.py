import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import numpy as np
import shap

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test, plot_type="bar")

def visualize_data():
    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)
    
    # Load the original data
    try:
        df = pd.read_csv('data/telco_churn.csv')
        print("Original data loaded successfully")
    except FileNotFoundError:
        print("Error: Could not find the original data file")
        return
    
    # Churn distribution
    plt.figure(figsize=(8, 6))
    sns.countplot(x='Churn', data=df)
    plt.title('Churn Distribution')
    plt.savefig('results/churn_distribution.png')
    plt.close()
    
    # Try to load the model
    try:
        model = joblib.load('models/churn_model.joblib')
        print("Model loaded successfully")
    except FileNotFoundError:
        print("Error: Could not find the model file")
        return
    
    # Visualizations based on raw data without requiring the processed features
    
    # Contract Type vs Churn
    plt.figure(figsize=(10, 6))
    if 'Contract' in df.columns:
        contract_churn = df.groupby(['Contract', 'Churn']).size().unstack()
        contract_churn.plot(kind='bar', stacked=True)
        plt.title('Contract Type vs Churn')
        plt.xlabel('Contract Type')
        plt.ylabel('Count')
        plt.savefig('results/contract_vs_churn.png')
        plt.close()
        print("Contract vs Churn visualization created")
    
    # Monthly Charges vs Churn
    plt.figure(figsize=(10, 6))
    if 'MonthlyCharges' in df.columns:
        sns.boxplot(x='Churn', y='MonthlyCharges', data=df)
        plt.title('Monthly Charges vs Churn')
        plt.savefig('results/monthly_charges_vs_churn.png')
        plt.close()
        print("Monthly Charges vs Churn visualization created")
    
    # Tenure vs Churn
    plt.figure(figsize=(10, 6))
    if 'tenure' in df.columns:
        sns.boxplot(x='Churn', y='tenure', data=df)
        plt.title('Tenure vs Churn')
        plt.savefig('results/tenure_vs_churn.png')
        plt.close()
        print("Tenure vs Churn visualization created")
    
    # Internet Service vs Churn
    plt.figure(figsize=(10, 6))
    if 'InternetService' in df.columns:
        internet_churn = df.groupby(['InternetService', 'Churn']).size().unstack()
        internet_churn.plot(kind='bar', stacked=True)
        plt.title('Internet Service vs Churn')
        plt.xlabel('Internet Service')
        plt.ylabel('Count')
        plt.savefig('results/internet_service_vs_churn.png')
        plt.close()
        print("Internet Service vs Churn visualization created")
    
    # Payment Method vs Churn
    plt.figure(figsize=(10, 6))
    if 'PaymentMethod' in df.columns:
        payment_churn = df.groupby(['PaymentMethod', 'Churn']).size().unstack()
        payment_churn.plot(kind='bar', stacked=True)
        plt.title('Payment Method vs Churn')
        plt.xlabel('Payment Method')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('results/payment_method_vs_churn.png')
        plt.close()
        print("Payment Method vs Churn visualization created")
    
    # Correlation Heatmap
    plt.figure(figsize=(12, 10))
    # Select only numeric columns
    numeric_df = df.select_dtypes(include=[np.number])
    # Create a correlation matrix
    corr = numeric_df.corr()
    # Plot the heatmap
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Correlation Heatmap of Numeric Features')
    plt.tight_layout()
    plt.savefig('results/correlation_heatmap.png')
    plt.close()
    print("Correlation heatmap created")
    
    print("All visualizations completed successfully")

if __name__ == "__main__":
    visualize_data()
