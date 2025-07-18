import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import numpy as np
import shap

def create_shap_visualizations():
    """Generate SHAP value plots to explain model predictions."""
    print("Generating SHAP visualizations...")
    
    try:
        # Load the model and data
        model = joblib.load('models/churn_model.joblib')
        feature_names = joblib.load('models/feature_names.joblib')
        df = pd.read_csv('data/engineered_data.csv')
        
        # Convert Churn to numeric if needed
        if df['remainder__Churn'].dtype == 'object':
            df['remainder__Churn'] = df['remainder__Churn'].map({'Yes': 1, 'No': 0})
        
        # Use only the features used for training
        X = df[feature_names]
        
        # Create a sample for faster computation
        X_sample = X.sample(min(500, len(X)), random_state=42)
        
        # Initialize the SHAP explainer
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
        
        # If shap_values is a list (for multiclass), use the positive class (index 1)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Use positive class values
        
        # Create SHAP summary plot
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
        plt.title('Feature Importance (SHAP values)')
        plt.tight_layout()
        plt.savefig('results/shap_feature_importance.png')
        plt.close()
        
        print("SHAP visualizations created successfully")
        
    except Exception as e:
        print(f"Error creating SHAP visualizations: {e}")
        import traceback
        traceback.print_exc()

def create_shap_plot():
    try:
        model = joblib.load('models/churn_model.joblib')
        feature_names = joblib.load('models/feature_names.joblib')
        df = pd.read_csv('data/engineered_data.csv')
        
        # Convert Churn to numeric
        if df['remainder__Churn'].dtype == 'object':
            df['remainder__Churn'] = df['remainder__Churn'].map({'Yes': 1, 'No': 0})
        
        # Use only features used during training
        X = df[feature_names]
        
        # Create SHAP explainer
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X, plot_type="bar", show=False)
        plt.title('SHAP Feature Importance')
        plt.tight_layout()
        plt.savefig('results/shap_feature_importance.png')
        plt.close()
        print("SHAP visualization created")
    except Exception as e:
        print(f"Error creating SHAP plot: {e}")

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
    create_shap_visualizations()
#By Messoj
