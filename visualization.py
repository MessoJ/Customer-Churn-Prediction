import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

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
    
    # Load the preprocessed features - we need to load or recreate the same features used for training
    # Option 1: If you have the processed features saved, load them
    try:
        # Try to load processed features if available
        X_processed = pd.read_csv('data/processed_features.csv')
        feature_importance = pd.Series(model.feature_importances_, index=X_processed.columns)
    except FileNotFoundError:
        # Option 2: Recreate the feature processing pipeline
        # This is a simplified approach - ideally you should use the same preprocessing as in training
        from data_preprocessing import preprocess_data
        from feature_engineering import engineer_features
        
        processed_data = preprocess_data()
        X_processed = engineer_features(processed_data)
        
        # Remove the target variable if present
        if 'remainder__Churn' in X_processed.columns:
            X_processed = X_processed.drop('remainder__Churn', axis=1)
        
        feature_importance = pd.Series(model.feature_importances_, index=X_processed.columns)
    
    # Plot feature importance
    plt.figure(figsize=(10, 8))
    feature_importance.nlargest(10).plot(kind='barh')
    plt.title('Top 10 Feature Importance')
    plt.savefig('results/feature_importance.png')
    plt.close()

if __name__ == "__main__":
    visualize_data()
