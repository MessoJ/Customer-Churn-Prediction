import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_model():
    model = joblib.load('models/churn_model.joblib')
    df = pd.read_csv('data/engineered_data.csv')
    
    # Print columns to see what we're working with
    print(df.columns)
    
    # Drop customerID as it's not a feature
    if 'remainder__customerID' in df.columns:
        df = df.drop('remainder__customerID', axis=1)
    
    # Convert target to numeric if needed
    if df['remainder__Churn'].dtype == 'object':
        df['remainder__Churn'] = df['remainder__Churn'].map({'Yes': 1, 'No': 0})
    
    # Prepare features and target with correct column name
    X = df.drop('remainder__Churn', axis=1)
    y = df['remainder__Churn']
    
    # Convert any remaining object columns to numeric
    for col in X.select_dtypes(include=['object']).columns:
        X[col] = pd.to_numeric(X[col], errors='coerce')
    
    # Fill any NaN values that might have been created
    X = X.fillna(X.mean())
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    print(f"Precision: {precision_score(y_test, y_pred):.2f}")
    print(f"Recall: {recall_score(y_test, y_pred):.2f}")
    print(f"F1-Score: {f1_score(y_test, y_pred):.2f}")
    print(f"AUC-ROC: {roc_auc_score(y_test, y_proba):.2f}")
    
    # Confusion matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig('results/confusion_matrix.png')
    plt.close()
    
    # ROC Curve
    plt.figure(figsize=(8, 6))
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc_score(y_test, y_proba):.2f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.savefig('results/roc_curve.png')
    plt.close()
    
    # Feature importance
    plt.figure(figsize=(12, 8))
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    sns.barplot(x='Importance', y='Feature', data=feature_importance.head(15))
    plt.title('Top 15 Feature Importances')
    plt.tight_layout()
    plt.savefig('results/feature_importance.png')
    plt.close()
    
    print("Model evaluation completed.")

if __name__ == "__main__":
    evaluate_model()
