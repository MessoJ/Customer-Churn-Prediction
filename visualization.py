import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

def visualize_data():
    df = pd.read_csv('data/telco_churn.csv')
    
    # Churn distribution
    sns.countplot(x='Churn', data=df)
    plt.title('Churn Distribution')
    plt.savefig('results/churn_distribution.png')
    plt.close()
    
    # Feature importance
    model = joblib.load('models/churn_model.joblib')
    feature_importance = pd.Series(model.feature_importances_, index=X.columns)
    feature_importance.nlargest(10).plot(kind='barh')
    plt.title('Top 10 Feature Importance')
    plt.savefig('results/feature_importance.png')
    plt.close()

if __name__ == "__main__":
    visualize_data()
