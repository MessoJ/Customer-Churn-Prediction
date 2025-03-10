import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

def train_model():
    df = pd.read_csv('data/engineered_data.csv')
    
    # Check column names first
    print(df.columns)
    
    # Use the correct column name for Churn (likely 'remainder__Churn')
    X = df.drop('remainder__Churn', axis=1)
    y = df['remainder__Churn']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    joblib.dump(model, 'models/churn_model.joblib')
    print("Model training completed.")

if __name__ == "__main__":
    train_model()
