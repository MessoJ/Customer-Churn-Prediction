import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

def train_model():
    df = pd.read_csv('data/engineered_data.csv')
    
    # Print column names for debugging
    print(df.columns)
    
    # Drop the customerID column first
    df = df.drop('remainder__customerID', axis=1)
    
    # Check data types to see if any other columns need conversion
    print(df.dtypes)
    
    # Make sure target is properly formatted
    df['remainder__Churn'] = df['remainder__Churn'].map({'Yes': 1, 'No': 0})
    
    # Separate features and target
    X = df.drop('remainder__Churn', axis=1)
    y = df['remainder__Churn']
    
    # Convert any remaining object columns to numeric
    for col in X.select_dtypes(include=['object']).columns:
        X[col] = pd.to_numeric(X[col], errors='coerce')
    
    # Fill any NaN values that might have been created
    X = X.fillna(X.mean())
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    important_features = [
        'remainder__tenure',
        'remainder__MonthlyCharges',
        'cat__Contract_Month-to-month',
        'AvgMonthlyCharge'
    ]
    X = df[important_features]
    
    
    
    model = RandomForestClassifier(n_estimators=150, max_depth=8, class_weight='balanced', sampling_strategy='minority', random_state=42)
    model.fit(X_train, y_train)

    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [5, 10],
        'min_samples_leaf': [1, 2]
    }
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)
    grid_search.fit(X_train, y_train)
    
    # Calculate and print accuracy
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    print(f"Training accuracy: {train_score:.4f}")
    print(f"Testing accuracy: {test_score:.4f}")
    
    joblib.dump(model, 'models/churn_model.joblib')
    print("Model training completed.")

if __name__ == "__main__":
    train_model()
