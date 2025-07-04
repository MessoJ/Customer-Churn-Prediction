import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV  # GridSearchCV import
from sklearn.ensemble import RandomForestClassifier
import joblib

def train_model():
    df = pd.read_csv('data/engineered_data.csv')
    
    # Print column names for debugging
    print("Columns in engineered data:", df.columns.tolist())
    
    # Drop the customerID column first
    df = df.drop('remainder__customerID', axis=1)
    
    # Check data types to see if any other columns need conversion
    print("\nData types before conversion:")
    print(df.dtypes)
    
    # Make sure target is properly formatted
    df['remainder__Churn'] = df['remainder__Churn'].map({'Yes': 1, 'No': 0})
    
    # Handle remaining categorical columns
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = pd.factorize(df[col])[0]  # Convert remaining object columns to numeric
    
    # Separate features and target
    X = df.drop('remainder__Churn', axis=1)
    y = df['remainder__Churn']
    
    # Feature selection should happen BEFORE train/test split
    important_features = [
        'remainder__tenure',
        'remainder__MonthlyCharges',
        'cat__Contract_Month-to-month',
        'AvgMonthlyCharge'
    ]
    X = X[important_features]
    
    # Split data after feature selection
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Initialize model with correct parameters
    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=8,
        class_weight='balanced',
        random_state=42
    )
    
    # Set up parameter grid
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [5, 10],
        'min_samples_leaf': [1, 2]
    }
    
    # Initialize and fit grid search
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=5,
        scoring='accuracy'
    )
    grid_search.fit(X_train, y_train)
    
    # Get best model
    best_model = grid_search.best_estimator_
    
    # Calculate and print accuracy
    train_score = best_model.score(X_train, y_train)
    test_score = best_model.score(X_test, y_test)
    print(f"\nTraining accuracy: {train_score:.4f}")
    print(f"Testing accuracy: {test_score:.4f}")
    
    # Save the best model
    feature_names = X.columns.tolist()
    joblib.dump(feature_names, 'models/feature_names.joblib')
    joblib.dump(best_model, 'models/churn_model.joblib')
    print("Model training completed.")

if __name__ == "__main__":
    train_model()
#By Messoj
