import pandas as pd
import numpy as np
import joblib
import os

def preprocess_customer_data(customer_data):
    """
    Preprocess customer data to match the format used during training.
    
    Parameters:
    -----------
    customer_data : dict or pd.DataFrame
        Single customer data or batch of customers
    
    Returns:
    --------
    pd.DataFrame
        Preprocessed data ready for prediction
    """
    # Convert to DataFrame if dictionary
    if isinstance(customer_data, dict):
        customer_data = pd.DataFrame([customer_data])
    
    # Load feature names required by the model
    feature_names = joblib.load('models/feature_names.joblib')
    
    # Create empty DataFrame with required columns
    processed_data = pd.DataFrame(index=customer_data.index)
    
    # Process each column as needed
    for feature in feature_names:
        if feature in customer_data.columns:
            processed_data[feature] = customer_data[feature]
        elif feature.startswith('cat__InternetService_'):
            # Handle one-hot encoded internet service
            service_type = feature.split('_')[-1]
            if 'InternetService' in customer_data:
                # Create a Series for comparison to handle both dict and DataFrame
                is_service = customer_data['InternetService'].eq(service_type)
                processed_data[feature] = is_service.astype(float)
            else:
                processed_data[feature] = 0.0
        elif feature.startswith('cat__Contract_'):
            # Handle one-hot encoded contract
            contract_type = feature.split('_')[-1]
            if 'Contract' in customer_data:
                # Create a Series for comparison to handle both dict and DataFrame
                is_contract = customer_data['Contract'].eq(contract_type)
                processed_data[feature] = is_contract.astype(float)
            else:
                processed_data[feature] = 0.0
        elif feature.startswith('cat__PaymentMethod_'):
            # Handle one-hot encoded payment method
            payment_type = ' '.join(feature.split('_')[2:])
            if 'PaymentMethod' in customer_data:
                # Create a Series for comparison to handle both dict and DataFrame
                is_payment = customer_data['PaymentMethod'].eq(payment_type)
                processed_data[feature] = is_payment.astype(float)
            else:
                processed_data[feature] = 0.0
        elif feature == 'AvgMonthlyCharge':
            # Calculate average monthly charge
            if 'TotalCharges' in customer_data and 'tenure' in customer_data:
                processed_data[feature] = customer_data['TotalCharges'] / (customer_data['tenure'] + 1e-6)
            else:
                processed_data[feature] = 0.0
        elif feature == 'TenureToChargeRatio':
            # Calculate tenure to charge ratio
            if 'tenure' in customer_data and 'MonthlyCharges' in customer_data:
                processed_data[feature] = customer_data['tenure'] / (customer_data['MonthlyCharges'] + 1e-6)
            else:
                processed_data[feature] = 0.0
        else:
            # For any missing features, fill with 0
            processed_data[feature] = 0.0
    
    # Ensure all columns are present in the correct order
    for col in feature_names:
        if col not in processed_data.columns:
            processed_data[col] = 0.0
    
    return processed_data[feature_names]

def predict_churn(customer_data):
    """
    Predict churn probability for a customer or batch of customers.
    
    Parameters:
    -----------
    customer_data : dict or pd.DataFrame
        Single customer data or batch of customers
    
    Returns:
    --------
    dict or pd.DataFrame
        Predictions with churn probability and risk category
    """
    try:
        # Load the model
        model = joblib.load('models/churn_model.joblib')
        
        # Preprocess customer data
        processed_data = preprocess_customer_data(customer_data)
        
        # Make predictions
        churn_proba = model.predict_proba(processed_data)[:, 1]
        
        # Create results
        is_single_customer = isinstance(customer_data, dict)
        if is_single_customer:
            result = {
                'churn_probability': float(churn_proba[0]),
                'churn_risk': 'High' if churn_proba[0] > 0.7 else 
                             'Medium' if churn_proba[0] > 0.3 else 'Low',
                'is_likely_to_churn': bool(churn_proba[0] > 0.5)  # Explicitly convert to bool
            }
        else:
            result = pd.DataFrame({
                'churn_probability': churn_proba,
                'churn_risk': ['High' if p > 0.7 else 'Medium' if p > 0.3 else 'Low' for p in churn_proba],
                'is_likely_to_churn': [bool(p > 0.5) for p in churn_proba]  # Create list of booleans
            }, index=customer_data.index)
        
        return result
    
    except Exception as e:
        print(f"Error predicting churn: {e}")
        import traceback
        traceback.print_exc()
        if isinstance(customer_data, dict):
            return {'error': str(e)}
        else:
            return pd.DataFrame({'error': [str(e)] * len(customer_data)})

# Example usage
if __name__ == "__main__":
    # Example of a single customer
    new_customer = {
        'gender': 'Male',
        'SeniorCitizen': 0,
        'Partner': 'Yes',
        'Dependents': 'No',
        'tenure': 24,
        'PhoneService': 'Yes',
        'MultipleLines': 'Yes',
        'InternetService': 'Fiber optic',
        'OnlineSecurity': 'No',
        'OnlineBackup': 'Yes',
        'DeviceProtection': 'No',
        'TechSupport': 'No',
        'StreamingTV': 'Yes',
        'StreamingMovies': 'Yes',
        'Contract': 'Month-to-month',
        'PaperlessBilling': 'Yes',
        'PaymentMethod': 'Electronic check',
        'MonthlyCharges': 94.95,
        'TotalCharges': 2279.8
    }
    
    # Convert column names to match model features
    new_customer_processed = {
        'remainder__' + k if not k.startswith('cat__') else k: v 
        for k, v in new_customer.items()
    }
    
    # Get prediction
    prediction = predict_churn(new_customer_processed)
    print("Churn Prediction:", prediction)
#By MessoJ
