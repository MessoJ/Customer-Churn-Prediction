# Customer Churn Prediction

A comprehensive machine learning pipeline for predicting customer churn in the telecommunications industry.

## Overview

This repository contains an end-to-end solution for predicting customer churn, helping businesses identify customers at risk of leaving their service. The pipeline implements best practices in data preprocessing, feature engineering, model training, and evaluation.

## Features

- **Robust Data Preprocessing**: Handles categorical variables, missing values, and data transformations for 7,043 customer records
- **Advanced Feature Engineering**: Creates 9 powerful derived features that capture customer behavior patterns
- **Cross-Validated Model Training**: Ensures model reliability and stability
- **Comprehensive Evaluation**: Multiple metrics including accuracy, precision, recall, F1-score, and AUC-ROC
- **Insightful Visualizations**: Various plots to understand churn drivers and model decisions
- **Production-Ready Prediction**: Returns probability, risk level, and binary prediction

## Pipeline Components

1. **Data Preprocessing** (`data_preprocessing.py`)
   - Standardizes data formats
   - Converts categorical variables to numeric
   - One-hot encodes multi-class categorical features

2. **Feature Engineering** (`feature_engineering.py`)
   - Creates basic and advanced derived features:
     - `AvgMonthlyCharge`: Average charge per month
     - `HighValueFlag`: Identifies high-value customers
     - `TenureToChargeRatio`: Relationship between tenure and charges
     - `ServiceDiversity`: Variety of services used
     - `TenureByContract`: Interaction between tenure and contract type
     - `TotalServices`: Count of services subscribed
     - `ServiceDensity`: Service concentration
     - `PotentialLTV`: Estimated lifetime value
     - `ChurnRiskScore`: Aggregated risk score

3. **Model Training** (`model_training.py`)
   - Implements machine learning model with cross-validation
   - Current performance metrics:
     - Accuracy: 76%
     - Precision: 53%
     - Recall: 73%
     - F1-Score: 61%
     - AUC-ROC: 84%

4. **Visualization** (`visualization.py`)
   - Creates multiple visualizations:
     - Contract vs Churn
     - Monthly Charges vs Churn
     - Tenure vs Churn
     - Internet Service vs Churn
     - Payment Method vs Churn
     - Correlation heatmap
     - SHAP visualizations for model explainability

## Installation

```bash
# Clone the repository
git clone https://github.com/username/Customer-Churn-Prediction.git
cd Customer-Churn-Prediction

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Running the Full Pipeline

```python
# Run the full pipeline
python main.py
```

### Making Predictions

```python
# Make a prediction for a single customer
from prediction import predict_churn

customer_data = {
    'gender': 'Male',
    'SeniorCitizen': 0,
    'Partner': 'Yes',
    'Dependents': 'No',
    'tenure': 24,
    'PhoneService': 'Yes',
    'MultipleLines': 'No',
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
    'MonthlyCharges': 94.85,
    'TotalCharges': 2275.4
}

result = predict_churn(customer_data)
print(result)
# Output: {'churn_probability': 0.21, 'churn_risk': 'Low', 'is_likely_to_churn': False}
```

## Model Insights

The current model prioritizes recall (73%) over precision (53%), making it well-suited for identifying as many potential churners as possible, even at the cost of some false positives. This approach is business-oriented, as the cost of missing a potential churner typically exceeds the cost of incorrectly flagging a loyal customer.

Key findings from the analysis:
- Contract type is a strong predictor of churn (month-to-month contracts have higher churn)
- Fiber optic internet service users show higher churn rates
- Electronic check payment method is associated with higher churn
- Short tenure strongly predicts churn likelihood

## Project Structure

```
Customer-Churn-Prediction/
├── data/
│   └── telco_customer_churn.csv
├── data_preprocessing.py
├── feature_engineering.py
├── model_training.py
├── model_evaluation.py
├── visualization.py
├── prediction.py
├── main.py
├── models/
│   └── churn_model.pkl
├── requirements.txt
└── README.md
```

## Future Improvements

- Implement hyperparameter tuning
- Explore ensemble models (Random Forest, XGBoost)
- Add more interaction features
- Develop customer segmentation before classification
- Deploy as an API service

## License

MIT

## Acknowledgments

This project was developed using telecommunication customer data to help businesses improve customer retention strategies.
