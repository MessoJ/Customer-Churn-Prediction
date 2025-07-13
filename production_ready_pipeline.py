import pandas as pd
import numpy as np
import joblib
import json
import os
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/production_pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ProductionChurnPredictor:
    """
    Production-ready customer churn prediction system with MLOps best practices.
    """
    
    def __init__(self, model_path: str = 'models/overall_best_churn_model.joblib'):
        self.model_path = model_path
        self.model = None
        self.feature_names = None
        self.preprocessor = None
        self.model_metadata = {}
        self.prediction_history = []
        
        # Load model and metadata
        self.load_model()
        
    def load_model(self):
        """Load the trained model and associated metadata."""
        try:
            # Load model
            self.model = joblib.load(self.model_path)
            logger.info(f"Model loaded successfully from {self.model_path}")
            
            # Load feature names
            try:
                self.feature_names = joblib.load('models/feature_names.joblib')
                logger.info("Feature names loaded successfully")
            except FileNotFoundError:
                logger.warning("Feature names file not found, will use all available features")
                self.feature_names = None
            
            # Load model metadata
            self.load_model_metadata()
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def load_model_metadata(self):
        """Load model metadata for versioning and tracking."""
        metadata_path = 'models/model_metadata.json'
        try:
            with open(metadata_path, 'r') as f:
                self.model_metadata = json.load(f)
            logger.info("Model metadata loaded successfully")
        except FileNotFoundError:
            # Create default metadata
            self.model_metadata = {
                'model_version': '1.0.0',
                'training_date': datetime.now().isoformat(),
                'model_type': type(self.model).__name__,
                'feature_count': len(self.feature_names) if self.feature_names else 0,
                'performance_metrics': {},
                'data_schema': {}
            }
            logger.info("Created default model metadata")
    
    def preprocess_input(self, customer_data: Dict) -> pd.DataFrame:
        """
        Preprocess customer data for prediction.
        
        Args:
            customer_data: Dictionary containing customer features
            
        Returns:
            Preprocessed DataFrame ready for prediction
        """
        try:
            # Convert to DataFrame
            df = pd.DataFrame([customer_data])
            
            # Apply the same preprocessing as training
            df = self.apply_feature_engineering(df)
            
            # Select features if feature names are specified
            if self.feature_names:
                available_features = [f for f in self.feature_names if f in df.columns]
                df = df[available_features]
                
                # Add missing features with default values
                missing_features = set(self.feature_names) - set(df.columns)
                for feature in missing_features:
                    df[feature] = 0
            
            # Handle missing values
            df = df.fillna(0)
            
            # Convert categorical variables
            for col in df.select_dtypes(include=['object']).columns:
                df[col] = pd.factorize(df[col])[0]
            
            logger.info(f"Data preprocessed successfully. Shape: {df.shape}")
            return df
            
        except Exception as e:
            logger.error(f"Error preprocessing data: {str(e)}")
            raise
    
    def apply_feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply the same feature engineering as training."""
        try:
            # Basic feature engineering
            if 'remainder__TotalCharges' in df.columns:
                df['remainder__TotalCharges'] = pd.to_numeric(
                    df['remainder__TotalCharges'].replace(' ', np.nan),
                    errors='coerce'
                )
            
            # Create engineered features
            if 'remainder__tenure' in df.columns and 'remainder__MonthlyCharges' in df.columns:
                df['TenureToChargeRatio'] = df['remainder__tenure'] / (df['remainder__MonthlyCharges'] + 1)
            
            if 'remainder__TotalCharges' in df.columns and 'remainder__tenure' in df.columns:
                df['AvgMonthlyCharge'] = df['remainder__TotalCharges'] / (df['remainder__tenure'] + 1)
            
            # Service utilization
            service_cols = [
                'remainder__OnlineSecurity', 'remainder__OnlineBackup', 
                'remainder__DeviceProtection', 'remainder__TechSupport',
                'remainder__StreamingTV', 'remainder__StreamingMovies'
            ]
            
            available_service_cols = [col for col in service_cols if col in df.columns]
            if available_service_cols:
                # Convert to numeric
                for col in available_service_cols:
                    if df[col].dtype == 'object':
                        df[col] = df[col].replace({'Yes': 1, 'No': 0, 'No internet service': 0})
                
                df['ServiceUtilization'] = df[available_service_cols].sum(axis=1)
            
            # Risk scoring
            df['RiskScore'] = 0
            if 'cat__Contract_Month-to-month' in df.columns:
                df.loc[df['cat__Contract_Month-to-month'] == 1, 'RiskScore'] += 3
            if 'remainder__tenure' in df.columns:
                df.loc[df['remainder__tenure'] < 6, 'RiskScore'] += 2
            if 'remainder__MonthlyCharges' in df.columns:
                df.loc[df['remainder__MonthlyCharges'] > 80, 'RiskScore'] += 1
            if 'cat__PaymentMethod_Electronic check' in df.columns:
                df.loc[df['cat__PaymentMethod_Electronic check'] == 1, 'RiskScore'] += 1
            
            return df
            
        except Exception as e:
            logger.error(f"Error in feature engineering: {str(e)}")
            raise
    
    def predict_churn(self, customer_data: Dict) -> Dict:
        """
        Predict customer churn probability.
        
        Args:
            customer_data: Dictionary containing customer features
            
        Returns:
            Dictionary with prediction results and metadata
        """
        try:
            # Preprocess data
            X = self.preprocess_input(customer_data)
            
            # Make prediction
            churn_probability = self.model.predict_proba(X)[0, 1]
            churn_prediction = self.model.predict(X)[0]
            
            # Determine risk level
            risk_level = self.determine_risk_level(churn_probability)
            
            # Create prediction result
            prediction_result = {
                'customer_id': customer_data.get('customerID', 'unknown'),
                'churn_probability': float(churn_probability),
                'churn_prediction': bool(churn_prediction),
                'risk_level': risk_level,
                'confidence_score': self.calculate_confidence_score(churn_probability),
                'prediction_timestamp': datetime.now().isoformat(),
                'model_version': self.model_metadata.get('model_version', 'unknown'),
                'recommended_actions': self.get_recommended_actions(churn_probability, customer_data)
            }
            
            # Log prediction
            self.log_prediction(prediction_result)
            
            logger.info(f"Prediction completed for customer {prediction_result['customer_id']}")
            return prediction_result
            
        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}")
            raise
    
    def determine_risk_level(self, probability: float) -> str:
        """Determine risk level based on churn probability."""
        if probability >= 0.8:
            return 'Very High'
        elif probability >= 0.6:
            return 'High'
        elif probability >= 0.4:
            return 'Medium'
        elif probability >= 0.2:
            return 'Low'
        else:
            return 'Very Low'
    
    def calculate_confidence_score(self, probability: float) -> float:
        """Calculate confidence score based on probability distance from 0.5."""
        return abs(probability - 0.5) * 2
    
    def get_recommended_actions(self, probability: float, customer_data: Dict) -> List[str]:
        """Get recommended actions based on churn probability and customer data."""
        actions = []
        
        if probability >= 0.6:
            actions.append("Immediate retention campaign")
            actions.append("Personalized offer")
            actions.append("Account review call")
        
        elif probability >= 0.4:
            actions.append("Targeted marketing campaign")
            actions.append("Service quality check")
        
        else:
            actions.append("Regular monitoring")
            actions.append("Standard engagement")
        
        # Add specific actions based on customer characteristics
        if customer_data.get('cat__Contract_Month-to-month') == 1:
            actions.append("Contract upgrade incentive")
        
        if customer_data.get('remainder__tenure', 0) < 6:
            actions.append("Onboarding support")
        
        if customer_data.get('remainder__MonthlyCharges', 0) > 80:
            actions.append("Premium service review")
        
        return actions
    
    def log_prediction(self, prediction_result: Dict):
        """Log prediction for monitoring and analysis."""
        self.prediction_history.append(prediction_result)
        
        # Keep only last 1000 predictions in memory
        if len(self.prediction_history) > 1000:
            self.prediction_history = self.prediction_history[-1000:]
    
    def batch_predict(self, customer_data_list: List[Dict]) -> List[Dict]:
        """
        Make batch predictions for multiple customers.
        
        Args:
            customer_data_list: List of customer data dictionaries
            
        Returns:
            List of prediction results
        """
        results = []
        
        for customer_data in customer_data_list:
            try:
                result = self.predict_churn(customer_data)
                results.append(result)
            except Exception as e:
                logger.error(f"Error predicting for customer {customer_data.get('customerID', 'unknown')}: {str(e)}")
                results.append({
                    'customer_id': customer_data.get('customerID', 'unknown'),
                    'error': str(e),
                    'prediction_timestamp': datetime.now().isoformat()
                })
        
        return results
    
    def get_model_performance_metrics(self) -> Dict:
        """Get current model performance metrics."""
        return {
            'model_version': self.model_metadata.get('model_version', 'unknown'),
            'training_date': self.model_metadata.get('training_date', 'unknown'),
            'total_predictions': len(self.prediction_history),
            'recent_predictions': len([p for p in self.prediction_history 
                                     if datetime.fromisoformat(p['prediction_timestamp']) > 
                                     datetime.now() - timedelta(hours=24)]),
            'average_confidence': np.mean([p.get('confidence_score', 0) 
                                         for p in self.prediction_history[-100:]]) if self.prediction_history else 0
        }
    
    def export_predictions(self, filepath: str = None):
        """Export prediction history to file."""
        if not filepath:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filepath = f'results/predictions_export_{timestamp}.json'
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(self.prediction_history, f, indent=2)
        
        logger.info(f"Predictions exported to {filepath}")

class ChurnPredictionAPI:
    """
    API wrapper for the churn prediction system.
    """
    
    def __init__(self, model_path: str = 'models/overall_best_churn_model.joblib'):
        self.predictor = ProductionChurnPredictor(model_path)
    
    def predict_single(self, customer_data: Dict) -> Dict:
        """API endpoint for single customer prediction."""
        return self.predictor.predict_churn(customer_data)
    
    def predict_batch(self, customer_data_list: List[Dict]) -> List[Dict]:
        """API endpoint for batch predictions."""
        return self.predictor.batch_predict(customer_data_list)
    
    def get_health_status(self) -> Dict:
        """API health check endpoint."""
        return {
            'status': 'healthy',
            'model_loaded': self.predictor.model is not None,
            'model_version': self.predictor.model_metadata.get('model_version', 'unknown'),
            'total_predictions': len(self.predictor.prediction_history),
            'timestamp': datetime.now().isoformat()
        }

def run_production_pipeline():
    """Run the production pipeline with example data."""
    logger.info("Starting production churn prediction pipeline")
    
    # Initialize predictor
    predictor = ProductionChurnPredictor()
    
    # Example customer data
    example_customers = [
        {
            'customerID': '1234-ABCD',
            'remainder__tenure': 12,
            'remainder__MonthlyCharges': 75.0,
            'remainder__TotalCharges': 900.0,
            'cat__Contract_Month-to-month': 1,
            'cat__PaymentMethod_Electronic check': 1,
            'remainder__OnlineSecurity': 'No',
            'remainder__TechSupport': 'No'
        },
        {
            'customerID': '5678-EFGH',
            'remainder__tenure': 48,
            'remainder__MonthlyCharges': 45.0,
            'remainder__TotalCharges': 2160.0,
            'cat__Contract_Two year': 1,
            'cat__PaymentMethod_Bank transfer (automatic)': 1,
            'remainder__OnlineSecurity': 'Yes',
            'remainder__TechSupport': 'Yes'
        }
    ]
    
    # Make predictions
    logger.info("Making predictions for example customers")
    for customer in example_customers:
        try:
            result = predictor.predict_churn(customer)
            logger.info(f"Customer {result['customer_id']}: "
                       f"Churn Probability: {result['churn_probability']:.3f}, "
                       f"Risk Level: {result['risk_level']}")
        except Exception as e:
            logger.error(f"Error predicting for customer {customer['customerID']}: {str(e)}")
    
    # Export predictions
    predictor.export_predictions()
    
    # Get performance metrics
    metrics = predictor.get_model_performance_metrics()
    logger.info(f"Model Performance Metrics: {metrics}")
    
    logger.info("Production pipeline completed successfully")

if __name__ == "__main__":
    run_production_pipeline()