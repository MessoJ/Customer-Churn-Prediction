"""
Advanced ML Integration System
Combines all advanced ML techniques for global scale deployment:
1. Real-Time Adaptation with Streaming Features
2. Cold-Start Handling with Graph Neural Networks
3. Counterfactual Robustness with Alibi
4. Adversarial Hardening with GAN-based Training

Implements global scale best practices with monitoring, observability, and fault tolerance
"""

import logging
import time
import json
import os
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import redis
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import numpy as np
import pandas as pd

# Import our advanced ML modules
try:
    from streaming_features import RealTimeFeatureEngine, UserEvent, StreamingFeature
    from cold_start_handling import ColdStartHandler, create_sample_cold_start_data
    from counterfactual_robustness import CounterfactualAnalyzer, InterventionScenario
    from adversarial_hardening import AdversarialHardener, create_sample_adversarial_data
    ADVANCED_ML_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Some advanced ML modules not available: {e}")
    ADVANCED_ML_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Prometheus metrics
INTEGRATION_REQUESTS = Counter('integration_requests_total', 'Total integration requests')
PREDICTION_TIME = Histogram('prediction_time_seconds', 'Time spent on predictions')
SYSTEM_HEALTH = Gauge('system_health_score', 'Overall system health score')
ACTIVE_MODELS = Gauge('active_models', 'Number of active ML models')

@dataclass
class CustomerPrediction:
    """Comprehensive customer prediction with all advanced features"""
    customer_id: str
    churn_probability: float
    confidence: float
    prediction_method: str  # 'ensemble', 'streaming', 'cold_start', 'robust'
    real_time_features: Dict[str, float]
    cold_start_features: Dict[str, float]
    counterfactual_analysis: Dict[str, Any]
    adversarial_protection: Dict[str, Any]
    metadata: Dict[str, Any]

class AdvancedMLIntegration:
    """
    Main integration system combining all advanced ML techniques
    Implements global scale best practices with monitoring and observability
    """
    
    def __init__(self, config_path: str = 'config/advanced_ml_config.json'):
        self.config_path = config_path
        self.config = self._load_config()
        
        # Initialize Redis for distributed caching
        self.redis_client = redis.Redis(
            host=self.config.get('redis_host', 'localhost'),
            port=self.config.get('redis_port', 6379),
            decode_responses=True
        )
        
        # Initialize advanced ML components
        self.streaming_engine = None
        self.cold_start_handler = None
        self.counterfactual_analyzer = None
        self.adversarial_hardener = None
        
        # Initialize components if available
        if ADVANCED_ML_AVAILABLE:
            self._initialize_components()
        
        # Start Prometheus metrics server
        self._start_metrics_server()
        
        # System health tracking
        self.health_checks = {
            'streaming_engine': False,
            'cold_start_handler': False,
            'counterfactual_analyzer': False,
            'adversarial_hardener': False
        }
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from JSON file"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    return json.load(f)
            else:
                # Default configuration
                return {
                    'redis_host': 'localhost',
                    'redis_port': 6379,
                    'kafka_bootstrap_servers': 'localhost:9092',
                    'model_path': 'models/',
                    'prometheus_port': 8000,
                    'enable_streaming': True,
                    'enable_cold_start': True,
                    'enable_counterfactual': True,
                    'enable_adversarial': True,
                    'cache_ttl': 3600,
                    'batch_size': 100,
                    'prediction_timeout': 30
                }
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return {}
    
    def _initialize_components(self):
        """Initialize all advanced ML components"""
        try:
            model_path = self.config.get('model_path', 'models/')
            
            # Initialize streaming engine
            if self.config.get('enable_streaming', True):
                self.streaming_engine = RealTimeFeatureEngine(
                    kafka_bootstrap_servers=self.config.get('kafka_bootstrap_servers', 'localhost:9092'),
                    redis_host=self.config.get('redis_host', 'localhost')
                )
                self.health_checks['streaming_engine'] = True
                logger.info("Streaming engine initialized")
            
            # Initialize cold-start handler
            if self.config.get('enable_cold_start', True):
                self.cold_start_handler = ColdStartHandler(
                    model_path=model_path,
                    redis_host=self.config.get('redis_host', 'localhost')
                )
                self.health_checks['cold_start_handler'] = True
                logger.info("Cold-start handler initialized")
            
            # Initialize counterfactual analyzer
            if self.config.get('enable_counterfactual', True):
                self.counterfactual_analyzer = CounterfactualAnalyzer(
                    model_path=model_path,
                    redis_host=self.config.get('redis_host', 'localhost')
                )
                self.health_checks['counterfactual_analyzer'] = True
                logger.info("Counterfactual analyzer initialized")
            
            # Initialize adversarial hardener
            if self.config.get('enable_adversarial', True):
                self.adversarial_hardener = AdversarialHardener(
                    model_path=model_path,
                    redis_host=self.config.get('redis_host', 'localhost')
                )
                self.health_checks['adversarial_hardener'] = True
                logger.info("Adversarial hardener initialized")
            
            # Update metrics
            active_models = sum(self.health_checks.values())
            ACTIVE_MODELS.set(active_models)
            
            logger.info(f"Initialized {active_models} advanced ML components")
            
        except Exception as e:
            logger.error(f"Error initializing components: {e}")
    
    def _start_metrics_server(self):
        """Start Prometheus metrics server"""
        try:
            port = self.config.get('prometheus_port', 8000)
            start_http_server(port)
            logger.info(f"Prometheus metrics server started on port {port}")
        except Exception as e:
            logger.error(f"Error starting metrics server: {e}")
    
    def predict_customer_churn(self, customer_data: Dict[str, Any]) -> CustomerPrediction:
        """
        Comprehensive customer churn prediction using all advanced ML techniques
        Implements ensemble approach with multiple prediction methods
        """
        try:
            start_time = time.time()
            INTEGRATION_REQUESTS.inc()
            
            customer_id = customer_data.get('customer_id', 'unknown')
            
            # Initialize result containers
            real_time_features = {}
            cold_start_features = {}
            counterfactual_analysis = {}
            adversarial_protection = {}
            
            # Method 1: Real-time streaming features
            streaming_prediction = self._get_streaming_prediction(customer_data)
            if streaming_prediction:
                real_time_features = streaming_prediction
            
            # Method 2: Cold-start handling
            cold_start_prediction = self._get_cold_start_prediction(customer_data)
            if cold_start_prediction:
                cold_start_features = cold_start_prediction
            
            # Method 3: Counterfactual analysis
            counterfactual_result = self._get_counterfactual_analysis(customer_data)
            if counterfactual_result:
                counterfactual_analysis = counterfactual_result
            
            # Method 4: Adversarial protection
            adversarial_result = self._get_adversarial_protection(customer_data)
            if adversarial_result:
                adversarial_protection = adversarial_result
            
            # Ensemble prediction
            ensemble_result = self._ensemble_prediction(
                streaming_prediction, cold_start_prediction, 
                counterfactual_result, adversarial_result
            )
            
            # Calculate prediction time
            prediction_time = time.time() - start_time
            PREDICTION_TIME.observe(prediction_time)
            
            # Create comprehensive prediction result
            prediction = CustomerPrediction(
                customer_id=customer_id,
                churn_probability=ensemble_result['churn_probability'],
                confidence=ensemble_result['confidence'],
                prediction_method=ensemble_result['method'],
                real_time_features=real_time_features,
                cold_start_features=cold_start_features,
                counterfactual_analysis=counterfactual_analysis,
                adversarial_protection=adversarial_protection,
                metadata={
                    'prediction_time': prediction_time,
                    'components_used': list(self.health_checks.keys()),
                    'timestamp': datetime.now().isoformat()
                }
            )
            
            # Cache result
            self._cache_prediction_result(customer_id, prediction)
            
            return prediction
            
        except Exception as e:
            logger.error(f"Error predicting customer churn: {e}")
            return self._create_fallback_prediction(customer_data)
    
    def _get_streaming_prediction(self, customer_data: Dict[str, Any]) -> Optional[Dict[str, float]]:
        """Get real-time streaming features prediction"""
        try:
            if not self.streaming_engine:
                return None
            
            # Create user event from customer data
            user_event = UserEvent(
                user_id=customer_data.get('customer_id', 'unknown'),
                event_type=customer_data.get('event_type', 'page_view'),
                timestamp=datetime.now(),
                session_id=customer_data.get('session_id', 'default'),
                page_url=customer_data.get('page_url'),
                product_id=customer_data.get('product_id'),
                cart_value=customer_data.get('cart_value'),
                click_coordinates=customer_data.get('click_coordinates'),
                time_on_page=customer_data.get('time_on_page')
            )
            
            # Compute real-time features
            features = self.streaming_engine.compute_real_time_features(user_event)
            
            if features:
                # Extract churn probability from features
                churn_feature = next((f for f in features if f.feature_name == 'real_time_churn_probability'), None)
                if churn_feature:
                    return {
                        'churn_probability': churn_feature.feature_value,
                        'confidence': churn_feature.confidence,
                        'session_urgency': next((f.feature_value for f in features if f.feature_name == 'current_session_urgency'), 0.0),
                        'click_velocity': next((f.feature_value for f in features if f.feature_name == 'click_velocity'), 0.0),
                        'cart_abandonment_risk': next((f.feature_value for f in features if f.feature_name == 'cart_abandonment_risk'), 0.0)
                    }
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting streaming prediction: {e}")
            return None
    
    def _get_cold_start_prediction(self, customer_data: Dict[str, Any]) -> Optional[Dict[str, float]]:
        """Get cold-start prediction using GNNs and transfer learning"""
        try:
            if not self.cold_start_handler:
                return None
            
            # Convert customer data to DataFrame format
            customer_df = pd.DataFrame([customer_data])
            
            # Get cold-start prediction
            predictions = self.cold_start_handler.predict_cold_start(customer_df)
            
            if predictions and customer_data.get('customer_id') in predictions:
                pred = predictions[customer_data['customer_id']]
                return {
                    'churn_probability': pred['churn_probability'],
                    'confidence': pred['confidence'],
                    'gnn_prediction': pred.get('gnn_prediction'),
                    'demographic_prediction': pred.get('demographic_prediction'),
                    'prediction_method': pred.get('prediction_method', 'fallback')
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting cold-start prediction: {e}")
            return None
    
    def _get_counterfactual_analysis(self, customer_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Get counterfactual analysis for intervention scenarios"""
        try:
            if not self.counterfactual_analyzer:
                return None
            
            # Create intervention scenario
            scenario = InterventionScenario(
                scenario_id="discount_10_percent",
                intervention_type="discount",
                intervention_value=0.1,
                target_metric="churn_probability",
                target_threshold=0.3,
                cost_per_customer=6.5,
                max_budget=5000.0,
                priority_customers=[customer_data.get('customer_id', 'unknown')]
            )
            
            # Convert customer data to DataFrame
            customer_df = pd.DataFrame([customer_data])
            
            # Analyze intervention scenario
            results = self.counterfactual_analyzer.analyze_intervention_scenario(scenario, customer_df)
            
            if results and customer_data.get('customer_id') in results:
                result = results[customer_data['customer_id']]
                return {
                    'original_prediction': result.original_prediction,
                    'counterfactual_prediction': result.counterfactual_prediction,
                    'intervention_applied': result.intervention_applied,
                    'cost_benefit_ratio': result.cost_benefit_ratio,
                    'recommended_action': result.recommended_action
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting counterfactual analysis: {e}")
            return None
    
    def _get_adversarial_protection(self, customer_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Get adversarial protection and robust prediction"""
        try:
            if not self.adversarial_hardener:
                return None
            
            # Extract features for adversarial protection
            feature_columns = [col for col in customer_data.keys() 
                             if col not in ['customer_id', 'event_type', 'session_id', 'page_url', 'product_id']]
            
            features = []
            for col in feature_columns:
                features.append(float(customer_data.get(col, 0)))
            
            features = np.array(features, dtype=np.float32)
            
            # Get protected prediction
            result = self.adversarial_hardener.predict_with_protection(
                features, customer_data.get('customer_id', 'unknown')
            )
            
            return {
                'prediction': result['prediction'],
                'confidence': result['confidence'],
                'anomaly_detected': result['anomaly_detected'],
                'anomaly_score': result['anomaly_score'],
                'attack_type': result['attack_type'],
                'recommended_action': result['recommended_action']
            }
            
        except Exception as e:
            logger.error(f"Error getting adversarial protection: {e}")
            return None
    
    def _ensemble_prediction(self, streaming_pred: Optional[Dict], 
                           cold_start_pred: Optional[Dict],
                           counterfactual_pred: Optional[Dict],
                           adversarial_pred: Optional[Dict]) -> Dict[str, Any]:
        """Combine predictions from all methods using ensemble approach"""
        try:
            predictions = []
            weights = []
            methods = []
            
            # Collect all available predictions
            if streaming_pred:
                predictions.append(streaming_pred['churn_probability'])
                weights.append(0.3)  # Real-time features get higher weight
                methods.append('streaming')
            
            if cold_start_pred:
                predictions.append(cold_start_pred['churn_probability'])
                weights.append(0.25)
                methods.append('cold_start')
            
            if counterfactual_pred:
                # Use original prediction from counterfactual analysis
                predictions.append(counterfactual_pred['original_prediction'])
                weights.append(0.25)
                methods.append('counterfactual')
            
            if adversarial_pred:
                # Convert binary prediction to probability
                adv_prob = 0.8 if adversarial_pred['prediction'] == 1 else 0.2
                predictions.append(adv_prob)
                weights.append(0.2)
                methods.append('adversarial')
            
            if not predictions:
                # Fallback prediction
                return {
                    'churn_probability': 0.5,
                    'confidence': 0.3,
                    'method': 'fallback'
                }
            
            # Weighted ensemble
            total_weight = sum(weights)
            if total_weight > 0:
                ensemble_probability = sum(p * w for p, w in zip(predictions, weights)) / total_weight
                ensemble_confidence = min(sum(weights) / len(weights), 1.0)
            else:
                ensemble_probability = np.mean(predictions)
                ensemble_confidence = 0.5
            
            return {
                'churn_probability': ensemble_probability,
                'confidence': ensemble_confidence,
                'method': f"ensemble_{'_'.join(methods)}"
            }
            
        except Exception as e:
            logger.error(f"Error in ensemble prediction: {e}")
            return {
                'churn_probability': 0.5,
                'confidence': 0.3,
                'method': 'fallback'
            }
    
    def _cache_prediction_result(self, customer_id: str, prediction: CustomerPrediction):
        """Cache prediction result for quick access"""
        try:
            cache_key = f"prediction:{customer_id}"
            cache_data = {
                'customer_id': prediction.customer_id,
                'churn_probability': prediction.churn_probability,
                'confidence': prediction.confidence,
                'prediction_method': prediction.prediction_method,
                'timestamp': datetime.now().isoformat()
            }
            
            self.redis_client.setex(
                cache_key,
                self.config.get('cache_ttl', 3600),
                json.dumps(cache_data)
            )
            
        except Exception as e:
            logger.error(f"Error caching prediction result: {e}")
    
    def _create_fallback_prediction(self, customer_data: Dict[str, Any]) -> CustomerPrediction:
        """Create fallback prediction when advanced ML is unavailable"""
        return CustomerPrediction(
            customer_id=customer_data.get('customer_id', 'unknown'),
            churn_probability=0.5,
            confidence=0.3,
            prediction_method='fallback',
            real_time_features={},
            cold_start_features={},
            counterfactual_analysis={},
            adversarial_protection={},
            metadata={
                'error': 'Advanced ML components unavailable',
                'timestamp': datetime.now().isoformat()
            }
        )
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get comprehensive system health status"""
        try:
            health_score = sum(self.health_checks.values()) / len(self.health_checks)
            SYSTEM_HEALTH.set(health_score)
            
            return {
                'overall_health': health_score,
                'component_status': self.health_checks.copy(),
                'active_models': sum(self.health_checks.values()),
                'total_components': len(self.health_checks),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting system health: {e}")
            return {
                'overall_health': 0.0,
                'component_status': {},
                'active_models': 0,
                'total_components': 0,
                'error': str(e)
            }
    
    def train_all_models(self, training_data: pd.DataFrame):
        """Train all advanced ML models"""
        try:
            logger.info("Starting training for all advanced ML models...")
            
            # Train cold-start models
            if self.cold_start_handler:
                logger.info("Training cold-start models...")
                self.cold_start_handler.train_models(training_data, training_data)
            
            # Train adversarial models
            if self.adversarial_hardener:
                logger.info("Training adversarial models...")
                self.adversarial_hardener.train_adversarially_robust_model(training_data, training_data)
            
            # Train counterfactual models (if needed)
            if self.counterfactual_analyzer:
                logger.info("Counterfactual models ready for analysis...")
            
            logger.info("All model training completed")
            
        except Exception as e:
            logger.error(f"Error training models: {e}")

def create_sample_customer_data():
    """Create sample customer data for testing"""
    return {
        'customer_id': 'user_001',
        'event_type': 'click',
        'session_id': 'session_001',
        'page_url': '/products/phone',
        'product_id': 'phone_001',
        'cart_value': 150.0,
        'click_coordinates': (100, 200),
        'time_on_page': 45.5,
        'tenure': 24.0,
        'monthly_charges': 65.0,
        'total_charges': 1500.0,
        'contract_month_to_month': 0,
        'internet_service_fiber': 1,
        'payment_method_electronic': 0,
        'online_security': 1,
        'tech_support': 0,
        'streaming_tv': 1,
        'streaming_movies': 1,
        'paperless_billing': 1,
        'dependents': 0,
        'partner': 1,
        'phone_service': 1,
        'multiple_lines': 0,
        'online_backup': 1,
        'device_protection': 0
    }

def main():
    """Main function to demonstrate advanced ML integration"""
    try:
        # Initialize advanced ML integration
        integration = AdvancedMLIntegration()
        
        # Check system health
        health = integration.get_system_health()
        print(f"System Health: {health['overall_health']:.2%}")
        print(f"Active Models: {health['active_models']}/{health['total_components']}")
        
        # Create sample data
        customer_data = create_sample_customer_data()
        
        # Make comprehensive prediction
        print("\nMaking comprehensive customer prediction...")
        prediction = integration.predict_customer_churn(customer_data)
        
        print(f"\nPrediction Results for {prediction.customer_id}:")
        print(f"Churn Probability: {prediction.churn_probability:.3f}")
        print(f"Confidence: {prediction.confidence:.3f}")
        print(f"Method: {prediction.prediction_method}")
        
        if prediction.real_time_features:
            print(f"Real-time Features: {prediction.real_time_features}")
        
        if prediction.cold_start_features:
            print(f"Cold-start Features: {prediction.cold_start_features}")
        
        if prediction.counterfactual_analysis:
            print(f"Counterfactual Analysis: {prediction.counterfactual_analysis}")
        
        if prediction.adversarial_protection:
            print(f"Adversarial Protection: {prediction.adversarial_protection}")
        
        print(f"\nPrediction completed in {prediction.metadata['prediction_time']:.3f} seconds")
        
    except Exception as e:
        logger.error(f"Error in main: {e}")
        print(f"Error: {e}")

if __name__ == "__main__":
    main()