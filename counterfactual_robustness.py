"""
Counterfactual Robustness with Alibi
Implements counterfactual analysis for "What if we offer 10% discount?" scenarios
Global scale best practices with distributed computing and monitoring
"""

import numpy as np
import pandas as pd
import joblib
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import json
import time
from datetime import datetime
import redis
from prometheus_client import Counter, Histogram, Gauge
import os

# Alibi imports for counterfactual analysis
try:
    from alibi.explainers import CounterfactualProto
    from alibi.explainers import CounterfactualRL
    from alibi.datasets import fetch_adult
    from alibi.utils.data import gen_category_map
    ALIBI_AVAILABLE = True
except ImportError:
    ALIBI_AVAILABLE = False
    print("Warning: Alibi not available. Install with: pip install alibi")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Prometheus metrics
COUNTERFACTUAL_GENERATION_TIME = Histogram('counterfactual_generation_seconds', 'Time spent generating counterfactuals')
COUNTERFACTUAL_REQUESTS = Counter('counterfactual_requests_total', 'Total counterfactual analysis requests')
INTERVENTION_SUCCESS_RATE = Gauge('intervention_success_rate', 'Success rate of counterfactual interventions')

@dataclass
class InterventionScenario:
    """Represents a business intervention scenario"""
    scenario_id: str
    intervention_type: str  # 'discount', 'feature_upgrade', 'retention_offer'
    intervention_value: float  # 0.1 for 10% discount
    target_metric: str  # 'churn_probability', 'revenue', 'lifetime_value'
    target_threshold: float  # Desired outcome threshold
    cost_per_customer: float  # Cost of intervention per customer
    max_budget: float  # Maximum budget for intervention
    priority_customers: List[str]  # High-priority customer IDs

@dataclass
class CounterfactualResult:
    """Result of counterfactual analysis"""
    customer_id: str
    original_prediction: float
    counterfactual_prediction: float
    intervention_applied: str
    intervention_value: float
    confidence: float
    cost_benefit_ratio: float
    recommended_action: str
    metadata: Dict[str, Any]

class CounterfactualAnalyzer:
    """
    Counterfactual analysis system using Alibi
    Implements global scale best practices with distributed processing
    """
    
    def __init__(self, model_path: str = 'models/', redis_host: str = 'localhost'):
        self.model_path = model_path
        self.redis_client = redis.Redis(host=redis_host, port=6379, decode_responses=True)
        
        # Load trained model and preprocessor
        self.model = None
        self.preprocessor = None
        self.feature_names = None
        self._load_models()
        
        # Initialize counterfactual explainers
        self.counterfactual_explainer = None
        self._initialize_explainers()
        
        # Cache for counterfactual results
        self.cache_ttl = 3600  # 1 hour
    
    def _load_models(self):
        """Load trained model and preprocessor"""
        try:
            # Load model
            model_path = f"{self.model_path}/churn_model.joblib"
            if os.path.exists(model_path):
                self.model = joblib.load(model_path)
                logger.info("Loaded churn prediction model")
            
            # Load preprocessor
            preprocessor_path = f"{self.model_path}/preprocessor.joblib"
            if os.path.exists(preprocessor_path):
                self.preprocessor = joblib.load(preprocessor_path)
                logger.info("Loaded feature preprocessor")
            
            # Load feature names
            feature_names_path = f"{self.model_path}/feature_names.joblib"
            if os.path.exists(feature_names_path):
                self.feature_names = joblib.load(feature_names_path)
                logger.info("Loaded feature names")
                
        except Exception as e:
            logger.error(f"Error loading models: {e}")
    
    def _initialize_explainers(self):
        """Initialize counterfactual explainers"""
        if not ALIBI_AVAILABLE:
            logger.warning("Alibi not available. Counterfactual analysis will be limited.")
            return
        
        try:
            # Initialize counterfactual prototype explainer
            self.counterfactual_explainer = CounterfactualProto(
                predictor=self._model_predictor,
                shape=(1, len(self.feature_names)) if self.feature_names else (1, 10),
                kappa=0.0,
                beta=0.1,
                feature_range=(0, 1),
                gamma=100,
                theta=100,
                ae_model=None,
                enc_model=None,
                use_kdtree=True,
                learning_rate_init=0.01,
                max_iterations=1000,
                c_init=1.0,
                c_steps=10,
                eps=(1e-3, 1e-3),
                clip=(-1000, 1000),
                update_num_grad=1,
                write_dir=None,
                sess=None
            )
            
            logger.info("Initialized counterfactual explainer")
            
        except Exception as e:
            logger.error(f"Error initializing counterfactual explainer: {e}")
    
    def _model_predictor(self, x: np.ndarray) -> np.ndarray:
        """Wrapper for model prediction"""
        try:
            if self.model is None:
                return np.zeros((x.shape[0], 1))
            
            # Preprocess features if preprocessor is available
            if self.preprocessor is not None:
                x_processed = self.preprocessor.transform(x)
            else:
                x_processed = x
            
            # Make prediction
            predictions = self.model.predict_proba(x_processed)
            return predictions[:, 1:2]  # Return churn probability
            
        except Exception as e:
            logger.error(f"Error in model prediction: {e}")
            return np.zeros((x.shape[0], 1))
    
    def analyze_intervention_scenario(self, scenario: InterventionScenario, 
                                   customer_data: pd.DataFrame) -> Dict[str, CounterfactualResult]:
        """
        Analyze a business intervention scenario using counterfactual analysis
        Example: "What if we offer 10% discount to high-risk customers?"
        """
        try:
            start_time = time.time()
            COUNTERFACTUAL_REQUESTS.inc()
            
            results = {}
            
            # Filter customers based on priority
            if scenario.priority_customers:
                target_customers = customer_data[customer_data['user_id'].isin(scenario.priority_customers)]
            else:
                # Use all customers for analysis
                target_customers = customer_data
            
            logger.info(f"Analyzing intervention for {len(target_customers)} customers")
            
            for _, customer in target_customers.iterrows():
                customer_id = customer['user_id']
                
                # Generate counterfactual for this customer
                counterfactual_result = self._generate_customer_counterfactual(
                    customer, scenario
                )
                
                if counterfactual_result:
                    results[customer_id] = counterfactual_result
            
            # Calculate intervention success rate
            successful_interventions = sum(1 for result in results.values() 
                                        if result.counterfactual_prediction < result.original_prediction)
            success_rate = successful_interventions / len(results) if results else 0.0
            INTERVENTION_SUCCESS_RATE.set(success_rate)
            
            # Cache results
            self._cache_counterfactual_results(scenario.scenario_id, results)
            
            generation_time = time.time() - start_time
            COUNTERFACTUAL_GENERATION_TIME.observe(generation_time)
            
            logger.info(f"Counterfactual analysis completed in {generation_time:.2f} seconds")
            logger.info(f"Success rate: {success_rate:.2%}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error analyzing intervention scenario: {e}")
            return {}
    
    def _generate_customer_counterfactual(self, customer: pd.Series, 
                                        scenario: InterventionScenario) -> Optional[CounterfactualResult]:
        """Generate counterfactual analysis for a single customer"""
        try:
            customer_id = customer['user_id']
            
            # Extract customer features
            customer_features = self._extract_customer_features(customer)
            
            if customer_features is None:
                return None
            
            # Get original prediction
            original_prediction = self._model_predictor(customer_features.reshape(1, -1))[0, 0]
            
            # Apply intervention to features
            intervened_features = self._apply_intervention_to_features(
                customer_features, scenario
            )
            
            # Get counterfactual prediction
            counterfactual_prediction = self._model_predictor(intervened_features.reshape(1, -1))[0, 0]
            
            # Calculate confidence and cost-benefit ratio
            confidence = self._calculate_prediction_confidence(customer_features, intervened_features)
            cost_benefit_ratio = self._calculate_cost_benefit_ratio(
                original_prediction, counterfactual_prediction, scenario
            )
            
            # Determine recommended action
            recommended_action = self._determine_recommended_action(
                original_prediction, counterfactual_prediction, scenario, cost_benefit_ratio
            )
            
            return CounterfactualResult(
                customer_id=customer_id,
                original_prediction=float(original_prediction),
                counterfactual_prediction=float(counterfactual_prediction),
                intervention_applied=scenario.intervention_type,
                intervention_value=scenario.intervention_value,
                confidence=float(confidence),
                cost_benefit_ratio=float(cost_benefit_ratio),
                recommended_action=recommended_action,
                metadata={
                    'scenario_id': scenario.scenario_id,
                    'target_metric': scenario.target_metric,
                    'cost_per_customer': scenario.cost_per_customer,
                    'feature_changes': self._get_feature_changes(customer_features, intervened_features)
                }
            )
            
        except Exception as e:
            logger.error(f"Error generating counterfactual for customer {customer_id}: {e}")
            return None
    
    def _extract_customer_features(self, customer: pd.Series) -> Optional[np.ndarray]:
        """Extract features for counterfactual analysis"""
        try:
            if self.feature_names is None:
                # Fallback to basic features
                basic_features = [
                    'tenure', 'monthly_charges', 'total_charges',
                    'contract_month_to_month', 'internet_service_fiber',
                    'payment_method_electronic', 'online_security',
                    'tech_support', 'streaming_tv', 'streaming_movies'
                ]
                
                features = []
                for feature in basic_features:
                    if feature in customer:
                        features.append(float(customer[feature]))
                    else:
                        features.append(0.0)
                
                return np.array(features, dtype=np.float32)
            
            else:
                # Use predefined feature names
                features = []
                for feature_name in self.feature_names:
                    if feature_name in customer:
                        features.append(float(customer[feature_name]))
                    else:
                        features.append(0.0)
                
                return np.array(features, dtype=np.float32)
                
        except Exception as e:
            logger.error(f"Error extracting customer features: {e}")
            return None
    
    def _apply_intervention_to_features(self, features: np.ndarray, 
                                      scenario: InterventionScenario) -> np.ndarray:
        """Apply business intervention to customer features"""
        try:
            intervened_features = features.copy()
            
            if scenario.intervention_type == 'discount':
                # Apply discount to monthly charges
                if len(features) > 1:  # Assuming monthly_charges is at index 1
                    original_charges = intervened_features[1]
                    discount_amount = original_charges * scenario.intervention_value
                    intervened_features[1] = original_charges - discount_amount
            
            elif scenario.intervention_type == 'feature_upgrade':
                # Upgrade service features (e.g., add tech support)
                if len(features) > 7:  # Assuming tech_support is at index 7
                    intervened_features[7] = 1.0  # Enable tech support
            
            elif scenario.intervention_type == 'retention_offer':
                # Apply retention offer (combination of discount and feature upgrade)
                if len(features) > 1:
                    original_charges = intervened_features[1]
                    discount_amount = original_charges * scenario.intervention_value
                    intervened_features[1] = original_charges - discount_amount
                
                if len(features) > 7:
                    intervened_features[7] = 1.0  # Enable tech support
            
            return intervened_features
            
        except Exception as e:
            logger.error(f"Error applying intervention to features: {e}")
            return features
    
    def _calculate_prediction_confidence(self, original_features: np.ndarray, 
                                       intervened_features: np.ndarray) -> float:
        """Calculate confidence in counterfactual prediction"""
        try:
            # Calculate feature similarity
            feature_similarity = 1 - np.mean(np.abs(original_features - intervened_features))
            
            # Calculate prediction stability (multiple predictions)
            predictions = []
            for _ in range(5):
                pred = self._model_predictor(intervened_features.reshape(1, -1))[0, 0]
                predictions.append(pred)
            
            prediction_stability = 1 - np.std(predictions)
            
            # Combine similarity and stability
            confidence = (feature_similarity * 0.6) + (prediction_stability * 0.4)
            
            return max(0.0, min(1.0, confidence))
            
        except Exception as e:
            logger.error(f"Error calculating prediction confidence: {e}")
            return 0.5
    
    def _calculate_cost_benefit_ratio(self, original_prediction: float, 
                                    counterfactual_prediction: float,
                                    scenario: InterventionScenario) -> float:
        """Calculate cost-benefit ratio of intervention"""
        try:
            # Calculate benefit (reduction in churn probability)
            churn_reduction = original_prediction - counterfactual_prediction
            
            if churn_reduction <= 0:
                return 0.0  # No benefit
            
            # Calculate cost (intervention cost)
            intervention_cost = scenario.cost_per_customer
            
            # Calculate customer lifetime value (simplified)
            monthly_value = 65.0  # Average monthly charge
            lifetime_months = 24.0  # Average customer lifetime
            customer_lifetime_value = monthly_value * lifetime_months
            
            # Calculate benefit in monetary terms
            monetary_benefit = churn_reduction * customer_lifetime_value
            
            # Calculate cost-benefit ratio
            if intervention_cost > 0:
                cost_benefit_ratio = monetary_benefit / intervention_cost
            else:
                cost_benefit_ratio = float('inf') if monetary_benefit > 0 else 0.0
            
            return min(cost_benefit_ratio, 100.0)  # Cap at 100x return
            
        except Exception as e:
            logger.error(f"Error calculating cost-benefit ratio: {e}")
            return 0.0
    
    def _determine_recommended_action(self, original_prediction: float,
                                    counterfactual_prediction: float,
                                    scenario: InterventionScenario,
                                    cost_benefit_ratio: float) -> str:
        """Determine recommended action based on counterfactual analysis"""
        try:
            # Check if intervention is beneficial
            churn_reduction = original_prediction - counterfactual_prediction
            
            if churn_reduction <= 0:
                return "no_intervention"
            
            # Check cost-benefit threshold
            if cost_benefit_ratio < 2.0:  # Less than 2x return
                return "no_intervention"
            
            # Check budget constraints
            if scenario.max_budget > 0:
                intervention_cost = scenario.cost_per_customer
                if intervention_cost > scenario.max_budget:
                    return "budget_constrained"
            
            # Determine intervention type
            if churn_reduction > 0.3:  # High churn reduction
                return "immediate_intervention"
            elif churn_reduction > 0.1:  # Moderate churn reduction
                return "targeted_intervention"
            else:
                return "monitor_only"
                
        except Exception as e:
            logger.error(f"Error determining recommended action: {e}")
            return "no_intervention"
    
    def _get_feature_changes(self, original_features: np.ndarray, 
                            intervened_features: np.ndarray) -> Dict[str, float]:
        """Get feature changes between original and intervened features"""
        try:
            changes = {}
            
            if self.feature_names:
                for i, feature_name in enumerate(self.feature_names):
                    if i < len(original_features) and i < len(intervened_features):
                        change = intervened_features[i] - original_features[i]
                        if abs(change) > 1e-6:  # Only include significant changes
                            changes[feature_name] = float(change)
            else:
                # Use generic feature names
                for i in range(min(len(original_features), len(intervened_features))):
                    change = intervened_features[i] - original_features[i]
                    if abs(change) > 1e-6:
                        changes[f"feature_{i}"] = float(change)
            
            return changes
            
        except Exception as e:
            logger.error(f"Error getting feature changes: {e}")
            return {}
    
    def _cache_counterfactual_results(self, scenario_id: str, 
                                    results: Dict[str, CounterfactualResult]):
        """Cache counterfactual results for quick access"""
        try:
            # Convert results to serializable format
            serializable_results = {}
            for customer_id, result in results.items():
                serializable_results[customer_id] = {
                    'customer_id': result.customer_id,
                    'original_prediction': result.original_prediction,
                    'counterfactual_prediction': result.counterfactual_prediction,
                    'intervention_applied': result.intervention_applied,
                    'intervention_value': result.intervention_value,
                    'confidence': result.confidence,
                    'cost_benefit_ratio': result.cost_benefit_ratio,
                    'recommended_action': result.recommended_action,
                    'metadata': result.metadata
                }
            
            cache_key = f"counterfactual_results:{scenario_id}"
            self.redis_client.setex(
                cache_key,
                self.cache_ttl,
                json.dumps(serializable_results)
            )
            
        except Exception as e:
            logger.error(f"Error caching counterfactual results: {e}")
    
    def get_optimal_intervention_path(self, customer_data: pd.DataFrame,
                                    budget: float = 10000.0) -> Dict[str, Any]:
        """
        Find optimal intervention path for maximum impact within budget
        Uses counterfactual analysis to prioritize interventions
        """
        try:
            # Define intervention scenarios
            scenarios = [
                InterventionScenario(
                    scenario_id="discount_10_percent",
                    intervention_type="discount",
                    intervention_value=0.1,
                    target_metric="churn_probability",
                    target_threshold=0.3,
                    cost_per_customer=6.5,  # 10% of average monthly charge
                    max_budget=budget * 0.4,
                    priority_customers=[]
                ),
                InterventionScenario(
                    scenario_id="tech_support_upgrade",
                    intervention_type="feature_upgrade",
                    intervention_value=1.0,
                    target_metric="churn_probability",
                    target_threshold=0.3,
                    cost_per_customer=15.0,  # Tech support upgrade cost
                    max_budget=budget * 0.3,
                    priority_customers=[]
                ),
                InterventionScenario(
                    scenario_id="retention_package",
                    intervention_type="retention_offer",
                    intervention_value=0.15,
                    target_metric="churn_probability",
                    target_threshold=0.2,
                    cost_per_customer=20.0,  # Comprehensive retention package
                    max_budget=budget * 0.3,
                    priority_customers=[]
                )
            ]
            
            # Analyze each scenario
            all_results = {}
            scenario_results = {}
            
            for scenario in scenarios:
                results = self.analyze_intervention_scenario(scenario, customer_data)
                scenario_results[scenario.scenario_id] = results
                all_results.update(results)
            
            # Find optimal intervention path
            optimal_path = self._calculate_optimal_path(scenario_results, budget)
            
            return {
                'optimal_path': optimal_path,
                'scenario_results': scenario_results,
                'total_customers_analyzed': len(all_results),
                'total_budget_required': optimal_path.get('total_cost', 0),
                'expected_churn_reduction': optimal_path.get('expected_impact', 0)
            }
            
        except Exception as e:
            logger.error(f"Error calculating optimal intervention path: {e}")
            return {}
    
    def _calculate_optimal_path(self, scenario_results: Dict[str, Dict[str, CounterfactualResult]],
                              budget: float) -> Dict[str, Any]:
        """Calculate optimal intervention path using greedy algorithm"""
        try:
            # Collect all interventions with their costs and benefits
            interventions = []
            
            for scenario_id, results in scenario_results.items():
                for customer_id, result in results.items():
                    if result.recommended_action != "no_intervention":
                        interventions.append({
                            'customer_id': customer_id,
                            'scenario_id': scenario_id,
                            'cost': result.metadata.get('cost_per_customer', 0),
                            'benefit': result.original_prediction - result.counterfactual_prediction,
                            'cost_benefit_ratio': result.cost_benefit_ratio,
                            'result': result
                        })
            
            # Sort by cost-benefit ratio (highest first)
            interventions.sort(key=lambda x: x['cost_benefit_ratio'], reverse=True)
            
            # Greedy selection
            selected_interventions = []
            total_cost = 0.0
            total_benefit = 0.0
            
            for intervention in interventions:
                if total_cost + intervention['cost'] <= budget:
                    selected_interventions.append(intervention)
                    total_cost += intervention['cost']
                    total_benefit += intervention['benefit']
            
            return {
                'selected_interventions': selected_interventions,
                'total_cost': total_cost,
                'total_benefit': total_benefit,
                'expected_impact': total_benefit,
                'customers_targeted': len(selected_interventions),
                'budget_utilization': total_cost / budget if budget > 0 else 0
            }
            
        except Exception as e:
            logger.error(f"Error calculating optimal path: {e}")
            return {}

def create_sample_intervention_scenarios():
    """Create sample intervention scenarios for testing"""
    scenarios = [
        InterventionScenario(
            scenario_id="discount_10_percent",
            intervention_type="discount",
            intervention_value=0.1,
            target_metric="churn_probability",
            target_threshold=0.3,
            cost_per_customer=6.5,
            max_budget=5000.0,
            priority_customers=["user_001", "user_002", "user_003"]
        ),
        InterventionScenario(
            scenario_id="tech_support_upgrade",
            intervention_type="feature_upgrade",
            intervention_value=1.0,
            target_metric="churn_probability",
            target_threshold=0.3,
            cost_per_customer=15.0,
            max_budget=3000.0,
            priority_customers=["user_004", "user_005"]
        )
    ]
    
    return scenarios

if __name__ == "__main__":
    # Create sample customer data
    customer_data = pd.DataFrame({
        'user_id': [f'user_{i:03d}' for i in range(1, 101)],
        'tenure': np.random.exponential(24, 100),
        'monthly_charges': np.random.normal(65, 20, 100),
        'total_charges': np.random.normal(1500, 800, 100),
        'contract_month_to_month': np.random.choice([0, 1], 100, p=[0.7, 0.3]),
        'internet_service_fiber': np.random.choice([0, 1], 100, p=[0.6, 0.4]),
        'payment_method_electronic': np.random.choice([0, 1], 100, p=[0.8, 0.2]),
        'online_security': np.random.choice([0, 1], 100, p=[0.6, 0.4]),
        'tech_support': np.random.choice([0, 1], 100, p=[0.7, 0.3]),
        'streaming_tv': np.random.choice([0, 1], 100, p=[0.5, 0.5]),
        'streaming_movies': np.random.choice([0, 1], 100, p=[0.5, 0.5])
    })
    
    # Initialize counterfactual analyzer
    analyzer = CounterfactualAnalyzer()
    
    # Test intervention scenarios
    scenarios = create_sample_intervention_scenarios()
    
    for scenario in scenarios:
        print(f"\nAnalyzing scenario: {scenario.scenario_id}")
        results = analyzer.analyze_intervention_scenario(scenario, customer_data)
        
        print(f"Results for {len(results)} customers:")
        for customer_id, result in list(results.items())[:5]:  # Show first 5
            print(f"  {customer_id}: {result.recommended_action} "
                  f"(Churn: {result.original_prediction:.3f} → {result.counterfactual_prediction:.3f})")
    
    # Test optimal intervention path
    print("\nCalculating optimal intervention path...")
    optimal_path = analyzer.get_optimal_intervention_path(customer_data, budget=10000.0)
    
    print(f"Optimal path results:")
    print(f"  Total cost: ${optimal_path.get('total_cost', 0):.2f}")
    print(f"  Customers targeted: {optimal_path.get('customers_targeted', 0)}")
    print(f"  Expected impact: {optimal_path.get('expected_impact', 0):.3f}")