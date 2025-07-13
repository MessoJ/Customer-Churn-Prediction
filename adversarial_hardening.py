"""
Adversarial Hardening with GAN-based Training
Implements GAN-based training with malicious feature perturbations and anomaly detection
Global scale best practices with distributed training and real-time protection
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
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
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Prometheus metrics
ADVERSARIAL_TRAINING_TIME = Histogram('adversarial_training_seconds', 'Time spent on adversarial training')
ANOMALY_DETECTION_REQUESTS = Counter('anomaly_detection_requests_total', 'Total anomaly detection requests')
ADVERSARIAL_ATTACKS_DETECTED = Counter('adversarial_attacks_detected_total', 'Total adversarial attacks detected')
MODEL_ROBUSTNESS_SCORE = Gauge('model_robustness_score', 'Model robustness against adversarial attacks')

@dataclass
class AdversarialAttack:
    """Represents an adversarial attack attempt"""
    attack_id: str
    attack_type: str  # 'perturbation', 'injection', 'evasion'
    target_customer_id: str
    original_features: np.ndarray
    perturbed_features: np.ndarray
    attack_success: bool
    detection_confidence: float
    timestamp: datetime
    metadata: Dict[str, Any]

@dataclass
class AnomalyDetectionResult:
    """Result of anomaly detection"""
    customer_id: str
    is_anomaly: bool
    anomaly_score: float
    confidence: float
    detected_attack_type: Optional[str]
    recommended_action: str
    metadata: Dict[str, Any]

class AdversarialGenerator(nn.Module):
    """
    Generative Adversarial Network for creating adversarial examples
    Trains generator to create realistic but malicious perturbations
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 128, noise_dim: int = 50):
        super(AdversarialGenerator, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.noise_dim = noise_dim
        
        # Generator architecture
        self.generator = nn.Sequential(
            nn.Linear(noise_dim + input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim // 2, input_dim),
            nn.Tanh()  # Output perturbations in [-1, 1] range
        )
        
        # Discriminator architecture
        self.discriminator = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """Generate adversarial perturbations"""
        combined_input = torch.cat([x, noise], dim=1)
        perturbations = self.generator(combined_input)
        return perturbations
    
    def discriminate(self, x: torch.Tensor) -> torch.Tensor:
        """Discriminate between real and generated samples"""
        return self.discriminator(x)

class RobustModel(nn.Module):
    """
    Adversarially robust model with built-in protection
    Implements multiple defense mechanisms
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 128, num_classes: int = 2):
        super(RobustModel, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        
        # Main classifier with adversarial training
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        # Adversarial detection head
        self.anomaly_detector = nn.Sequential(
            nn.Linear(input_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Feature denoising layer
        self.denoiser = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Tanh()
        )
    
    def forward(self, x: torch.Tensor, training_mode: str = 'normal') -> Dict[str, torch.Tensor]:
        """
        Forward pass with different training modes
        Args:
            x: Input features
            training_mode: 'normal', 'adversarial', 'defense'
        """
        outputs = {}
        
        if training_mode == 'defense':
            # Apply feature denoising
            x_denoised = self.denoiser(x)
            x = x_denoised
        
        # Main classification
        logits = self.classifier(x)
        outputs['logits'] = logits
        outputs['probabilities'] = F.softmax(logits, dim=1)
        
        # Anomaly detection
        anomaly_score = self.anomaly_detector(x)
        outputs['anomaly_score'] = anomaly_score
        
        return outputs

class AnomalyDetector:
    """
    Anomaly detection system for identifying adversarial attacks
    Uses multiple detection methods for robust protection
    """
    
    def __init__(self, input_dim: int):
        self.input_dim = input_dim
        self.scaler = StandardScaler()
        self.isolation_forest = IsolationForest(
            contamination=0.1,
            random_state=42,
            n_estimators=100
        )
        
        # Statistical anomaly detection
        self.feature_means = None
        self.feature_stds = None
        self.threshold_multiplier = 3.0
        
        # Ensemble of detection methods
        self.detection_methods = ['isolation_forest', 'statistical', 'distance_based']
    
    def fit(self, training_data: np.ndarray):
        """Fit anomaly detection models"""
        try:
            # Fit scaler
            self.scaler.fit(training_data)
            scaled_data = self.scaler.transform(training_data)
            
            # Fit isolation forest
            self.isolation_forest.fit(scaled_data)
            
            # Calculate statistical thresholds
            self.feature_means = np.mean(scaled_data, axis=0)
            self.feature_stds = np.std(scaled_data, axis=0)
            
            logger.info("Anomaly detection models fitted successfully")
            
        except Exception as e:
            logger.error(f"Error fitting anomaly detection models: {e}")
    
    def detect_anomaly(self, features: np.ndarray) -> Tuple[bool, float, Dict[str, float]]:
        """Detect anomalies using ensemble of methods"""
        try:
            # Scale features
            scaled_features = self.scaler.transform(features.reshape(1, -1)).flatten()
            
            # Method 1: Isolation Forest
            if_anomaly = self.isolation_forest.predict(scaled_features.reshape(1, -1))[0]
            if_score = self.isolation_forest.decision_function(scaled_features.reshape(1, -1))[0]
            
            # Method 2: Statistical detection
            statistical_anomaly = self._statistical_detection(scaled_features)
            
            # Method 3: Distance-based detection
            distance_anomaly, distance_score = self._distance_based_detection(scaled_features)
            
            # Ensemble decision
            anomaly_scores = {
                'isolation_forest': -if_score,  # Negative because isolation forest returns negative for anomalies
                'statistical': statistical_anomaly,
                'distance_based': distance_score
            }
            
            # Weighted ensemble
            ensemble_score = (
                anomaly_scores['isolation_forest'] * 0.4 +
                anomaly_scores['statistical'] * 0.3 +
                anomaly_scores['distance_based'] * 0.3
            )
            
            is_anomaly = ensemble_score > 0.5
            
            return is_anomaly, ensemble_score, anomaly_scores
            
        except Exception as e:
            logger.error(f"Error detecting anomaly: {e}")
            return False, 0.0, {}
    
    def _statistical_detection(self, features: np.ndarray) -> float:
        """Statistical anomaly detection using z-scores"""
        try:
            if self.feature_means is None:
                return 0.0
            
            z_scores = np.abs((features - self.feature_means) / (self.feature_stds + 1e-8))
            max_z_score = np.max(z_scores)
            
            # Normalize to [0, 1] range
            return min(max_z_score / self.threshold_multiplier, 1.0)
            
        except Exception as e:
            logger.error(f"Error in statistical detection: {e}")
            return 0.0
    
    def _distance_based_detection(self, features: np.ndarray) -> Tuple[bool, float]:
        """Distance-based anomaly detection"""
        try:
            if self.feature_means is None:
                return False, 0.0
            
            # Calculate Mahalanobis distance
            diff = features - self.feature_means
            inv_cov = np.linalg.inv(np.diag(self.feature_stds ** 2))
            mahalanobis_dist = np.sqrt(diff @ inv_cov @ diff)
            
            # Normalize distance
            normalized_distance = min(mahalanobis_dist / 10.0, 1.0)
            
            return normalized_distance > 0.7, normalized_distance
            
        except Exception as e:
            logger.error(f"Error in distance-based detection: {e}")
            return False, 0.0

class AdversarialHardener:
    """
    Main adversarial hardening system
    Implements GAN-based training and anomaly detection for global scale protection
    """
    
    def __init__(self, model_path: str = 'models/', redis_host: str = 'localhost'):
        self.model_path = model_path
        self.redis_client = redis.Redis(host=redis_host, port=6379, decode_responses=True)
        
        # Initialize components
        self.robust_model = None
        self.adversarial_generator = None
        self.anomaly_detector = None
        
        # Load pre-trained models if available
        self._load_models()
        
        # Attack tracking
        self.attack_history = []
        self.attack_cache_ttl = 3600  # 1 hour
    
    def _load_models(self):
        """Load pre-trained robust models"""
        try:
            # Load robust model
            robust_model_path = f"{self.model_path}/robust_model.pth"
            if os.path.exists(robust_model_path):
                self.robust_model = RobustModel(input_dim=21)
                self.robust_model.load_state_dict(torch.load(robust_model_path))
                self.robust_model.eval()
                logger.info("Loaded pre-trained robust model")
            
            # Load adversarial generator
            generator_path = f"{self.model_path}/adversarial_generator.pth"
            if os.path.exists(generator_path):
                self.adversarial_generator = AdversarialGenerator(input_dim=21)
                self.adversarial_generator.load_state_dict(torch.load(generator_path))
                self.adversarial_generator.eval()
                logger.info("Loaded pre-trained adversarial generator")
            
            # Load anomaly detector
            anomaly_path = f"{self.model_path}/anomaly_detector.pkl"
            if os.path.exists(anomaly_path):
                self.anomaly_detector = joblib.load(anomaly_path)
                logger.info("Loaded pre-trained anomaly detector")
                
        except Exception as e:
            logger.error(f"Error loading models: {e}")
    
    def train_adversarially_robust_model(self, training_data: pd.DataFrame, 
                                       validation_data: pd.DataFrame):
        """Train adversarially robust model using GAN-based training"""
        try:
            start_time = time.time()
            
            # Prepare data
            X_train = self._prepare_features(training_data)
            y_train = training_data['churn'].values
            
            X_val = self._prepare_features(validation_data)
            y_val = validation_data['churn'].values
            
            # Initialize models
            input_dim = X_train.shape[1]
            self.robust_model = RobustModel(input_dim=input_dim)
            self.adversarial_generator = AdversarialGenerator(input_dim=input_dim)
            
            # Training parameters
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.robust_model.to(device)
            self.adversarial_generator.to(device)
            
            # Optimizers
            robust_optimizer = torch.optim.Adam(self.robust_model.parameters(), lr=0.001)
            generator_optimizer = torch.optim.Adam(self.adversarial_generator.parameters(), lr=0.0002)
            discriminator_optimizer = torch.optim.Adam(self.adversarial_generator.discriminator.parameters(), lr=0.0002)
            
            # Loss functions
            classification_criterion = nn.CrossEntropyLoss()
            adversarial_criterion = nn.BCELoss()
            
            # Training loop
            num_epochs = 100
            batch_size = 32
            
            for epoch in range(num_epochs):
                self.robust_model.train()
                self.adversarial_generator.train()
                
                # Create data loader
                train_dataset = TensorDataset(
                    torch.FloatTensor(X_train),
                    torch.LongTensor(y_train)
                )
                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                
                epoch_losses = {'robust': 0, 'generator': 0, 'discriminator': 0}
                
                for batch_idx, (data, targets) in enumerate(train_loader):
                    data, targets = data.to(device), targets.to(device)
                    
                    # Train robust model with adversarial examples
                    robust_optimizer.zero_grad()
                    
                    # Normal forward pass
                    normal_outputs = self.robust_model(data, training_mode='normal')
                    normal_loss = classification_criterion(normal_outputs['logits'], targets)
                    
                    # Generate adversarial examples
                    noise = torch.randn(data.size(0), self.adversarial_generator.noise_dim).to(device)
                    perturbations = self.adversarial_generator(data, noise)
                    adversarial_data = data + perturbations * 0.1  # Small perturbations
                    
                    # Adversarial forward pass
                    adversarial_outputs = self.robust_model(adversarial_data, training_mode='defense')
                    adversarial_loss = classification_criterion(adversarial_outputs['logits'], targets)
                    
                    # Combined loss
                    total_robust_loss = normal_loss + 0.5 * adversarial_loss
                    total_robust_loss.backward()
                    robust_optimizer.step()
                    
                    epoch_losses['robust'] += total_robust_loss.item()
                    
                    # Train adversarial generator
                    generator_optimizer.zero_grad()
                    
                    # Generate fake perturbations
                    fake_perturbations = self.adversarial_generator(data, noise)
                    
                    # Train discriminator
                    discriminator_optimizer.zero_grad()
                    
                    real_labels = torch.ones(data.size(0), 1).to(device)
                    fake_labels = torch.zeros(data.size(0), 1).to(device)
                    
                    # Real data
                    real_outputs = self.adversarial_generator.discriminate(data)
                    real_loss = adversarial_criterion(real_outputs, real_labels)
                    
                    # Fake data
                    fake_data = data + fake_perturbations * 0.1
                    fake_outputs = self.adversarial_generator.discriminate(fake_data)
                    fake_loss = adversarial_criterion(fake_outputs, fake_labels)
                    
                    discriminator_loss = real_loss + fake_loss
                    discriminator_loss.backward()
                    discriminator_optimizer.step()
                    
                    epoch_losses['discriminator'] += discriminator_loss.item()
                    
                    # Train generator
                    generator_optimizer.zero_grad()
                    
                    fake_outputs = self.adversarial_generator.discriminate(fake_data)
                    generator_loss = adversarial_criterion(fake_outputs, real_labels)
                    generator_loss.backward()
                    generator_optimizer.step()
                    
                    epoch_losses['generator'] += generator_loss.item()
                
                # Validation
                if epoch % 10 == 0:
                    self.robust_model.eval()
                    with torch.no_grad():
                        val_data = torch.FloatTensor(X_val).to(device)
                        val_targets = torch.LongTensor(y_val).to(device)
                        
                        val_outputs = self.robust_model(val_data, training_mode='normal')
                        val_loss = classification_criterion(val_outputs['logits'], val_targets)
                        
                        predictions = torch.argmax(val_outputs['logits'], dim=1)
                        accuracy = accuracy_score(val_targets.cpu(), predictions.cpu())
                        
                        logger.info(f"Epoch {epoch}: Val Loss: {val_loss:.4f}, Accuracy: {accuracy:.4f}")
            
            # Save models
            self._save_models()
            
            training_time = time.time() - start_time
            ADVERSARIAL_TRAINING_TIME.observe(training_time)
            
            logger.info(f"Adversarial training completed in {training_time:.2f} seconds")
            
        except Exception as e:
            logger.error(f"Error training adversarially robust model: {e}")
    
    def _prepare_features(self, data: pd.DataFrame) -> np.ndarray:
        """Prepare features for training"""
        try:
            # Select feature columns
            feature_columns = [col for col in data.columns 
                             if col not in ['user_id', 'churn']]
            
            features = data[feature_columns].values
            
            # Handle missing values
            features = np.nan_to_num(features, nan=0.0)
            
            # Normalize features
            features = (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-8)
            
            return features
            
        except Exception as e:
            logger.error(f"Error preparing features: {e}")
            return np.zeros((len(data), 21))
    
    def detect_adversarial_attack(self, customer_features: np.ndarray, 
                                customer_id: str) -> AnomalyDetectionResult:
        """Detect adversarial attacks using multiple methods"""
        try:
            ANOMALY_DETECTION_REQUESTS.inc()
            
            # Method 1: Anomaly detection
            if self.anomaly_detector is not None:
                is_anomaly, anomaly_score, detection_scores = self.anomaly_detector.detect_anomaly(customer_features)
            else:
                is_anomaly, anomaly_score = False, 0.0
                detection_scores = {}
            
            # Method 2: Model-based detection
            if self.robust_model is not None:
                with torch.no_grad():
                    features_tensor = torch.FloatTensor(customer_features.reshape(1, -1))
                    outputs = self.robust_model(features_tensor, training_mode='defense')
                    model_anomaly_score = outputs['anomaly_score'].item()
                    
                    # Combine anomaly scores
                    combined_anomaly_score = (anomaly_score + model_anomaly_score) / 2
                    is_anomaly = combined_anomaly_score > 0.5
            else:
                combined_anomaly_score = anomaly_score
                model_anomaly_score = 0.0
            
            # Determine attack type and recommended action
            attack_type = self._classify_attack_type(customer_features, detection_scores)
            recommended_action = self._determine_defense_action(combined_anomaly_score, attack_type)
            
            # Record attack if detected
            if is_anomaly:
                self._record_adversarial_attack(customer_id, customer_features, attack_type, combined_anomaly_score)
                ADVERSARIAL_ATTACKS_DETECTED.inc()
            
            return AnomalyDetectionResult(
                customer_id=customer_id,
                is_anomaly=is_anomaly,
                anomaly_score=combined_anomaly_score,
                confidence=combined_anomaly_score,
                detected_attack_type=attack_type,
                recommended_action=recommended_action,
                metadata={
                    'detection_scores': detection_scores,
                    'model_anomaly_score': model_anomaly_score,
                    'timestamp': datetime.now().isoformat()
                }
            )
            
        except Exception as e:
            logger.error(f"Error detecting adversarial attack: {e}")
            return AnomalyDetectionResult(
                customer_id=customer_id,
                is_anomaly=False,
                anomaly_score=0.0,
                confidence=0.0,
                detected_attack_type=None,
                recommended_action="allow",
                metadata={'error': str(e)}
            )
    
    def _classify_attack_type(self, features: np.ndarray, detection_scores: Dict[str, float]) -> Optional[str]:
        """Classify the type of adversarial attack"""
        try:
            # Analyze feature patterns to determine attack type
            feature_variance = np.var(features)
            feature_range = np.max(features) - np.min(features)
            
            if feature_variance > 2.0:
                return "perturbation"
            elif feature_range > 5.0:
                return "injection"
            elif detection_scores.get('statistical', 0) > 0.8:
                return "evasion"
            else:
                return "unknown"
                
        except Exception as e:
            logger.error(f"Error classifying attack type: {e}")
            return "unknown"
    
    def _determine_defense_action(self, anomaly_score: float, attack_type: Optional[str]) -> str:
        """Determine appropriate defense action"""
        try:
            if anomaly_score > 0.8:
                return "block"
            elif anomaly_score > 0.6:
                return "challenge"
            elif anomaly_score > 0.4:
                return "monitor"
            else:
                return "allow"
                
        except Exception as e:
            logger.error(f"Error determining defense action: {e}")
            return "allow"
    
    def _record_adversarial_attack(self, customer_id: str, features: np.ndarray, 
                                 attack_type: Optional[str], confidence: float):
        """Record detected adversarial attack"""
        try:
            attack = AdversarialAttack(
                attack_id=f"attack_{int(time.time())}",
                attack_type=attack_type or "unknown",
                target_customer_id=customer_id,
                original_features=features,
                perturbed_features=features,  # Same for detection
                attack_success=False,  # Detected, so not successful
                detection_confidence=confidence,
                timestamp=datetime.now(),
                metadata={'detection_method': 'ensemble'}
            )
            
            self.attack_history.append(attack)
            
            # Cache attack record
            attack_data = {
                'attack_id': attack.attack_id,
                'attack_type': attack.attack_type,
                'target_customer_id': attack.target_customer_id,
                'detection_confidence': attack.detection_confidence,
                'timestamp': attack.timestamp.isoformat(),
                'metadata': attack.metadata
            }
            
            cache_key = f"adversarial_attack:{attack.attack_id}"
            self.redis_client.setex(
                cache_key,
                self.attack_cache_ttl,
                json.dumps(attack_data)
            )
            
        except Exception as e:
            logger.error(f"Error recording adversarial attack: {e}")
    
    def predict_with_protection(self, customer_features: np.ndarray, 
                              customer_id: str) -> Dict[str, Any]:
        """Make prediction with adversarial protection"""
        try:
            # Detect adversarial attacks
            detection_result = self.detect_adversarial_attack(customer_features, customer_id)
            
            # Make prediction if no attack detected or attack is allowed
            if not detection_result.is_anomaly or detection_result.recommended_action == "allow":
                if self.robust_model is not None:
                    with torch.no_grad():
                        features_tensor = torch.FloatTensor(customer_features.reshape(1, -1))
                        outputs = self.robust_model(features_tensor, training_mode='defense')
                        
                        probabilities = outputs['probabilities'].numpy()[0]
                        prediction = np.argmax(probabilities)
                        confidence = np.max(probabilities)
                else:
                    # Fallback to simple prediction
                    prediction = 0
                    confidence = 0.5
            else:
                # Attack detected, return safe prediction
                prediction = 0  # Default to no churn
                confidence = 0.3  # Low confidence due to attack
            
            return {
                'customer_id': customer_id,
                'prediction': int(prediction),
                'confidence': float(confidence),
                'anomaly_detected': detection_result.is_anomaly,
                'anomaly_score': detection_result.anomaly_score,
                'attack_type': detection_result.detected_attack_type,
                'recommended_action': detection_result.recommended_action,
                'metadata': detection_result.metadata
            }
            
        except Exception as e:
            logger.error(f"Error making protected prediction: {e}")
            return {
                'customer_id': customer_id,
                'prediction': 0,
                'confidence': 0.0,
                'anomaly_detected': False,
                'anomaly_score': 0.0,
                'attack_type': None,
                'recommended_action': 'allow',
                'metadata': {'error': str(e)}
            }
    
    def _save_models(self):
        """Save trained models"""
        try:
            os.makedirs(self.model_path, exist_ok=True)
            
            # Save robust model
            if self.robust_model is not None:
                torch.save(self.robust_model.state_dict(), f"{self.model_path}/robust_model.pth")
            
            # Save adversarial generator
            if self.adversarial_generator is not None:
                torch.save(self.adversarial_generator.state_dict(), f"{self.model_path}/adversarial_generator.pth")
            
            # Save anomaly detector
            if self.anomaly_detector is not None:
                joblib.dump(self.anomaly_detector, f"{self.model_path}/anomaly_detector.pkl")
            
            logger.info("Models saved successfully")
            
        except Exception as e:
            logger.error(f"Error saving models: {e}")

def create_sample_adversarial_data():
    """Create sample data for testing adversarial hardening"""
    np.random.seed(42)
    
    # Generate normal customer data
    n_customers = 1000
    normal_data = pd.DataFrame({
        'user_id': [f'user_{i:04d}' for i in range(n_customers)],
        'tenure': np.random.exponential(24, n_customers),
        'monthly_charges': np.random.normal(65, 20, n_customers),
        'total_charges': np.random.normal(1500, 800, n_customers),
        'contract_month_to_month': np.random.choice([0, 1], n_customers, p=[0.7, 0.3]),
        'internet_service_fiber': np.random.choice([0, 1], n_customers, p=[0.6, 0.4]),
        'payment_method_electronic': np.random.choice([0, 1], n_customers, p=[0.8, 0.2]),
        'online_security': np.random.choice([0, 1], n_customers, p=[0.6, 0.4]),
        'tech_support': np.random.choice([0, 1], n_customers, p=[0.7, 0.3]),
        'streaming_tv': np.random.choice([0, 1], n_customers, p=[0.5, 0.5]),
        'streaming_movies': np.random.choice([0, 1], n_customers, p=[0.5, 0.5]),
        'paperless_billing': np.random.choice([0, 1], n_customers, p=[0.6, 0.4]),
        'dependents': np.random.choice([0, 1], n_customers, p=[0.8, 0.2]),
        'partner': np.random.choice([0, 1], n_customers, p=[0.6, 0.4]),
        'phone_service': np.random.choice([0, 1], n_customers, p=[0.2, 0.8]),
        'multiple_lines': np.random.choice([0, 1], n_customers, p=[0.7, 0.3]),
        'online_backup': np.random.choice([0, 1], n_customers, p=[0.6, 0.4]),
        'device_protection': np.random.choice([0, 1], n_customers, p=[0.6, 0.4]),
        'churn': np.random.choice([0, 1], n_customers, p=[0.8, 0.2])
    })
    
    # Generate adversarial examples
    adversarial_data = normal_data.copy()
    
    # Perturbation attack: Add noise to features
    noise = np.random.normal(0, 0.5, (len(adversarial_data), 21))
    feature_columns = [col for col in adversarial_data.columns if col not in ['user_id', 'churn']]
    for i, col in enumerate(feature_columns):
        if i < noise.shape[1]:
            adversarial_data[col] += noise[:, i]
    
    # Injection attack: Insert extreme values
    injection_mask = np.random.choice([0, 1], len(adversarial_data), p=[0.9, 0.1])
    for i in range(len(adversarial_data)):
        if injection_mask[i]:
            # Inject extreme values
            adversarial_data.loc[i, 'monthly_charges'] = np.random.uniform(200, 500)
            adversarial_data.loc[i, 'tenure'] = np.random.uniform(0, 1)
    
    return normal_data, adversarial_data

if __name__ == "__main__":
    # Create sample data
    normal_data, adversarial_data = create_sample_adversarial_data()
    
    # Initialize adversarial hardener
    hardener = AdversarialHardener()
    
    # Train robust model
    print("Training adversarially robust model...")
    hardener.train_adversarially_robust_model(normal_data, normal_data)
    
    # Test adversarial detection
    print("\nTesting adversarial detection...")
    test_customers = adversarial_data.head(10)
    
    for _, customer in test_customers.iterrows():
        customer_id = customer['user_id']
        features = customer.drop(['user_id', 'churn']).values
        
        # Test protected prediction
        result = hardener.predict_with_protection(features, customer_id)
        
        print(f"{customer_id}: Prediction={result['prediction']}, "
              f"Anomaly={result['anomaly_detected']}, "
              f"Action={result['recommended_action']}")
    
    # Update model robustness score
    MODEL_ROBUSTNESS_SCORE.set(0.85)  # Example robustness score