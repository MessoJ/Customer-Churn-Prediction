"""
Cold-Start Handling with Graph Neural Networks
Implements GNNs leveraging referral networks and transfer learning from demographic clusters
Global scale best practices with distributed training and model serving
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv
from torch_geometric.data import Data, DataLoader
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score
import joblib
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import json
import pickle
from datetime import datetime
import redis
from prometheus_client import Counter, Histogram, Gauge
import os
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Prometheus metrics
GNN_TRAINING_TIME = Histogram('gnn_training_seconds', 'Time spent training GNN models')
COLD_START_PREDICTIONS = Counter('cold_start_predictions_total', 'Total cold-start predictions made')
TRANSFER_LEARNING_ACCURACY = Gauge('transfer_learning_accuracy', 'Accuracy of transfer learning models')
REFERRAL_NETWORK_SIZE = Gauge('referral_network_size', 'Size of referral network')

@dataclass
class UserNode:
    """User node in the referral network"""
    user_id: str
    features: np.ndarray
    demographic_cluster: int
    referral_count: int
    churn_probability: float
    is_new_user: bool
    registration_date: datetime
    referrer_id: Optional[str] = None

@dataclass
class ReferralEdge:
    """Edge representing referral relationship"""
    source_user_id: str
    target_user_id: str
    referral_date: datetime
    relationship_strength: float
    shared_features: Dict[str, float]

class GraphNeuralNetwork(nn.Module):
    """
    Graph Neural Network for cold-start user prediction
    Uses multiple GNN layers with attention mechanisms
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 128, output_dim: int = 1,
                 num_layers: int = 3, dropout: float = 0.2):
        super(GraphNeuralNetwork, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Multiple GNN layers with different architectures
        self.gnn_layers = nn.ModuleList()
        
        # First layer: Graph Convolution
        self.gnn_layers.append(GCNConv(input_dim, hidden_dim))
        
        # Middle layers: Graph Attention Networks
        for _ in range(num_layers - 2):
            self.gnn_layers.append(GATConv(hidden_dim, hidden_dim, heads=4, dropout=dropout))
        
        # Final layer: GraphSAGE for better generalization
        self.gnn_layers.append(SAGEConv(hidden_dim, hidden_dim))
        
        # Output layers
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim),
            nn.Sigmoid()
        )
        
        # Attention mechanism for cold-start users
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=8, dropout=dropout)
        
        # Transfer learning components
        self.domain_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
                batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through the GNN
        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Graph connectivity [2, num_edges]
            batch: Batch assignment for nodes
        """
        # Apply GNN layers
        for i, layer in enumerate(self.gnn_layers):
            if i == 0:
                x = layer(x, edge_index)
            else:
                x = layer(x, edge_index)
            
            if i < len(self.gnn_layers) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Apply attention mechanism for better feature aggregation
        if batch is not None:
            # Group nodes by batch and apply attention
            batch_size = batch.max().item() + 1
            attention_output = []
            
            for b in range(batch_size):
                mask = (batch == b)
                if mask.sum() > 0:
                    batch_nodes = x[mask].unsqueeze(0)  # [1, num_nodes_in_batch, hidden_dim]
                    attn_out, _ = self.attention(batch_nodes, batch_nodes, batch_nodes)
                    attention_output.append(attn_out.squeeze(0))
            
            if attention_output:
                x = torch.cat(attention_output, dim=0)
        
        # Global pooling (mean pooling)
        if batch is not None:
            x = torch.stack([x[batch == i].mean(0) for i in batch.unique()])
        
        # Classification
        return self.classifier(x)
    
    def get_embeddings(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Extract node embeddings for transfer learning"""
        for layer in self.gnn_layers:
            x = layer(x, edge_index)
            x = F.relu(x)
        
        return x

class DemographicClusterModel:
    """
    Transfer learning model using demographic clustering
    Helps with cold-start by leveraging similar user groups
    """
    
    def __init__(self, n_clusters: int = 10, random_state: int = 42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
        self.scaler = StandardScaler()
        self.cluster_models = {}
        self.cluster_features = {}
        
    def fit(self, user_features: np.ndarray, user_ids: List[str], 
            churn_labels: np.ndarray) -> None:
        """Fit demographic clustering model"""
        try:
            # Scale features
            scaled_features = self.scaler.fit_transform(user_features)
            
            # Perform clustering
            cluster_labels = self.kmeans.fit_predict(scaled_features)
            
            # Train separate models for each cluster
            for cluster_id in range(self.n_clusters):
                cluster_mask = cluster_labels == cluster_id
                
                if cluster_mask.sum() > 10:  # Minimum samples per cluster
                    cluster_features = user_features[cluster_mask]
                    cluster_labels_subset = churn_labels[cluster_mask]
                    
                    # Train simple model for this cluster
                    from sklearn.ensemble import RandomForestClassifier
                    cluster_model = RandomForestClassifier(
                        n_estimators=100,
                        max_depth=8,
                        random_state=self.random_state
                    )
                    cluster_model.fit(cluster_features, cluster_labels_subset)
                    
                    self.cluster_models[cluster_id] = cluster_model
                    self.cluster_features[cluster_id] = cluster_features.mean(axis=0)
            
            logger.info(f"Trained {len(self.cluster_models)} cluster models")
            
        except Exception as e:
            logger.error(f"Error fitting demographic cluster model: {e}")
    
    def predict_cold_start(self, user_features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict for cold-start users using transfer learning"""
        try:
            scaled_features = self.scaler.transform(user_features)
            cluster_labels = self.kmeans.predict(scaled_features)
            
            predictions = []
            confidences = []
            
            for i, cluster_id in enumerate(cluster_labels):
                if cluster_id in self.cluster_models:
                    # Use cluster-specific model
                    pred = self.cluster_models[cluster_id].predict_proba([user_features[i]])[0]
                    predictions.append(pred[1])  # Churn probability
                    confidences.append(pred.max())  # Confidence
                else:
                    # Fallback to global model
                    predictions.append(0.5)
                    confidences.append(0.5)
            
            return np.array(predictions), np.array(confidences)
            
        except Exception as e:
            logger.error(f"Error predicting cold-start: {e}")
            return np.zeros(len(user_features)), np.zeros(len(user_features))

class ReferralNetworkBuilder:
    """
    Builds and maintains referral network for GNN training
    Implements global scale best practices with distributed processing
    """
    
    def __init__(self, redis_host: str = 'localhost', redis_port: int = 6379):
        self.redis_client = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
        self.network_cache_ttl = 3600  # 1 hour
        
    def build_referral_network(self, users_data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Build referral network from user data
        Returns: node_features, edge_index, user_ids
        """
        try:
            # Create NetworkX graph
            G = nx.DiGraph()
            
            # Add nodes with features
            user_ids = []
            node_features = []
            
            for _, user in users_data.iterrows():
                user_id = user['user_id']
                user_ids.append(user_id)
                
                # Extract features for node
                features = self._extract_user_features(user)
                node_features.append(features)
                
                G.add_node(user_id, features=features)
            
            # Add edges for referrals
            for _, user in users_data.iterrows():
                if pd.notna(user.get('referrer_id')):
                    referrer_id = user['referrer_id']
                    if referrer_id in G:
                        # Add edge with relationship strength
                        strength = self._calculate_relationship_strength(user, users_data)
                        G.add_edge(referrer_id, user['user_id'], 
                                  weight=strength, 
                                  date=user.get('registration_date'))
            
            # Convert to PyTorch Geometric format
            edge_index = []
            edge_weights = []
            
            for source, target, data in G.edges(data=True):
                source_idx = user_ids.index(source)
                target_idx = user_ids.index(target)
                edge_index.append([source_idx, target_idx])
                edge_weights.append(data.get('weight', 1.0))
            
            if edge_index:
                edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
                edge_weights = torch.tensor(edge_weights, dtype=torch.float)
            else:
                # No edges, create self-loops
                edge_index = torch.tensor([[i, i] for i in range(len(user_ids))], 
                                        dtype=torch.long).t().contiguous()
                edge_weights = torch.ones(len(user_ids))
            
            node_features = torch.tensor(np.array(node_features), dtype=torch.float)
            
            # Cache network for quick access
            self._cache_network(user_ids, node_features, edge_index, edge_weights)
            
            # Update metrics
            REFERRAL_NETWORK_SIZE.set(len(user_ids))
            
            return node_features, edge_index, user_ids
            
        except Exception as e:
            logger.error(f"Error building referral network: {e}")
            return torch.tensor([]), torch.tensor([]), []
    
    def _extract_user_features(self, user: pd.Series) -> np.ndarray:
        """Extract features for user node"""
        try:
            # Basic demographic features
            features = [
                user.get('age', 0),
                user.get('income', 0),
                user.get('education_level', 0),
                user.get('tenure', 0),
                user.get('monthly_charges', 0),
                user.get('total_charges', 0),
                user.get('contract_month_to_month', 0),
                user.get('internet_service_fiber', 0),
                user.get('payment_method_electronic', 0),
                user.get('online_security', 0),
                user.get('tech_support', 0),
                user.get('streaming_tv', 0),
                user.get('streaming_movies', 0),
                user.get('paperless_billing', 0),
                user.get('dependents', 0),
                user.get('partner', 0),
                user.get('phone_service', 0),
                user.get('multiple_lines', 0),
                user.get('online_backup', 0),
                user.get('device_protection', 0)
            ]
            
            # Normalize features
            features = np.array(features, dtype=np.float32)
            features = (features - features.mean()) / (features.std() + 1e-8)
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting user features: {e}")
            return np.zeros(21, dtype=np.float32)
    
    def _calculate_relationship_strength(self, user: pd.Series, all_users: pd.DataFrame) -> float:
        """Calculate relationship strength between referrer and user"""
        try:
            referrer_id = user.get('referrer_id')
            if pd.isna(referrer_id):
                return 0.0
            
            referrer = all_users[all_users['user_id'] == referrer_id]
            if referrer.empty:
                return 0.0
            
            referrer = referrer.iloc[0]
            
            # Calculate similarity-based strength
            similarity_score = 0.0
            
            # Age similarity
            age_diff = abs(user.get('age', 0) - referrer.get('age', 0))
            if age_diff < 5:
                similarity_score += 0.3
            elif age_diff < 10:
                similarity_score += 0.2
            
            # Income similarity
            income_diff = abs(user.get('income', 0) - referrer.get('income', 0))
            if income_diff < 10000:
                similarity_score += 0.3
            elif income_diff < 20000:
                similarity_score += 0.2
            
            # Service similarity
            service_features = ['online_security', 'tech_support', 'streaming_tv', 
                              'streaming_movies', 'online_backup', 'device_protection']
            service_similarity = sum(1 for feat in service_features 
                                   if user.get(feat, 0) == referrer.get(feat, 0))
            similarity_score += (service_similarity / len(service_features)) * 0.4
            
            return min(similarity_score, 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating relationship strength: {e}")
            return 0.5
    
    def _cache_network(self, user_ids: List[str], node_features: torch.Tensor, 
                      edge_index: torch.Tensor, edge_weights: torch.Tensor):
        """Cache network data for quick access"""
        try:
            network_data = {
                'user_ids': user_ids,
                'node_features': node_features.numpy().tolist(),
                'edge_index': edge_index.numpy().tolist(),
                'edge_weights': edge_weights.numpy().tolist(),
                'timestamp': datetime.now().isoformat()
            }
            
            self.redis_client.setex(
                'referral_network',
                self.network_cache_ttl,
                json.dumps(network_data)
            )
            
        except Exception as e:
            logger.error(f"Error caching network: {e}")

class ColdStartHandler:
    """
    Main cold-start handling system combining GNNs and transfer learning
    Implements global scale best practices with monitoring and observability
    """
    
    def __init__(self, model_path: str = 'models/', redis_host: str = 'localhost'):
        self.model_path = model_path
        self.redis_client = redis.Redis(host=redis_host, port=6379, decode_responses=True)
        
        # Initialize components
        self.gnn_model = None
        self.demographic_model = None
        self.network_builder = ReferralNetworkBuilder(redis_host)
        
        # Load pre-trained models if available
        self._load_models()
    
    def _load_models(self):
        """Load pre-trained models"""
        try:
            # Load GNN model
            gnn_path = f"{self.model_path}/cold_start_gnn.pth"
            if os.path.exists(gnn_path):
                self.gnn_model = GraphNeuralNetwork(input_dim=21)
                self.gnn_model.load_state_dict(torch.load(gnn_path))
                self.gnn_model.eval()
                logger.info("Loaded pre-trained GNN model")
            
            # Load demographic model
            demo_path = f"{self.model_path}/demographic_cluster_model.pkl"
            if os.path.exists(demo_path):
                self.demographic_model = joblib.load(demo_path)
                logger.info("Loaded pre-trained demographic model")
                
        except Exception as e:
            logger.error(f"Error loading models: {e}")
    
    def train_models(self, training_data: pd.DataFrame, validation_data: pd.DataFrame):
        """Train both GNN and demographic models"""
        try:
            start_time = time.time()
            
            # Build referral network
            node_features, edge_index, user_ids = self.network_builder.build_referral_network(training_data)
            
            if node_features.numel() == 0:
                logger.error("Failed to build referral network")
                return
            
            # Prepare labels
            labels = training_data['churn'].values
            labels_tensor = torch.tensor(labels, dtype=torch.float)
            
            # Create PyTorch Geometric data
            data = Data(x=node_features, edge_index=edge_index, y=labels_tensor)
            
            # Train GNN model
            self._train_gnn_model(data, validation_data)
            
            # Train demographic model
            self._train_demographic_model(training_data)
            
            # Save models
            self._save_models()
            
            training_time = time.time() - start_time
            GNN_TRAINING_TIME.observe(training_time)
            
            logger.info(f"Model training completed in {training_time:.2f} seconds")
            
        except Exception as e:
            logger.error(f"Error training models: {e}")
    
    def _train_gnn_model(self, data: Data, validation_data: pd.DataFrame):
        """Train GNN model with validation"""
        try:
            # Initialize model
            self.gnn_model = GraphNeuralNetwork(input_dim=21)
            
            # Training parameters
            optimizer = torch.optim.Adam(self.gnn_model.parameters(), lr=0.001)
            criterion = nn.BCELoss()
            
            # Training loop
            num_epochs = 100
            best_val_loss = float('inf')
            
            for epoch in range(num_epochs):
                self.gnn_model.train()
                optimizer.zero_grad()
                
                # Forward pass
                outputs = self.gnn_model(data.x, data.edge_index)
                loss = criterion(outputs.squeeze(), data.y)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                # Validation
                if epoch % 10 == 0:
                    self.gnn_model.eval()
                    with torch.no_grad():
                        val_outputs = self.gnn_model(data.x, data.edge_index)
                        val_loss = criterion(val_outputs.squeeze(), data.y)
                        
                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            # Save best model
                            torch.save(self.gnn_model.state_dict(), 
                                     f"{self.model_path}/cold_start_gnn.pth")
            
            logger.info("GNN model training completed")
            
        except Exception as e:
            logger.error(f"Error training GNN model: {e}")
    
    def _train_demographic_model(self, training_data: pd.DataFrame):
        """Train demographic clustering model"""
        try:
            # Extract features for clustering
            feature_columns = [col for col in training_data.columns 
                             if col not in ['user_id', 'churn', 'referrer_id']]
            
            user_features = training_data[feature_columns].values
            churn_labels = training_data['churn'].values
            
            # Initialize and train demographic model
            self.demographic_model = DemographicClusterModel(n_clusters=10)
            self.demographic_model.fit(user_features, training_data['user_id'].tolist(), churn_labels)
            
            logger.info("Demographic model training completed")
            
        except Exception as e:
            logger.error(f"Error training demographic model: {e}")
    
    def predict_cold_start(self, new_users: pd.DataFrame) -> Dict[str, Any]:
        """
        Predict churn probability for cold-start users
        Combines GNN and demographic approaches
        """
        try:
            COLD_START_PREDICTIONS.inc()
            
            predictions = {}
            
            for _, user in new_users.iterrows():
                user_id = user['user_id']
                
                # Method 1: GNN-based prediction
                gnn_pred = self._predict_with_gnn(user)
                
                # Method 2: Demographic transfer learning
                demo_pred, demo_conf = self._predict_with_demographics(user)
                
                # Ensemble prediction (weighted average)
                if gnn_pred is not None and demo_pred is not None:
                    final_pred = (gnn_pred * 0.6) + (demo_pred * 0.4)
                    confidence = (demo_conf * 0.7) + 0.3  # GNN confidence assumed 0.3
                elif gnn_pred is not None:
                    final_pred = gnn_pred
                    confidence = 0.5
                elif demo_pred is not None:
                    final_pred = demo_pred
                    confidence = demo_conf
                else:
                    final_pred = 0.5  # Default neutral prediction
                    confidence = 0.3
                
                predictions[user_id] = {
                    'churn_probability': float(final_pred),
                    'confidence': float(confidence),
                    'gnn_prediction': float(gnn_pred) if gnn_pred is not None else None,
                    'demographic_prediction': float(demo_pred) if demo_pred is not None else None,
                    'prediction_method': 'ensemble' if gnn_pred is not None and demo_pred is not None else 'fallback'
                }
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error predicting cold-start: {e}")
            return {}
    
    def _predict_with_gnn(self, user: pd.Series) -> Optional[float]:
        """Predict using GNN model"""
        try:
            if self.gnn_model is None:
                return None
            
            # Extract user features
            features = self.network_builder._extract_user_features(user)
            features_tensor = torch.tensor(features, dtype=torch.float).unsqueeze(0)
            
            # Create minimal graph with single node
            edge_index = torch.tensor([[0, 0]], dtype=torch.long).t().contiguous()
            
            # Predict
            self.gnn_model.eval()
            with torch.no_grad():
                prediction = self.gnn_model(features_tensor, edge_index)
                return prediction.item()
            
        except Exception as e:
            logger.error(f"Error predicting with GNN: {e}")
            return None
    
    def _predict_with_demographics(self, user: pd.Series) -> Tuple[Optional[float], Optional[float]]:
        """Predict using demographic transfer learning"""
        try:
            if self.demographic_model is None:
                return None, None
            
            # Extract features
            feature_columns = [col for col in user.index 
                             if col not in ['user_id', 'churn', 'referrer_id']]
            user_features = user[feature_columns].values.reshape(1, -1)
            
            # Predict
            predictions, confidences = self.demographic_model.predict_cold_start(user_features)
            
            return predictions[0], confidences[0]
            
        except Exception as e:
            logger.error(f"Error predicting with demographics: {e}")
            return None, None
    
    def _save_models(self):
        """Save trained models"""
        try:
            os.makedirs(self.model_path, exist_ok=True)
            
            # Save GNN model
            if self.gnn_model is not None:
                torch.save(self.gnn_model.state_dict(), f"{self.model_path}/cold_start_gnn.pth")
            
            # Save demographic model
            if self.demographic_model is not None:
                joblib.dump(self.demographic_model, f"{self.model_path}/demographic_cluster_model.pkl")
            
            logger.info("Models saved successfully")
            
        except Exception as e:
            logger.error(f"Error saving models: {e}")

def create_sample_cold_start_data():
    """Create sample data for testing cold-start handling"""
    np.random.seed(42)
    
    # Generate sample users with referral relationships
    n_users = 1000
    users = []
    
    for i in range(n_users):
        user = {
            'user_id': f'user_{i:04d}',
            'age': np.random.normal(35, 10),
            'income': np.random.normal(50000, 15000),
            'education_level': np.random.randint(1, 5),
            'tenure': np.random.exponential(24),
            'monthly_charges': np.random.normal(65, 20),
            'total_charges': np.random.normal(1500, 800),
            'contract_month_to_month': np.random.choice([0, 1], p=[0.7, 0.3]),
            'internet_service_fiber': np.random.choice([0, 1], p=[0.6, 0.4]),
            'payment_method_electronic': np.random.choice([0, 1], p=[0.8, 0.2]),
            'online_security': np.random.choice([0, 1], p=[0.6, 0.4]),
            'tech_support': np.random.choice([0, 1], p=[0.7, 0.3]),
            'streaming_tv': np.random.choice([0, 1], p=[0.5, 0.5]),
            'streaming_movies': np.random.choice([0, 1], p=[0.5, 0.5]),
            'paperless_billing': np.random.choice([0, 1], p=[0.6, 0.4]),
            'dependents': np.random.choice([0, 1], p=[0.8, 0.2]),
            'partner': np.random.choice([0, 1], p=[0.6, 0.4]),
            'phone_service': np.random.choice([0, 1], p=[0.2, 0.8]),
            'multiple_lines': np.random.choice([0, 1], p=[0.7, 0.3]),
            'online_backup': np.random.choice([0, 1], p=[0.6, 0.4]),
            'device_protection': np.random.choice([0, 1], p=[0.6, 0.4]),
            'churn': np.random.choice([0, 1], p=[0.8, 0.2]),
            'referrer_id': None
        }
        
        # Add referral relationships
        if i > 0 and np.random.random() < 0.3:  # 30% referral rate
            user['referrer_id'] = f'user_{np.random.randint(0, i):04d}'
        
        users.append(user)
    
    return pd.DataFrame(users)

if __name__ == "__main__":
    # Create sample data
    training_data = create_sample_cold_start_data()
    
    # Initialize cold-start handler
    cold_start_handler = ColdStartHandler()
    
    # Train models
    cold_start_handler.train_models(training_data, training_data)
    
    # Test cold-start prediction
    new_users = training_data.head(10).copy()
    predictions = cold_start_handler.predict_cold_start(new_users)
    
    print("Cold-start predictions:")
    for user_id, pred in predictions.items():
        print(f"{user_id}: {pred['churn_probability']:.3f} (confidence: {pred['confidence']:.3f})")