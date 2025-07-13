"""
Real-Time Adaptation with Streaming Features
Implements Kafka + Flink architecture for real-time feature engineering
Global scale best practices with monitoring, observability, and fault tolerance
"""

import json
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd
from kafka import KafkaProducer, KafkaConsumer
from kafka.errors import KafkaError
import redis
from prometheus_client import Counter, Histogram, Gauge
import opentelemetry
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.jaeger.thrift import JaegerExporter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure OpenTelemetry for distributed tracing
trace.set_tracer_provider(TracerProvider())
tracer = trace.get_tracer(__name__)

# Prometheus metrics
FEATURE_PROCESSING_TIME = Histogram('feature_processing_seconds', 'Time spent processing features')
STREAMING_EVENTS_TOTAL = Counter('streaming_events_total', 'Total streaming events processed')
ACTIVE_SESSIONS = Gauge('active_sessions', 'Number of active user sessions')
FEATURE_QUEUE_SIZE = Gauge('feature_queue_size', 'Size of feature processing queue')

@dataclass
class UserEvent:
    """User interaction event for real-time feature engineering"""
    user_id: str
    event_type: str  # 'click', 'purchase', 'cart_add', 'page_view'
    timestamp: datetime
    session_id: str
    page_url: Optional[str] = None
    product_id: Optional[str] = None
    cart_value: Optional[float] = None
    click_coordinates: Optional[tuple] = None
    time_on_page: Optional[float] = None

@dataclass
class StreamingFeature:
    """Real-time computed feature"""
    user_id: str
    feature_name: str
    feature_value: float
    timestamp: datetime
    confidence: float = 1.0
    metadata: Dict[str, Any] = None

class RealTimeFeatureEngine:
    """
    Real-time feature engineering engine using Kafka + Flink architecture
    Implements global scale best practices with monitoring and observability
    """
    
    def __init__(self, kafka_bootstrap_servers: str = 'localhost:9092',
                 redis_host: str = 'localhost', redis_port: int = 6379):
        self.kafka_bootstrap_servers = kafka_bootstrap_servers
        self.redis_client = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
        
        # Initialize Kafka producer and consumer
        self.producer = KafkaProducer(
            bootstrap_servers=kafka_bootstrap_servers,
            value_serializer=lambda v: json.dumps(v, default=str).encode('utf-8'),
            acks='all',  # Ensure all replicas acknowledge
            retries=3,
            max_in_flight_requests_per_connection=1
        )
        
        self.consumer = KafkaConsumer(
            'user_events',
            bootstrap_servers=kafka_bootstrap_servers,
            value_deserializer=lambda m: json.loads(m.decode('utf-8')),
            auto_offset_reset='latest',
            enable_auto_commit=True,
            group_id='feature_engineering_group'
        )
        
        # Session management
        self.session_timeout = 1800  # 30 minutes
        self.feature_cache_ttl = 300  # 5 minutes
        
        # Initialize metrics
        self._setup_metrics()
    
    def _setup_metrics(self):
        """Setup Prometheus metrics for monitoring"""
        self.feature_processing_time = FEATURE_PROCESSING_TIME
        self.streaming_events_total = STREAMING_EVENTS_TOTAL
        self.active_sessions = ACTIVE_SESSIONS
        self.feature_queue_size = FEATURE_QUEUE_SIZE
    
    def compute_session_urgency(self, user_id: str, session_id: str) -> float:
        """
        Compute session urgency based on recent activity
        Higher values indicate higher urgency (likely to churn)
        """
        with tracer.start_as_current_span("compute_session_urgency"):
            try:
                # Get recent events for this session
                recent_events = self._get_recent_session_events(user_id, session_id, minutes=5)
                
                if not recent_events:
                    return 0.0
                
                # Calculate urgency factors
                time_since_last_click = self._get_time_since_last_click(recent_events)
                cart_value = self._get_current_cart_value(user_id)
                page_view_frequency = len(recent_events) / 5.0  # events per minute
                
                # Urgency formula: (time_since_last_click < 5s) & (cart_value > $100)
                urgency_score = 0.0
                
                if time_since_last_click < 5.0:  # Less than 5 seconds
                    urgency_score += 0.4
                
                if cart_value > 100.0:  # High cart value
                    urgency_score += 0.3
                
                if page_view_frequency > 2.0:  # High activity
                    urgency_score += 0.3
                
                # Normalize to 0-1 range
                return min(urgency_score, 1.0)
                
            except Exception as e:
                logger.error(f"Error computing session urgency: {e}")
                return 0.0
    
    def compute_real_time_features(self, user_event: UserEvent) -> List[StreamingFeature]:
        """
        Compute real-time features from user events
        Implements advanced feature engineering with monitoring
        """
        with tracer.start_as_current_span("compute_real_time_features"):
            start_time = time.time()
            
            try:
                features = []
                user_id = user_event.user_id
                session_id = user_event.session_id
                
                # 1. Session urgency (primary feature)
                urgency = self.compute_session_urgency(user_id, session_id)
                features.append(StreamingFeature(
                    user_id=user_id,
                    feature_name="current_session_urgency",
                    feature_value=urgency,
                    timestamp=user_event.timestamp,
                    confidence=0.9 if urgency > 0.5 else 0.7
                ))
                
                # 2. Click velocity (clicks per minute)
                click_velocity = self._compute_click_velocity(user_id, minutes=10)
                features.append(StreamingFeature(
                    user_id=user_id,
                    feature_name="click_velocity",
                    feature_value=click_velocity,
                    timestamp=user_event.timestamp
                ))
                
                # 3. Cart abandonment risk
                cart_abandonment_risk = self._compute_cart_abandonment_risk(user_id)
                features.append(StreamingFeature(
                    user_id=user_id,
                    feature_name="cart_abandonment_risk",
                    feature_value=cart_abandonment_risk,
                    timestamp=user_event.timestamp
                ))
                
                # 4. Page engagement score
                engagement_score = self._compute_engagement_score(user_event)
                features.append(StreamingFeature(
                    user_id=user_id,
                    feature_name="page_engagement_score",
                    feature_value=engagement_score,
                    timestamp=user_event.timestamp
                ))
                
                # 5. Real-time churn probability
                churn_probability = self._compute_real_time_churn_probability(user_id)
                features.append(StreamingFeature(
                    user_id=user_id,
                    feature_name="real_time_churn_probability",
                    feature_value=churn_probability,
                    timestamp=user_event.timestamp,
                    confidence=0.85
                ))
                
                # Update metrics
                processing_time = time.time() - start_time
                self.feature_processing_time.observe(processing_time)
                self.streaming_events_total.inc()
                
                # Cache features for quick access
                self._cache_features(user_id, features)
                
                return features
                
            except Exception as e:
                logger.error(f"Error computing real-time features: {e}")
                return []
    
    def _compute_click_velocity(self, user_id: str, minutes: int = 10) -> float:
        """Compute clicks per minute over recent time window"""
        try:
            recent_clicks = self._get_recent_events_by_type(user_id, 'click', minutes)
            return len(recent_clicks) / minutes
        except Exception as e:
            logger.error(f"Error computing click velocity: {e}")
            return 0.0
    
    def _compute_cart_abandonment_risk(self, user_id: str) -> float:
        """Compute risk of cart abandonment based on behavior patterns"""
        try:
            # Get cart events and purchase events
            cart_events = self._get_recent_events_by_type(user_id, 'cart_add', minutes=30)
            purchase_events = self._get_recent_events_by_type(user_id, 'purchase', minutes=30)
            
            if not cart_events:
                return 0.0
            
            # Calculate abandonment rate
            cart_count = len(cart_events)
            purchase_count = len(purchase_events)
            
            if cart_count == 0:
                return 0.0
            
            abandonment_rate = 1 - (purchase_count / cart_count)
            
            # Factor in time since last cart addition
            if cart_events:
                last_cart_time = max(event['timestamp'] for event in cart_events)
                time_since_cart = (datetime.now() - last_cart_time).total_seconds() / 60
                
                # Higher risk if cart is older
                time_factor = min(time_since_cart / 60, 1.0)  # Normalize to 0-1
                return (abandonment_rate * 0.7) + (time_factor * 0.3)
            
            return abandonment_rate
            
        except Exception as e:
            logger.error(f"Error computing cart abandonment risk: {e}")
            return 0.0
    
    def _compute_engagement_score(self, user_event: UserEvent) -> float:
        """Compute page engagement score based on user interaction"""
        try:
            engagement_score = 0.0
            
            # Time on page factor
            if user_event.time_on_page:
                if user_event.time_on_page > 300:  # 5+ minutes
                    engagement_score += 0.4
                elif user_event.time_on_page > 60:  # 1+ minutes
                    engagement_score += 0.2
            
            # Click coordinates factor (indicates active engagement)
            if user_event.click_coordinates:
                engagement_score += 0.3
            
            # Product interaction factor
            if user_event.product_id:
                engagement_score += 0.3
            
            return min(engagement_score, 1.0)
            
        except Exception as e:
            logger.error(f"Error computing engagement score: {e}")
            return 0.0
    
    def _compute_real_time_churn_probability(self, user_id: str) -> float:
        """Compute real-time churn probability using multiple signals"""
        try:
            # Get various risk factors
            urgency = self.compute_session_urgency(user_id, "current")
            cart_abandonment = self._compute_cart_abandonment_risk(user_id)
            click_velocity = self._compute_click_velocity(user_id, 5)
            
            # Historical churn risk from Redis cache
            historical_risk = float(self.redis_client.get(f"churn_risk:{user_id}") or 0.0)
            
            # Weighted combination of factors
            churn_probability = (
                urgency * 0.3 +
                cart_abandonment * 0.25 +
                (1 - min(click_velocity / 5.0, 1.0)) * 0.2 +  # Lower velocity = higher risk
                historical_risk * 0.25
            )
            
            return min(churn_probability, 1.0)
            
        except Exception as e:
            logger.error(f"Error computing real-time churn probability: {e}")
            return 0.5  # Default to neutral probability
    
    def _get_recent_session_events(self, user_id: str, session_id: str, minutes: int) -> List[Dict]:
        """Get recent events for a specific session"""
        try:
            # In production, this would query a time-series database
            # For demo, we'll use Redis with TTL
            key = f"session_events:{user_id}:{session_id}"
            events_json = self.redis_client.get(key)
            
            if events_json:
                events = json.loads(events_json)
                cutoff_time = datetime.now() - timedelta(minutes=minutes)
                return [e for e in events if datetime.fromisoformat(e['timestamp']) > cutoff_time]
            
            return []
            
        except Exception as e:
            logger.error(f"Error getting recent session events: {e}")
            return []
    
    def _get_time_since_last_click(self, events: List[Dict]) -> float:
        """Calculate time since last click event"""
        try:
            if not events:
                return float('inf')
            
            click_events = [e for e in events if e.get('event_type') == 'click']
            if not click_events:
                return float('inf')
            
            last_click_time = max(datetime.fromisoformat(e['timestamp']) for e in click_events)
            return (datetime.now() - last_click_time).total_seconds()
            
        except Exception as e:
            logger.error(f"Error calculating time since last click: {e}")
            return float('inf')
    
    def _get_current_cart_value(self, user_id: str) -> float:
        """Get current cart value for user"""
        try:
            cart_value = self.redis_client.get(f"cart_value:{user_id}")
            return float(cart_value) if cart_value else 0.0
        except Exception as e:
            logger.error(f"Error getting cart value: {e}")
            return 0.0
    
    def _get_recent_events_by_type(self, user_id: str, event_type: str, minutes: int) -> List[Dict]:
        """Get recent events of specific type for user"""
        try:
            # In production, this would query a time-series database
            key = f"user_events:{user_id}:{event_type}"
            events_json = self.redis_client.get(key)
            
            if events_json:
                events = json.loads(events_json)
                cutoff_time = datetime.now() - timedelta(minutes=minutes)
                return [e for e in events if datetime.fromisoformat(e['timestamp']) > cutoff_time]
            
            return []
            
        except Exception as e:
            logger.error(f"Error getting recent events by type: {e}")
            return []
    
    def _cache_features(self, user_id: str, features: List[StreamingFeature]):
        """Cache computed features for quick access"""
        try:
            feature_dict = {f.feature_name: f.feature_value for f in features}
            self.redis_client.setex(
                f"streaming_features:{user_id}",
                self.feature_cache_ttl,
                json.dumps(feature_dict)
            )
        except Exception as e:
            logger.error(f"Error caching features: {e}")
    
    def process_event_stream(self):
        """Process incoming event stream from Kafka"""
        logger.info("Starting event stream processing...")
        
        try:
            for message in self.consumer:
                with tracer.start_as_current_span("process_event"):
                    try:
                        # Parse event
                        event_data = message.value
                        user_event = UserEvent(
                            user_id=event_data['user_id'],
                            event_type=event_data['event_type'],
                            timestamp=datetime.fromisoformat(event_data['timestamp']),
                            session_id=event_data['session_id'],
                            page_url=event_data.get('page_url'),
                            product_id=event_data.get('product_id'),
                            cart_value=event_data.get('cart_value'),
                            click_coordinates=event_data.get('click_coordinates'),
                            time_on_page=event_data.get('time_on_page')
                        )
                        
                        # Compute real-time features
                        features = self.compute_real_time_features(user_event)
                        
                        # Publish features to output topic
                        for feature in features:
                            self.producer.send(
                                'streaming_features',
                                value=asdict(feature)
                            )
                        
                        # Update session tracking
                        self._update_session_tracking(user_event)
                        
                        # Update metrics
                        self.active_sessions.inc()
                        
                    except Exception as e:
                        logger.error(f"Error processing event: {e}")
                        continue
                        
        except KeyboardInterrupt:
            logger.info("Stopping event stream processing...")
        finally:
            self.consumer.close()
            self.producer.close()
    
    def _update_session_tracking(self, user_event: UserEvent):
        """Update session tracking and metrics"""
        try:
            session_key = f"session:{user_event.user_id}:{user_event.session_id}"
            self.redis_client.setex(session_key, self.session_timeout, "active")
            
            # Update session events
            events_key = f"session_events:{user_event.user_id}:{user_event.session_id}"
            events_json = self.redis_client.get(events_key)
            events = json.loads(events_json) if events_json else []
            
            events.append(asdict(user_event))
            self.redis_client.setex(events_key, self.session_timeout, json.dumps(events))
            
        except Exception as e:
            logger.error(f"Error updating session tracking: {e}")

def create_sample_event_stream():
    """Create sample event stream for testing"""
    producer = KafkaProducer(
        bootstrap_servers='localhost:9092',
        value_serializer=lambda v: json.dumps(v, default=str).encode('utf-8')
    )
    
    sample_events = [
        {
            'user_id': 'user_001',
            'event_type': 'click',
            'timestamp': datetime.now().isoformat(),
            'session_id': 'session_001',
            'page_url': '/products/phone',
            'click_coordinates': (100, 200),
            'time_on_page': 45.5
        },
        {
            'user_id': 'user_001',
            'event_type': 'cart_add',
            'timestamp': datetime.now().isoformat(),
            'session_id': 'session_001',
            'product_id': 'phone_001',
            'cart_value': 150.0
        }
    ]
    
    for event in sample_events:
        producer.send('user_events', value=event)
    
    producer.flush()
    producer.close()

if __name__ == "__main__":
    # Initialize the real-time feature engine
    feature_engine = RealTimeFeatureEngine()
    
    # Start processing event stream
    feature_engine.process_event_stream()