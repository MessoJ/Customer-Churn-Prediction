import logging
import os
import time
from functools import wraps

# Configure logging
def setup_logging():
    """Set up logging configuration for the project."""
    os.makedirs('logs', exist_ok=True)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Setup file handler
    file_handler = logging.FileHandler(f'logs/churn_prediction_{time.strftime("%Y%m%d")}.log')
    file_handler.setFormatter(formatter)
    
    # Setup console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    return root_logger

# Create a decorator to time functions
def time_it(func):
    """Decorator to measure and log the execution time of functions."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger = logging.getLogger(func.__module__)
        logger.info(f"Starting {func.__name__}")
        
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        duration = end_time - start_time
        logger.info(f"Finished {func.__name__} in {duration:.2f} seconds")
        
        return result
    return wrapper

# Create a project configuration class
class ChurnConfig:
    """Configuration class for the churn prediction project."""
    
    # Paths
    DATA_DIR = 'data'
    MODELS_DIR = 'models'
    RESULTS_DIR = 'results'
    
    # Ensure directories exist
    @classmethod
    def init_directories(cls):
        """Initialize project directories."""
        for directory in [cls.DATA_DIR, cls.MODELS_DIR, cls.RESULTS_DIR, 'logs']:
            os.makedirs(directory, exist_ok=True)
    
    # Important features used across modules
    IMPORTANT_FEATURES = [
        'remainder__tenure',
        'remainder__MonthlyCharges',
        'cat__Contract_Month-to-month',
        'AvgMonthlyCharge'
    ]
    
    # Model hyperparameters
    MODEL_PARAMS = {
        'n_estimators': 200,
        'max_depth': 10,
        'min_samples_leaf': 2,
        'class_weight': 'balanced',
        'random_state': 42
    }
    
    # Column prefixes
    PREFIX_CAT = 'cat__'
    PREFIX_REMAINDER = 'remainder__'
    
    # Target column
    TARGET_COL = 'remainder__Churn'
    
    # File paths
    RAW_DATA_PATH = f'{DATA_DIR}/telco_churn.csv'
    PROCESSED_DATA_PATH = f'{DATA_DIR}/processed_data.csv'
    ENGINEERED_DATA_PATH = f'{DATA_DIR}/engineered_data.csv'
    MODEL_PATH = f'{MODELS_DIR}/churn_model.joblib'
    FEATURE_NAMES_PATH = f'{MODELS_DIR}/feature_names.joblib'

# Example usage in a module:
# 
# from utils import setup_logging, time_it, ChurnConfig
# 
# logger = setup_logging()
# 
# @time_it
# def process_data():
#     logger.info("Processing data...")
#     # Your existing code here
#     logger.info("Data processing complete")
