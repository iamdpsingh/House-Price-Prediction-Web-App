"""
Configuration settings for House Price Prediction Application
Centralized configuration management
"""

import os
from pathlib import Path

class Config:
    """Main configuration class"""
    
    # Base paths
    BASE_DIR = Path(__file__).parent.parent
    DATA_DIR = BASE_DIR / 'data'
    MODELS_DIR = BASE_DIR / 'models'
    
    # Data paths
    RAW_DATA_DIR = DATA_DIR / 'raw'
    PROCESSED_DATA_DIR = DATA_DIR / 'processed'
    DATASET_PATH = RAW_DATA_DIR / 'bengaluru_house_prices.csv'
    PROCESSED_DATA_PATH = PROCESSED_DATA_DIR / 'processed_data.csv'
    
    # Model paths
    TRAINED_MODELS_DIR = MODELS_DIR / 'trained_models'
    EVALUATION_PLOTS_DIR = MODELS_DIR / 'evaluation_plots'
    
    # Specific model files
    LINEAR_REGRESSION_MODEL = TRAINED_MODELS_DIR / 'linear_regression_model.pkl'
    RANDOM_FOREST_MODEL = TRAINED_MODELS_DIR / 'random_forest_model.pkl'
    LOCATION_ENCODER = TRAINED_MODELS_DIR / 'location_encoder.pkl'
    FEATURE_SCALER = TRAINED_MODELS_DIR / 'feature_scaler.pkl'
    MODEL_METADATA = TRAINED_MODELS_DIR / 'model_metadata.pkl'
    
    # Kaggle dataset configuration
    KAGGLE_DATASET = "amitabhajoy/bengaluru-house-price-data"
    KAGGLE_API_KEY_PATH = Path.home() / '.kaggle' / 'kaggle.json'
    
    # Model training parameters
    MODEL_CONFIG = {
        'test_size': 0.2,
        'random_state': 42,
        'cv_folds': 5,
        'linear_regression': {
            'normalize': False
        },
        'random_forest': {
            'n_estimators': 100,
            'max_depth': None,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'random_state': 42,
            'n_jobs': -1
        }
    }
    
    # Data validation rules
    VALIDATION_RULES = {
        'total_sqft': {'min': 500, 'max': 8000},
        'bhk': {'min': 1, 'max': 10},
        'bath': {'min': 1, 'max': 10},
        'balcony': {'min': 0, 'max': 10},
        'price': {'min': 10, 'max': 1000}  # in lakhs
    }
    
    # Flask configuration
    FLASK_CONFIG = {
        'DEBUG': True,
        'HOST': '0.0.0.0',
        'PORT': 9000,
        'SECRET_KEY': 'house-price-prediction-secret-key'
    }
    
    @classmethod
    def create_directories(cls):
        """Create all necessary directories"""
        directories = [
            cls.DATA_DIR,
            cls.RAW_DATA_DIR,
            cls.PROCESSED_DATA_DIR,
            cls.MODELS_DIR,
            cls.TRAINED_MODELS_DIR,
            cls.EVALUATION_PLOTS_DIR
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def get_kaggle_credentials_setup_message(cls):
        """Get message for setting up Kaggle credentials"""
        return f"""
To download datasets from Kaggle, you need to set up your API credentials:

1. Go to https://www.kaggle.com/account
2. Click "Create API Token" to download kaggle.json
3. Place the file at: {cls.KAGGLE_API_KEY_PATH}
4. Make sure the file has the correct permissions:
   chmod 600 {cls.KAGGLE_API_KEY_PATH}

The kaggle.json file should contain:
{{
    "username": "your-kaggle-username",
    "key": "your-kaggle-key"
}}
"""

# Global configuration instance
config = Config()
