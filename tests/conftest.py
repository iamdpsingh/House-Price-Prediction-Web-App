"""
Pytest Configuration and Fixtures
Shared test configuration and utilities for House Price Prediction tests
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import sys
import os
from pathlib import Path
from unittest.mock import Mock, patch
import json

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Configure pandas for testing
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

@pytest.fixture(scope="session")
def project_root_path():
    """Get the project root path"""
    return Path(__file__).parent.parent

@pytest.fixture(scope="session") 
def sample_dataset():
    """Create a sample dataset for testing"""
    np.random.seed(42)  # For reproducible tests
    
    n_samples = 100
    locations = ['Whitefield', 'Electronic City', 'Marathahalli', 'BTM Layout', 'Koramangala']
    
    data = {
        'location': np.random.choice(locations, n_samples),
        'total_sqft': np.random.normal(1200, 300, n_samples),
        'size': [f"{np.random.choice([1,2,3,4])} BHK" for _ in range(n_samples)],
        'bath': np.random.randint(1, 5, n_samples),
        'balcony': np.random.randint(0, 4, n_samples),
        'price': np.random.normal(80, 25, n_samples)
    }
    
    # Ensure positive values
    data['total_sqft'] = np.maximum(data['total_sqft'], 500)
    data['price'] = np.maximum(data['price'], 20)
    
    return pd.DataFrame(data)

@pytest.fixture(scope="session")
def sample_processed_dataset(sample_dataset):
    """Create a processed version of sample dataset"""
    df = sample_dataset.copy()
    
    # Extract BHK from size
    df['bhk'] = df['size'].str.extract(r'(\d+)').astype(int)    
    
    # Add derived features
    df['price_per_sqft'] = (df['price'] * 100000) / df['total_sqft']
    df['sqft_per_room'] = df['total_sqft'] / df['bhk']
    df['bath_per_bhk'] = df['bath'] / df['bhk']
    
    return df

@pytest.fixture
def valid_prediction_input():
    """Valid input data for prediction testing"""
    return {
        'location': 'Whitefield',
        'total_sqft': 1200,
        'bhk': 2,
        'bath': 2,
        'balcony': 1
    }

@pytest.fixture
def invalid_prediction_inputs():
    """Various invalid input data for testing"""
    return [
        # Missing required fields
        {
            'location': 'Whitefield',
            'total_sqft': 1200
            # Missing bhk, bath
        },
        
        # Invalid data types
        {
            'location': 'Whitefield',
            'total_sqft': 'invalid',
            'bhk': 2,
            'bath': 2,
            'balcony': 1
        },
        
        # Out of range values
        {
            'location': 'Whitefield',
            'total_sqft': -100,
            'bhk': 0,
            'bath': 20,
            'balcony': -5
        },
        
        # Empty location
        {
            'location': '',
            'total_sqft': 1200,
            'bhk': 2,
            'bath': 2,
            'balcony': 1
        }
    ]

@pytest.fixture
def mock_model():
    """Create a mock machine learning model"""
    model = Mock()
    model.predict.return_value = np.array([85.5])
    model.feature_importances_ = np.array([0.3, 0.25, 0.2, 0.15, 0.1])
    return model

@pytest.fixture
def mock_location_encoder():
    """Create a mock location encoder"""
    encoder = Mock()
    encoder.classes_ = np.array(['Whitefield', 'Electronic City', 'Marathahalli', 'BTM Layout'])
    encoder.transform.return_value = np.array([0])
    encoder.inverse_transform.return_value = np.array(['Whitefield'])
    return encoder

@pytest.fixture
def mock_scaler():
    """Create a mock feature scaler"""
    scaler = Mock()
    scaler.transform.return_value = np.array([[0.5, -0.2, 1.1, 0.8, -0.3]])
    scaler.inverse_transform.return_value = np.array([[1200, 2, 2, 1, 0.5]])
    return scaler

@pytest.fixture
def mock_predictor(mock_model, mock_location_encoder, mock_scaler):
    """Create a mock house price predictor"""
    try:
        from api.model_loader import HousePricePredictor
        
        predictor = HousePricePredictor()
        predictor.is_loaded = True
        predictor.random_forest_model = mock_model
        predictor.location_encoder = mock_location_encoder
        predictor.feature_scaler = mock_scaler
        predictor.best_model_name = 'random_forest'
        
        return predictor
    except ImportError:
        # Return a basic mock if the actual class isn't available
        predictor = Mock()
        predictor.is_loaded = True
        predictor.predict_price.return_value = {
            'predicted_price': 85.5,
            'price_category': 'Premium',
            'model_used': 'random_forest'
        }
        return predictor

@pytest.fixture
def temporary_model_files():
    """Create temporary model files for testing"""
    import tempfile
    import joblib
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create mock model files
        mock_model = Mock()
        mock_encoder = Mock()
        mock_scaler = Mock()
        
        # Save mock objects
        joblib.dump(mock_model, temp_path / 'best_model.pkl')
        joblib.dump(mock_encoder, temp_path / 'location_encoder.pkl')
        joblib.dump(mock_scaler, temp_path / 'feature_scaler.pkl')
        
        metadata = {
            'best_model': 'random_forest',
            'model_performance': {'test_r2': 0.89},
            'feature_columns': ['location_encoded', 'total_sqft', 'bhk', 'bath', 'balcony'],
            'location_classes': ['Whitefield', 'Electronic City']
        }
        joblib.dump(metadata, temp_path / 'model_metadata.pkl')
        
        yield temp_path

@pytest.fixture
def flask_test_app():
    """Create Flask test application"""
    try:
        from api.app import create_app
        
        app = create_app()
        app.config['TESTING'] = True
        app.config['DEBUG'] = False
        
        return app
    except ImportError:
        pytest.skip("Flask app not available")

@pytest.fixture
def flask_test_client(flask_test_app):
    """Create Flask test client"""
    return flask_test_app.test_client()

@pytest.fixture(autouse=True)
def setup_test_environment():
    """Set up test environment before each test"""
    # Set random seeds for reproducible tests
    np.random.seed(42)
    
    # Configure warnings
    import warnings
    warnings.filterwarnings('ignore', category=DeprecationWarning)
    warnings.filterwarnings('ignore', category=FutureWarning)
    
    yield
    
    # Cleanup after test (if needed)
    pass

@pytest.fixture
def sample_api_responses():
    """Sample API response data for testing"""
    return {
        'successful_prediction': {
            'success': True,
            'prediction': {
                'predicted_price': 85.5,
                'formatted_price': '₹85.50 lakhs',
                'price_range': {'min': 77.0, 'max': 94.0}
            },
            'property_details': {
                'location': 'Whitefield',
                'total_sqft': 1200,
                'bhk': 2,
                'bath': 2,
                'balcony': 1
            },
            'metrics': {
                'price_per_sqft': 7125.0,
                'sqft_per_room': 600.0
            },
            'category': {
                'name': 'Premium',
                'description': 'Upper-middle-class segment'
            }
        },
        
        'validation_error': {
            'success': False,
            'errors': [
                'Total square feet must be between 300-8000',
                'BHK must be at least 1'
            ]
        },
        
        'model_unavailable': {
            'success': False,
            'error': 'Model not available. Please ensure models are trained.'
        }
    }

# Test data generators
def generate_test_predictions(n_samples=10):
    """Generate test prediction data"""
    np.random.seed(42)
    
    locations = ['Whitefield', 'Electronic City', 'Marathahalli']
    
    predictions = []
    for _ in range(n_samples):
        predictions.append({
            'location': np.random.choice(locations),
            'total_sqft': np.random.randint(800, 2500),
            'bhk': np.random.randint(1, 5),
            'bath': np.random.randint(1, 4),
            'balcony': np.random.randint(0, 3),
            'expected_price_range': (50, 150)  # Expected price range in lakhs
        })
    
    return predictions

# Pytest configuration
def pytest_configure(config):
    """Configure pytest"""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )
    config.addinivalue_line(
        "markers", "api: marks tests as API tests"
    )

def pytest_collection_modifyitems(config, items):
    """Modify test collection"""
    # Add markers to tests based on file names
    for item in items:
        if "test_api" in item.nodeid:
            item.add_marker(pytest.mark.api)
        if "test_models" in item.nodeid:
            item.add_marker(pytest.mark.unit)
        if "integration" in item.nodeid:
            item.add_marker(pytest.mark.integration)

# Custom assertions
class CustomAssertions:
    """Custom assertion helpers for testing"""
    
    @staticmethod
    def assert_valid_price(price):
        """Assert that price is valid"""
        assert isinstance(price, (int, float))
        assert 10 <= price <= 500  # Reasonable price range in lakhs
    
    @staticmethod
    def assert_valid_prediction_response(response):
        """Assert that prediction response has valid structure"""
        assert 'success' in response
        
        if response['success']:
            assert 'prediction' in response or 'predicted_price' in response
            
            if 'prediction' in response:
                pred = response['prediction']
                assert 'predicted_price' in pred
                CustomAssertions.assert_valid_price(pred['predicted_price'])
        else:
            assert 'error' in response or 'errors' in response
    
    @staticmethod
    def assert_valid_model_structure(model):
        """Assert that model has required methods"""
        assert hasattr(model, 'predict')
        assert callable(getattr(model, 'predict'))

@pytest.fixture
def assertions():
    """Provide custom assertions"""
    return CustomAssertions

# Test utilities
class TestUtils:
    """Utility functions for testing"""
    
    @staticmethod
    def create_test_config():
        """Create test configuration"""
        return {
            'test_mode': True,
            'model_path': '/tmp/test_models',
            'data_path': '/tmp/test_data'
        }
    
    @staticmethod
    def mock_file_system(temp_dir):
        """Mock file system structure"""
        dirs_to_create = [
            'data/raw',
            'data/processed', 
            'models/trained_models',
            'models/evaluation_plots'
        ]
        
        for dir_path in dirs_to_create:
            (Path(temp_dir) / dir_path).mkdir(parents=True, exist_ok=True)
    
    @staticmethod
    def generate_mock_dataset(n_samples=100):
        """Generate mock dataset for testing"""
        np.random.seed(42)
        
        return pd.DataFrame({
            'location': np.random.choice(['Loc1', 'Loc2', 'Loc3'], n_samples),
            'total_sqft': np.random.randint(500, 3000, n_samples),
            'bhk': np.random.randint(1, 5, n_samples),
            'bath': np.random.randint(1, 4, n_samples),
            'balcony': np.random.randint(0, 3, n_samples),
            'price': np.random.uniform(30, 150, n_samples)
        })

@pytest.fixture
def test_utils():
    """Provide test utilities"""
    return TestUtils

# Performance testing helpers
@pytest.fixture
def performance_timer():
    """Timer for performance testing"""
    import time
    
    class Timer:
        def __init__(self):
            self.start_time = None
            self.end_time = None
        
        def start(self):
            self.start_time = time.time()
        
        def stop(self):
            self.end_time = time.time()
            return self.elapsed
        
        @property
        def elapsed(self):
            if self.start_time and self.end_time:
                return self.end_time - self.start_time
            return None
    
    return Timer()

# Database/file cleanup
@pytest.fixture(autouse=True)
def cleanup_test_files():
    """Clean up test files after each test"""
    yield
    
    # Clean up any temporary files created during testing
    temp_files = [
        'test_data.csv',
        'test_model.pkl',
        'test_results.json'
    ]
    
    for file_name in temp_files:
        if Path(file_name).exists():
            try:
                Path(file_name).unlink()
            except:
                pass  # Ignore cleanup errors

# Parameterized test data
@pytest.fixture(params=[
    {'location': 'Whitefield', 'sqft': 1000, 'bhk': 2, 'expected_range': (60, 100)},
    {'location': 'Electronic City', 'sqft': 1500, 'bhk': 3, 'expected_range': (80, 120)},
    {'location': 'Marathahalli', 'sqft': 800, 'bhk': 1, 'expected_range': (45, 75)},
])
def prediction_test_cases(request):
    """Parameterized test cases for predictions"""
    return request.param

# Skip conditions
def skip_if_no_models():
    """Skip test if model files are not available"""
    models_dir = Path('models/trained_models')
    if not models_dir.exists() or not list(models_dir.glob('*.pkl')):
        pytest.skip("Model files not available")

def skip_if_no_data():
    """Skip test if data files are not available"""
    data_dir = Path('data')
    if not data_dir.exists():
        pytest.skip("Data directory not available")

# Environment setup
@pytest.fixture(scope="session", autouse=True)
def setup_test_session():
    """Set up test session"""
    print("\n🧪 Starting test session for House Price Prediction App")
    
    # Set environment variables for testing
    os.environ['TESTING'] = '1'
    os.environ['LOG_LEVEL'] = 'ERROR'  # Reduce logging during tests
    
    yield
    
    print("\n✅ Test session completed")
    
    # Clean up environment
    if 'TESTING' in os.environ:
        del os.environ['TESTING']
    if 'LOG_LEVEL' in os.environ:
        del os.environ['LOG_LEVEL']
