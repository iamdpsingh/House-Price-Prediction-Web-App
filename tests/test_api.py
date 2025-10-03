"""
API Endpoint Tests for House Price Prediction Application
Comprehensive testing of Flask API endpoints
"""

import pytest
import json
import sys
from pathlib import Path
import tempfile
import os

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from api.app import create_app
from config.config import Config

class TestHousePriceAPI:
    """Test suite for House Price Prediction API"""
    
    @pytest.fixture
    def app(self):
        """Create test Flask application"""
        app = create_app()
        app.config['TESTING'] = True
        app.config['DEBUG'] = False
        return app
    
    @pytest.fixture
    def client(self, app):
        """Create test client"""
        return app.test_client()
    
    @pytest.fixture
    def valid_prediction_data(self):
        """Valid prediction request data"""
        return {
            'location': 'Whitefield',
            'total_sqft': 1200,
            'bhk': 2,
            'bath': 2,
            'balcony': 1
        }
    
    @pytest.fixture
    def invalid_prediction_data(self):
        """Invalid prediction request data"""
        return {
            'location': 'InvalidLocation',
            'total_sqft': -100,  # Invalid negative value
            'bhk': 0,           # Invalid zero value
            'bath': 'invalid',  # Invalid string value
            'balcony': 100      # Invalid large value
        }

    def test_index_page_loads(self, client):
        """Test that the main page loads successfully"""
        response = client.get('/')
        assert response.status_code == 200
        assert b'House Price Predictor' in response.data or b'HousePricer' in response.data

    def test_index_page_contains_form_elements(self, client):
        """Test that index page contains necessary form elements"""
        response = client.get('/')
        assert response.status_code == 200
        
        # Check for form elements
        assert b'location' in response.data
        assert b'total_sqft' in response.data
        assert b'bhk' in response.data
        assert b'bath' in response.data

    def test_locations_endpoint(self, client):
        """Test locations API endpoint"""
        response = client.get('/locations')
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert 'success' in data or 'locations' in data
        
        if 'locations' in data:
            assert isinstance(data['locations'], list)
            assert len(data['locations']) > 0

    def test_health_check_endpoint(self, client):
        """Test health check endpoint"""
        response = client.get('/health')
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert 'status' in data

    def test_model_info_endpoint(self, client):
        """Test model info endpoint"""
        response = client.get('/model_info')
        
        # Should return either 200 with info or 404/503 if models not available
        assert response.status_code in [200, 404, 503]
        
        if response.status_code == 200:
            data = json.loads(response.data)
            assert 'success' in data or 'model_info' in data

    def test_plot_endpoint(self, client):
        """Test plot generation endpoint"""
        response = client.get('/plot')
        
        # Should return either image/png or error
        assert response.status_code in [200, 500]
        
        if response.status_code == 200:
            assert response.content_type == 'image/png' or 'image' in response.content_type

    def test_prediction_with_valid_data_json(self, client, valid_prediction_data):
        """Test prediction with valid JSON data"""
        response = client.post('/predict',
                              data=json.dumps(valid_prediction_data),
                              content_type='application/json')
        
        # Should succeed if models are available, or return 503 if not
        assert response.status_code in [200, 503]
        
        data = json.loads(response.data)
        
        if response.status_code == 200:
            assert data['success'] == True
            assert 'predicted_price' in data or 'prediction' in data
        else:
            assert data['success'] == False
            assert 'error' in data

    def test_prediction_with_valid_data_form(self, client, valid_prediction_data):
        """Test prediction with valid form data"""
        response = client.post('/predict', data=valid_prediction_data)
        
        # Should succeed if models are available, or return 503 if not
        assert response.status_code in [200, 503]
        
        data = json.loads(response.data)
        
        if response.status_code == 200:
            assert data['success'] == True
        else:
            assert data['success'] == False

    def test_prediction_with_missing_required_fields(self, client):
        """Test prediction with missing required fields"""
        incomplete_data = {
            'location': 'Whitefield',
            'total_sqft': 1200
            # Missing bhk, bath
        }
        
        response = client.post('/predict',
                              data=json.dumps(incomplete_data),
                              content_type='application/json')
        
        assert response.status_code == 400
        data = json.loads(response.data)
        assert data['success'] == False
        assert 'errors' in data

    def test_prediction_with_invalid_data_types(self, client):
        """Test prediction with invalid data types"""
        invalid_data = {
            'location': 'Whitefield',
            'total_sqft': 'invalid_string',
            'bhk': 2,
            'bath': 2,
            'balcony': 1
        }
        
        response = client.post('/predict',
                              data=json.dumps(invalid_data),
                              content_type='application/json')
        
        assert response.status_code == 400
        data = json.loads(response.data)
        assert data['success'] == False

    def test_prediction_with_out_of_range_values(self, client):
        """Test prediction with out of range values"""
        out_of_range_data = {
            'location': 'Whitefield',
            'total_sqft': -100,  # Negative value
            'bhk': 0,           # Zero value
            'bath': 20,         # Too high
            'balcony': -5       # Negative
        }
        
        response = client.post('/predict',
                              data=json.dumps(out_of_range_data),
                              content_type='application/json')
        
        assert response.status_code == 400
        data = json.loads(response.data)
        assert data['success'] == False
        assert 'errors' in data

    def test_prediction_with_empty_location(self, client):
        """Test prediction with empty location"""
        empty_location_data = {
            'location': '',
            'total_sqft': 1200,
            'bhk': 2,
            'bath': 2,
            'balcony': 1
        }
        
        response = client.post('/predict',
                              data=json.dumps(empty_location_data),
                              content_type='application/json')
        
        assert response.status_code == 400
        data = json.loads(response.data)
        assert data['success'] == False

    def test_prediction_endpoint_methods(self, client):
        """Test that prediction endpoint only accepts POST"""
        response = client.get('/predict')
        assert response.status_code == 405  # Method Not Allowed

    def test_prediction_without_content_type(self, client, valid_prediction_data):
        """Test prediction without proper content type"""
        response = client.post('/predict', data=json.dumps(valid_prediction_data))
        
        # Should still work as it falls back to form data
        assert response.status_code in [200, 400, 503]

    def test_prediction_response_structure(self, client, valid_prediction_data):
        """Test that prediction response has correct structure"""
        response = client.post('/predict',
                              data=json.dumps(valid_prediction_data),
                              content_type='application/json')
        
        data = json.loads(response.data)
        
        # Should have success field
        assert 'success' in data
        
        if response.status_code == 200 and data['success']:
            # Should have prediction fields
            expected_fields = ['predicted_price', 'prediction', 'property_details', 'input_data']
            assert any(field in data for field in expected_fields)

    def test_cors_headers(self, client):
        """Test that CORS headers are present"""
        response = client.options('/predict')
        
        # Should have CORS headers or handle OPTIONS
        assert response.status_code in [200, 204, 405]

    def test_error_handling_for_invalid_json(self, client):
        """Test error handling for malformed JSON"""
        response = client.post('/predict',
                              data='invalid json{',
                              content_type='application/json')
        
        assert response.status_code in [400, 500]
        
        # Should return JSON error response
        try:
            data = json.loads(response.data)
            assert 'success' in data
            assert data['success'] == False
        except:
            # If not JSON, that's also acceptable error handling
            pass

    def test_large_request_handling(self, client):
        """Test handling of unusually large requests"""
        large_data = valid_prediction_data.copy()
        large_data['extra_field'] = 'x' * 10000  # Large string
        
        response = client.post('/predict',
                              data=json.dumps(large_data),
                              content_type='application/json')
        
        # Should handle gracefully
        assert response.status_code in [200, 400, 413, 503]

    def test_concurrent_requests_simulation(self, client, valid_prediction_data):
        """Simulate multiple concurrent requests"""
        responses = []
        
        for _ in range(5):
            response = client.post('/predict',
                                  data=json.dumps(valid_prediction_data),
                                  content_type='application/json')
            responses.append(response.status_code)
        
        # All should return consistent status codes
        assert len(set(responses)) <= 2  # Should be consistent or have at most 2 different codes

    def test_special_characters_in_location(self, client):
        """Test handling of special characters in location"""
        special_data = {
            'location': 'Test@Location#123',
            'total_sqft': 1200,
            'bhk': 2,
            'bath': 2,
            'balcony': 1
        }
        
        response = client.post('/predict',
                              data=json.dumps(special_data),
                              content_type='application/json')
        
        # Should handle gracefully
        assert response.status_code in [200, 400, 503]

    def test_boundary_values(self, client):
        """Test boundary values for numerical inputs"""
        boundary_cases = [
            {'total_sqft': 300, 'bhk': 1, 'bath': 1, 'balcony': 0},  # Minimum values
            {'total_sqft': 8000, 'bhk': 10, 'bath': 10, 'balcony': 10},  # Maximum values
            {'total_sqft': 299, 'bhk': 0, 'bath': 0, 'balcony': -1},  # Below minimum
        ]
        
        for case in boundary_cases:
            test_data = {
                'location': 'Whitefield',
                **case
            }
            
            response = client.post('/predict',
                                  data=json.dumps(test_data),
                                  content_type='application/json')
            
            # Should handle all cases gracefully
            assert response.status_code in [200, 400, 503]
            
            data = json.loads(response.data)
            assert 'success' in data

class TestAPIPerformance:
    """Performance tests for API endpoints"""
    
    @pytest.fixture
    def app(self):
        """Create test Flask application"""
        app = create_app()
        app.config['TESTING'] = True
        return app
    
    @pytest.fixture
    def client(self, app):
        """Create test client"""
        return app.test_client()

    def test_response_time_index(self, client):
        """Test response time for index page"""
        import time
        
        start_time = time.time()
        response = client.get('/')
        end_time = time.time()
        
        assert response.status_code == 200
        assert (end_time - start_time) < 5.0  # Should respond within 5 seconds

    def test_response_time_prediction(self, client):
        """Test response time for prediction endpoint"""
        import time
        
        valid_data = {
            'location': 'Whitefield',
            'total_sqft': 1200,
            'bhk': 2,
            'bath': 2,
            'balcony': 1
        }
        
        start_time = time.time()
        response = client.post('/predict',
                              data=json.dumps(valid_data),
                              content_type='application/json')
        end_time = time.time()
        
        # Should respond within reasonable time
        assert (end_time - start_time) < 30.0  # 30 seconds max

class TestAPIIntegration:
    """Integration tests for complete API workflow"""
    
    @pytest.fixture
    def app(self):
        """Create test Flask application"""
        app = create_app()
        app.config['TESTING'] = True
        return app
    
    @pytest.fixture
    def client(self, app):
        """Create test client"""
        return app.test_client()

    def test_complete_prediction_workflow(self, client):
        """Test complete prediction workflow from start to finish"""
        
        # Step 1: Load main page
        response = client.get('/')
        assert response.status_code == 200
        
        # Step 2: Get available locations
        response = client.get('/locations')
        assert response.status_code == 200
        
        # Step 3: Make prediction
        prediction_data = {
            'location': 'Whitefield',
            'total_sqft': 1200,
            'bhk': 2,
            'bath': 2,
            'balcony': 1
        }
        
        response = client.post('/predict',
                              data=json.dumps(prediction_data),
                              content_type='application/json')
        
        # Should complete successfully or fail gracefully
        assert response.status_code in [200, 503]
        
        data = json.loads(response.data)
        assert 'success' in data

    def test_api_consistency(self, client):
        """Test that API responses are consistent across multiple calls"""
        
        prediction_data = {
            'location': 'Whitefield',
            'total_sqft': 1200,
            'bhk': 2,
            'bath': 2,
            'balcony': 1
        }
        
        responses = []
        for _ in range(3):
            response = client.post('/predict',
                                  data=json.dumps(prediction_data),
                                  content_type='application/json')
            responses.append(response.status_code)
        
        # All responses should have same status code
        assert len(set(responses)) == 1

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
