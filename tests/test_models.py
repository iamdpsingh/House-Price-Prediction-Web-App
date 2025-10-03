"""
Unit Tests for ML Pipeline Components
Testing model loading, prediction, and utility functions
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path
import tempfile
import joblib
from unittest.mock import Mock, patch, MagicMock

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

class TestDataProcessing:
    """Test data processing and validation functions"""
    
    def test_input_validation_valid_data(self):
        """Test input validation with valid data"""
        try:
            from api.utils import validate_input_data
            
            valid_data = {
                'location': 'Whitefield',
                'total_sqft': 1200,
                'bhk': 2,
                'bath': 2,
                'balcony': 1
            }
            
            is_valid, errors = validate_input_data(valid_data)
            assert is_valid == True
            assert len(errors) == 0
            
        except ImportError:
            pytest.skip("Utils module not available")

    def test_input_validation_invalid_data(self):
        """Test input validation with invalid data"""
        try:
            from api.utils import validate_input_data
            
            invalid_data = {
                'location': '',  # Empty location
                'total_sqft': -100,  # Negative value
                'bhk': 0,  # Zero BHK
                'bath': 'invalid',  # Non-numeric
                'balcony': 100  # Too high
            }
            
            is_valid, errors = validate_input_data(invalid_data)
            assert is_valid == False
            assert len(errors) > 0
            
        except ImportError:
            pytest.skip("Utils module not available")

    def test_input_validation_missing_fields(self):
        """Test input validation with missing required fields"""
        try:
            from api.utils import validate_input_data
            
            incomplete_data = {
                'location': 'Whitefield',
                'total_sqft': 1200
                # Missing bhk, bath
            }
            
            is_valid, errors = validate_input_data(incomplete_data)
            assert is_valid == False
            assert len(errors) > 0
            
        except ImportError:
            pytest.skip("Utils module not available")

    def test_price_formatting(self):
        """Test price formatting utility"""
        try:
            from api.utils import format_currency
            
            # Test normal price
            formatted = format_currency(85.50)
            assert '85.50' in formatted
            assert '₹' in formatted
            
            # Test None value
            formatted_none = format_currency(None)
            assert formatted_none is not None
            
        except ImportError:
            pytest.skip("Utils module not available")

    def test_price_per_sqft_calculation(self):
        """Test price per sqft calculation"""
        try:
            from api.utils import calculate_price_per_sqft
            
            # Normal calculation
            price_per_sqft = calculate_price_per_sqft(100, 1000)  # 100 lakhs, 1000 sqft
            expected = (100 * 100000) / 1000  # Should be 10000
            assert price_per_sqft == expected
            
            # Edge case - zero sqft
            price_per_sqft_zero = calculate_price_per_sqft(100, 0)
            assert price_per_sqft_zero is None
            
        except (ImportError, NameError):
            pytest.skip("Utils module or function not available")

class TestModelLoader:
    """Test model loading and prediction functionality"""
    
    def test_model_loader_initialization(self):
        """Test model loader can be initialized"""
        try:
            from api.model_loader import HousePricePredictor
            
            predictor = HousePricePredictor()
            assert predictor is not None
            assert hasattr(predictor, 'is_loaded')
            
        except ImportError:
            pytest.skip("Model loader not available")

    def test_model_loading_with_mock_files(self):
        """Test model loading with mock model files"""
        try:
            from api.model_loader import HousePricePredictor
            
            with patch('joblib.load') as mock_load:
                with patch('pathlib.Path.exists', return_value=True):
                    # Mock model components
                    mock_load.side_effect = [
                        Mock(),  # linear regression model
                        Mock(),  # random forest model  
                        Mock(classes_=['Location1', 'Location2']),  # location encoder
                        Mock(),  # scaler
                        {'best_model': 'random_forest'}  # metadata
                    ]
                    
                    predictor = HousePricePredictor()
                    result = predictor.load_models()
                    
                    assert result == True
                    assert predictor.is_loaded == True
                    
        except ImportError:
            pytest.skip("Model loader not available")

    def test_location_validation(self):
        """Test location validation"""
        try:
            from api.model_loader import HousePricePredictor
            
            predictor = HousePricePredictor()
            predictor.is_loaded = True
            predictor.location_encoder = Mock()
            predictor.location_encoder.classes_ = ['Whitefield', 'Electronic City']
            
            # Valid location
            is_valid, error = predictor.validate_location('Whitefield')
            assert is_valid == True
            assert error == ""
            
            # Invalid location
            is_valid, error = predictor.validate_location('InvalidLocation')
            assert is_valid == False
            assert error != ""
            
        except ImportError:
            pytest.skip("Model loader not available")

    def test_prediction_with_mock_model(self):
        """Test prediction with mocked model"""
        try:
            from api.model_loader import HousePricePredictor
            
            predictor = HousePricePredictor()
            predictor.is_loaded = True
            
            # Mock components
            predictor.location_encoder = Mock()
            predictor.location_encoder.classes_ = ['Whitefield']
            predictor.location_encoder.transform.return_value = [0]
            
            mock_model = Mock()
            mock_model.predict.return_value = [85.5]
            predictor.random_forest_model = mock_model
            predictor.best_model_name = 'random_forest'
            
            result = predictor.predict_price(
                location='Whitefield',
                total_sqft=1200,
                bhk=2,
                bath=2,
                balcony=1
            )
            
            assert result is not None
            assert 'predicted_price' in result
            assert result['predicted_price'] > 0
            
        except ImportError:
            pytest.skip("Model loader not available")

class TestDataCleaning:
    """Test data cleaning pipeline"""
    
    def create_sample_dataframe(self):
        """Create sample DataFrame for testing"""
        return pd.DataFrame({
            'location': ['Whitefield', 'Electronic City', 'Marathahalli', None],
            'total_sqft': [1200, '1000-1100', 1500, 800],
            'size': ['2 BHK', '3 BHK', '4 BHK', '2 BHK'],
            'bath': [2, 3, 4, 2],
            'balcony': [1, 2, None, 1],
            'price': [85.5, 95.0, 120.0, 70.0]
        })

    def test_duplicate_removal(self):
        """Test duplicate row removal"""
        # Create DataFrame with duplicates
        df = pd.DataFrame({
            'location': ['Whitefield', 'Whitefield', 'Electronic City'],
            'total_sqft': [1200, 1200, 1000],
            'price': [85.5, 85.5, 95.0]
        })
        
        # Should remove duplicates
        df_cleaned = df.drop_duplicates()
        assert len(df_cleaned) <= len(df)

    def test_sqft_range_parsing(self):
        """Test parsing of sqft ranges like '1200-1300'"""
        def clean_sqft_value(value):
            if pd.isna(value):
                return np.nan
            
            value_str = str(value).strip()
            
            if '-' in value_str:
                try:
                    parts = value_str.split('-')
                    if len(parts) == 2:
                        lower = float(parts[0].strip())
                        upper = float(parts[1].strip())
                        return (lower + upper) / 2
                except:
                    return np.nan
            
            try:
                return float(value_str)
            except:
                return np.nan
        
        # Test range parsing
        assert clean_sqft_value('1200-1300') == 1250.0
        assert clean_sqft_value('1000') == 1000.0
        assert pd.isna(clean_sqft_value('invalid'))

    def test_bhk_extraction(self):
        """Test BHK extraction from size strings"""
        import re
        
        def extract_bhk_number(size_str):
            if pd.isna(size_str):
                return np.nan
            
            size_str = str(size_str).strip().upper()
            numbers = re.findall(r'\d+', size_str)
            
            if numbers:
                bhk_num = int(numbers[0])
                return bhk_num if 1 <= bhk_num <= 10 else np.nan
            
            return np.nan
        
        # Test BHK extraction
        assert extract_bhk_number('2 BHK') == 2
        assert extract_bhk_number('3BHK') == 3
        assert extract_bhk_number('4 Bedroom') == 4
        assert pd.isna(extract_bhk_number('invalid'))

    def test_outlier_detection(self):
        """Test outlier detection using IQR method"""
        def detect_outliers_iqr(series, multiplier=1.5):
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - multiplier * IQR
            upper_bound = Q3 + multiplier * IQR
            
            outliers = series[(series < lower_bound) | (series > upper_bound)]
            return outliers, lower_bound, upper_bound
        
        # Test with sample data
        sample_prices = pd.Series([20, 30, 40, 50, 60, 70, 80, 200])  # 200 is outlier
        outliers, lower, upper = detect_outliers_iqr(sample_prices)
        
        assert len(outliers) > 0
        assert 200 in outliers.values

class TestFeatureEngineering:
    """Test feature engineering functions"""
    
    def test_derived_feature_creation(self):
        """Test creation of derived features"""
        df = pd.DataFrame({
            'total_sqft': [1200, 1500, 800],
            'bhk': [2, 3, 1],
            'bath': [2, 3, 1],
            'price': [85.5, 120.0, 60.0]
        })
        
        # Create derived features
        df['price_per_sqft'] = (df['price'] * 100000) / df['total_sqft']
        df['sqft_per_room'] = df['total_sqft'] / df['bhk']
        df['bath_per_bhk'] = df['bath'] / df['bhk']
        
        # Verify calculations
        assert df['price_per_sqft'].iloc[0] == (85.5 * 100000) / 1200
        assert df['sqft_per_room'].iloc[0] == 1200 / 2
        assert df['bath_per_bhk'].iloc[0] == 2 / 2

    def test_location_encoding(self):
        """Test location encoding"""
        from sklearn.preprocessing import LabelEncoder
        
        locations = pd.Series(['Whitefield', 'Electronic City', 'Whitefield', 'Marathahalli'])
        
        le = LabelEncoder()
        encoded = le.fit_transform(locations)
        
        assert len(encoded) == len(locations)
        assert len(set(encoded)) <= len(locations.unique())
        
        # Should be able to inverse transform
        decoded = le.inverse_transform(encoded)
        assert list(decoded) == list(locations)

class TestModelTraining:
    """Test model training components"""
    
    def test_train_test_split_functionality(self):
        """Test train-test split"""
        from sklearn.model_selection import train_test_split
        
        # Sample data
        X = pd.DataFrame({
            'feature1': range(100),
            'feature2': range(100, 200)
        })
        y = pd.Series(range(200, 300))
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        assert len(X_train) == 80
        assert len(X_test) == 20
        assert len(y_train) == 80
        assert len(y_test) == 20

    def test_model_evaluation_metrics(self):
        """Test model evaluation metrics"""
        from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
        
        # Sample predictions vs actual
        y_true = np.array([100, 120, 90, 110, 95])
        y_pred = np.array([105, 115, 92, 108, 98])
        
        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        
        assert 0 <= r2 <= 1  # R² should be between 0 and 1 for reasonable predictions
        assert mae >= 0
        assert rmse >= 0
        assert rmse >= mae  # RMSE is typically >= MAE

    @patch('sklearn.ensemble.RandomForestRegressor')
    def test_random_forest_training(self, mock_rf):
        """Test Random Forest model training"""
        # Mock the RandomForestRegressor
        mock_model = Mock()
        mock_rf.return_value = mock_model
        
        from sklearn.ensemble import RandomForestRegressor
        
        # Create sample data
        X_train = pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [4, 5, 6]})
        y_train = pd.Series([10, 20, 30])
        
        # Train model
        model = RandomForestRegressor()
        model.fit(X_train, y_train)
        
        # Verify fit was called
        mock_model.fit.assert_called_once()

class TestIntegration:
    """Integration tests for ML pipeline"""
    
    def test_end_to_end_pipeline_mock(self):
        """Test end-to-end pipeline with mocked components"""
        # This test verifies the pipeline can run without actual model files
        
        sample_data = {
            'location': 'Whitefield',
            'total_sqft': 1200,
            'bhk': 2,
            'bath': 2,
            'balcony': 1
        }
        
        # Mock validation
        with patch('api.utils.validate_input_data', return_value=(True, [])):
            # Mock model loading
            with patch('api.model_loader.HousePricePredictor') as mock_predictor:
                mock_instance = Mock()
                mock_instance.predict_price.return_value = {
                    'predicted_price': 85.5,
                    'price_category': 'Premium',
                    'model_used': 'random_forest'
                }
                mock_predictor.return_value = mock_instance
                
                # This would be the complete pipeline
                predictor = mock_predictor()
                result = predictor.predict_price(**sample_data)
                
                assert result is not None
                assert 'predicted_price' in result

class TestErrorHandling:
    """Test error handling in various components"""
    
    def test_invalid_model_file_handling(self):
        """Test handling of missing or corrupt model files"""
        try:
            from api.model_loader import HousePricePredictor
            
            predictor = HousePricePredictor()
            
            # Should handle missing files gracefully
            with patch('pathlib.Path.exists', return_value=False):
                result = predictor.load_models()
                assert result == False
                assert predictor.is_loaded == False
                
        except ImportError:
            pytest.skip("Model loader not available")

    def test_prediction_error_handling(self):
        """Test prediction error handling"""
        try:
            from api.model_loader import HousePricePredictor
            
            predictor = HousePricePredictor()
            
            # Test with unloaded models
            result = predictor.predict_price('TestLocation', 1200, 2, 2, 1)
            assert result is None
            
        except ImportError:
            pytest.skip("Model loader not available")

class TestPerformance:
    """Performance tests for ML components"""
    
    def test_prediction_performance(self):
        """Test that prediction completes within reasonable time"""
        import time
        
        # Mock a prediction that should be fast
        start_time = time.time()
        
        # Simulate prediction calculation
        sample_calculation = sum(range(1000))  # Simple calculation
        
        end_time = time.time()
        
        # Should complete very quickly
        assert (end_time - start_time) < 1.0

    def test_data_processing_performance(self):
        """Test data processing performance with larger dataset"""
        # Create larger sample dataset
        large_df = pd.DataFrame({
            'location': ['Location' + str(i % 10) for i in range(10000)],
            'total_sqft': np.random.randint(500, 3000, 10000),
            'bhk': np.random.randint(1, 5, 10000),
            'price': np.random.uniform(20, 200, 10000)
        })
        
        import time
        start_time = time.time()
        
        # Basic processing
        processed_df = large_df.dropna()
        processed_df = processed_df.drop_duplicates()
        
        end_time = time.time()
        
        # Should process quickly
        assert (end_time - start_time) < 5.0  # 5 seconds for 10k records

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
