"""
Model Loader for House Price Prediction
Loads and manages trained ML models for price prediction
"""

import joblib
import numpy as np
import pandas as pd
import logging
import sys
from pathlib import Path

# Add config to path
sys.path.append(str(Path(__file__).parent.parent))
from config.config import Config

logger = logging.getLogger(__name__)

class HousePricePredictor:
    """Main predictor class that loads models and makes predictions"""
    
    def __init__(self):
        self.config = Config()
        
        # Model components
        self.linear_regression_model = None
        self.random_forest_model = None
        self.location_encoder = None
        self.feature_scaler = None
        self.metadata = None
        
        # Status
        self.is_loaded = False
        self.best_model_name = None
        
    def load_models(self):
        """Load all trained models and preprocessors"""
        try:
            logger.info("📂 Loading trained models...")
            
            # Load models
            if self.config.LINEAR_REGRESSION_MODEL.exists():
                self.linear_regression_model = joblib.load(self.config.LINEAR_REGRESSION_MODEL)
                logger.info("✅ Linear Regression model loaded")
            else:
                logger.warning("⚠️ Linear Regression model not found")
            
            if self.config.RANDOM_FOREST_MODEL.exists():
                self.random_forest_model = joblib.load(self.config.RANDOM_FOREST_MODEL)
                logger.info("✅ Random Forest model loaded")
            else:
                logger.warning("⚠️ Random Forest model not found")
            
            # Load preprocessors
            if self.config.LOCATION_ENCODER.exists():
                self.location_encoder = joblib.load(self.config.LOCATION_ENCODER)
                logger.info("✅ Location encoder loaded")
            else:
                logger.error("❌ Location encoder not found")
                return False
            
            if self.config.FEATURE_SCALER.exists():
                self.feature_scaler = joblib.load(self.config.FEATURE_SCALER)
                logger.info("✅ Feature scaler loaded")
            else:
                logger.warning("⚠️ Feature scaler not found")
            
            # Load metadata
            if self.config.MODEL_METADATA.exists():
                self.metadata = joblib.load(self.config.MODEL_METADATA)
                self.best_model_name = self.metadata.get('best_model', 'random_forest')
                logger.info(f"✅ Metadata loaded - Best model: {self.best_model_name}")
            else:
                logger.warning("⚠️ Model metadata not found, using Random Forest as default")
                self.best_model_name = 'random_forest'
            
            # Check if at least one model is available
            if self.linear_regression_model is None and self.random_forest_model is None:
                logger.error("❌ No models available")
                return False
            
            self.is_loaded = True
            logger.info("🎉 All models loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error loading models: {e}")
            return False
    
    def get_available_locations(self):
        """Get list of available locations from the encoder"""
        if not self.is_loaded or self.location_encoder is None:
            logger.warning("⚠️ Models not loaded or location encoder not available")
            return []
        
        return list(self.location_encoder.classes_)
    
    def validate_location(self, location):
        """Validate if location is available in trained model"""
        if not self.is_loaded:
            return False, "Models not loaded"
        
        available_locations = self.get_available_locations()
        if location not in available_locations:
            return False, f"Location '{location}' not found in available locations: {available_locations}"
        
        return True, ""
    
    def preprocess_input(self, location, total_sqft, bhk, bath, balcony):
        """Preprocess input data for model prediction"""
        try:
            # Validate location
            is_valid, error_msg = self.validate_location(location)
            if not is_valid:
                raise ValueError(error_msg)
            
            # Encode location
            location_encoded = self.location_encoder.transform([location])[0]
            
            # Create derived features (matching training pipeline)
            bhk_bath_ratio = bath / bhk if bhk > 0 else 0
            room_sqft_ratio = total_sqft / bhk if bhk > 0 else 0
            
            # Calculate location frequency (use median frequency if not available)
            if self.metadata and 'location_frequency_map' in self.metadata:
                location_frequency = self.metadata['location_frequency_map'].get(location, 10)
            else:
                # Estimate frequency based on common locations
                common_locations = ['Whitefield', 'Electronic City', 'Sarjapur Road', 'Marathahalli']
                location_frequency = 15 if location in common_locations else 8
            
            # Create feature vector (must match training feature order)
            features = np.array([[
                location_encoded,
                total_sqft,
                bhk,
                bath,
                balcony,
                bhk_bath_ratio,
                room_sqft_ratio,
                location_frequency
            ]])
            
            return features
            
        except Exception as e:
            logger.error(f"Error preprocessing input: {e}")
            raise
    
    def get_best_model(self):
        """Get the best performing model"""
        if self.best_model_name == 'linear_regression' and self.linear_regression_model:
            return self.linear_regression_model, 'linear_regression', True  # needs scaling
        elif self.best_model_name == 'random_forest' and self.random_forest_model:
            return self.random_forest_model, 'random_forest', False  # no scaling needed
        elif self.random_forest_model:
            return self.random_forest_model, 'random_forest', False
        elif self.linear_regression_model:
            return self.linear_regression_model, 'linear_regression', True
        else:
            return None, None, False
    
    def predict_price(self, location, total_sqft, bhk, bath, balcony=0):
        """Make price prediction using the best available model"""
        if not self.is_loaded:
            logger.error("❌ Models not loaded")
            return None
        
        try:
            # Preprocess input
            features = self.preprocess_input(location, total_sqft, bhk, bath, balcony)
            
            # Get best model
            model, model_name, needs_scaling = self.get_best_model()
            
            if model is None:
                logger.error("❌ No models available for prediction")
                return None
            
            # Apply scaling if needed
            if needs_scaling and self.feature_scaler:
                features = self.feature_scaler.transform(features)
            
            # Make prediction
            prediction = model.predict(features)[0]
            
            # Apply reasonable bounds
            prediction = max(10, min(prediction, 500))  # Between 10 and 500 lakhs
            
            # Calculate additional metrics
            price_per_sqft = (prediction * 100000) / total_sqft  # Convert lakhs to rupees
            
            # Determine price category
            if prediction < 30:
                category = 'Budget'
            elif prediction < 60:
                category = 'Mid-range'
            elif prediction < 100:
                category = 'Premium'
            elif prediction < 200:
                category = 'Luxury'
            else:
                category = 'Ultra-luxury'
            
            # Create result dictionary
            result = {
                'predicted_price': round(prediction, 2),
                'price_per_sqft': round(price_per_sqft, 2),
                'price_category': category,
                'model_used': model_name,
                'confidence': 'High' if model_name == self.best_model_name else 'Medium'
            }
            
            logger.info(f"✅ Prediction successful: ₹{prediction:.2f} lakhs using {model_name}")
            return result
            
        except Exception as e:
            logger.error(f"❌ Prediction error: {e}")
            return None
    
    def predict_with_all_models(self, location, total_sqft, bhk, bath, balcony=0):
        """Make predictions with all available models for comparison"""
        if not self.is_loaded:
            return None
        
        results = {}
        
        try:
            features = self.preprocess_input(location, total_sqft, bhk, bath, balcony)
            
            # Linear Regression prediction
            if self.linear_regression_model:
                try:
                    lr_features = self.feature_scaler.transform(features) if self.feature_scaler else features
                    lr_pred = self.linear_regression_model.predict(lr_features)[0]
                    lr_pred = max(10, min(lr_pred, 500))
                    results['linear_regression'] = round(lr_pred, 2)
                except Exception as e:
                    logger.error(f"Linear Regression prediction error: {e}")
            
            # Random Forest prediction
            if self.random_forest_model:
                try:
                    rf_pred = self.random_forest_model.predict(features)[0]
                    rf_pred = max(10, min(rf_pred, 500))
                    results['random_forest'] = round(rf_pred, 2)
                except Exception as e:
                    logger.error(f"Random Forest prediction error: {e}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in multi-model prediction: {e}")
            return None
    
    def get_model_info(self):
        """Get information about loaded models"""
        if not self.is_loaded:
            return {'status': 'Models not loaded'}
        
        info = {
            'status': 'Models loaded',
            'best_model': self.best_model_name,
            'available_models': [],
            'available_locations_count': len(self.get_available_locations()),
            'available_locations': self.get_available_locations()
        }
        
        if self.linear_regression_model:
            info['available_models'].append('linear_regression')
        if self.random_forest_model:
            info['available_models'].append('random_forest')
        
        if self.metadata:
            info['model_scores'] = self.metadata.get('model_scores', {})
            info['dataset_info'] = self.metadata.get('dataset_info', {})
            info['feature_columns'] = self.metadata.get('feature_columns', [])
        
        return info
    
    def get_feature_importance(self):
        """Get feature importance from Random Forest model"""
        if not self.is_loaded or not self.random_forest_model:
            return None
        
        try:
            if hasattr(self.random_forest_model, 'feature_importances_'):
                feature_names = [
                    'location_encoded', 'total_sqft', 'bhk', 'bath', 'balcony',
                    'bhk_bath_ratio', 'room_sqft_ratio', 'location_frequency'
                ]
                
                importance_dict = dict(zip(feature_names, self.random_forest_model.feature_importances_))
                
                # Sort by importance
                sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
                return sorted_importance
            
        except Exception as e:
            logger.error(f"Error getting feature importance: {e}")
        
        return None

# Convenience functions for backward compatibility
def load_models():
    """Load models and return predictor instance"""
    predictor = HousePricePredictor()
    success = predictor.load_models()
    return predictor if success else None

def predict_house_price(location, total_sqft, bhk, bath, balcony=0):
    """Quick prediction function"""
    predictor = load_models()
    if predictor:
        return predictor.predict_price(location, total_sqft, bhk, bath, balcony)
    return None

if __name__ == "__main__":
    # Test the predictor
    predictor = HousePricePredictor()
    if predictor.load_models():
        print("✅ Models loaded successfully")
        
        # Test prediction
        result = predictor.predict_price(
            location="Whitefield",
            total_sqft=1200,
            bhk=2,
            bath=2,
            balcony=1
        )
        
        if result:
            print(f"🏠 Test prediction: ₹{result['predicted_price']} lakhs")
            print(f"📊 Category: {result['price_category']}")
            print(f"🔧 Model used: {result['model_used']}")
        else:
            print("❌ Test prediction failed")
    else:
        print("❌ Failed to load models")
