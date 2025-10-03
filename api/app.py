"""
Flask Web Application for House Price Prediction
Main API server with web interface
"""

from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS
import os
import sys
import logging
import traceback
from pathlib import Path
import pandas as pd

# Add paths for imports
sys.path.append(str(Path(__file__).parent.parent))

# Import components
try:
    from api.model_loader import HousePricePredictor
    from api.utils import validate_input_data, format_response
    from config.config import Config
    MODEL_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Model components not available: {e}")
    MODEL_AVAILABLE = False

logger = logging.getLogger(__name__)

def create_app():
    """Create and configure Flask application"""
    app = Flask(__name__,
                template_folder='../frontend/templates',
                static_folder='../frontend/static')
    
    # Configuration
    config = Config()
    app.config.update(config.FLASK_CONFIG)
    
    # Enable CORS
    CORS(app)
    
    # Initialize predictor
    if MODEL_AVAILABLE:
        try:
            predictor = HousePricePredictor()
            predictor.load_models()
            logger.info("✅ Models loaded successfully")
        except Exception as e:
            logger.error(f"❌ Failed to load models: {e}")
            predictor = None
    else:
        predictor = None
    
    @app.route('/')
    def index():
        """Main page with prediction form"""
        try:
            if predictor and predictor.is_loaded:
                locations = predictor.get_available_locations()
            else:
                # Fallback locations if model not available
                locations = [
                    'Whitefield', 'Sarjapur Road', 'Electronic City', 'Kanakpura Road',
                    'Thanisandra', 'Yelahanka', 'Uttarahalli', 'Hebbal', 'Marathahalli',
                    'Raja Rajeshwari Nagar', 'Bannerghatta Road', 'Kalyan nagar',
                    'Balagere', 'Haralur Road', 'KR Puram', 'Kothanur', 'Bommanahalli'
                ]
            
            return render_template('index.html', locations=locations)
            
        except Exception as e:
            logger.error(f"Error rendering index: {e}")
            return f"Application Error: {str(e)}", 500
    
    @app.route('/predict', methods=['POST'])
    def predict():
        """Make house price prediction"""
        try:
            # Check if predictor is available
            if not predictor or not predictor.is_loaded:
                return jsonify({
                    'success': False,
                    'error': 'Model not available. Please ensure models are trained.'
                }), 503
            
            # Get input data
            if request.is_json:
                data = request.get_json()
            else:
                data = request.form.to_dict()
            
            # Validate input
            is_valid, validation_errors = validate_input_data(data)
            if not is_valid:
                return jsonify({
                    'success': False,
                    'errors': validation_errors
                }), 400
            
            # Extract and convert data
            location = data['location']
            total_sqft = float(data['total_sqft'])
            bhk = int(data['bhk'])
            bath = int(data['bath'])
            balcony = int(data.get('balcony', 0))
            
            # Make prediction
            result = predictor.predict_price(
            location=data['location'],
            total_sqft=float(data['total_sqft']),
            bhk=int(data['bhk']),
            bath=int(data['bath']),
            balcony=int(data.get('balcony', 0)))
        
            # Handle prediction result
            if result is None:
                return jsonify({
                    'success': False,
                    'error': 'Model not available. Please ensure models are trained.'
                }), 503
            
            # Check if result contains an error
            if isinstance(result, dict) and result.get('success') == False:
                return jsonify(result), 400  # Return 400 for validation errors
            
            # Success case - return prediction
            return jsonify({
                'success': True,
                'prediction': result,
                'input_data': data
            })
        
        except Exception as e:
            logger.error(f"Prediction endpoint error: {str(e)}")
            return jsonify({
                'success': False,
                'error': 'Internal server error during prediction'
            }), 500
        
    @app.route('/locations')
    def get_locations():
        """Get available locations"""
        try:
            if predictor and predictor.is_loaded:
                locations = predictor.get_available_locations()
            else:
                locations = [
                    'Whitefield', 'Sarjapur Road', 'Electronic City', 'Kanakpura Road',
                    'Thanisandra', 'Yelahanka', 'Uttarahalli', 'Hebbal', 'Marathahalli'
                ]
            
            return jsonify({
                'success': True,
                'locations': sorted(locations),
                'count': len(locations)
            })
            
        except Exception as e:
            logger.error(f"Error getting locations: {e}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    @app.route('/model_info')
    def model_info():
        """Get model information and performance metrics"""
        try:
            if not predictor.is_loaded:
                return jsonify({
                    'success': False,
                    'error': 'Models not available'
                }), 503
            
            # Convert numpy arrays to lists for JSON serialization
            model_info = {
                'success': True,
                'model_info': {
                    'best_model': predictor.best_model_name,
                    'available_models': ['linear_regression', 'random_forest'],
                    'locations_count': len(predictor.location_encoder.classes_) if predictor.location_encoder else 0,
                    'available_locations': predictor.location_encoder.classes_.tolist() if predictor.location_encoder else [],
                    'model_loaded': predictor.is_loaded,
                    'features': predictor.feature_columns if hasattr(predictor, 'feature_columns') else []
                }
            }
            
            return jsonify(model_info)
            
        except Exception as e:
            logger.error(f"Error getting model info: {str(e)}")
            return jsonify({
                'success': False,
                'error': 'Unable to retrieve model information'
            }), 500

    
    @app.route('/plot')
    def generate_plot():
        """Generate model performance visualization"""
        try:
            import matplotlib
            matplotlib.use('Agg')  # Use non-interactive backend
            import matplotlib.pyplot as plt
            import numpy as np
            from io import BytesIO
            
            # Create model comparison plot
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            if predictor and predictor.is_loaded:
                model_info = predictor.get_model_info()
                
                if 'model_scores' in model_info:
                    scores = model_info['model_scores']
                    models = list(scores.keys())
                    r2_scores = [scores[model].get('test_r2', 0) for model in models]
                    mae_scores = [scores[model].get('test_mae', 0) for model in models]
                else:
                    # Default values if no scores available
                    models = ['Linear Regression', 'Random Forest']
                    r2_scores = [0.83, 0.89]
                    mae_scores = [18.5, 15.2]
            else:
                # Demo data if models not loaded
                models = ['Linear Regression', 'Random Forest']
                r2_scores = [0.83, 0.89]
                mae_scores = [18.5, 15.2]
            
            # Clean model names for display
            display_names = [name.replace('_', ' ').title() for name in models]
            
            # R² Score comparison
            bars1 = ax1.bar(display_names, r2_scores, color=['skyblue', 'lightcoral'], alpha=0.8)
            ax1.set_title('Model R² Score Comparison', fontweight='bold')
            ax1.set_ylabel('R² Score')
            ax1.set_ylim(0, 1)
            ax1.grid(True, alpha=0.3)
            
            for bar, score in zip(bars1, r2_scores):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
            
            # MAE comparison
            bars2 = ax2.bar(display_names, mae_scores, color=['lightgreen', 'orange'], alpha=0.8)
            ax2.set_title('Model MAE Comparison', fontweight='bold')
            ax2.set_ylabel('MAE (₹ lakhs)')
            ax2.grid(True, alpha=0.3)
            
            for bar, score in zip(bars2, mae_scores):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                        f'{score:.1f}', ha='center', va='bottom', fontweight='bold')
            
            plt.tight_layout()
            
            # Save to BytesIO
            img_buffer = BytesIO()
            plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
            img_buffer.seek(0)
            plt.close()
            
            return send_file(img_buffer, mimetype='image/png')
            
        except Exception as e:
            logger.error(f"Error generating plot: {e}")
            return jsonify({
                'success': False,
                'error': 'Failed to generate plot'
            }), 500
    
    @app.route('/health')
    def health_check():
        """Application health check"""
        try:
            health_status = {
                'status': 'healthy',
                'model_available': MODEL_AVAILABLE,
                'model_loaded': predictor.is_loaded if predictor else False,
                'timestamp': pd.Timestamp.now().isoformat()
            }
            
            if predictor and predictor.is_loaded:
                health_status['locations_count'] = len(predictor.get_available_locations())
            
            return jsonify(health_status)
            
        except Exception as e:
            return jsonify({
                'status': 'unhealthy',
                'error': str(e),
                'timestamp': pd.Timestamp.now().isoformat()
            }), 500
    
    @app.route('/results')
    def results():
        """Results page (for future use)"""
        return render_template('results.html')
    
    @app.errorhandler(404)
    def not_found_error(error):
        """Handle 404 errors"""
        if request.path.startswith('/api/'):
            return jsonify({
                'success': False,
                'error': 'Endpoint not found'
            }), 404
        return render_template('index.html', 
                             locations=['Error: Page not found'],
                             error="Page not found"), 404
    
    @app.errorhandler(500)
    def internal_error(error):
        """Handle 500 errors"""
        logger.error(f"Internal server error: {error}")
        if request.path.startswith('/api/'):
            return jsonify({
                'success': False,
                'error': 'Internal server error'
            }), 500
        return render_template('index.html',
                             locations=['Error: Server error'],
                             error="Internal server error"), 500
    
    return app

if __name__ == '__main__':
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    app = create_app()
    logger.info("🚀 Starting Flask application...")
    app.run(
        host='0.0.0.0', 
        port=9000, 
        debug=True
    )
