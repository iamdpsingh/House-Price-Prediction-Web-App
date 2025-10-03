#!/usr/bin/env python3
"""
Complete House Price Prediction Web Application Runner
Downloads dataset, trains models, and runs web application
"""

import os
import sys
import subprocess
import logging
from pathlib import Path

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('app.log', mode='a')
        ]
    )
    return logging.getLogger(__name__)

def create_directories(logger):
    """Create all necessary directories"""
    directories = [
        'data/raw',
        'data/processed',
        'models/trained_models',
        'models/evaluation_plots',
        'api',
        'frontend/static/css',
        'frontend/static/js',
        'frontend/templates',
        'notebooks',
        'tests',
        'config'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.info(f"✅ Created directory: {directory}")

def install_requirements(logger):
    """Install Python requirements"""
    try:
        logger.info("📦 Installing requirements...")
        subprocess.check_call([
            sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt', '--quiet'
        ])
        logger.info("✅ Requirements installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Failed to install requirements: {e}")
        return False

def download_dataset(logger):
    """Download dataset from Kaggle"""
    try:
        logger.info("🔽 Downloading dataset from Kaggle...")
        from data.download_dataset import main as download_main
        download_main()
        logger.info("✅ Dataset downloaded successfully")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to download dataset: {e}")
        logger.info("💡 Please check your Kaggle API credentials")
        return False

def train_models(logger):
    """Train machine learning models"""
    try:
        logger.info("🤖 Training machine learning models...")
        from models.train_model import main as train_main
        train_main()
        logger.info("✅ Models trained successfully")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to train models: {e}")
        return False

def run_flask_app(logger):
    """Run the Flask web application"""
    try:
        logger.info("🚀 Starting Flask web application...")
        logger.info("🌐 Access the application at: http://localhost:5000")
        from api.app import create_app
        
        app = create_app()
        app.run(host='0.0.0.0', port=5000, debug=True)
        return True
    except Exception as e:
        logger.error(f"❌ Failed to run Flask app: {e}")
        return False

def main():
    """Main application runner"""
    print("🏠" + "="*60 + "🏠")
    print("    HOUSE PRICE PREDICTION WEB APPLICATION")
    print("         Complete ML Pipeline with Kaggle Data")
    print("🏠" + "="*60 + "🏠")
    
    logger = setup_logging()
    
    steps = [
        ("Creating Project Structure", create_directories),
        ("Installing Requirements", install_requirements),
        ("Downloading Dataset", download_dataset),
        ("Training ML Models", train_models),
        ("Starting Web Application", run_flask_app)
    ]
    
    for step_name, step_func in steps:
        logger.info(f"\n{'='*50}")
        logger.info(f"STEP: {step_name}")
        logger.info(f"{'='*50}")
        
        if step_func == create_directories:
            step_func(logger)
        elif not step_func(logger):
            logger.error(f"❌ Failed at step: {step_name}")
            return
    
    logger.info("\n🎉 Application setup completed successfully!")

if __name__ == "__main__":
    main()
