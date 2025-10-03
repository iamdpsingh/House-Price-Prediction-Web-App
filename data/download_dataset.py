"""
Kaggle Dataset Downloader for House Price Prediction
Downloads Bengaluru house price data from Kaggle
"""

import os
import sys
import zipfile
import pandas as pd
from pathlib import Path
import logging
from tqdm import tqdm

# Add config to path
sys.path.append(str(Path(__file__).parent.parent))
from config.config import Config

logger = logging.getLogger(__name__)

class KaggleDatasetDownloader:
    """Download and process Kaggle datasets"""
    
    def __init__(self):
        self.config = Config()
        self.config.create_directories()
    
    def check_kaggle_credentials(self):
        """Check if Kaggle credentials are properly set up"""
        try:
            import kaggle
            kaggle.api.authenticate()
            logger.info("✅ Kaggle credentials found and authenticated")
            return True
        except Exception as e:
            logger.error(f"❌ Kaggle authentication failed: {e}")
            logger.info(self.config.get_kaggle_credentials_setup_message())
            return False
    
    def download_from_kaggle(self):
        """Download dataset from Kaggle"""
        if not self.check_kaggle_credentials():
            return False
        
        try:
            import kaggle
            
            # Download dataset
            logger.info(f"📥 Downloading dataset: {self.config.KAGGLE_DATASET}")
            kaggle.api.dataset_download_files(
                self.config.KAGGLE_DATASET,
                path=self.config.RAW_DATA_DIR,
                unzip=True
            )
            
            logger.info("✅ Dataset downloaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to download from Kaggle: {e}")
            return False
    
    def create_sample_dataset(self):
        """Create a sample dataset if Kaggle download fails"""
        logger.info("📊 Creating sample Bengaluru house price dataset...")
        
        import numpy as np
        
        # Set random seed for reproducibility
        np.random.seed(42)
        
        # Define Bengaluru locations with realistic price ranges
        locations_data = {
            'Whitefield': {'base_price': 4000, 'variance': 0.3},
            'Sarjapur Road': {'base_price': 3500, 'variance': 0.25},
            'Electronic City': {'base_price': 3800, 'variance': 0.28},
            'Kanakpura Road': {'base_price': 3200, 'variance': 0.3},
            'Thanisandra': {'base_price': 3600, 'variance': 0.25},
            'Yelahanka': {'base_price': 3400, 'variance': 0.27},
            'Uttarahalli': {'base_price': 3100, 'variance': 0.32},
            'Hebbal': {'base_price': 4200, 'variance': 0.22},
            'Marathahalli': {'base_price': 4500, 'variance': 0.25},
            'Raja Rajeshwari Nagar': {'base_price': 2800, 'variance': 0.35},
            'Bannerghatta Road': {'base_price': 3300, 'variance': 0.28},
            'Kalyan nagar': {'base_price': 4000, 'variance': 0.24},
            'Balagere': {'base_price': 2900, 'variance': 0.3},
            'Haralur Road': {'base_price': 3700, 'variance': 0.26},
            'KR Puram': {'base_price': 3500, 'variance': 0.29},
            'Kothanur': {'base_price': 3200, 'variance': 0.31},
            'Bommanahalli': {'base_price': 3600, 'variance': 0.27},
            'CV Raman Nagar': {'base_price': 4100, 'variance': 0.23},
            'Mysore Road': {'base_page': 3000, 'variance': 0.33},
            'Old Airport Road': {'base_price': 4800, 'variance': 0.28}
        }
        
        # Generate dataset
        n_samples = 13320  # Similar to original Bengaluru dataset size
        data = []
        
        locations = list(locations_data.keys())
        weights = np.random.dirichlet(np.ones(len(locations)), size=1)[0]
        
        for _ in tqdm(range(n_samples), desc="Generating samples"):
            # Select location
            location = np.random.choice(locations, p=weights)
            location_info = locations_data[location]
            
            # Generate area_type
            area_type = np.random.choice(
                ['Super built-up Area', 'Plot Area', 'Built-up Area', 'Carpet Area'],
                p=[0.6, 0.15, 0.15, 0.1]
            )
            
            # Generate availability
            availability = np.random.choice(
                ['Ready To Move', '18-Jun', '19-Dec', '20-Mar', 'immediate'],
                p=[0.5, 0.2, 0.15, 0.1, 0.05]
            )
            
            # Generate size (BHK)
            size = np.random.choice(
                ['2 BHK', '3 BHK', '4 BHK', '1 BHK', '5 BHK', '6 BHK'],
                p=[0.35, 0.30, 0.15, 0.12, 0.06, 0.02]
            )
            
            # Extract BHK number
            bhk = int(size.split()[0])
            
            # Generate society name
            society_names = [
                'Prestige High Fields', 'Brigade Meadows', 'Sobha Indraprastha',
                'Total Environment', 'Godrej Properties', 'Purva Panorama',
                'Mantri Serenity', 'Adarsh Palm Retreat', 'Salarpuria Sattva',
                'Embassy Springs', 'Prestige Lakeside Habitat', 'Nitesh Hyde Park'
            ]
            society = np.random.choice(society_names)
            
            # Generate total_sqft based on BHK
            if bhk == 1:
                base_sqft = np.random.normal(650, 100)
            elif bhk == 2:
                base_sqft = np.random.normal(1100, 150)
            elif bhk == 3:
                base_sqft = np.random.normal(1400, 200)
            elif bhk == 4:
                base_sqft = np.random.normal(1900, 250)
            elif bhk == 5:
                base_sqft = np.random.normal(2400, 300)
            else:  # 6+ BHK
                base_sqft = np.random.normal(3000, 400)
            
            # Sometimes add range format
            if np.random.random() < 0.3:
                variance = base_sqft * 0.1
                lower = base_sqft - variance
                upper = base_sqft + variance
                total_sqft = f"{int(lower)} - {int(upper)}"
            else:
                total_sqft = str(int(max(500, base_sqft)))
            
            # Generate bath
            if bhk <= 2:
                bath = np.random.choice([1, 2], p=[0.3, 0.7])
            elif bhk == 3:
                bath = np.random.choice([2, 3], p=[0.4, 0.6])
            elif bhk == 4:
                bath = np.random.choice([3, 4], p=[0.6, 0.4])
            else:
                bath = np.random.choice([4, 5, 6], p=[0.5, 0.3, 0.2])
            
            # Generate balcony
            balcony = np.random.choice([0, 1, 2, 3], p=[0.1, 0.5, 0.3, 0.1])
            
            # Calculate price
            base_price_per_sqft = location_info['base_price']
            
            # Get numeric sqft for price calculation
            if ' - ' in str(total_sqft):
                sqft_range = total_sqft.split(' - ')
                numeric_sqft = (int(sqft_range[0]) + int(sqft_range[1])) / 2
            else:
                numeric_sqft = float(total_sqft)
            
            # Calculate price with variations
            bhk_multiplier = {1: 0.9, 2: 1.0, 3: 1.1, 4: 1.2, 5: 1.3, 6: 1.4}.get(bhk, 1.0)
            base_price = numeric_sqft * base_price_per_sqft * bhk_multiplier
            
            # Add market variations
            market_factor = np.random.normal(1.0, location_info['variance'])
            market_factor = max(0.5, min(market_factor, 2.0))  # Bound the variation
            
            final_price = base_price * market_factor / 100000  # Convert to lakhs
            final_price = max(20, min(final_price, 500))  # Reasonable bounds
            
            data.append({
                'area_type': area_type,
                'availability': availability,
                'location': location,
                'size': size,
                'society': society,
                'total_sqft': total_sqft,
                'bath': bath,
                'balcony': balcony,
                'price': round(final_price, 2)
            })
        
        # Create DataFrame
        df = pd.DataFrame(data)
        
        # Save to CSV
        df.to_csv(self.config.DATASET_PATH, index=False)
        
        logger.info(f"✅ Sample dataset created: {df.shape}")
        logger.info(f"📁 Saved to: {self.config.DATASET_PATH}")
        
        return True
    
    def process_downloaded_data(self):
        """Process and validate downloaded data"""
        try:
            # Look for CSV files in raw data directory
            csv_files = list(self.config.RAW_DATA_DIR.glob('*.csv'))
            
            if not csv_files:
                logger.warning("No CSV files found in downloaded data")
                return False
            
            # Use the first CSV file found
            source_file = csv_files[0]
            
            # Load and inspect data
            df = pd.read_csv(source_file)
            logger.info(f"📊 Dataset loaded: {df.shape}")
            logger.info(f"📋 Columns: {list(df.columns)}")
            
            # Copy to standard location if not already there
            if source_file != self.config.DATASET_PATH:
                df.to_csv(self.config.DATASET_PATH, index=False)
                logger.info(f"📁 Dataset copied to: {self.config.DATASET_PATH}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error processing downloaded data: {e}")
            return False
    
    def download_dataset(self):
        """Main method to download dataset"""
        logger.info("🔽 Starting dataset download process...")
        
        # Check if dataset already exists
        if self.config.DATASET_PATH.exists():
            logger.info(f"📁 Dataset already exists at: {self.config.DATASET_PATH}")
            return True
        
        # Try to download from Kaggle first
        if self.download_from_kaggle():
            if self.process_downloaded_data():
                return True
        
        # If Kaggle download fails, create sample data
        logger.info("🔄 Kaggle download failed, creating sample dataset...")
        return self.create_sample_dataset()

def main():
    """Main function"""
    downloader = KaggleDatasetDownloader()
    success = downloader.download_dataset()
    
    if success:
        logger.info("✅ Dataset download completed successfully!")
    else:
        logger.error("❌ Dataset download failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
