"""
Machine Learning Model Training Pipeline
Train Linear Regression and Random Forest models for house price prediction
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import logging
import sys
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
import warnings
warnings.filterwarnings('ignore')

# Add config to path
sys.path.append(str(Path(__file__).parent.parent))
from config.config import Config

logger = logging.getLogger(__name__)

class HousePriceModelTrainer:
    """Complete ML pipeline for house price prediction"""
    
    def __init__(self):
        self.config = Config()
        self.config.create_directories()
        
        # Data components
        self.raw_data = None
        self.processed_data = None
        
        # ML components
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
        # Models
        self.models = {}
        self.model_scores = {}
        self.best_model_name = None
        self.best_model = None
        
        # Preprocessors
        self.location_encoder = LabelEncoder()
        self.feature_scaler = StandardScaler()
        
    def load_data(self):
        """Load the dataset"""
        try:
            logger.info(f"📂 Loading dataset from: {self.config.DATASET_PATH}")
            self.raw_data = pd.read_csv(self.config.DATASET_PATH)
            logger.info(f"✅ Dataset loaded successfully: {self.raw_data.shape}")
            logger.info(f"📋 Columns: {list(self.raw_data.columns)}")
            return True
        except FileNotFoundError:
            logger.error(f"❌ Dataset not found: {self.config.DATASET_PATH}")
            return False
        except Exception as e:
            logger.error(f"❌ Error loading dataset: {e}")
            return False
    
    def explore_data(self):
        """Basic data exploration"""
        logger.info("🔍 Exploring dataset...")
        
        print("\n" + "="*50)
        print("DATASET OVERVIEW")
        print("="*50)
        print(f"Shape: {self.raw_data.shape}")
        print(f"Memory usage: {self.raw_data.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        print(f"\nMissing values:")
        missing = self.raw_data.isnull().sum()
        for col, count in missing.items():
            if count > 0:
                print(f"  {col}: {count} ({count/len(self.raw_data)*100:.1f}%)")
        
        print(f"\nData types:")
        for col, dtype in self.raw_data.dtypes.items():
            print(f"  {col}: {dtype}")
        
        print(f"\nFirst 5 rows:")
        print(self.raw_data.head())
        
        if 'price' in self.raw_data.columns:
            print(f"\nPrice statistics:")
            print(f"  Mean: ₹{self.raw_data['price'].mean():.2f} lakhs")
            print(f"  Median: ₹{self.raw_data['price'].median():.2f} lakhs")
            print(f"  Range: ₹{self.raw_data['price'].min():.2f} - ₹{self.raw_data['price'].max():.2f} lakhs")
    
    def clean_data(self):
        """Clean and preprocess the data"""
        logger.info("🧹 Cleaning data...")
        
        df = self.raw_data.copy()
        initial_shape = df.shape
        
        # Handle missing values
        logger.info(f"  Removing rows with missing values...")
        df = df.dropna()
        
        # Clean total_sqft column (handle ranges like "1200 - 1300")
        if 'total_sqft' in df.columns:
            logger.info(f"  Processing total_sqft column...")
            def clean_sqft(x):
                if pd.isna(x):
                    return np.nan
                if isinstance(x, (int, float)):
                    return float(x)
                
                x = str(x).strip()
                if '-' in x:
                    try:
                        parts = x.split('-')
                        return (float(parts[0].strip()) + float(parts[1].strip())) / 2
                    except:
                        return np.nan
                else:
                    try:
                        return float(x)
                    except:
                        return np.nan
            
            df['total_sqft'] = df['total_sqft'].apply(clean_sqft)
            df = df.dropna(subset=['total_sqft'])
        
        # Extract BHK from size column
        if 'size' in df.columns:
            logger.info(f"  Extracting BHK from size column...")
            def extract_bhk(size_str):
                if pd.isna(size_str):
                    return np.nan
                try:
                    return int(str(size_str).split()[0])
                except:
                    return np.nan
            
            df['bhk'] = df['size'].apply(extract_bhk)
            df = df.dropna(subset=['bhk'])
        
        # Ensure required columns exist
        required_columns = ['location', 'total_sqft', 'bhk', 'bath', 'price']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            logger.error(f"❌ Missing required columns: {missing_columns}")
            return False
        
        # Convert data types
        df['bhk'] = df['bhk'].astype(int)
        df['bath'] = df['bath'].astype(int)
        df['total_sqft'] = df['total_sqft'].astype(float)
        df['price'] = df['price'].astype(float)
        
        # Add balcony column if not present
        if 'balcony' not in df.columns:
            logger.info(f"  Adding default balcony column...")
            df['balcony'] = np.random.randint(0, 3, len(df))
        
        # Remove outliers
        logger.info(f"  Removing outliers...")
        
        # Price outliers (using IQR method)
        Q1 = df['price'].quantile(0.25)
        Q3 = df['price'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[(df['price'] >= lower_bound) & (df['price'] <= upper_bound)]
        
        # Total sqft outliers
        df = df[(df['total_sqft'] >= 500) & (df['total_sqft'] <= 8000)]
        
        # Logical filtering
        df = df[df['bath'] <= df['bhk'] + 3]  # Reasonable bath to BHK ratio
        df = df[df['bhk'] <= 10]  # Maximum 10 BHK
        df = df[df['price'] >= 10]  # Minimum 10 lakhs
        
        self.processed_data = df
        
        logger.info(f"  Data cleaning completed:")
        logger.info(f"    Original shape: {initial_shape}")
        logger.info(f"    Final shape: {df.shape}")
        logger.info(f"    Rows removed: {initial_shape[0] - df.shape[0]} ({(initial_shape[0] - df.shape[0])/initial_shape[0]*100:.1f}%)")
        
        return True
    
    def feature_engineering(self):
        """Create additional features"""
        logger.info("⚙️ Engineering features...")
        
        df = self.processed_data.copy()
        
        # Encode location
        df['location_encoded'] = self.location_encoder.fit_transform(df['location'])
        
        # Create derived features
        df['price_per_sqft'] = (df['price'] * 100000) / df['total_sqft']  # Price per sqft in rupees
        df['bhk_bath_ratio'] = df['bath'] / df['bhk']
        df['room_sqft_ratio'] = df['total_sqft'] / df['bhk']
        
        # Location-based features (frequency encoding)
        location_counts = df['location'].value_counts()
        df['location_frequency'] = df['location'].map(location_counts)
        
        # Price category (for analysis)
        df['price_category'] = pd.cut(df['price'], 
                                    bins=[0, 50, 100, 200, float('inf')], 
                                    labels=['Budget', 'Mid-range', 'Premium', 'Luxury'])
        
        self.processed_data = df
        
        logger.info(f"✅ Feature engineering completed")
        logger.info(f"   New features created: price_per_sqft, bhk_bath_ratio, room_sqft_ratio, location_frequency")
        logger.info(f"   Final dataset shape: {df.shape}")
        
        return True
    
    def prepare_features(self):
        """Prepare features for model training"""
        logger.info("🔧 Preparing features for training...")
        
        # Select features for training
        feature_columns = [
            'location_encoded',
            'total_sqft', 
            'bhk', 
            'bath', 
            'balcony',
            'bhk_bath_ratio',
            'room_sqft_ratio',
            'location_frequency'
        ]
        
        X = self.processed_data[feature_columns].copy()
        y = self.processed_data['price'].copy()
        
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, 
            test_size=self.config.MODEL_CONFIG['test_size'],
            random_state=self.config.MODEL_CONFIG['random_state'],
            stratify=pd.qcut(y, q=5, labels=False, duplicates='drop')  # Stratify by price quintiles
        )
        
        logger.info(f"✅ Feature preparation completed:")
        logger.info(f"   Features: {feature_columns}")
        logger.info(f"   Training set: {self.X_train.shape}")
        logger.info(f"   Test set: {self.X_test.shape}")
        
        # Save feature scaler
        self.X_train_scaled = self.feature_scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.feature_scaler.transform(self.X_test)
        
        return True
    
    def train_linear_regression(self):
        """Train Linear Regression model"""
        logger.info("📈 Training Linear Regression model...")
        
        # Train model
        lr_model = LinearRegression()
        lr_model.fit(self.X_train_scaled, self.y_train)
        
        # Make predictions
        y_train_pred = lr_model.predict(self.X_train_scaled)
        y_test_pred = lr_model.predict(self.X_test_scaled)
        
        # Calculate metrics
        train_r2 = r2_score(self.y_train, y_train_pred)
        test_r2 = r2_score(self.y_test, y_test_pred)
        test_mae = mean_absolute_error(self.y_test, y_test_pred)
        test_rmse = np.sqrt(mean_squared_error(self.y_test, y_test_pred))
        
        # Cross-validation
        cv_scores = cross_val_score(
            lr_model, self.X_train_scaled, self.y_train,
            cv=self.config.MODEL_CONFIG['cv_folds'], scoring='r2'
        )
        
        # Store results
        self.models['linear_regression'] = lr_model
        self.model_scores['linear_regression'] = {
            'train_r2': train_r2,
            'test_r2': test_r2,
            'test_mae': test_mae,
            'test_rmse': test_rmse,
            'cv_r2_mean': cv_scores.mean(),
            'cv_r2_std': cv_scores.std(),
            'predictions': y_test_pred
        }
        
        logger.info(f"✅ Linear Regression trained:")
        logger.info(f"   Test R²: {test_r2:.4f}")
        logger.info(f"   Test MAE: {test_mae:.2f} lakhs")
        logger.info(f"   CV R² (mean±std): {cv_scores.mean():.4f}±{cv_scores.std():.4f}")
    
    def train_random_forest(self):
        """Train Random Forest model"""
        logger.info("🌳 Training Random Forest model...")
        
        # Train model (using original features, not scaled)
        rf_config = self.config.MODEL_CONFIG['random_forest']
        rf_model = RandomForestRegressor(**rf_config)
        rf_model.fit(self.X_train, self.y_train)
        
        # Make predictions
        y_train_pred = rf_model.predict(self.X_train)
        y_test_pred = rf_model.predict(self.X_test)
        
        # Calculate metrics
        train_r2 = r2_score(self.y_train, y_train_pred)
        test_r2 = r2_score(self.y_test, y_test_pred)
        test_mae = mean_absolute_error(self.y_test, y_test_pred)
        test_rmse = np.sqrt(mean_squared_error(self.y_test, y_test_pred))
        
        # Cross-validation
        cv_scores = cross_val_score(
            rf_model, self.X_train, self.y_train,
            cv=self.config.MODEL_CONFIG['cv_folds'], scoring='r2'
        )
        
        # Feature importance
        feature_importance = dict(zip(self.X_train.columns, rf_model.feature_importances_))
        
        # Store results
        self.models['random_forest'] = rf_model
        self.model_scores['random_forest'] = {
            'train_r2': train_r2,
            'test_r2': test_r2,
            'test_mae': test_mae,
            'test_rmse': test_rmse,
            'cv_r2_mean': cv_scores.mean(),
            'cv_r2_std': cv_scores.std(),
            'predictions': y_test_pred,
            'feature_importance': feature_importance
        }
        
        logger.info(f"✅ Random Forest trained:")
        logger.info(f"   Test R²: {test_r2:.4f}")
        logger.info(f"   Test MAE: {test_mae:.2f} lakhs")
        logger.info(f"   CV R² (mean±std): {cv_scores.mean():.4f}±{cv_scores.std():.4f}")
        
        # Log feature importance
        logger.info(f"   Top 5 important features:")
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        for feature, importance in sorted_features[:5]:
            logger.info(f"     {feature}: {importance:.4f}")
    
    def evaluate_models(self):
        """Compare model performance"""
        logger.info("📊 Evaluating and comparing models...")
        
        print("\n" + "="*60)
        print("MODEL PERFORMANCE COMPARISON")
        print("="*60)
        
        comparison_data = []
        for model_name, scores in self.model_scores.items():
            comparison_data.append({
                'Model': model_name.replace('_', ' ').title(),
                'Test_R²': f"{scores['test_r2']:.4f}",
                'Test_MAE': f"{scores['test_mae']:.2f}",
                'Test_RMSE': f"{scores['test_rmse']:.2f}",
                'CV_R²_Mean': f"{scores['cv_r2_mean']:.4f}",
                'CV_R²_Std': f"{scores['cv_r2_std']:.4f}"
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        print(comparison_df.to_string(index=False))
        
        # Find best model
        best_test_r2 = max(self.model_scores.values(), key=lambda x: x['test_r2'])
        for name, scores in self.model_scores.items():
            if scores['test_r2'] == best_test_r2['test_r2']:
                self.best_model_name = name
                self.best_model = self.models[name]
                break
        
        print(f"\n🏆 Best Model: {self.best_model_name.replace('_', ' ').title()}")
        print(f"   Test R² Score: {best_test_r2['test_r2']:.4f}")
        print(f"   Test MAE: {best_test_r2['test_mae']:.2f} lakhs")
    
    def create_visualizations(self):
        """Create model evaluation plots"""
        logger.info("📈 Creating evaluation visualizations...")
        
        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Model Performance Comparison
        ax1 = plt.subplot(2, 3, 1)
        models = list(self.model_scores.keys())
        r2_scores = [self.model_scores[model]['test_r2'] for model in models]
        mae_scores = [self.model_scores[model]['test_mae'] for model in models]
        
        model_names = [name.replace('_', ' ').title() for name in models]
        bars = ax1.bar(model_names, r2_scores, color=['skyblue', 'lightcoral'])
        ax1.set_title('Model R² Score Comparison', fontsize=14, fontweight='bold')
        ax1.set_ylabel('R² Score')
        ax1.set_ylim(0, 1)
        
        # Add value labels
        for bar, score in zip(bars, r2_scores):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 2. MAE Comparison
        ax2 = plt.subplot(2, 3, 2)
        bars2 = ax2.bar(model_names, mae_scores, color=['lightgreen', 'orange'])
        ax2.set_title('Model MAE Comparison', fontsize=14, fontweight='bold')
        ax2.set_ylabel('MAE (₹ lakhs)')
        
        for bar, score in zip(bars2, mae_scores):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{score:.1f}', ha='center', va='bottom', fontweight='bold')
        
        # 3. Predictions vs Actual (Best Model)
        ax3 = plt.subplot(2, 3, 3)
        best_predictions = self.model_scores[self.best_model_name]['predictions']
        
        ax3.scatter(self.y_test, best_predictions, alpha=0.6, color='purple')
        min_price = min(self.y_test.min(), best_predictions.min())
        max_price = max(self.y_test.max(), best_predictions.max())
        ax3.plot([min_price, max_price], [min_price, max_price], 'r--', lw=2, label='Perfect Prediction')
        
        ax3.set_xlabel('Actual Price (₹ lakhs)')
        ax3.set_ylabel('Predicted Price (₹ lakhs)')
        ax3.set_title(f'Predictions vs Actual - {self.best_model_name.replace("_", " ").title()}')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Residuals Plot
        ax4 = plt.subplot(2, 3, 4)
        residuals = self.y_test - best_predictions
        ax4.scatter(best_predictions, residuals, alpha=0.6, color='red')
        ax4.axhline(y=0, color='black', linestyle='--')
        ax4.set_xlabel('Predicted Price (₹ lakhs)')
        ax4.set_ylabel('Residuals (₹ lakhs)')
        ax4.set_title('Residual Plot - Best Model')
        ax4.grid(True, alpha=0.3)
        
        # 5. Feature Importance (if Random Forest is available)
        if 'random_forest' in self.model_scores and 'feature_importance' in self.model_scores['random_forest']:
            ax5 = plt.subplot(2, 3, 5)
            importance = self.model_scores['random_forest']['feature_importance']
            features = list(importance.keys())
            importances = list(importance.values())
            
            # Sort by importance
            sorted_idx = np.argsort(importances)[::-1]
            features_sorted = [features[i] for i in sorted_idx]
            importances_sorted = [importances[i] for i in sorted_idx]
            
            bars5 = ax5.barh(features_sorted, importances_sorted, color='lightblue')
            ax5.set_title('Feature Importance - Random Forest')
            ax5.set_xlabel('Importance')
            
            for i, (bar, imp) in enumerate(zip(bars5, importances_sorted)):
                width = bar.get_width()
                ax5.text(width + max(importances_sorted)*0.01, bar.get_y() + bar.get_height()/2,
                        f'{imp:.3f}', ha='left', va='center', fontsize=9)
        
        # 6. Price Distribution
        ax6 = plt.subplot(2, 3, 6)
        ax6.hist(self.processed_data['price'], bins=50, alpha=0.7, color='gold', edgecolor='black')
        ax6.axvline(self.processed_data['price'].mean(), color='red', linestyle='--', 
                   label=f'Mean: ₹{self.processed_data["price"].mean():.1f}L')
        ax6.axvline(self.processed_data['price'].median(), color='green', linestyle='--', 
                   label=f'Median: ₹{self.processed_data["price"].median():.1f}L')
        ax6.set_xlabel('Price (₹ lakhs)')
        ax6.set_ylabel('Frequency')
        ax6.set_title('Price Distribution')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save the plot
        plot_path = self.config.EVALUATION_PLOTS_DIR / 'model_evaluation_comprehensive.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        logger.info(f"📊 Comprehensive evaluation plot saved: {plot_path}")
        
        # Also save individual comparison plot
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        bars = plt.bar(model_names, r2_scores, color=['skyblue', 'lightcoral'], alpha=0.8)
        plt.title('Model R² Score Comparison', fontsize=14, fontweight='bold')
        plt.ylabel('R² Score')
        plt.ylim(0, 1)
        
        for bar, score in zip(bars, r2_scores):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.subplot(1, 2, 2)
        bars2 = plt.bar(model_names, mae_scores, color=['lightgreen', 'orange'], alpha=0.8)
        plt.title('Model MAE Comparison', fontsize=14, fontweight='bold')
        plt.ylabel('MAE (₹ lakhs)')
        
        for bar, score in zip(bars2, mae_scores):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{score:.1f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        comparison_plot_path = self.config.EVALUATION_PLOTS_DIR / 'model_comparison.png'
        plt.savefig(comparison_plot_path, dpi=300, bbox_inches='tight')
        logger.info(f"📊 Model comparison plot saved: {comparison_plot_path}")
        
        plt.close('all')  # Close all figures to save memory
    
    def save_models(self):
        """Save trained models and preprocessors"""
        logger.info("💾 Saving trained models...")
        
        try:
            # Save individual models
            joblib.dump(self.models['linear_regression'], self.config.LINEAR_REGRESSION_MODEL)
            joblib.dump(self.models['random_forest'], self.config.RANDOM_FOREST_MODEL)
            
            # Save preprocessors
            joblib.dump(self.location_encoder, self.config.LOCATION_ENCODER)
            joblib.dump(self.feature_scaler, self.config.FEATURE_SCALER)
            
            # Save processed data for future reference
            self.processed_data.to_csv(self.config.PROCESSED_DATA_PATH, index=False)
            
            # Save model metadata
            metadata = {
                'best_model': self.best_model_name,
                'model_scores': self.model_scores,
                'feature_columns': list(self.X_train.columns),
                'location_classes': list(self.location_encoder.classes_),
                'dataset_info': {
                    'original_shape': self.raw_data.shape,
                    'processed_shape': self.processed_data.shape,
                    'train_shape': self.X_train.shape,
                    'test_shape': self.X_test.shape
                }
            }
            
            joblib.dump(metadata, self.config.MODEL_METADATA)
            
            logger.info(f"✅ Models saved successfully:")
            logger.info(f"   Linear Regression: {self.config.LINEAR_REGRESSION_MODEL}")
            logger.info(f"   Random Forest: {self.config.RANDOM_FOREST_MODEL}")
            logger.info(f"   Location Encoder: {self.config.LOCATION_ENCODER}")
            logger.info(f"   Feature Scaler: {self.config.FEATURE_SCALER}")
            logger.info(f"   Metadata: {self.config.MODEL_METADATA}")
            
        except Exception as e:
            logger.error(f"❌ Error saving models: {e}")
            raise
    
    def run_complete_pipeline(self):
        """Run the complete ML pipeline"""
        logger.info("🚀 Starting complete ML pipeline...")
        
        pipeline_steps = [
            ("Load Data", self.load_data),
            ("Explore Data", self.explore_data),
            ("Clean Data", self.clean_data),
            ("Feature Engineering", self.feature_engineering),
            ("Prepare Features", self.prepare_features),
            ("Train Linear Regression", self.train_linear_regression),
            ("Train Random Forest", self.train_random_forest),
            ("Evaluate Models", self.evaluate_models),
            ("Create Visualizations", self.create_visualizations),
            ("Save Models", self.save_models)
        ]
        
        for step_name, step_func in pipeline_steps:
            logger.info(f"\n📋 {step_name}...")
            try:
                if step_name in ["Explore Data", "Evaluate Models"]:
                    step_func()  # These don't return boolean
                else:
                    result = step_func()
                    if result is False:
                        logger.error(f"❌ Pipeline failed at: {step_name}")
                        return False
            except Exception as e:
                logger.error(f"❌ Error in {step_name}: {e}")
                return False
        
        logger.info("\n🎉 ML Pipeline completed successfully!")
        logger.info(f"🏆 Best model: {self.best_model_name.replace('_', ' ').title()}")
        logger.info(f"📊 Best R² score: {self.model_scores[self.best_model_name]['test_r2']:.4f}")
        
        return True

def main():
    """Main function to run the training pipeline"""
    trainer = HousePriceModelTrainer()
    success = trainer.run_complete_pipeline()
    
    if not success:
        logger.error("❌ Training pipeline failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
