"""
E-COMMERCE SALES PREDICTION MODEL
=================================
Professional-grade, production-ready code structure.
Implements data science best practices.

Author: AI Assistant
Date: 2025
"""

# =============================================================================
# LIBRARY IMPORTS
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any
import logging
import joblib
import pickle
import os

# Machine Learning libraries
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import lightgbm as lgb

# Suppress warnings
warnings.filterwarnings('ignore')

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Visualization settings
plt.style.use('seaborn-v0_8')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

# =============================================================================
# CONSTANTS AND CONFIGURATION
# =============================================================================

class Config:
    """Model configuration constants"""
    RANDOM_STATE = 42
    TEST_SIZE = 0.2
    CV_FOLDS = 5
    TARGET_COLUMN = 'sales'
    DATE_COLUMN = 'date'
    
    # Model saving paths
    MODEL_DIR = 'models'
    BEST_MODEL_PATH = 'models/best_model.pkl'
    ALL_MODELS_PATH = 'models/all_models.pkl'
    FEATURE_NAMES_PATH = 'models/feature_names.pkl'
    SCALER_PATH = 'models/scaler.pkl'
    PIPELINE_CONFIG_PATH = 'models/pipeline_config.pkl'
    
    # Feature categories
    CATEGORICAL_FEATURES = ['category', 'region']
    NUMERICAL_FEATURES = ['price', 'discount', 'cost', 'day_of_week', 'is_promo_day']
    
    # Model hyperparameters
    MODEL_PARAMS = {
        'xgboost': {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.2],
            'random_state': [RANDOM_STATE]
        },
        'lightgbm': {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.2],
            'random_state': [RANDOM_STATE]
        },
        'random_forest': {
            'n_estimators': [100, 200],
            'max_depth': [5, 10, 15],
            'min_samples_split': [2, 5],
            'random_state': [RANDOM_STATE]
        }
    }

# =============================================================================
# DATA LOADER AND VALIDATOR CLASS
# =============================================================================

class DataLoader:
    """Data loading and basic validation operations"""
    
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.data = None
        
    def load_data(self) -> pd.DataFrame:
        """Load dataset and perform basic checks"""
        try:
            logger.info(f"Loading dataset: {self.file_path}")
            self.data = pd.read_csv(self.file_path)
            logger.info(f"Dataset loaded successfully. Shape: {self.data.shape}")
            return self.data
        except Exception as e:
            logger.error(f"Data loading error: {e}")
            raise
    
    def validate_data(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Data quality checks"""
        validation_report = {}
        
        # Basic information
        validation_report['shape'] = data.shape
        validation_report['columns'] = list(data.columns)
        
        # Missing values
        missing_values = data.isnull().sum()
        validation_report['missing_values'] = missing_values[missing_values > 0].to_dict()
        
        # Data types
        validation_report['data_types'] = data.dtypes.to_dict()
        
        # Business logic checks
        business_checks = {}
        
        # Revenue calculation accuracy check
        if all(col in data.columns for col in ['price', 'discount', 'sales', 'revenue']):
            calculated_revenue = data['price'] * (1 - data['discount']) * data['sales']
            revenue_diff = abs(data['revenue'] - calculated_revenue)
            business_checks['revenue_inconsistency'] = (revenue_diff > 0.01).sum()
        
        # Negative value checks
        for col in ['price', 'sales', 'cost']:
            if col in data.columns:
                business_checks[f'negative_{col}'] = (data[col] < 0).sum()
        
        validation_report['business_checks'] = business_checks
        
        logger.info("Data validation completed")
        return validation_report

# =============================================================================
# DATA PREPROCESSOR CLASS
# =============================================================================

class DataPreprocessor:
    """Data cleaning and preparation operations"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Data cleaning operations"""
        logger.info("Starting data cleaning...")
        
        # Create copy
        df = data.copy()
        
        # Convert date column to datetime format
        df[Config.DATE_COLUMN] = pd.to_datetime(df[Config.DATE_COLUMN])
        
        # Fill missing values (according to business logic)
        if 'discount' in df.columns:
            df['discount'] = df['discount'].fillna(0)  # No discount = 0
        
        # Detect and handle outliers
        df = self._handle_outliers(df)
        
        logger.info("Data cleaning completed")
        return df
    
    def _handle_outliers(self, data: pd.DataFrame) -> pd.DataFrame:
        """Outlier detection and handling"""
        df = data.copy()
        
        for column in ['sales', 'price', 'cost']:
            if column in df.columns:
                Q1 = df[column].quantile(0.25)
                Q3 = df[column].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # Log outliers
                outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
                logger.info(f"Detected {len(outliers)} outliers in {column} column")
                
                # Cap outliers (limit values)
                df[column] = df[column].clip(lower=lower_bound, upper=upper_bound)
        
        return df

# =============================================================================
# FEATURE ENGINEERING CLASS
# =============================================================================

class FeatureEngineer:
    """Feature engineering operations"""
    
    def __init__(self):
        pass
    
    def create_time_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features"""
        logger.info("Creating time-based features...")
        
        df = data.copy()
        date_col = df[Config.DATE_COLUMN]
        
        # Basic time features
        df['year'] = date_col.dt.year
        df['month'] = date_col.dt.month
        df['day'] = date_col.dt.day
        df['quarter'] = date_col.dt.quarter
        df['week_of_year'] = date_col.dt.isocalendar().week
        df['day_of_year'] = date_col.dt.dayofyear
        
        # Weekday/weekend
        df['is_weekend'] = (date_col.dt.weekday >= 5).astype(int)
        
        # Cyclical features (with sine/cosine)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['day_sin'] = np.sin(2 * np.pi * df['day'] / 31)
        df['day_cos'] = np.cos(2 * np.pi * df['day'] / 31)
        
        logger.info("Time-based features completed")
        return df
    
    def create_lag_features(self, data: pd.DataFrame, lags: List[int] = [1, 7, 14, 30]) -> pd.DataFrame:
        """Create lag features"""
        logger.info("Creating lag features...")
        
        df = data.copy()
        
        # Sort dataset by date and product ID
        df = df.sort_values([Config.DATE_COLUMN, 'product_id']).reset_index(drop=True)
        
        # Check minimum data count for each product
        product_counts = df['product_id'].value_counts()
        min_required_data = max(lags) + 7  # Largest lag + rolling window
        
        sufficient_products = product_counts[product_counts >= min_required_data].index
        logger.info(f"Products with sufficient data: {len(sufficient_products)}/{len(product_counts)}")
        
        # Lag features for each product
        for lag in lags:
            lag_col = f'sales_lag_{lag}'
            df[lag_col] = df.groupby('product_id')[Config.TARGET_COLUMN].shift(lag)
            
            # Rolling statistics (only if sufficient data)
            if lag >= 7:
                rolling_mean_col = f'sales_rolling_mean_{lag}'
                rolling_std_col = f'sales_rolling_std_{lag}'
                
                df[rolling_mean_col] = (df.groupby('product_id')[Config.TARGET_COLUMN]
                                       .shift(1)
                                       .rolling(window=lag, min_periods=min(3, lag//2))
                                       .mean())
                
                df[rolling_std_col] = (df.groupby('product_id')[Config.TARGET_COLUMN]
                                      .shift(1)
                                      .rolling(window=lag, min_periods=min(3, lag//2))
                                      .std())
                
                # Fill NaNs with forward fill
                df[rolling_std_col] = df.groupby('product_id')[rolling_std_col].fillna(method='ffill')
        
        # Check NaN counts
        nan_counts_before = df.isnull().sum()
        critical_nan_features = [f'sales_lag_{lags[0]}']  # At least lag-1 should exist
        
        for feature in critical_nan_features:
            if feature in nan_counts_before and nan_counts_before[feature] > 0:
                logger.info(f"{feature} feature has {nan_counts_before[feature]} NaN values")
        
        logger.info("Lag features completed")
        return df
    
    def create_business_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create business logic-based features"""
        logger.info("Creating business logic features...")
        
        df = data.copy()
        
        # Profit margin
        if all(col in df.columns for col in ['price', 'cost']):
            df['profit_margin'] = (df['price'] - df['cost']) / df['price']
            df['profit_margin'] = df['profit_margin'].clip(0, 1)  # Limit to 0-1 range
        
        # Effective price (discounted price)
        if all(col in df.columns for col in ['price', 'discount']):
            df['effective_price'] = df['price'] * (1 - df['discount'])
        
        # Price category (numerical encoding)
        if 'price' in df.columns:
            # Quantile-based categories with numerical codes
            df['price_quartile'] = pd.qcut(df['price'], q=4, labels=False, duplicates='drop')
            # 0: lowest, 3: highest price group
            
            # Price level (0-1 normalized)
            price_min, price_max = df['price'].min(), df['price'].max()
            df['price_normalized'] = (df['price'] - price_min) / (price_max - price_min)
        
        # Product-based statistics
        product_stats = df.groupby('product_id')[Config.TARGET_COLUMN].agg(['mean', 'std']).reset_index()
        product_stats.columns = ['product_id', 'product_avg_sales', 'product_std_sales']
        # Fill NaN values with 0
        product_stats['product_std_sales'] = product_stats['product_std_sales'].fillna(0)
        df = df.merge(product_stats, on='product_id', how='left')
        
        # Category-based statistics
        if 'category' in df.columns:
            category_stats = df.groupby('category')[Config.TARGET_COLUMN].agg(['mean', 'std']).reset_index()
            category_stats.columns = ['category', 'category_avg_sales', 'category_std_sales']
            # Fill NaN values with 0
            category_stats['category_std_sales'] = category_stats['category_std_sales'].fillna(0)
            df = df.merge(category_stats, on='category', how='left')
        
        # Discount indicator (binary)
        if 'discount' in df.columns:
            df['has_discount'] = (df['discount'] > 0).astype(int)
            # Discount level categories
            df['discount_level'] = pd.cut(df['discount'], 
                                        bins=[-0.01, 0, 0.1, 0.2, 1.0], 
                                        labels=[0, 1, 2, 3]).astype(float)
        
        # Price-cost ratio
        if all(col in df.columns for col in ['price', 'cost']):
            df['price_cost_ratio'] = df['price'] / (df['cost'] + 0.01)  # Protection against division by zero
        
        logger.info("Business logic features completed")
        return df
    
    def encode_categorical_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical variables"""
        logger.info("Encoding categorical variables...")
        
        df = data.copy()
        
        # Categorical columns for one-hot encoding
        categorical_cols = [col for col in Config.CATEGORICAL_FEATURES if col in df.columns]
        
        if categorical_cols:
            df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
        
        logger.info("Categorical variables encoded")
        return df

# =============================================================================
# MODEL CLASS
# =============================================================================

class SalesPredictor:
    """Sales prediction model"""
    
    def __init__(self):
        self.models = {
            'linear_regression': LinearRegression(),
            'random_forest': RandomForestRegressor(random_state=Config.RANDOM_STATE),
            'xgboost': xgb.XGBRegressor(random_state=Config.RANDOM_STATE, verbosity=0),
            'lightgbm': lgb.LGBMRegressor(random_state=Config.RANDOM_STATE, verbose=-1)
        }
        self.best_model = None
        self.best_model_name = None
        self.feature_names = None
        self.results = {}
    
    def prepare_features(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features for modeling"""
        df = data.copy()
        
        # First check if target variable exists
        if Config.TARGET_COLUMN not in df.columns:
            raise ValueError(f"Target variable '{Config.TARGET_COLUMN}' not found!")
        
        # Clean NaN values - this operation can change indices
        initial_count = len(df)
        df = df.dropna()
        final_count = len(df)
        
        if initial_count != final_count:
            logger.info(f"NaN cleaning: {initial_count - final_count} rows removed")
        
        # Define feature columns
        excluded_cols = [
            Config.TARGET_COLUMN, Config.DATE_COLUMN, 'product_id', 'revenue'
        ]
        feature_cols = [col for col in df.columns if col not in excluded_cols]
        
        # Data type checking and conversion
        for col in feature_cols:
            if df[col].dtype == 'object':
                logger.warning(f"'{col}' column is object type - attempting to encode")
                # Convert to numerical codes if categorical
                try:
                    df[col] = pd.Categorical(df[col]).codes
                    logger.info(f"'{col}' column converted to numerical codes")
                except:
                    logger.error(f"'{col}' column could not be converted, removing from features")
                    feature_cols.remove(col)
            elif df[col].dtype == 'category':
                # Convert category type to numerical codes
                df[col] = df[col].cat.codes
                logger.info(f"'{col}' categorical column converted to numerical codes")
        
        # Check and clean infinite values
        inf_cols = []
        for col in feature_cols:
            if df[col].dtype in ['float64', 'int64', 'float32', 'int32']:
                inf_count = np.isinf(df[col]).sum()
                if inf_count > 0:
                    inf_cols.append(col)
                    logger.warning(f"Found {inf_count} infinite values in '{col}' column")
        
        if inf_cols:
            df = df.replace([np.inf, -np.inf], np.nan).dropna()
            logger.info(f"Infinite values cleaned, remaining rows: {len(df)}")
        
        # Final check: empty DataFrame check
        if len(df) == 0:
            raise ValueError("No rows left after data cleaning!")
        
        # Separate features and target
        X = df[feature_cols].copy()
        y = df[Config.TARGET_COLUMN].copy()
        
        # Final data type check
        non_numeric_cols = []
        for col in X.columns:
            if not pd.api.types.is_numeric_dtype(X[col]):
                non_numeric_cols.append(col)
                logger.warning(f"'{col}' column is still non-numeric: {X[col].dtype}")
        
        if non_numeric_cols:
            # Last resort: remove problematic columns
            X = X.drop(columns=non_numeric_cols)
            feature_cols = [col for col in feature_cols if col not in non_numeric_cols]
            logger.warning(f"Removed {len(non_numeric_cols)} non-numeric columns")
        
        # Empty features check
        if len(X.columns) == 0:
            raise ValueError("No numeric features remaining!")
        
        self.feature_names = list(X.columns)
        logger.info(f"Prepared {len(self.feature_names)} features for modeling")
        logger.info(f"Available rows count: {len(X)}")
        
        return X, y
    
    def split_data(self, X: pd.DataFrame, y: pd.Series, data: pd.DataFrame) -> Tuple:
        """Appropriate data splitting for time series"""
        # Reset index to prevent index mismatch
        # This solves index problems after dropna()
        
        # Find common indices (remaining data after NaN cleaning)
        common_indices = X.index.intersection(data.index)
        
        # Use only common indices
        data_filtered = data.loc[common_indices].copy()
        X_filtered = X.loc[common_indices].copy()
        y_filtered = y.loc[common_indices].copy()
        
        # Sort by date and reset index
        data_sorted = data_filtered.sort_values(Config.DATE_COLUMN).reset_index(drop=True)
        X_sorted = X_filtered.loc[data_filtered.sort_values(Config.DATE_COLUMN).index].reset_index(drop=True)
        y_sorted = y_filtered.loc[data_filtered.sort_values(Config.DATE_COLUMN).index].reset_index(drop=True)
        
        # Split last 20% as test set
        split_idx = int(len(X_sorted) * (1 - Config.TEST_SIZE))
        
        X_train = X_sorted.iloc[:split_idx]
        X_test = X_sorted.iloc[split_idx:]
        y_train = y_sorted.iloc[:split_idx]
        y_test = y_sorted.iloc[split_idx:]
        
        logger.info(f"Data split: Train={len(X_train)}, Test={len(X_test)}")
        logger.info(f"Index mismatch resolved: {len(common_indices)} records used")
        
        return X_train, X_test, y_train, y_test
    
    def train_models(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict:
        """Train models and evaluate with cross-validation"""
        logger.info("Training models...")
        
        results = {}
        
        # Time Series Cross Validation
        tscv = TimeSeriesSplit(n_splits=Config.CV_FOLDS)
        
        for model_name, model in self.models.items():
            logger.info(f"Training {model_name}...")
            
            # Cross-validation scores
            cv_scores = []
            
            for train_idx, val_idx in tscv.split(X_train):
                X_train_fold = X_train.iloc[train_idx]
                X_val_fold = X_train.iloc[val_idx]
                y_train_fold = y_train.iloc[train_idx]
                y_val_fold = y_train.iloc[val_idx]
                
                # Train model
                model.fit(X_train_fold, y_train_fold)
                
                # Make predictions
                y_pred_fold = model.predict(X_val_fold)
                
                # Calculate score
                rmse = np.sqrt(mean_squared_error(y_val_fold, y_pred_fold))
                cv_scores.append(rmse)
            
            # Train final model with all training data
            model.fit(X_train, y_train)
            
            results[model_name] = {
                'model': model,
                'cv_scores': cv_scores,
                'cv_mean': np.mean(cv_scores),
                'cv_std': np.std(cv_scores)
            }
            
            logger.info(f"{model_name} - CV RMSE: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores)*2:.4f})")
        
        self.results = results
        
        # Select best model
        best_model_name = min(results.keys(), key=lambda x: results[x]['cv_mean'])
        self.best_model = results[best_model_name]['model']
        self.best_model_name = best_model_name
        
        logger.info(f"Best model: {best_model_name}")
        
        # Save models
        self._save_models()
        
        return results
    
    def _save_models(self):
        """Save trained models and related objects"""
        try:
            # Create models directory if it doesn't exist
            os.makedirs(Config.MODEL_DIR, exist_ok=True)
            
            # Save best model
            logger.info(f"Saving best model ({self.best_model_name}) to {Config.BEST_MODEL_PATH}")
            joblib.dump(self.best_model, Config.BEST_MODEL_PATH)
            
            # Save all models
            logger.info(f"Saving all models to {Config.ALL_MODELS_PATH}")
            all_models_dict = {name: info['model'] for name, info in self.results.items()}
            joblib.dump(all_models_dict, Config.ALL_MODELS_PATH)
            
            # Save feature names
            logger.info(f"Saving feature names to {Config.FEATURE_NAMES_PATH}")
            joblib.dump(self.feature_names, Config.FEATURE_NAMES_PATH)
            
            # Save model metadata
            model_metadata = {
                'best_model_name': self.best_model_name,
                'feature_names': self.feature_names,
                'model_performances': {name: {
                    'cv_mean': info['cv_mean'],
                    'cv_std': info['cv_std']
                } for name, info in self.results.items()},
                'training_date': datetime.now().isoformat(),
                'data_shape': None  # Will be filled by pipeline
            }
            
            with open('models/model_metadata.pkl', 'wb') as f:
                pickle.dump(model_metadata, f)
                
            logger.info("All models and metadata saved successfully!")
            
        except Exception as e:
            logger.error(f"Error saving models: {e}")
            raise
    
    def load_model(self, model_path: str = None):
        """Load a saved model"""
        if model_path is None:
            model_path = Config.BEST_MODEL_PATH
            
        try:
            logger.info(f"Loading model from {model_path}")
            self.best_model = joblib.load(model_path)
            
            # Load feature names if available
            if os.path.exists(Config.FEATURE_NAMES_PATH):
                self.feature_names = joblib.load(Config.FEATURE_NAMES_PATH)
                logger.info("Feature names loaded successfully")
            
            # Load metadata if available
            if os.path.exists('models/model_metadata.pkl'):
                with open('models/model_metadata.pkl', 'rb') as f:
                    metadata = pickle.load(f)
                    self.best_model_name = metadata.get('best_model_name', 'loaded_model')
                logger.info("Model metadata loaded successfully")
                
            logger.info("Model loaded successfully!")
            return self.best_model
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def evaluate_models(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
        """Evaluate models on test set"""
        logger.info("Testing models...")
        
        evaluation_results = {}
        
        for model_name, model_info in self.results.items():
            model = model_info['model']
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            evaluation_results[model_name] = {
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'predictions': y_pred
            }
            
            logger.info(f"{model_name} - Test RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
        
        return evaluation_results

# =============================================================================
# VISUALIZATION CLASS
# =============================================================================

class Visualizer:
    """Data visualization operations"""
    
    @staticmethod
    def plot_data_overview(data: pd.DataFrame):
        """General overview of the dataset"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Sales distribution
        axes[0,0].hist(data[Config.TARGET_COLUMN], bins=50, edgecolor='black', alpha=0.7)
        axes[0,0].set_title('Sales Distribution')
        axes[0,0].set_xlabel('Sales Amount')
        axes[0,0].set_ylabel('Frequency')
        
        # Time series
        daily_sales = data.groupby(Config.DATE_COLUMN)[Config.TARGET_COLUMN].sum()
        axes[0,1].plot(daily_sales.index, daily_sales.values)
        axes[0,1].set_title('Daily Total Sales')
        axes[0,1].set_xlabel('Date')
        axes[0,1].set_ylabel('Sales Amount')
        axes[0,1].tick_params(axis='x', rotation=45)
        
        # Category-based sales
        if 'category' in data.columns:
            category_sales = data.groupby('category')[Config.TARGET_COLUMN].sum()
            axes[1,0].bar(category_sales.index, category_sales.values)
            axes[1,0].set_title('Category-Based Total Sales')
            axes[1,0].set_xlabel('Category')
            axes[1,0].set_ylabel('Sales Amount')
        
        # Region-based sales
        if 'region' in data.columns:
            region_sales = data.groupby('region')[Config.TARGET_COLUMN].sum()
            axes[1,1].bar(region_sales.index, region_sales.values, color='orange')
            axes[1,1].set_title('Region-Based Total Sales')
            axes[1,1].set_xlabel('Region')
            axes[1,1].set_ylabel('Sales Amount')
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_model_performance(evaluation_results: Dict, y_test: pd.Series):
        """Model performance comparison"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # RMSE comparison
        model_names = list(evaluation_results.keys())
        rmse_scores = [evaluation_results[name]['rmse'] for name in model_names]
        
        axes[0,0].bar(model_names, rmse_scores, color='skyblue', edgecolor='navy')
        axes[0,0].set_title('Model RMSE Comparison')
        axes[0,0].set_xlabel('Model')
        axes[0,0].set_ylabel('RMSE')
        axes[0,0].tick_params(axis='x', rotation=45)
        
        # R² comparison
        r2_scores = [evaluation_results[name]['r2'] for name in model_names]
        axes[0,1].bar(model_names, r2_scores, color='lightcoral', edgecolor='darkred')
        axes[0,1].set_title('Model R² Comparison')
        axes[0,1].set_xlabel('Model')
        axes[0,1].set_ylabel('R²')
        axes[0,1].tick_params(axis='x', rotation=45)
        
        # Best model predictions vs actual
        best_model = min(model_names, key=lambda x: evaluation_results[x]['rmse'])
        best_predictions = evaluation_results[best_model]['predictions']
        
        axes[1,0].scatter(y_test, best_predictions, alpha=0.6)
        axes[1,0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        axes[1,0].set_title(f'{best_model} - Actual vs Predicted')
        axes[1,0].set_xlabel('Actual Values')
        axes[1,0].set_ylabel('Predicted Values')
        
        # Residuals plot
        residuals = y_test - best_predictions
        axes[1,1].scatter(best_predictions, residuals, alpha=0.6)
        axes[1,1].axhline(y=0, color='r', linestyle='--')
        axes[1,1].set_title(f'{best_model} - Residuals')
        axes[1,1].set_xlabel('Predicted Values')
        axes[1,1].set_ylabel('Residuals')
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_feature_importance(model, feature_names: List[str], top_n: int = 15):
        """Visualize feature importance"""
        if hasattr(model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False).head(top_n)
            
            plt.figure(figsize=(10, 8))
            sns.barplot(data=importance_df, y='feature', x='importance')
            plt.title(f'Top {top_n} Feature Importance')
            plt.xlabel('Importance')
            plt.ylabel('Feature')
            plt.tight_layout()
            plt.show()

# =============================================================================
# MAIN PIPELINE CLASS
# =============================================================================

class EcommerceSalesPipeline:
    """Main pipeline class - coordinates all operations"""
    
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.data_loader = DataLoader(data_path)
        self.preprocessor = DataPreprocessor()
        self.feature_engineer = FeatureEngineer()
        self.predictor = SalesPredictor()
        self.visualizer = Visualizer()
        
        self.raw_data = None
        self.processed_data = None
        self.evaluation_results = None
    
    def run_pipeline(self):
        """Run complete pipeline"""
        logger.info("Pipeline starting...")
        
        try:
            # 1. Data loading
            logger.info("Step 1/10: Data loading...")
            self.raw_data = self.data_loader.load_data()
            
            # 2. Data validation
            logger.info("Step 2/10: Data validation...")
            validation_report = self.data_loader.validate_data(self.raw_data)
            self._print_validation_report(validation_report)
            
            # 3. Data visualization (overview)
            logger.info("Step 3/10: Data visualization...")
            self.visualizer.plot_data_overview(self.raw_data)
            
            # 4. Data preprocessing
            logger.info("Step 4/10: Data cleaning...")
            cleaned_data = self.preprocessor.clean_data(self.raw_data)
            
            # 5. Feature engineering
            logger.info("Step 5/10: Feature engineering...")
            data_with_time_features = self.feature_engineer.create_time_features(cleaned_data)
            data_with_lag_features = self.feature_engineer.create_lag_features(data_with_time_features)
            data_with_business_features = self.feature_engineer.create_business_features(data_with_lag_features)
            self.processed_data = self.feature_engineer.encode_categorical_features(data_with_business_features)
            
            logger.info(f"Feature engineering completed. Final data shape: {self.processed_data.shape}")
            
            # 6. Model preparation
            logger.info("Step 6/10: Model data preparation...")
            X, y = self.predictor.prepare_features(self.processed_data)
            
            # Minimum data check
            if len(X) < 100:
                raise ValueError(f"Insufficient data! Only {len(X)} rows remaining. At least 100 rows required.")
            
            X_train, X_test, y_train, y_test = self.predictor.split_data(X, y, self.processed_data)
            
            # 7. Model training (models will be saved automatically)
            logger.info("Step 7/10: Model training...")
            training_results = self.predictor.train_models(X_train, y_train)
            
            # 8. Model evaluation
            logger.info("Step 8/10: Model evaluation...")
            self.evaluation_results = self.predictor.evaluate_models(X_test, y_test)
            
            # 9. Save pipeline state
            logger.info("Step 9/10: Saving pipeline state...")
            self._save_pipeline_state()
            
            # 10. Visualize results
            logger.info("Step 10/10: Results visualization...")
            self.visualizer.plot_model_performance(self.evaluation_results, y_test)
            
            # Show best model feature importance
            if hasattr(self.predictor.best_model, 'feature_importances_'):
                self.visualizer.plot_feature_importance(
                    self.predictor.best_model, 
                    self.predictor.feature_names
                )
            else:
                logger.info("Selected model does not support feature importance")
            
            logger.info("Pipeline completed successfully!")
            
            return self.evaluation_results
            
        except Exception as e:
            logger.error(f"Pipeline error: {str(e)}")
            logger.error(f"Error type: {type(e).__name__}")
            
            # Debug information
            if hasattr(self, 'processed_data') and self.processed_data is not None:
                logger.error(f"Processed data shape: {self.processed_data.shape}")
                logger.error(f"NaN value counts: {self.processed_data.isnull().sum().sum()}")
            
            raise
    
    def _save_pipeline_state(self):
        """Save complete pipeline state for reproducibility"""
        try:
            # Create models directory
            os.makedirs(Config.MODEL_DIR, exist_ok=True)
            
            # Save pipeline configuration
            pipeline_state = {
                'config': {
                    'RANDOM_STATE': Config.RANDOM_STATE,
                    'TEST_SIZE': Config.TEST_SIZE,
                    'CV_FOLDS': Config.CV_FOLDS,
                    'TARGET_COLUMN': Config.TARGET_COLUMN,
                    'DATE_COLUMN': Config.DATE_COLUMN
                },
                'data_info': {
                    'raw_data_shape': self.raw_data.shape if self.raw_data is not None else None,
                    'processed_data_shape': self.processed_data.shape if self.processed_data is not None else None,
                    'feature_count': len(self.predictor.feature_names) if self.predictor.feature_names else 0
                },
                'model_results': self.evaluation_results,
                'best_model_name': self.predictor.best_model_name,
                'timestamp': datetime.now().isoformat()
            }
            
            with open(Config.PIPELINE_CONFIG_PATH, 'wb') as f:
                pickle.dump(pipeline_state, f)
                
            logger.info("Pipeline state saved successfully!")
            
        except Exception as e:
            logger.error(f"Error saving pipeline state: {e}")
    
    def load_trained_pipeline(self, model_path: str = None):
        """Load a previously trained pipeline"""
        try:
            # Load the trained model
            self.predictor.load_model(model_path)
            
            # Load pipeline configuration if available
            if os.path.exists(Config.PIPELINE_CONFIG_PATH):
                with open(Config.PIPELINE_CONFIG_PATH, 'rb') as f:
                    pipeline_state = pickle.load(f)
                    
                logger.info("Trained pipeline loaded successfully!")
                logger.info(f"Best model: {pipeline_state.get('best_model_name', 'Unknown')}")
                logger.info(f"Training date: {pipeline_state.get('timestamp', 'Unknown')}")
                
                return pipeline_state
            else:
                logger.warning("Pipeline configuration not found, only model loaded")
                return None
                
        except Exception as e:
            logger.error(f"Error loading trained pipeline: {e}")
            raise
    
    def _print_validation_report(self, report: Dict):
        """Print validation report"""
        print("\n" + "="*50)
        print("DATA VALIDATION REPORT")
        print("="*50)
        
        print(f"Data shape: {report['shape']}")
        print(f"Column count: {len(report['columns'])}")
        
        if report['missing_values']:
            print("\nMissing values:")
            for col, count in report['missing_values'].items():
                print(f"  {col}: {count}")
        else:
            print("\nNo missing values ✓")
        
        print("\nBusiness logic checks:")
        for check, result in report['business_checks'].items():
            status = "⚠️" if result > 0 else "✓"
            print(f"  {check}: {result} {status}")
        
        print("="*50 + "\n")
    
    def predict_future_sales(self, days: int = 30) -> pd.DataFrame:
        """Predict future sales"""
        if self.predictor.best_model is None:
            raise ValueError("Model not trained yet. Run run_pipeline() first.")
        
        logger.info(f"Predicting sales for next {days} days...")
        
        # Find last date
        last_date = self.processed_data[Config.DATE_COLUMN].max()
        
        # Create future dates
        future_dates = pd.date_range(
            start=last_date + timedelta(days=1), 
            periods=days, 
            freq='D'
        )
        
        # Create future data for each product
        future_predictions = []
        unique_products = self.processed_data['product_id'].unique()
        
        for product_id in unique_products:
            product_data = self.processed_data[
                self.processed_data['product_id'] == product_id
            ].sort_values(Config.DATE_COLUMN)
            
            if len(product_data) == 0:
                continue
                
            last_row = product_data.iloc[-1]
            
            for future_date in future_dates:
                # Create new row
                future_row = {}
                
                # Date features
                future_row['year'] = future_date.year
                future_row['month'] = future_date.month
                future_row['day'] = future_date.day
                future_row['quarter'] = future_date.quarter
                future_row['week_of_year'] = future_date.isocalendar().week
                future_row['day_of_year'] = future_date.dayofyear
                future_row['is_weekend'] = 1 if future_date.weekday() >= 5 else 0
                
                # Cyclical features
                future_row['month_sin'] = np.sin(2 * np.pi * future_date.month / 12)
                future_row['month_cos'] = np.cos(2 * np.pi * future_date.month / 12)
                future_row['day_sin'] = np.sin(2 * np.pi * future_date.day / 31)
                future_row['day_cos'] = np.cos(2 * np.pi * future_date.day / 31)
                
                # Static features (use last values)
                for col in ['price', 'discount', 'cost', 'day_of_week', 'is_promo_day']:
                    if col in last_row:
                        future_row[col] = last_row[col]
                
                # Lag features (from existing data)
                recent_sales = product_data[Config.TARGET_COLUMN].values
                if len(recent_sales) >= 1:
                    future_row['sales_lag_1'] = recent_sales[-1]
                if len(recent_sales) >= 7:
                    future_row['sales_lag_7'] = recent_sales[-7]
                    future_row['sales_rolling_mean_7'] = np.mean(recent_sales[-7:])
                    future_row['sales_rolling_std_7'] = np.std(recent_sales[-7:])
                if len(recent_sales) >= 14:
                    future_row['sales_lag_14'] = recent_sales[-14]
                    future_row['sales_rolling_mean_14'] = np.mean(recent_sales[-14:])
                    future_row['sales_rolling_std_14'] = np.std(recent_sales[-14:])
                if len(recent_sales) >= 30:
                    future_row['sales_lag_30'] = recent_sales[-30]
                    future_row['sales_rolling_mean_30'] = np.mean(recent_sales[-30:])
                    future_row['sales_rolling_std_30'] = np.std(recent_sales[-30:])
                
                # Business logic features
                if 'profit_margin' in last_row:
                    future_row['profit_margin'] = last_row['profit_margin']
                if 'effective_price' in last_row:
                    future_row['effective_price'] = last_row['effective_price']
                if 'price_quartile' in last_row:
                    future_row['price_quartile'] = last_row['price_quartile']
                if 'price_normalized' in last_row:
                    future_row['price_normalized'] = last_row['price_normalized']
                if 'has_discount' in last_row:
                    future_row['has_discount'] = 0  # Assumption: no discount in future
                if 'discount_level' in last_row:
                    future_row['discount_level'] = 0  # Discount level 0
                if 'price_cost_ratio' in last_row:
                    future_row['price_cost_ratio'] = last_row['price_cost_ratio']
                
                # Product and category statistics
                for col in ['product_avg_sales', 'product_std_sales', 
                           'category_avg_sales', 'category_std_sales']:
                    if col in last_row:
                        future_row[col] = last_row[col]
                
                # Categorical features (one-hot encoded)
                for col in self.processed_data.columns:
                    if col.startswith(('category_', 'region_')) and col in last_row:
                        future_row[col] = last_row[col]
                
                # Meta information
                future_row['date'] = future_date
                future_row['product_id'] = product_id
                
                future_predictions.append(future_row)
        
        # Convert to DataFrame
        future_df = pd.DataFrame(future_predictions)
        
        # Select model features
        feature_cols = [col for col in self.predictor.feature_names if col in future_df.columns]
        missing_features = set(self.predictor.feature_names) - set(feature_cols)
        
        if missing_features:
            logger.warning(f"Missing features will be filled with zeros: {missing_features}")
            for col in missing_features:
                future_df[col] = 0
        
        # Make predictions
        X_future = future_df[self.predictor.feature_names]
        future_predictions = self.predictor.best_model.predict(X_future)
        
        # Create result DataFrame
        result_df = pd.DataFrame({
            'date': future_df['date'],
            'product_id': future_df['product_id'],
            'predicted_sales': future_predictions
        })
        
        logger.info(f"Future sales predictions completed: {len(result_df)} predictions")
        
        return result_df
    
    def generate_report(self) -> str:
        """Generate detailed analysis report"""
        if self.evaluation_results is None:
            raise ValueError("Model not trained yet. Run run_pipeline() first.")
        
        report = []
        report.append("E-COMMERCE SALES PREDICTION MODEL REPORT")
        report.append("=" * 50)
        report.append("")
        
        # Dataset information
        report.append("DATASET INFORMATION:")
        report.append(f"- Total records: {len(self.raw_data):,}")
        report.append(f"- Number of features: {len(self.predictor.feature_names)}")
        report.append(f"- Date range: {self.raw_data[Config.DATE_COLUMN].min()} - {self.raw_data[Config.DATE_COLUMN].max()}")
        report.append(f"- Number of products: {self.raw_data['product_id'].nunique()}")
        report.append("")
        
        # Model performances
        report.append("MODEL PERFORMANCES:")
        for model_name, results in self.evaluation_results.items():
            report.append(f"- {model_name}:")
            report.append(f"  * RMSE: {results['rmse']:.4f}")
            report.append(f"  * MAE: {results['mae']:.4f}")
            report.append(f"  * R²: {results['r2']:.4f}")
        report.append("")
        
        # Best model
        best_model = min(self.evaluation_results.keys(), 
                        key=lambda x: self.evaluation_results[x]['rmse'])
        report.append(f"BEST MODEL: {best_model}")
        report.append(f"- RMSE: {self.evaluation_results[best_model]['rmse']:.4f}")
        report.append(f"- MAE: {self.evaluation_results[best_model]['mae']:.4f}")
        report.append(f"- R²: {self.evaluation_results[best_model]['r2']:.4f}")
        report.append("")
        
        # Recommendations
        report.append("RECOMMENDATIONS:")
        report.append("- Add more external features (weather, holidays)")
        report.append("- Implement hyperparameter tuning to improve model performance")
        report.append("- Try ensemble methods")
        report.append("- Perform seasonal decomposition analysis")
        report.append("")
        
        report_text = "\n".join(report)
        logger.info("Report generated")
        
        return report_text

# =============================================================================
# USAGE EXAMPLE AND MAIN FUNCTION
# =============================================================================

def main():
    """Main function - run the pipeline"""
    
    # Initialize pipeline
    pipeline = EcommerceSalesPipeline('ecommerce_sales.csv')
    
    try:
        # Run pipeline
        results = pipeline.run_pipeline()
        
        # Generate and print report
        report = pipeline.generate_report()
        print(report)
        
        # Make future predictions
        future_predictions = pipeline.predict_future_sales(days=30)
        
        # Save results
        future_predictions.to_csv('future_sales_predictions.csv', index=False)
        logger.info("Future predictions saved to 'future_sales_predictions.csv'")
        
        # Save detailed report
        with open('model_analysis_report.txt', 'w', encoding='utf-8') as f:
            f.write(report)
        logger.info("Detailed report saved to 'model_analysis_report.txt'")
        
        # Print model file locations
        print(f"\n🎯 MODEL FILES SAVED:")
        print(f"├── Best Model: {Config.BEST_MODEL_PATH}")
        print(f"├── All Models: {Config.ALL_MODELS_PATH}")
        print(f"├── Feature Names: {Config.FEATURE_NAMES_PATH}")
        print(f"├── Pipeline Config: {Config.PIPELINE_CONFIG_PATH}")
        print(f"└── Model Metadata: models/model_metadata.pkl")
        
        # Visualize predictions for top-selling products
        top_products = pipeline.raw_data.groupby('product_id')[Config.TARGET_COLUMN].sum().nlargest(3).index
        
        plt.figure(figsize=(15, 5))
        for i, product_id in enumerate(top_products):
            plt.subplot(1, 3, i+1)
            product_predictions = future_predictions[future_predictions['product_id'] == product_id]
            plt.plot(product_predictions['date'], product_predictions['predicted_sales'], marker='o')
            plt.title(f'Product {product_id} - 30-Day Forecast')
            plt.xlabel('Date')
            plt.ylabel('Predicted Sales')
            plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig('future_predictions_visualization.png', dpi=300, bbox_inches='tight')
        plt.show()
        logger.info("Prediction visualization saved to 'future_predictions_visualization.png'")
        
        return pipeline
        
    except Exception as e:
        logger.error(f"Pipeline error: {e}")
        raise

def load_and_predict(data_path: str, model_path: str = None):
    """Load trained model and make predictions on new data"""
    
    # Initialize pipeline
    pipeline = EcommerceSalesPipeline(data_path)
    
    try:
        # Load trained model
        pipeline.load_trained_pipeline(model_path)
        
        # Process new data (without training)
        logger.info("Processing new data...")
        pipeline.raw_data = pipeline.data_loader.load_data()
        cleaned_data = pipeline.preprocessor.clean_data(pipeline.raw_data)
        data_with_time_features = pipeline.feature_engineer.create_time_features(cleaned_data)
        data_with_lag_features = pipeline.feature_engineer.create_lag_features(data_with_time_features)
        data_with_business_features = pipeline.feature_engineer.create_business_features(data_with_lag_features)
        pipeline.processed_data = pipeline.feature_engineer.encode_categorical_features(data_with_business_features)
        
        # Make predictions
        logger.info("Making predictions...")
        future_predictions = pipeline.predict_future_sales(days=30)
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f'predictions_{timestamp}.csv'
        future_predictions.to_csv(output_file, index=False)
        logger.info(f"Predictions saved to '{output_file}'")
        
        return future_predictions
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise

# =============================================================================
# DEVELOPMENT AND DEBUGGING HELPER FUNCTIONS
# =============================================================================

def analyze_model_errors(pipeline, threshold: float = 100):
    """Analyze model errors"""
    if pipeline.evaluation_results is None:
        print("Model not trained yet!")
        return
    
    # Get best model predictions
    best_model_name = min(pipeline.evaluation_results.keys(), 
                         key=lambda x: pipeline.evaluation_results[x]['rmse'])
    predictions = pipeline.evaluation_results[best_model_name]['predictions']
    
    # Get test set - use correct size
    X, y = pipeline.predictor.prepare_features(pipeline.processed_data)
    X_train, X_test, y_train, y_test = pipeline.predictor.split_data(X, y, pipeline.processed_data)
    
    # Check if same size
    if len(y_test) != len(predictions):
        print(f"Size mismatch: y_test={len(y_test)}, predictions={len(predictions)}")
        # Cut to shorter size
        min_len = min(len(y_test), len(predictions))
        y_test = y_test.iloc[:min_len]
        predictions = predictions[:min_len]
    
    # Calculate errors
    errors = np.abs(y_test.values - predictions)
    high_error_indices = np.where(errors > threshold)[0]
    
    print(f"High error ({threshold}+) predictions: {len(high_error_indices)}")
    print(f"Average error: {np.mean(errors):.2f}")
    print(f"Maximum error: {np.max(errors):.2f}")
    
    if len(high_error_indices) > 0:
        print(f"\nTop 5 highest errors:")
        error_df = pd.DataFrame({
            'actual': y_test.iloc[high_error_indices].values,
            'predicted': predictions[high_error_indices],
            'error': errors[high_error_indices]
        }).sort_values('error', ascending=False).head()
        print(error_df)

def feature_correlation_analysis(pipeline):
    """Feature correlation analysis"""
    if pipeline.processed_data is None:
        print("Data not processed yet!")
        return
    
    # Select numeric columns
    numeric_cols = pipeline.processed_data.select_dtypes(include=[np.number]).columns
    correlation_matrix = pipeline.processed_data[numeric_cols].corr()
    
    # Correlation with target variable
    target_corr = correlation_matrix[Config.TARGET_COLUMN].abs().sort_values(ascending=False)
    
    print("Highest correlation with target variable:")
    print(target_corr.head(10))
    
    # Visualize correlation matrix
    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(correlation_matrix))
    sns.heatmap(correlation_matrix, mask=mask, cmap='coolwarm', center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": .8})
    plt.title('Feature Correlation Matrix')
    plt.show()

# Run main function if script is executed
if __name__ == "__main__":
    # Example usage
    print("E-Commerce Sales Prediction Model starting...")
    
    # Option 1: Train new model
    print("\n🚀 TRAINING NEW MODEL...")
    pipeline = main()
    
    # Optional analyses
    print("\n📊 RUNNING ADDITIONAL ANALYSES...")
    print("\nError analysis:")
    analyze_model_errors(pipeline)
    
    print("\nCorrelation analysis:")
    feature_correlation_analysis(pipeline)
    
    # Option 2: Load existing model and predict (example)
    print("\n🔄 EXAMPLE: LOADING EXISTING MODEL...")
    try:
        # This would work if you have a trained model
        # new_predictions = load_and_predict('new_data.csv', 'models/best_model.pkl')
        # print("New predictions generated successfully!")
        print("To use existing model: load_and_predict('new_data.csv', 'models/best_model.pkl')")
    except Exception as e:
        print(f"Model loading example: {e}")
    
    print("\n✅ PIPELINE COMPLETED!")
    print("📁 Check the 'models/' directory for saved model files")
    print("📈 Check output files: future_sales_predictions.csv, model_analysis_report.txt")