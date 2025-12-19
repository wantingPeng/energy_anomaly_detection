"""
Data loader for Random Forest time series anomaly detection.

This module loads pre-computed window features and prepares data for Random Forest training.
"""

import os
import sys
import numpy as np
import pandas as pd
import pickle
from typing import Dict, List, Tuple, Optional
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from src.utils.logger import logger


# Identity scaler for Random Forest (which doesn't need feature scaling)
class IdentityScaler:
    """
    A dummy scaler that returns the original data unchanged.
    Used for model types that don't require feature scaling, like Random Forest.
    """
    def transform(self, X):
        return X
        
    def inverse_transform(self, X):
        return X
        
    def fit(self, X):
        return self


class RandomForestDataLoader:
    """
    Data loader for Random Forest time series anomaly detection.
    
    Features:
    - Loads pre-computed window features
    - No feature engineering (already done)
    - Sequential train/val/test split for time series
    - Supports train/validation/test split for supervised learning
    """
    
    def __init__(
        self,
        data_path: str,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        target_column: str = "anomaly_label",
        exclude_columns: List[str] = None,
        balance_method: Optional[str] = None,
    ):
        """
        Initialize Random Forest data loader.
        
        Args:
            data_path: Path to parquet data file with window features
            train_ratio: Ratio of data for training
            val_ratio: Ratio of data for validation
            test_ratio: Ratio of data for testing
            target_column: Name of the target column
            exclude_columns: Additional columns to exclude from features
            balance_method: Method to balance classes (None or 'class_weight')
        """
        self.data_path = data_path
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.target_column = target_column
        self.balance_method = balance_method
        
        # Validate ratios
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError(f"Ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}")
        
        # Columns to exclude from features
        self.exclude_columns = exclude_columns or []
        # Always exclude target column
        if target_column not in self.exclude_columns:
            self.exclude_columns.append(target_column)
        
        # Initialize attributes
        self.raw_data = None
        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.feature_columns = None
        self.class_weights = None
        
        # Load and process data
        self._load_data()
        self._split_data()
        self._log_statistics()
        
        # Balance data if requested
        if self.balance_method:
            self._balance_data()
    
    def _load_data(self):
        """Load pre-computed window features from parquet file."""
        logger.info(f"Loading window features from: {self.data_path}")
        
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        
        self.raw_data = pd.read_parquet(self.data_path)
        logger.info(f"Loaded {len(self.raw_data)} rows with {len(self.raw_data.columns)} columns")
        
        # Verify target column exists
        if self.target_column not in self.raw_data.columns:
            raise ValueError(f"Target column '{self.target_column}' not found in data")
        
        # Get feature columns (all except excluded)
        self.feature_columns = [col for col in self.raw_data.columns 
                               if col not in self.exclude_columns]
        
        logger.info(f"Feature columns: {len(self.feature_columns)}")
        logger.info(f"Target column: {self.target_column}")
        logger.info(f"First few features: {self.feature_columns[:5]}")
        
        # Log overall anomaly statistics
        total_anomalies = self.raw_data[self.target_column].sum()
        total_samples = len(self.raw_data)
    
    def _split_data(self):
        """
        Split data into train, validation, and test sets sequentially.
        Sequential split preserves temporal order for time series data.
        """

        total_samples = len(self.raw_data)
        train_end = int(total_samples * self.train_ratio)
        val_end = int(total_samples * (self.train_ratio + self.val_ratio))
        
        # Sequential split to preserve temporal order
        self.train_data = self.raw_data.iloc[:train_end].copy().reset_index(drop=True)
        self.val_data = self.raw_data.iloc[train_end:val_end].copy().reset_index(drop=True)
        self.test_data = self.raw_data.iloc[val_end:].copy().reset_index(drop=True)
        
        logger.info(f"Train: {len(self.train_data)} samples")
        logger.info(f"Val: {len(self.val_data)} samples")
        logger.info(f"Test: {len(self.test_data)} samples")
    
    def _balance_data(self):
        """
        Balance training data using class weights.
        """
        from sklearn.utils import class_weight
        import numpy as np
        
        if self.balance_method == 'class_weight':
            # Compute class weights for imbalanced dataset
            train_labels = self.train_data[self.target_column].values
            self.class_weights = class_weight.compute_class_weight(
                'balanced', 
                classes=np.unique(train_labels), 
                y=train_labels
            )
            self.class_weights = dict(enumerate(self.class_weights))
            
            logger.info(f"Computed class weights: {self.class_weights}")
        else:
            logger.warning(f"Unknown balance method: {self.balance_method}")
            return
    
    def _log_statistics(self):
        """Log detailed statistics about the data splits."""
        logger.info("\n" + "=" * 60)
        logger.info("DATA STATISTICS")
        logger.info("=" * 60)
        
        total_samples = len(self.raw_data)
        train_samples = len(self.train_data)
        val_samples = len(self.val_data)
        test_samples = len(self.test_data)
        
        # Data quantity
        logger.info("\n--- Data Split ---")
        logger.info(f"Total samples: {total_samples}")
        logger.info(f"Train samples: {train_samples} ({train_samples/total_samples:.1%})")
        logger.info(f"Val samples: {val_samples} ({val_samples/total_samples:.1%})")
        logger.info(f"Test samples: {test_samples} ({test_samples/total_samples:.1%})")
        
        # Anomaly statistics
        train_anomalies = self.train_data[self.target_column].sum()
        val_anomalies = self.val_data[self.target_column].sum()
        test_anomalies = self.test_data[self.target_column].sum()
        total_anomalies = self.raw_data[self.target_column].sum()
        
        train_normal = len(self.train_data) - train_anomalies
        val_normal = len(self.val_data) - val_anomalies
        test_normal = len(self.test_data) - test_anomalies
        

        
        # Feature statistics
        logger.info("\n--- Feature Information ---")
        logger.info(f"Number of features: {len(self.feature_columns)}")
        logger.info(f"Feature names (first 10): {self.feature_columns[:10]}")
    
    def get_train_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get training data.
        
        Returns:
            Tuple of (features, labels) as numpy arrays
        """
        features = self.train_data[self.feature_columns].values
        labels = self.train_data[self.target_column].values
        return features, labels
    
    def get_val_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get validation data.
        
        Returns:
            Tuple of (features, labels) as numpy arrays
        """
        features = self.val_data[self.feature_columns].values
        labels = self.val_data[self.target_column].values
        return features, labels
    
    def get_test_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get test data.
        
        Returns:
            Tuple of (features, labels) as numpy arrays
        """
        features = self.test_data[self.feature_columns].values
        labels = self.test_data[self.target_column].values
        return features, labels
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature names."""
        return self.feature_columns
    
    def get_class_weights(self) -> Optional[Dict]:
        """
        Get class weights for imbalanced data.
        
        Returns:
            Dictionary of class weights or None if not computed
        """
        return self.class_weights
    
    
    def get_data_info(self) -> Dict:
        """
        Get summary information about the data.
        
        Returns:
            Dictionary with data information
        """
        return {
            'n_samples': len(self.raw_data),
            'n_train': len(self.train_data),
            'n_val': len(self.val_data),
            'n_test': len(self.test_data),
            'n_features': len(self.feature_columns),
            'feature_names': self.feature_columns,
            'train_anomaly_ratio': self.train_data[self.target_column].mean(),
            'val_anomaly_ratio': self.val_data[self.target_column].mean(),
            'test_anomaly_ratio': self.test_data[self.target_column].mean(),
            'class_weights': self.class_weights
        }
        
    def save_scaler(self, path: str) -> None:
        """
        Save scaler object to disk.
        
        For Random Forest, we don't actually use a scaler since tree-based models
        are invariant to monotonic transformations of features. This method exists
        for API compatibility with other model data loaders.
        
        Args:
            path: Path to save the scaler
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save the dummy scaler
        with open(path, 'wb') as f:
            pickle.dump(IdentityScaler(), f)
            
        logger.info(f"Saved identity scaler to: {path}")


def create_random_forest_data(
    data_path: str,
    config: Dict
) -> Tuple[RandomForestDataLoader, Dict]:
    """
    Create Random Forest data loader and data dictionary.
    
    Args:
        data_path: Path to window features parquet file
        config: Configuration dictionary
        
    Returns:
        Tuple of (data_loader, data_dict) where data_dict contains:
            - X_train, y_train: Training data
            - X_val, y_val: Validation data
            - X_test, y_test: Test data
            - feature_names: List of feature names
            - class_weights: Optional class weights for imbalanced data
    """
    logger.info("\n" + "=" * 60)
    logger.info("CREATING RANDOM FOREST DATA LOADER")
    logger.info("=" * 60)
    
    # Extract configurations
    data_config = config.get('data', {})
    
    # Create data loader
    data_loader = RandomForestDataLoader(
        data_path=data_path,
        train_ratio=data_config.get('train_ratio', 0.7),
        val_ratio=data_config.get('val_ratio', 0.15),
        test_ratio=data_config.get('test_ratio', 0.15),
        target_column=data_config.get('target_column', 'anomaly_label'),
        exclude_columns=data_config.get('exclude_columns', []),
        balance_method=data_config.get('balance_method', None)
    )
    
    # Get data arrays
    X_train, y_train = data_loader.get_train_data()
    X_val, y_val = data_loader.get_val_data()
    X_test, y_test = data_loader.get_test_data()
    
    # Create data dictionary
    data_dict = {
        'X_train': X_train,
        'y_train': y_train,
        'X_val': X_val,
        'y_val': y_val,
        'X_test': X_test,
        'y_test': y_test,
        'feature_names': data_loader.get_feature_names(),
        'class_weights': data_loader.get_class_weights()
    }
    
    # Log data shapes
    logger.info("\n--- Data Arrays ---")
    logger.info(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    logger.info(f"X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")
    logger.info(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
    logger.info(f"Number of features: {len(data_dict['feature_names'])}")
    
    if data_dict['class_weights']:
        logger.info(f"Class weights: {data_dict['class_weights']}")
    
    logger.info("=" * 60 + "\n")
    
    return data_loader, data_dict

