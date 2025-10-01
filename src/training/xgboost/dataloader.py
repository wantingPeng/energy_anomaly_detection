"""
Simplified data loader for XGBoost time series anomaly detection.

This module loads pre-computed window features and prepares data for XGBoost training.
XGBoost (tree-based model) does not require feature normalization.
"""

import os
import sys
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from src.utils.logger import logger


class XGBoostDataLoader:
    """
    Simplified data loader for XGBoost time series anomaly detection.
    
    Features:
    - Loads pre-computed window features
    - No feature engineering (already done)
    - No normalization (XGBoost doesn't need it)
    - Sequential train/val/test split for time series
    """
    
    def __init__(
        self,
        data_path: str,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        target_column: str = "anomaly_label",
        exclude_columns: List[str] = None
    ):
        """
        Initialize XGBoost data loader.
        
        Args:
            data_path: Path to parquet data file with window features
            train_ratio: Ratio of data for training
            val_ratio: Ratio of data for validation
            test_ratio: Ratio of data for testing
            target_column: Name of the target column
            exclude_columns: Additional columns to exclude from features
        """
        self.data_path = data_path
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.target_column = target_column
        
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
        
        # Load and process data
        self._load_data()
        self._split_data()
        self._log_statistics()
        
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
        logger.info(f"Overall anomaly ratio: {total_anomalies}/{total_samples} = {total_anomalies/total_samples:.4f}")
        
    def _split_data(self):
        """
        Split data into train, validation, and test sets sequentially.
        Sequential split preserves temporal order for time series data.
        """
        logger.info(f"Splitting data sequentially - Train: {self.train_ratio}, "
                   f"Val: {self.val_ratio}, Test: {self.test_ratio}")
        
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
        
        logger.info("\n--- Anomaly Distribution ---")
        logger.info(f"Total anomalies: {total_anomalies} ({total_anomalies/total_samples:.4f})")
        logger.info(f"Train - Normal: {train_normal}, Anomaly: {train_anomalies} "
                   f"(ratio: {train_anomalies/train_samples:.4f})")
        logger.info(f"Val   - Normal: {val_normal}, Anomaly: {val_anomalies} "
                   f"(ratio: {val_anomalies/val_samples:.4f})")
        logger.info(f"Test  - Normal: {test_normal}, Anomaly: {test_anomalies} "
                   f"(ratio: {test_anomalies/test_samples:.4f})")
        
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
    
    def get_scale_pos_weight(self) -> float:
        """
        Calculate scale_pos_weight for XGBoost to handle class imbalance.
        
        Returns:
            Ratio of negative to positive samples in training data
        """
        train_labels = self.train_data[self.target_column].values
        n_positive = train_labels.sum()
        n_negative = len(train_labels) - n_positive
        
        if n_positive == 0:
            logger.warning("No positive samples in training data!")
            return 1.0
        
        scale_pos_weight = n_negative / n_positive
        logger.info(f"\nCalculated scale_pos_weight: {scale_pos_weight:.2f}")
        logger.info(f"  Negative samples: {n_negative}")
        logger.info(f"  Positive samples: {n_positive}")
        
        return scale_pos_weight
    
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
            'scale_pos_weight': self.get_scale_pos_weight()
        }


def create_xgboost_data(
    data_path: str,
    config: Dict
) -> Tuple[XGBoostDataLoader, Dict]:
    """
    Create XGBoost data loader and data dictionary.
    
    Args:
        data_path: Path to window features parquet file
        config: Configuration dictionary
        
    Returns:
        Tuple of (data_loader, data_dict) where data_dict contains:
            - X_train, y_train: Training data
            - X_val, y_val: Validation data
            - X_test, y_test: Test data
            - feature_names: List of feature names
            - scale_pos_weight: Weight for positive class
    """
    logger.info("\n" + "=" * 60)
    logger.info("CREATING XGBOOST DATA LOADER")
    logger.info("=" * 60)
    
    # Extract configurations
    data_config = config.get('data', {})
    
    # Create data loader
    data_loader = XGBoostDataLoader(
        data_path=data_path,
        train_ratio=data_config.get('train_ratio', 0.7),
        val_ratio=data_config.get('val_ratio', 0.15),
        test_ratio=data_config.get('test_ratio', 0.15),
        target_column=data_config.get('target_column', 'anomaly_label'),
        exclude_columns=data_config.get('exclude_columns', [])
    )
    
    # Get data arrays (no normalization for XGBoost)
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
        'scale_pos_weight': data_loader.get_scale_pos_weight()
    }
    
    # Log data shapes
    logger.info("\n--- Data Arrays ---")
    logger.info(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    logger.info(f"X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")
    logger.info(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
    logger.info(f"Number of features: {len(data_dict['feature_names'])}")
    logger.info(f"Scale pos weight: {data_dict['scale_pos_weight']:.2f}")
    
    logger.info("=" * 60 + "\n")
    
    return data_loader, data_dict


# Keep backward compatibility with old function name
def create_data_loaders(*args, **kwargs):
    """
    Backward compatibility wrapper.
    Creates XGBoost data using create_xgboost_data.
    """
    logger.warning("create_data_loaders is deprecated, use create_xgboost_data instead")
    return create_xgboost_data(*args, **kwargs)
