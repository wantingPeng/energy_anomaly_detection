"""
Data loader with train/validation/test split, normalization, and sliding window functionality.

This module provides functionality to load energy data, split into train/validation/test sets
based on configurable ratios (default: 70% train, 15% val, 15% test), apply normalization, 
and create sliding windows for transformer model training.
"""

import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple, Optional, Union
import pickle
from datetime import datetime
import glob
from utils.logger import logger


class EnergyDataProcessor:
    """
    Processes energy data with train/validation/test split, normalization, and sliding windows.
    """
    
    def __init__(
        self,
        data_dir: str,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        window_size: int = 60,
        step_size: int = 1,
        exclude_columns: List[str] = None
    ):
        """
        Initialize the data processor.
        
        Args:
            data_dir: Directory containing parquet data files
            train_ratio: Ratio of data for training (default: 0.7)
            val_ratio: Ratio of data for validation (default: 0.15)
            test_ratio: Ratio of data for testing (default: 0.15)
            window_size: Size of sliding window (number of timesteps)
            step_size: Step size for sliding window (stride)
            exclude_columns: List of column names to exclude from features
        """
        self.data_dir = data_dir
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.window_size = window_size
        self.step_size = step_size
        
        # Validate ratios
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError(f"Train, validation, and test ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}")
        
        # Default columns to exclude
        if exclude_columns is None:
            exclude_columns = ["TimeStamp", "anomaly_label"]
        self.exclude_columns = exclude_columns
        
        # Initialize data attributes
        self.raw_data = None
        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.scaler = None
        self.feature_columns = None
        
        
        # Load and process data
        self._load_data()
        self._split_data()
        self._log_data_statistics()
        self._fit_scaler()
        
    def _load_data(self):
        """Load raw data from a parquet file or all parquet files in a directory."""
        target = self.data_dir
        logger.info(f"Loading data from: {target}")
        
        if not os.path.exists(target):
            raise FileNotFoundError(f"Path not found: {target}")
        
        # If a single parquet file path is provided, load it directl
        self.raw_data = pd.read_parquet(target)
        logger.info(f"Loaded single parquet file: {target}")
        
        # Ensure TimeStamp is datetime and sort to preserve temporal continuity
        if 'TimeStamp' in self.raw_data.columns:
            self.raw_data['TimeStamp'] = pd.to_datetime(self.raw_data['TimeStamp'])
            self.raw_data = self.raw_data.sort_values('TimeStamp').reset_index(drop=True)
        
        # Get feature columns
        self.feature_columns = [col for col in self.raw_data.columns if col not in self.exclude_columns]
        
        logger.info(f"Loaded {len(self.raw_data)} rows with {len(self.raw_data.columns)} columns")
        logger.info(f"Feature columns ({len(self.feature_columns)}): {self.feature_columns[:5]}{'...' if len(self.feature_columns) > 5 else ''}")
        
    def _split_data(self):
        """Split data into train, validation, and test sets sequentially to preserve time order."""
        logger.info(f"Splitting data sequentially - Train: {self.train_ratio}, Val: {self.val_ratio}, Test: {self.test_ratio}")
        
        total_samples = len(self.raw_data)
        
        # Calculate split indices (sequential)
        train_end = int(total_samples * self.train_ratio)
        val_end = int(total_samples * (self.train_ratio + self.val_ratio))
        
        # Create splits by order to keep temporal continuity
        self.train_data = self.raw_data.iloc[:train_end].reset_index(drop=True)
        self.val_data = self.raw_data.iloc[train_end:val_end].reset_index(drop=True)
        self.test_data = self.raw_data.iloc[val_end:].reset_index(drop=True)
        
        logger.info(f"Train data: {len(self.train_data)} rows")
        logger.info(f"Validation data: {len(self.val_data)} rows")
        logger.info(f"Test data: {len(self.test_data)} rows")
        
    def _log_data_statistics(self):
        """Log statistics about the data split."""
        total_samples = len(self.raw_data)
        train_samples = len(self.train_data)
        val_samples = len(self.val_data)
        test_samples = len(self.test_data)
        
        # Data quantity ratio
        train_ratio = train_samples / total_samples
        val_ratio = val_samples / total_samples
        test_ratio = test_samples / total_samples
        
        logger.info("=== Data Split Statistics ===")
        logger.info(f"Total samples: {total_samples}")
        logger.info(f"Train samples: {train_samples} ({train_ratio:.3f})")
        logger.info(f"Validation samples: {val_samples} ({val_ratio:.3f})")
        logger.info(f"Test samples: {test_samples} ({test_ratio:.3f})")
        
        # Anomaly ratio in each set
        train_anomalies = self.train_data['anomaly_label'].sum()
        val_anomalies = self.val_data['anomaly_label'].sum()
        test_anomalies = self.test_data['anomaly_label'].sum()
        total_anomalies = self.raw_data['anomaly_label'].sum()
        
        train_anomaly_ratio = train_anomalies / train_samples if train_samples > 0 else 0
        val_anomaly_ratio = val_anomalies / val_samples if val_samples > 0 else 0
        test_anomaly_ratio = test_anomalies / test_samples if test_samples > 0 else 0
        total_anomaly_ratio = total_anomalies / total_samples
        
        logger.info("=== Anomaly Statistics ===")
        logger.info(f"Total anomalies: {total_anomalies} ({total_anomaly_ratio:.4f})")
        logger.info(f"Train anomalies: {train_anomalies} ({train_anomaly_ratio:.4f})")
        logger.info(f"Validation anomalies: {val_anomalies} ({val_anomaly_ratio:.4f})")
        logger.info(f"Test anomalies: {test_anomalies} ({test_anomaly_ratio:.4f})")
        
        # Time range information
        logger.info("=== Time Range Information ===")
        logger.info(f"Train data range: {self.train_data['TimeStamp'].min()} to {self.train_data['TimeStamp'].max()}")
        logger.info(f"Validation data range: {self.val_data['TimeStamp'].min()} to {self.val_data['TimeStamp'].max()}")
        logger.info(f"Test data range: {self.test_data['TimeStamp'].min()} to {self.test_data['TimeStamp'].max()}")
        
    def _fit_scaler(self):
        """Fit StandardScaler on training data features."""
        logger.info("Fitting StandardScaler on training data")
        
        if len(self.train_data) == 0:
            raise ValueError("No training data available for fitting scaler")
        
        # Get training features
        train_features = self.train_data[self.feature_columns].values
        
        # Fit scaler on training data
        self.scaler = StandardScaler()
        self.scaler.fit(train_features)
        
        logger.info(f"Scaler fitted on {len(train_features)} training samples with {len(self.feature_columns)} features")
        logger.info(f"Feature means range: [{self.scaler.mean_.min():.3f}, {self.scaler.mean_.max():.3f}]")
        logger.info(f"Feature stds range: [{self.scaler.scale_.min():.3f}, {self.scaler.scale_.max():.3f}]")
        
    def _normalize_data(self, data: pd.DataFrame) -> np.ndarray:
        """
        Normalize feature data using the fitted scaler.
        
        Args:
            data: DataFrame to normalize
            
        Returns:
            Normalized feature array
        """
        features = data[self.feature_columns].values
        return self.scaler.transform(features)
        
    def get_train_data(self) -> Tuple[pd.DataFrame, np.ndarray]:
        """Get training data with normalized features."""
        return self.train_data, self._normalize_data(self.train_data)
        
    def get_val_data(self) -> Tuple[pd.DataFrame, np.ndarray]:
        """Get validation data with normalized features."""
        return self.val_data, self._normalize_data(self.val_data)
        
    def get_test_data(self) -> Tuple[pd.DataFrame, np.ndarray]:
        """Get test data with normalized features."""
        return self.test_data, self._normalize_data(self.test_data)
        


class SlidingWindowDataset(Dataset):
    """
    Dataset for transformer model with sliding window functionality and normalization.
    """
    
    def __init__(
        self,
        data: pd.DataFrame,
        normalized_features: np.ndarray,
        window_size: int = 60,
        step_size: int = 1,
        transform=None
    ):
        """
        Initialize the dataset.
        
        Args:
            data: Original DataFrame with labels
            normalized_features: Normalized feature array
            window_size: Size of sliding window (number of timesteps)
            step_size: Step size for sliding window (stride)
            transform: Optional transform to apply to the data
        """
        self.data = data
        self.normalized_features = normalized_features
        self.window_size = window_size
        self.step_size = step_size
        self.transform = transform
        
        # Create sliding windows
        self.windows, self.labels = self._create_sliding_windows()
        
        logger.info(f"Created {len(self.windows)} sliding windows")
        logger.info(f"Window shape: {self.windows.shape if len(self.windows) > 0 else 'Empty'}")
        logger.info(f"Labels shape: {self.labels.shape if len(self.labels) > 0 else 'Empty'}")
        
    def __len__(self):
        return len(self.windows)
    
    def __getitem__(self, idx):
        """
        Get a sample from the dataset.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Tuple of (window, label_sequence)
        """
        window = self.windows[idx]
        label_sequence = self.labels[idx]
        
        # Convert to torch tensors if they aren't already
        if not isinstance(window, torch.Tensor):
            window = torch.FloatTensor(window)
        
        if not isinstance(label_sequence, torch.Tensor):
            label_sequence = torch.LongTensor(label_sequence)
        
        if self.transform:
            window = self.transform(window)
        
        return window, label_sequence
    
    def _create_sliding_windows(self):
        """
        Create sliding windows from the normalized features and labels.
        
        Returns:
            Tuple of (windows, labels) as numpy arrays
        """
        if len(self.data) < self.window_size:
            logger.warning(f"Data length {len(self.data)} is smaller than window size {self.window_size}")
            return np.array([]), np.array([])
        
        # Get labels
        labels = self.data['anomaly_label'].values
        
        logger.info(f"Creating sliding windows from {len(self.data)} samples")
        logger.info(f"Features shape: {self.normalized_features.shape}")
        logger.info(f"Labels shape: {labels.shape}")
        
        # Create sliding windows
        windows = []
        window_labels = []
        
        # Calculate number of windows
        num_windows = (len(self.data) - self.window_size) // self.step_size + 1
        logger.info(f"Creating {num_windows} windows with size {self.window_size} and step {self.step_size}")
        
        for i in range(0, len(self.data) - self.window_size + 1, self.step_size):
            # Extract window of normalized features
            window = self.normalized_features[i:i + self.window_size]
            
            # Extract corresponding labels for the window
            window_label = labels[i:i + self.window_size]
            
            windows.append(window)
            window_labels.append(window_label)
        
        # Convert to numpy arrays
        windows = np.array(windows, dtype=np.float32)
        window_labels = np.array(window_labels, dtype=np.int64)
        
        logger.info(f"Created windows shape: {windows.shape}")
        logger.info(f"Created labels shape: {window_labels.shape}")
        
        # Log some statistics
        total_anomalies = np.sum(window_labels)
        total_timesteps = window_labels.size
        anomaly_ratio = total_anomalies / total_timesteps if total_timesteps > 0 else 0
        logger.info(f"Window anomaly ratio: {anomaly_ratio:.4f} ({total_anomalies}/{total_timesteps})")
        
        return windows, window_labels


def create_data_loaders(
    data_dir: str,
    batch_size: int = 64,
    num_workers: int = 4,
    window_size: int = 60,
    step_size: int = 1,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    exclude_columns: List[str] = None,
    scaler_save_path: str = None
) -> Dict[str, DataLoader]:
    """
    Create data loaders for training, validation, and test with normalization.
    
    Args:
        data_dir: Directory containing parquet data files
        batch_size: Batch size for data loaders
        num_workers: Number of worker processes for data loading
        window_size: Size of sliding window (number of timesteps)
        step_size: Step size for sliding window (stride)
        train_ratio: Ratio of data for training (default: 0.7)
        val_ratio: Ratio of data for validation (default: 0.15)
        test_ratio: Ratio of data for testing (default: 0.15)
        exclude_columns: List of column names to exclude from features
        scaler_save_path: Path to save the fitted scaler
        
    Returns:
        Dictionary of data loaders for 'train', 'val', and 'test'
    """
    logger.info("=== Creating Data Loaders ===")
    
    # Initialize data processor
    processor = EnergyDataProcessor(
        data_dir=data_dir,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        window_size=window_size,
        step_size=step_size,
        exclude_columns=exclude_columns
    )
    
    # Save scaler if path provided
    if scaler_save_path:
        processor.save_scaler(scaler_save_path)
    
    # Get normalized data
    train_data, train_features = processor.get_train_data()
    val_data, val_features = processor.get_val_data()
    test_data, test_features = processor.get_test_data()
    
    # Create datasets
    train_dataset = SlidingWindowDataset(
        data=train_data,
        normalized_features=train_features,
        window_size=window_size,
        step_size=step_size
    )
    
    val_dataset = SlidingWindowDataset(
        data=val_data,
        normalized_features=val_features,
        window_size=window_size,
        step_size=step_size
    )
    test_dataset = SlidingWindowDataset(
        data=test_data,
        normalized_features=test_features,
        window_size=window_size,
        step_size=step_size
    )
    # Create data loaders
    data_loaders = {}
    
    data_loaders['train'] = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False
    )
    
    data_loaders['val'] = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False
    )

    data_loaders['test'] = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False
    )
    
    logger.info(f"Created test data loader with {len(test_dataset)} samples")
    logger.info(f"Created train data loader with {len(train_dataset)} samples")
    logger.info(f"Created validation data loader with {len(val_dataset)} samples")
    
    # Create test data loader
   
    return data_loaders

