"""
TranAD specific data loader for anomaly detection.

This module provides functionality to load energy data, split into train/validation/test sets,
apply normalization, and create sliding windows for TranAD model training.

Key differences from standard dataloader:
1. Training data contains only normal samples (anomaly_label=0)
2. Validation and test data contain both normal and anomalous samples
3. Focus on reconstruction rather than classification
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


class TranADDataProcessor:
    """
    Processes energy data for TranAD with train/validation/test split and normalization.
    
    Key feature: Training data contains ONLY normal samples (anomaly_label=0)
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
        
        # Load parquet file
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
        """
        Split data into train, validation, and test sets with a key difference:
        - Training data contains ONLY normal samples (anomaly_label=0)
        - Validation and test data contain both normal and anomalous samples
        """
        logger.info("Splitting data for TranAD (train: normal data only, val/test: all data)")
        
        # First, separate normal and anomalous data
        normal_data = self.raw_data[self.raw_data['anomaly_label'] == 0].reset_index(drop=True)
        anomalous_data = self.raw_data[self.raw_data['anomaly_label'] == 1].reset_index(drop=True)
        
        logger.info(f"Total data: {len(self.raw_data)} rows")
        logger.info(f"Normal data: {len(normal_data)} rows")
        logger.info(f"Anomalous data: {len(anomalous_data)} rows")
        
        # Calculate split sizes for normal data
        total_normal = len(normal_data)
        
        # Adjust ratios to account for anomalous data in val/test
        total_samples = len(self.raw_data)
        anomaly_ratio = len(anomalous_data) / total_samples
        
        # Recalculate normal data ratios
        if anomaly_ratio > 0:
            # Adjust train ratio to ensure we have enough normal data for val/test
            adjusted_train_ratio = max(0, self.train_ratio - (anomaly_ratio / 2))
            adjusted_val_ratio = self.val_ratio / (1 - adjusted_train_ratio)
            adjusted_test_ratio = 1 - adjusted_val_ratio
        else:
            adjusted_train_ratio = self.train_ratio
            adjusted_val_ratio = self.val_ratio / (1 - adjusted_train_ratio)
            adjusted_test_ratio = 1 - adjusted_val_ratio
        
        # Calculate split indices for normal data
        normal_train_end = int(total_normal * adjusted_train_ratio)
        normal_val_end = normal_train_end + int((total_normal - normal_train_end) * adjusted_val_ratio)
        
        # Split normal data
        normal_train = normal_data.iloc[:normal_train_end].reset_index(drop=True)
        normal_val = normal_data.iloc[normal_train_end:normal_val_end].reset_index(drop=True)
        normal_test = normal_data.iloc[normal_val_end:].reset_index(drop=True)
        
        # Split anomalous data between validation and test
        anomaly_val_end = int(len(anomalous_data) * adjusted_val_ratio / (adjusted_val_ratio + adjusted_test_ratio))
        anomaly_val = anomalous_data.iloc[:anomaly_val_end].reset_index(drop=True)
        anomaly_test = anomalous_data.iloc[anomaly_val_end:].reset_index(drop=True)
        
        # Combine normal and anomalous data for validation and test
        self.train_data = normal_train  # Train data is ONLY normal data
        self.val_data = pd.concat([normal_val, anomaly_val]).sort_values('TimeStamp').reset_index(drop=True)
        self.test_data = pd.concat([normal_test, anomaly_test]).sort_values('TimeStamp').reset_index(drop=True)
        
        logger.info(f"Train data (normal only): {len(self.train_data)} rows")
        logger.info(f"Validation data (normal + anomalous): {len(self.val_data)} rows")
        logger.info(f"Test data (normal + anomalous): {len(self.test_data)} rows")
        
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
        
        logger.info("=== TranAD Data Split Statistics ===")
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
        logger.info(f"Train anomalies: {train_anomalies} ({train_anomaly_ratio:.4f}) - Should be 0")
        logger.info(f"Validation anomalies: {val_anomalies} ({val_anomaly_ratio:.4f})")
        logger.info(f"Test anomalies: {test_anomalies} ({test_anomaly_ratio:.4f})")
        
        # Time range information
        logger.info("=== Time Range Information ===")
        logger.info(f"Train data range: {self.train_data['TimeStamp'].min()} to {self.train_data['TimeStamp'].max()}")
        logger.info(f"Validation data range: {self.val_data['TimeStamp'].min()} to {self.val_data['TimeStamp'].max()}")
        logger.info(f"Test data range: {self.test_data['TimeStamp'].min()} to {self.test_data['TimeStamp'].max()}")
        
    def _fit_scaler(self):
        """Fit StandardScaler on training data features (normal data only)."""
        logger.info("Fitting StandardScaler on normal training data")
        
        if len(self.train_data) == 0:
            raise ValueError("No training data available for fitting scaler")
        
        # Get training features
        train_features = self.train_data[self.feature_columns].values
        
        # Fit scaler on training data
        self.scaler = StandardScaler()
        self.scaler.fit(train_features)
        
        logger.info(f"Scaler fitted on {len(train_features)} normal training samples with {len(self.feature_columns)} features")
        logger.info(f"Feature means range: [{self.scaler.mean_.min():.3f}, {self.scaler.mean_.max():.3f}]")
        logger.info(f"Feature stds range: [{self.scaler.scale_.min():.3f}, {self.scaler.scale_.max():.3f}]")
        
    def save_scaler(self, save_path):
        """Save the fitted scaler to disk."""
        if self.scaler is None:
            raise ValueError("Scaler has not been fitted yet")
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        
        logger.info(f"Saved scaler to {save_path}")
        
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


class TranADDataset(Dataset):
    """
    Dataset for TranAD model with sliding window functionality and normalization.
    
    This dataset is designed for reconstruction-based anomaly detection,
    where the model learns to reconstruct normal patterns.
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


def create_tranad_data_loaders(
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
    Create data loaders for TranAD training, validation, and test.
    
    Key difference: Training data contains ONLY normal samples (anomaly_label=0)
    
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
    logger.info("=== Creating TranAD Data Loaders ===")
    
    # Initialize data processor
    processor = TranADDataProcessor(
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
    train_dataset = TranADDataset(
        data=train_data,
        normalized_features=train_features,
        window_size=window_size,
        step_size=step_size
    )
    
    val_dataset = TranADDataset(
        data=val_data,
        normalized_features=val_features,
        window_size=window_size,
        step_size=step_size
    )
    
    test_dataset = TranADDataset(
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
        pin_memory=True
    )
    
    data_loaders['val'] = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    data_loaders['test'] = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    logger.info(f"Created train data loader with {len(train_dataset)} samples")
    logger.info(f"Created validation data loader with {len(val_dataset)} samples")
    logger.info(f"Created test data loader with {len(test_dataset)} samples")
    
    return data_loaders


if __name__ == "__main__":
    # Test the data loader
    import argparse
    
    parser = argparse.ArgumentParser(description="Test TranAD data loader")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to parquet data file")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--window_size", type=int, default=60, help="Window size")
    parser.add_argument("--step_size", type=int, default=1, help="Step size")
    
    args = parser.parse_args()
    
    # Create data loaders
    data_loaders = create_tranad_data_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        window_size=args.window_size,
        step_size=args.step_size
    )
    
    # Test data loaders
    for name, loader in data_loaders.items():
        batch = next(iter(loader))
        features, labels = batch
        
        print(f"{name} loader:")
        print(f"  Features shape: {features.shape}")
        print(f"  Labels shape: {labels.shape}")
        print(f"  Batch size: {features.size(0)}")
        print(f"  Window size: {features.size(1)}")
        print(f"  Feature dim: {features.size(2)}")
        print(f"  Anomaly ratio: {labels.float().mean().item():.4f}")
        print()
