"""
Data loader with sliding window functionality for transformer model training.

This module provides a flexible dataset class that loads raw energy data from parquet files
and creates sliding windows on-the-fly, supporting configurable window size and step size.
"""

import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import glob
from typing import Dict, List, Tuple, Optional, Union
from src.utils.logger import logger


class SlidingWindowDataset(Dataset):
    """
    Dataset for Transformer model with sliding window functionality.
    
    This dataset loads raw energy data from parquet files and creates sliding windows
    on-the-fly to feed into a transformer model. Data is formatted as 
    [batch_size, seq_len, features] to support batch_first=True in transformer models.
    Labels are sequences [batch_size, seq_len], with one label per timestep.
    """
    
    def __init__(
        self,
        data_dir: str,
        data_type: str = 'train',
        component: str = 'contact',
        window_size: int = 60,
        step_size: int = 1,
        exclude_columns: List[str] = None,
        transform=None
    ):
        """
        Initialize the dataset.
        
        Args:
            data_dir: Directory containing parquet data files
            data_type: Data type ('train', 'val', or 'test')
            component: Component type ('contact', 'pcb', or 'ring')
            window_size: Size of sliding window (number of timesteps)
            step_size: Step size for sliding window (stride)
            exclude_columns: List of column names to exclude from features
            transform: Optional transform to apply to the data
        """
        self.data_dir = data_dir
        self.data_type = data_type
        self.component = component
        self.window_size = window_size
        self.step_size = step_size
        self.transform = transform
        
        # Default columns to exclude
        if exclude_columns is None:
            exclude_columns = ["TimeStamp", "segment_id", "anomaly_label"]
        self.exclude_columns = exclude_columns
        
        # Path for the specific data type and component
        self.component_dir = os.path.join(data_dir, self.data_type, component)
        
        # Load and process data
        self.raw_data = self._load_raw_data()
        self.windows, self.labels = self._create_sliding_windows()
        
        logger.info(f"Created {len(self.windows)} sliding windows for {data_type}/{component}")
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
    
    def _load_raw_data(self):
        """
        Load raw data from parquet files.
        
        Returns:
            Concatenated DataFrame with all data
        """
        # Find all parquet files in the component directory
        parquet_files = sorted(glob.glob(os.path.join(self.component_dir, "*.parquet")))
        if not parquet_files:
            raise ValueError(f"No parquet files found in {self.component_dir}")
        
        # Load and concatenate data from all files
        all_data = []
        
        for parquet_file in parquet_files:
            logger.info(f"Loading {parquet_file}")
            df = pd.read_parquet(parquet_file)
            all_data.append(df)
        
        # Concatenate all data
        combined_data = pd.concat(all_data, ignore_index=True)
        
        # Sort by timestamp to ensure proper temporal order
        if 'TimeStamp' in combined_data.columns:
            combined_data = combined_data.sort_values('TimeStamp').reset_index(drop=True)
        
        logger.info(f"Loaded {len(combined_data)} rows with {len(combined_data.columns)} columns")
        logger.info(f"Columns: {list(combined_data.columns)}")
        
        return combined_data
    
    def _create_sliding_windows(self):
        """
        Create sliding windows from the raw data.
        
        Returns:
            Tuple of (windows, labels) as numpy arrays
        """
        if len(self.raw_data) < self.window_size:
            logger.warning(f"Data length {len(self.raw_data)} is smaller than window size {self.window_size}")
            return np.array([]), np.array([])
        
        # Separate features and labels
        feature_columns = [col for col in self.raw_data.columns if col not in self.exclude_columns]
        features = self.raw_data[feature_columns].values
        labels = self.raw_data['anomaly_label'].values
        
        logger.info(f"Feature columns ({len(feature_columns)}): {feature_columns}")
        logger.info(f"Features shape: {features.shape}")
        logger.info(f"Labels shape: {labels.shape}")
        
        # Create sliding windows
        windows = []
        window_labels = []
        
        # Calculate number of windows
        num_windows = (len(self.raw_data) - self.window_size) // self.step_size + 1
        logger.info(f"Creating {num_windows} windows with size {self.window_size} and step {self.step_size}")
        
        for i in range(0, len(self.raw_data) - self.window_size + 1, self.step_size):
            # Extract window of features
            window = features[i:i + self.window_size]
            
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
        logger.info(f"Anomaly ratio: {anomaly_ratio:.4f} ({total_anomalies}/{total_timesteps})")
        
        return windows, window_labels


def create_data_loaders(
    data_dir: str,
    batch_size: int = 64,
    num_workers: int = 4,
    component: str = 'contact',
    window_size: int = 60,
    step_size: int = 1,
    exclude_columns: List[str] = None
) -> Dict[str, DataLoader]:
    """
    Create data loaders for training, validation, and testing.
    
    Args:
        data_dir: Directory containing parquet data files
        batch_size: Batch size for data loaders
        num_workers: Number of worker processes for data loading
        component: Component type ('contact', 'pcb', or 'ring')
        window_size: Size of sliding window (number of timesteps)
        step_size: Step size for sliding window (stride)
        exclude_columns: List of column names to exclude from features
        
    Returns:
        Dictionary of data loaders for 'train', 'val', and 'test'
    """
    data_loaders = {}
    
    for data_type in ['train', 'val', 'test']:
        dataset = SlidingWindowDataset(
            data_dir=data_dir,
            data_type=data_type,
            component=component,
            window_size=window_size,
            step_size=step_size,
            exclude_columns=exclude_columns
        )
        
        shuffle = (data_type == 'train')
         
        data_loaders[data_type] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=False
        )
        
        logger.info(f"Created {data_type} data loader with {len(dataset)} samples")
    
    return data_loaders
