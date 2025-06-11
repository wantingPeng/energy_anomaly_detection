import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from datetime import datetime
from pathlib import Path
import glob
from typing import Dict, List, Tuple, Optional, Union
from src.utils.logger import logger

class LSTMLateFusionDataset(Dataset):
    """
    Dataset for LSTM model with Late Fusion that combines time series data
    with statistical features.
    
    This dataset loads both the LSTM sliding window data and the corresponding
    statistical features, which are already aligned by window timestamps.
    """
    
    def __init__(
        self,
        lstm_data_dir: str,
        stat_features_dir: str,
        data_type: str = 'train',
        component: str = 'contact',
        transform=None
    ):
        """
        Initialize the dataset.
        
        Args:
            lstm_data_dir: Directory containing LSTM sliding window data
            stat_features_dir: Directory containing statistical features
            data_type: Data type ('train', 'val', or 'test')
            component: Component type ('contact', 'pcb', or 'ring')
            transform: Optional transform to apply to the data
        """
        self.lstm_data_dir = lstm_data_dir
        self.stat_features_dir = stat_features_dir
        self.data_type = data_type
        self.component = component
        self.transform = transform
        
        # Paths for the specific data type and component
        self.lstm_component_dir = os.path.join(lstm_data_dir, data_type, component)
        self.stat_component_dir = os.path.join(stat_features_dir, data_type, component)
        
        # Load data
        self.windows, self.labels, self.stat_features = self._load_data()
        
        logger.info(f"Loaded {len(self.windows)} samples for {data_type}/{component}")
    
    def __len__(self):
        return len(self.windows)
    
    def __getitem__(self, idx):
        """
        Get a sample from the dataset.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Tuple of (window, statistical_features, label)
        """
        window = self.windows[idx]
        stat_feature = self.stat_features[idx]
        label = self.labels[idx]
        
        # Convert to torch tensors
        window = torch.FloatTensor(window)
        stat_feature = torch.FloatTensor(stat_feature)
        label = torch.LongTensor([label])[0]
        
        if self.transform:
            window = self.transform(window)
            stat_feature = self.transform(stat_feature)
        
        return window, stat_feature, label
    
    def _load_data(self):
        """
        Load LSTM data and statistical features.
        Since both datasets are created with the same window size and step size,
        they are already aligned by index.
        
        Returns:
            Tuple of (windows, labels, stat_features)
        """
        # Load LSTM data
        lstm_files = sorted(glob.glob(os.path.join(self.lstm_component_dir, "*.npz")))
        if not lstm_files:
            raise ValueError(f"No LSTM data files found in {self.lstm_component_dir}")
        
        # Load statistical features
        stat_files = sorted(glob.glob(os.path.join(self.stat_component_dir, "*.npz")))
        if not stat_files:
            raise ValueError(f"No statistical feature files found in {self.stat_component_dir}")
        
        # Ensure the number of batch files match
        if len(lstm_files) != len(stat_files):
            logger.warning(f"Number of LSTM files ({len(lstm_files)}) does not match "
                          f"number of statistical feature files ({len(stat_files)})")
        
        # Load all LSTM data
        all_windows = []
        all_labels = []
        
        for lstm_file in lstm_files:
            data = np.load(lstm_file)
            all_windows.append(data['windows'])
            all_labels.append(data['labels'])
        
        # Concatenate all LSTM data
        windows = np.concatenate(all_windows, axis=0)
        labels = np.concatenate(all_labels, axis=0)
        
        # Load all statistical features
        all_stat_features = []
        
        for stat_file in stat_files:
            data = np.load(stat_file)
            all_stat_features.append(data['stat_features'])
        
        # Concatenate all statistical features
        stat_features = np.concatenate(all_stat_features, axis=0)
        
          
        # Verify that the number of samples match
        if len(windows) != len(stat_features):
            logger.warning(f"Number of LSTM windows ({len(windows)}) does not match "
                          f"number of statistical features ({len(stat_features)})")
            # Use the minimum length to avoid index errors
            min_len = min(len(windows), len(stat_features))
            windows = windows[:min_len]
            labels = labels[:min_len]
            stat_features = stat_features[:min_len]
        
        logger.info(f"Loaded {len(windows)} LSTM windows and {len(stat_features)} statistical features")
        
        return windows, labels, stat_features


def create_data_loaders(
    lstm_data_dir: str,
    stat_features_dir: str,
    batch_size: int = 64,
    num_workers: int = 4,
    component: str = 'contact'
) -> Dict[str, DataLoader]:
    """
    Create data loaders for training, validation, and testing.
    
    Args:
        lstm_data_dir: Directory containing LSTM sliding window data
        stat_features_dir: Directory containing statistical features
        batch_size: Batch size for data loaders
        num_workers: Number of worker processes for data loading
        component: Component type ('contact', 'pcb', or 'ring')
        
    Returns:
        Dictionary of data loaders for 'train', 'val', and 'test'
    """
    data_loaders = {}
    
    for data_type in ['train_down_25%', 'val']:
        dataset = LSTMLateFusionDataset(
            lstm_data_dir=lstm_data_dir,
            stat_features_dir=stat_features_dir,
            data_type=data_type,
            component=component
        )
        
        shuffle = (data_type == 'train')
        
        data_loaders[data_type] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True
        )
        
        logger.info(f"Created {data_type} data loader with {len(dataset)} samples")
    
    return data_loaders

