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

class LSTMSequenceDataset(Dataset):
    """
    Dataset for LSTM model that processes time series data with sequence labels.
    
    This dataset loads the LSTM sliding window data with corresponding sequence labels
    (one label per time point).
    """
    
    def __init__(
        self,
        lstm_data_dir: str,
        data_type: str = 'train',
        component: str = 'contact',
        transform=None
    ):
        """
        Initialize the dataset.
        
        Args:
            lstm_data_dir: Directory containing LSTM sliding window data
            data_type: Data type ('train', 'val', or 'test')
            component: Component type ('contact', 'pcb', or 'ring')
            transform: Optional transform to apply to the data
        """
        self.lstm_data_dir = lstm_data_dir
        self.data_type = data_type
        self.component = component
        self.transform = transform
        
        # Path for the specific data type and component
        self.lstm_component_dir = os.path.join(lstm_data_dir, data_type, component)
        
        # Load data
        self.windows, self.labels = self._load_data()
        
        logger.info(f"Loaded {len(self.windows)} samples for {data_type}/{component}")
        logger.info(f"Window shape: {self.windows.shape}, Labels shape: {self.labels.shape}")
    
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
        label_sequence = self.labels[idx]  # Sequence of labels, shape (seq_len,)
        
        # Convert to torch tensors
        window = torch.FloatTensor(window)
        label_sequence = torch.LongTensor(label_sequence)  # Convert sequence of labels to tensor
        
        if self.transform:
            window = self.transform(window)
        
        return window, label_sequence
    
    def _load_data(self):
        """
        Load LSTM data.
        
        Returns:
            Tuple of (windows, labels)
        """
        # Load LSTM data
        lstm_files = sorted(glob.glob(os.path.join(self.lstm_component_dir, "*.npz")))
        if not lstm_files:
            raise ValueError(f"No LSTM data files found in {self.lstm_component_dir}")
        
        logger.info(f"Number of LSTM files: {len(lstm_files)}")
        
        # Load all LSTM data
        all_windows = []
        all_labels = []
        
        for lstm_file in lstm_files:
            data = np.load(lstm_file)
            all_windows.append(data['windows'])
            all_labels.append(data['labels'])  # Shape (n_windows, seq_len)
        
        # Concatenate all LSTM data
        windows = np.concatenate(all_windows, axis=0)
        labels = np.concatenate(all_labels, axis=0)
        
        logger.info(f"Windows shape: {windows.shape}, Labels shape: {labels.shape}")
        logger.info(f"Loaded {len(windows)} LSTM windows")
        
        return windows, labels


def create_data_loaders(
    lstm_data_dir: str,
    batch_size: int = 64,
    num_workers: int = 4,
    component: str = 'contact'
) -> Dict[str, DataLoader]:
    """
    Create data loaders for training, validation, and testing.
    
    Args:
        lstm_data_dir: Directory containing LSTM sliding window data
        batch_size: Batch size for data loaders
        num_workers: Number of worker processes for data loading
        component: Component type ('contact', 'pcb', or 'ring')
        
    Returns:
        Dictionary of data loaders for 'train', 'val', and 'test'
    """
    data_loaders = {}
    
    for data_type in ['train', 'val']:
        dataset = LSTMSequenceDataset(
            lstm_data_dir=lstm_data_dir,
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

