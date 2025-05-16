"""
PyTorch Dataset implementations for LSTM sliding window data.

This module provides dataset classes for loading and accessing
sliding window data for LSTM model training.
"""

import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Union, Optional

from src.utils.logger import logger


class LSTMWindowDataset(Dataset):
    """PyTorch Dataset for LSTM sliding windows."""
    
    def __init__(self, parquet_path: str):
        """
        Initialize the dataset.
        
        Args:
            parquet_path: Path to parquet file containing window data
        """
        logger.info(f"Loading dataset from {parquet_path}")
        self.data = pd.read_parquet(parquet_path)
        logger.info(f"Loaded {len(self.data)} samples")
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get item at index.
        
        Args:
            idx: Index
            
        Returns:
            Tuple containing:
            - Window features as torch.Tensor (shape: [window_size, num_features])
            - Label as torch.Tensor (shape: [1])
        """
        row = self.data.iloc[idx]
        
        # Deserialize window
        window = np.frombuffer(row['window'], dtype=np.float32).reshape(eval(row['window_shape']))
        window_tensor = torch.tensor(window, dtype=torch.float32)
        
        label = torch.tensor(row['label'], dtype=torch.float32)
        
        return window_tensor, label


def create_dataloader(
    dataset_path: str,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4
) -> DataLoader:
    """
    Create a DataLoader for the LSTM window dataset.
    
    Args:
        dataset_path: Path to parquet file containing window data
        batch_size: Batch size
        shuffle: Whether to shuffle the dataset
        num_workers: Number of worker processes for data loading
        
    Returns:
        PyTorch DataLoader
    """
    dataset = LSTMWindowDataset(dataset_path)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
    
    logger.info(f"Created DataLoader with {len(dataset)} samples, batch_size={batch_size}")
    
    return dataloader
