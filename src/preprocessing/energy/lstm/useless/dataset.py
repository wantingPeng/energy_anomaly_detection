"""
PyTorch Dataset classes for LSTM model training.

This module provides PyTorch Dataset implementations for handling 
sliding window data for LSTM models in energy anomaly detection.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Optional, Union


class LSTMWindowDataset(Dataset):
    """
    PyTorch Dataset for LSTM sliding window data.
    
    This dataset handles windows of time series data prepared for LSTM models,
    with each window having shape (window_size, num_features) and a corresponding label.
    
    Attributes:
        X (np.ndarray): Array of sliding windows with shape (n_samples, window_size, n_features)
        y (np.ndarray): Array of labels with shape (n_samples,)
        transform (callable, optional): Optional transform to be applied to each sample
    """
    
    def __init__(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        transform: Optional[callable] = None
    ):
        """
        Initialize the LSTM Window Dataset.
        
        Args:
            X: Array of sliding windows with shape (n_samples, window_size, n_features)
            y: Array of labels with shape (n_samples,)
            transform: Optional transform to be applied to each sample
        """
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        self.transform = transform
        
        # Validate shapes
        assert len(self.X) == len(self.y), f"X and y must have same length, got {len(self.X)} and {len(self.y)}"
        assert len(self.X.shape) == 3, f"X must have shape (n_samples, window_size, n_features), got {self.X.shape}"
        assert len(self.y.shape) == 1, f"y must have shape (n_samples,), got {self.y.shape}"
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.X)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Index of the sample to retrieve
            
        Returns:
            Tuple containing the window data and its label
        """
        window = self.X[idx]
        label = self.y[idx]
        
        if self.transform:
            window = self.transform(window)
        
        return window, label
    
    def get_dataloader(
        self, 
        batch_size: int = 32, 
        shuffle: bool = True, 
        num_workers: int = 4
    ) -> DataLoader:
        """
        Create a DataLoader for this dataset.
        
        Args:
            batch_size: Number of samples in each batch
            shuffle: Whether to shuffle the data
            num_workers: Number of worker processes for data loading
            
        Returns:
            PyTorch DataLoader configured with this dataset
        """
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available()
        )
    
    @property
    def windows(self) -> torch.Tensor:
        """
        Get the window data.
        
        Returns:
            Window data tensor
        """
        return self.X
    
    @property
    def labels(self) -> torch.Tensor:
        """
        Get the label data.
        
        Returns:
            Label data tensor
        """
        return self.y

    @classmethod
    def from_file(cls, file_path: str) -> 'LSTMWindowDataset':
        """
        Load a dataset from a PyTorch saved file.
        
        Args:
            file_path: Path to the saved dataset file
            
        Returns:
            Loaded LSTMWindowDataset instance
        """
        data = torch.load(file_path)
        return cls(data['windows'].numpy(), data['labels'].numpy())
    
    def to_file(self, file_path: str) -> None:
        """
        Save the dataset to a file.
        
        Args:
            file_path: Path where to save the dataset
        """
        torch.save({
            'windows': self.X,
            'labels': self.y
        }, file_path)
