import os
import torch
import math
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import glob
from typing import Dict, List, Tuple, Optional, Union
from src.utils.logger import logger


class PositionalEncoding(nn.Module):
    """
    Positional encoding for Transformer models.
    
    Implementation of the sinusoidal position encoding described in 
    "Attention Is All You Need" paper.
    """
    
    def __init__(self, d_model, max_seq_length=1200):
        """
        Initialize positional encoding.
        
        Args:
            d_model: Dimensionality of the model
            max_seq_length: Maximum sequence length
        """
        super(PositionalEncoding, self).__init__()
        
        # Create positional encoding matrix
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # Apply sine to even indices and cosine to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add batch dimension
        pe = pe.unsqueeze(0)
        
        # Register as buffer (not a parameter)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """
        Add positional encoding to input tensor.
        
        Args:
            x: Input tensor of shape [batch_size, seq_length, d_model]
            
        Returns:
            Tensor with positional encoding added
        """
        x = x + self.pe[:, :x.size(1), :]
        return x


class TransformerDataset(Dataset):
    """
    Dataset for Transformer model for anomaly detection.
    
    This dataset loads sliding window data from .npz files, applies linear projection
    and positional encoding, and feeds the processed data into a transformer model.
    Data is formatted as [batch_size, seq_len, features] to support batch_first=True in transformer models.
    """
    
    def __init__(
        self,
        data_dir: str,
        data_type: str = 'train',
        component: str = 'contact',
        transform=None,
        d_model: int = 256,
        max_seq_length: int = 1200
    ):
        """
        Initialize the dataset.
        
        Args:
            data_dir: Directory containing sliding window data
            data_type: Data type ('train', 'val', or 'test')
            component: Component type ('contact', 'pcb', or 'ring')
            transform: Optional transform to apply to the data
            d_model: Dimensionality of the model after linear projection
            max_seq_length: Maximum sequence length for positional encoding
        """
        self.data_dir = data_dir
        self.data_type = data_type
        self.component = component
        self.transform = transform
        self.d_model = d_model
        self.max_seq_length = max_seq_length


        # Path for the specific data type and component
        self.component_dir = os.path.join(data_dir, self.data_type, component)
        
        # Initialize linear projection and positional encoding
        self._initialize_projection_layers()
        
        # Load data
        self.windows, self.labels = self._load_data()
        
        logger.info(f"Loaded {len(self.windows)} samples for {data_type}/{component}")
    
    def _initialize_projection_layers(self):
        """
        Initialize linear projection and positional encoding layers.
        
        Checks if there are any .npz files to determine input feature dimension.
        """
        # Find a sample .npz file to determine input features
        npz_files = glob.glob(os.path.join(self.component_dir, "*.npz"))
        if not npz_files:
            raise ValueError(f"No .npz files found in {self.component_dir}")
        
        # Load a sample file to determine input features
        sample_file = npz_files[0]
        sample_data = np.load(sample_file, allow_pickle=True)
        input_features = sample_data['windows'].shape[2]
        logger.info(f"Input features from {os.path.basename(sample_file)}: {input_features}")
        
        # Create linear projection layer
        self.linear_projection = nn.Linear(input_features, self.d_model)
        nn.init.xavier_uniform_(self.linear_projection.weight)
        nn.init.zeros_(self.linear_projection.bias)
        
        # Create positional encoding layer
        self.positional_encoding = PositionalEncoding(self.d_model, self.max_seq_length)
    
    def __len__(self):
        return len(self.windows)
    
    def __getitem__(self, idx):
        """
        Get a sample from the dataset.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Tuple of (window, label)
        """
        window = self.windows[idx]
        label = self.labels[idx]
        
        # Convert to torch tensors if they aren't already
        if not isinstance(window, torch.Tensor):
            window = torch.FloatTensor(window)
        
        if not isinstance(label, torch.Tensor):
            label = torch.LongTensor([label])[0]
        
        if self.transform:
            window = self.transform(window)
        
        return window, label
    
    def _load_data(self):
        """
        Load sliding window data from .npz files, apply linear projection and positional encoding.
        
        Returns:
            Tuple of (windows, labels)
        """
        # Find all .npz files in the component directory
        npz_files = sorted(glob.glob(os.path.join(self.component_dir, "*.npz")))
        if not npz_files:
            raise ValueError(f"No .npz files found in {self.component_dir}")
        
        # Load and concatenate data from all files
        all_windows = []
        all_labels = []
        
        for npz_file in npz_files:
            logger.info(f"Loading {npz_file}")
            data = np.load(npz_file, allow_pickle=True)
            
            # Extract windows and labels
            windows = data['windows']
            labels = data['labels']
            
            # Convert to tensor
            windows_tensor = torch.FloatTensor(windows)
            labels_tensor = torch.LongTensor(labels)
            
            # Apply linear projection and positional encoding
            with torch.no_grad():
                projected = self.linear_projection(windows_tensor)
                transformed_windows = self.positional_encoding(projected)
            
            all_windows.append(transformed_windows)
            all_labels.append(labels_tensor)
            
            logger.info(f"Processed {npz_file}: {windows.shape} -> {transformed_windows.shape}")
        
        # Concatenate all windows and labels
        windows = torch.cat(all_windows, dim=0)
        labels = torch.cat(all_labels, dim=0)
        
        logger.info(f"Loaded {len(windows)} windows with shape {windows.shape} and {len(labels)} labels")
        
        return windows, labels


def create_data_loaders(
    data_dir: str = "Data/processed/lsmt_statisticalFeatures/sliding_window_1200s",
    batch_size: int = 64,
    num_workers: int = 4,
    component: str = 'contact',
    d_model: int = 256,
    max_seq_length: int = 1200
) -> Dict[str, DataLoader]:
    """
    Create data loaders for training, validation, and testing.
    
    Args:
        data_dir: Directory containing sliding window data
        batch_size: Batch size for data loaders
        num_workers: Number of worker processes for data loading
        component: Component type ('contact', 'pcb', or 'ring')
        d_model: Dimensionality of the model after linear projection
        max_seq_length: Maximum sequence length for positional encoding
        
    Returns:
        Dictionary of data loaders for 'train', 'val', and 'test'
    """
    data_loaders = {}
    
    for data_type in ['train', 'val']:
        dataset = TransformerDataset(
            data_dir=data_dir,
            data_type=data_type,
            component=component,
            d_model=d_model,
            max_seq_length=max_seq_length
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