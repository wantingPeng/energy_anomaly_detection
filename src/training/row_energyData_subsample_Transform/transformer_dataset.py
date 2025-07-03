import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import glob
from typing import Dict, List, Tuple, Optional, Union
from src.utils.logger import logger

class TransformerDataset(Dataset):
    """
    Dataset for Transformer model for anomaly detection.
    
    This dataset loads sliding window data from .npz files to feed into a transformer model.
    Data is formatted as [batch_size, seq_len, features] to support batch_first=True in transformer models.
    Labels are now sequences [batch_size, seq_len], with one label per timestep.
    """
    
    def __init__(
        self,
        data_dir: str,
        data_type: str = 'train',
        component: str = 'contact',
        transform=None
    ):
        """
        Initialize the dataset.
        
        Args:
            data_dir: Directory containing sliding window data
            data_type: Data type ('train', 'val', or 'test')
            component: Component type ('contact', 'pcb', or 'ring')
            transform: Optional transform to apply to the data
        """
        self.data_dir = data_dir
        self.data_type = data_type
        self.component = component
        self.transform = transform
        
        # Path for the specific data type and component
        self.component_dir = os.path.join(data_dir, self.data_type, component)
        
        # Load data
        self.windows, self.labels = self._load_data()
        
        logger.info(f"Loaded {len(self.windows)} samples for {data_type}/{component}")
    
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
    
    def _load_data(self):
        """
        Load sliding window data from .npz files.
        
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
            data = np.load(npz_file)
            if 'windows' in data and 'labels' in data:
                # Convert numpy arrays to torch tensors
                windows_tensor = torch.from_numpy(data['windows']).float()
                labels_tensor = torch.from_numpy(data['labels']).long()
                all_windows.append(windows_tensor)
                all_labels.append(labels_tensor)
            else:
                raise ValueError(f"Unknown data structure in {npz_file}")
           
        windows = torch.cat(all_windows, dim=0)
        labels = torch.cat(all_labels, dim=0)

        
        logger.info(f"Loaded {len(windows)} windows with shape {windows.shape} and labels shape {labels.shape}")
        
        return windows, labels


def create_data_loaders(
    data_dir: str,
    batch_size: int = 64,
    num_workers: int = 4,
    component: str = 'contact'
) -> Dict[str, DataLoader]:
    """
    Create data loaders for training, validation, and testing.
    
    Args:
        data_dir: Directory containing sliding window data
        batch_size: Batch size for data loaders
        num_workers: Number of worker processes for data loading
        component: Component type ('contact', 'pcb', or 'ring')
        
    Returns:
        Dictionary of data loaders for 'train', 'val', and 'test'
    """
    data_loaders = {}
    
    for data_type in ['train', 'val']:
        dataset = TransformerDataset(
            data_dir=data_dir,
            data_type=data_type,
            component=component
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