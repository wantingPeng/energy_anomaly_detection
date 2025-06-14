import os
import torch
from torch.utils.data import Dataset, DataLoader
import glob
from typing import Dict, List, Tuple, Optional, Union
from src.utils.logger import logger

class TransformerDataset(Dataset):
    """
    Dataset for Transformer model for anomaly detection.
    
    This dataset loads sliding window data from .pt files to feed into a transformer model.
    Data is formatted as [batch_size, seq_len, features] to support batch_first=True in transformer models.
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
            Tuple of (window, label)
        """
        window = self.windows[idx]
        label = self.labels[idx]
        
        '''# Convert to torch tensors if they aren't already
        if not isinstance(window, torch.Tensor):
            window = torch.FloatTensor(window)
        
        if not isinstance(label, torch.Tensor):
            label = torch.LongTensor([label])[0]
        
        if self.transform:
            window = self.transform(window)'''
        
        return window, label
    
    def _load_data(self):
        """
        Load sliding window data from .pt files.
        
        Returns:
            Tuple of (windows, labels)
        """
        # Find all .pt files in the component directory
        pt_files = sorted(glob.glob(os.path.join(self.component_dir, "batch_*.pt")))
        if not pt_files:
            raise ValueError(f"No .pt files found in {self.component_dir}")
        
        # Load and concatenate data from all files
        all_windows = []
        all_labels = []
        
        for pt_file in pt_files:
            logger.info(f"Loading {pt_file}")
            data = torch.load(pt_file)
            if 'windows' in data and 'labels' in data:
                all_windows.append(data['windows'])
                all_labels.append(data['labels'])
            else:
                raise ValueError(f"Unknown data structure in {pt_file}")
           
        windows = torch.cat(all_windows, dim=0)
        labels = torch.cat(all_labels, dim=0)

        
        logger.info(f"Loaded {len(windows)} windows with shape {windows.shape} and {len(labels)} labels")
        
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
    
    for data_type in ['train', 'val_200']:
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