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
        
        # Convert train_down_25% to the actual directory name
        if data_type == 'train':
            self.data_type_dir = 'train_down_25%'
        else:
            self.data_type_dir = data_type
        
        # Path for the specific data type and component
        self.component_dir = os.path.join(data_dir, self.data_type_dir, component)
        
        # Store file paths and indices for lazy loading
        self.file_indices = []
        self._prepare_file_indices()
        
        logger.info(f"Loaded {len(self.file_indices)} samples for {data_type}/{component}")
    
    def _prepare_file_indices(self):
        """
        Prepare file indices for lazy loading.
        """
        pt_files = sorted(glob.glob(os.path.join(self.component_dir, "batch_*.pt")))
        if not pt_files:
            raise ValueError(f"No .pt files found in {self.component_dir}")

        for pt_file in pt_files:
            data = torch.load(pt_file)
            num_samples = len(data['windows'])
            self.file_indices.extend([(pt_file, i) for i in range(num_samples)])
    
    def __len__(self):
        return len(self.file_indices)
    
    def __getitem__(self, idx):
        """
        Get a sample from the dataset using lazy loading.
        """
        pt_file, sample_idx = self.file_indices[idx]
        data = torch.load(pt_file)
        window = data['windows'][sample_idx]
        label = data['labels'][sample_idx]
        
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
        Deprecated: Load sliding window data from .pt files.
        """
        raise NotImplementedError("_load_data is deprecated due to lazy loading implementation.")


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
    
    for data_type in ['train_down_25%', 'val']:
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
            pin_memory=True
        )
        
        logger.info(f"Created {data_type} data loader with {len(dataset)} samples")
    
    return data_loaders 