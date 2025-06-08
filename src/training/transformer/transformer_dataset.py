import os
import torch
from torch.utils.data import Dataset, DataLoader
import glob
from typing import Dict, List, Tuple, Optional, Union
from src.utils.logger import logger
from functools import lru_cache

class TransformerDataset(Dataset):
    """
    Dataset for Transformer model for anomaly detection.
    
    This dataset loads sliding window data from .pt files to feed into a transformer model.
    Uses lazy loading to reduce memory usage.
    """
    
    def __init__(
        self,
        data_dir: str,
        data_type: str = 'train',
        component: str = 'contact',
        transform=None,
        cache_size: int = 3,
        device: str = 'cpu'
    ):
        """
        Initialize the dataset.
        
        Args:
            data_dir: Directory containing sliding window data
            data_type: Data type ('train', 'val', or 'test')
            component: Component type ('contact', 'pcb', or 'ring')
            transform: Optional transform to apply to the data
            cache_size: Number of batch files to cache in memory
            device: Device to load tensors to ('cpu' or 'cuda')
        """
        self.data_dir = data_dir
        self.data_type = data_type
        self.component = component
        self.transform = transform
        self.device = device
        
        # Convert train_down_25% to the actual directory name
        if data_type == 'train':
            self.data_type_dir = 'train_down_25%'
        else:
            self.data_type_dir = data_type
        
        # Path for the specific data type and component
        self.component_dir = os.path.join(data_dir, self.data_type_dir, component)
        
        # Find all .pt files in the component directory
        self.pt_files = sorted(glob.glob(os.path.join(self.component_dir, "batch_*.pt")))
        if not self.pt_files:
            raise ValueError(f"No .pt files found in {self.component_dir}")
        
        # Set up LRU cache for batch files
        self._load_batch_file = lru_cache(maxsize=cache_size)(self._load_batch_file_impl)
        
        # Build an index mapping to locate samples across batch files
        self._build_index()
        
        logger.info(f"Prepared dataset with {len(self)} samples for {data_type}/{component}")
    
    def _build_index(self):
        """
        Build an index mapping to locate samples across batch files.
        Instead of loading all data, we just record file paths and sample counts.
        """
        self.file_sample_counts = []
        self.file_offsets = []
        self.total_samples = 0
        
        for pt_file in self.pt_files:
            try:
                # Load metadata only
                data = torch.load(pt_file, map_location='cpu')
                
                if 'windows' in data and 'labels' in data:
                    num_samples = len(data['windows'])
                    self.file_sample_counts.append(num_samples)
                    self.file_offsets.append(self.total_samples)
                    self.total_samples += num_samples
                    # Immediately delete the loaded data to free memory
                    del data
                else:
                    logger.warning(f"Skipping {pt_file}: Unknown data structure")
            except Exception as e:
                logger.error(f"Error loading {pt_file}: {str(e)}")
        
        if not self.total_samples:
            raise ValueError(f"No valid data found in {self.component_dir}")
            
        logger.info(f"Indexed {len(self.pt_files)} batch files with {self.total_samples} total samples")
    
    def _load_batch_file_impl(self, file_path):
        """Implementation of batch file loading (wrapped with lru_cache)"""
        try:
            # Always load to CPU first to avoid OOM issues
            data = torch.load(file_path, map_location='cpu')
            return data
        except Exception as e:
            logger.error(f"Error loading batch file {file_path}: {str(e)}")
            # Return empty data as fallback
            return {'windows': torch.tensor([]), 'labels': torch.tensor([])}
    
    def __len__(self):
        return self.total_samples
    
    def __getitem__(self, idx):
        """
        Get a sample from the dataset.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Tuple of (window, label)
        """
        if idx >= self.total_samples:
            logger.warning(f"Index {idx} out of range, returning first item")
            idx = 0
            
        # Find which file contains this index
        file_idx = 0
        while file_idx < len(self.file_offsets) - 1 and self.file_offsets[file_idx + 1] <= idx:
            file_idx += 1
        
        # Calculate the local index within that file
        local_idx = idx - self.file_offsets[file_idx]
        
        try:
            # Load only the required file (using cache)
            data = self._load_batch_file(self.pt_files[file_idx])
            
            # Get the specific window and label
            window = data['windows'][local_idx]
            label = data['labels'][local_idx]
            
            # Convert to torch tensors if they aren't already
            if not isinstance(window, torch.Tensor):
                window = torch.FloatTensor(window)
            
            if not isinstance(label, torch.Tensor):
                label = torch.LongTensor([label])[0]
            
            if self.transform:
                window = self.transform(window)
            
            return window, label
        except Exception as e:
            logger.error(f"Error getting item at index {idx}: {str(e)}")
            # Return a default tensor in case of error
            shape = data['windows'][0].shape if 'windows' in data and len(data['windows']) > 0 else (100, 5)
            return torch.zeros(shape), torch.tensor(0)

def create_data_loaders(
    data_dir: str,
    batch_size: int = 64,
    num_workers: int = 4,
    component: str = 'contact',
    cache_size: int = 3,
    device: str = 'cpu'
) -> Dict[str, DataLoader]:
    """
    Create data loaders for training, validation, and testing.
    
    Args:
        data_dir: Directory containing sliding window data
        batch_size: Batch size for data loaders
        num_workers: Number of worker processes for data loading
        component: Component type ('contact', 'pcb', or 'ring')
        cache_size: Number of batch files to cache in memory
        device: Device to load tensors to ('cpu' or 'cuda')
        
    Returns:
        Dictionary of data loaders for 'train', 'val', and 'test'
    """
    data_loaders = {}
    
    for data_type in ['train', 'val', 'test']:
        try:
            dataset = TransformerDataset(
                data_dir=data_dir,
                data_type=data_type,
                component=component,
                cache_size=cache_size,
                device=device
            )
            
            shuffle = (data_type == 'train')
            
            data_loaders[data_type] = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                pin_memory=(device == 'cuda')
            )
            
            logger.info(f"Created {data_type} data loader with {len(dataset)} samples")
        except Exception as e:
            logger.error(f"Error creating {data_type} data loader: {str(e)}")
            data_loaders[data_type] = None
    
    return data_loaders 