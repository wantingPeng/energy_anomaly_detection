import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import glob
from typing import Dict, List, Tuple, Optional, Union
from src.utils.logger import logger

class TransformerLateFusionDataset(Dataset):
    """
    Dataset for Transformer model with Late Fusion for anomaly detection.
    
    This dataset loads both sliding window data and statistical features to feed into
    a transformer late fusion model. Data is formatted as [batch_size, seq_len, features]
    for transformer inputs.
    """
    
    def __init__(
        self,
        data_dir: str,
        stat_features_dir: str,
        data_type: str = 'train',
        component: str = 'contact',
        transform=None
    ):
        """
        Initialize the dataset.
        
        Args:
            data_dir: Directory containing sliding window data
            stat_features_dir: Directory containing statistical features
            data_type: Data type ('train', 'val', or 'test')
            component: Component type ('contact', 'pcb', or 'ring')
            transform: Optional transform to apply to the data
        """
        self.data_dir = data_dir
        self.stat_features_dir = stat_features_dir
        self.data_type = data_type
        self.component = component
        self.transform = transform
        
        # Path for the specific data type and component
        self.component_dir = os.path.join(data_dir, self.data_type, component)
        self.stat_component_dir = os.path.join(stat_features_dir, self.data_type, component)
        
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
        
        # Convert to torch tensors if they aren't already
        if not isinstance(window, torch.Tensor):
            window = torch.FloatTensor(window)
        
        if not isinstance(stat_feature, torch.Tensor):
            stat_feature = torch.FloatTensor(stat_feature)
            
        if not isinstance(label, torch.Tensor):
            label = torch.LongTensor([label])[0]
        
        if self.transform:
            window = self.transform(window)
            
        return window, stat_feature, label
    
    def _load_data(self):
        """
        Load sliding window data and statistical features.
        
        Returns:
            Tuple of (windows, labels, stat_features)
        """
        # Find all .pt files in the component directory
        pt_files = sorted(glob.glob(os.path.join(self.component_dir, "batch_*.pt")))
        if not pt_files:
            raise ValueError(f"No .pt files found in {self.component_dir}")
        
        # Find all statistical feature files
        stat_files = sorted(glob.glob(os.path.join(self.stat_component_dir, "*.npz")))
        if not stat_files:
            raise ValueError(f"No statistical feature files found in {self.stat_component_dir}")
        
        # Ensure the number of batch files makes sense
        if len(pt_files) == 0 or len(stat_files) == 0:
            raise ValueError(f"Missing data files in {self.component_dir} or {self.stat_component_dir}")
            
        logger.info(f"Found {len(pt_files)} window files and {len(stat_files)} stat feature files")
        
        # Load and concatenate window data from all files
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
        
        # Concatenate all window data
        windows = torch.cat(all_windows, dim=0)
        labels = torch.cat(all_labels, dim=0)
        
        # Load and concatenate statistical features
        all_stat_features = []
        
        for stat_file in stat_files:
            logger.info(f"Loading {stat_file}")
            data = np.load(stat_file)
            if 'stat_features' in data:
                all_stat_features.append(data['stat_features'])
            else:
                raise ValueError(f"Missing 'stat_features' in {stat_file}")
        
        # Concatenate all statistical features
        stat_features = np.concatenate(all_stat_features, axis=0)
        
        # Verify that the number of samples match
        if len(windows) != len(stat_features):
            logger.warning(f"Number of windows ({len(windows)}) does not match "
                          f"number of statistical features ({len(stat_features)})")
            # Use the minimum length to avoid index errors
            min_len = min(len(windows), len(stat_features))
            windows = windows[:min_len]
            labels = labels[:min_len]
            stat_features = stat_features[:min_len]
        
        logger.info(f"Loaded {len(windows)} windows with shape {windows.shape}, "
                   f"{len(stat_features)} statistical features with shape {stat_features.shape}, "
                   f"and {len(labels)} labels")
        
        return windows, labels, stat_features


def create_data_loaders(
    data_dir: str,
    stat_features_dir: str,
    batch_size: int = 64,
    num_workers: int = 4,
    component: str = 'contact'
) -> Dict[str, DataLoader]:
    """
    Create data loaders for training, validation, and testing.
    
    Args:
        data_dir: Directory containing sliding window data
        stat_features_dir: Directory containing statistical features
        batch_size: Batch size for data loaders
        num_workers: Number of worker processes for data loading
        component: Component type ('contact', 'pcb', or 'ring')
        
    Returns:
        Dictionary of data loaders for 'train', 'val', and 'test'
    """
    data_loaders = {}
    
    for data_type in ['train', 'val']:
        dataset = TransformerLateFusionDataset(
            data_dir=data_dir,
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