import glob
import os
from torch.utils.data import DataLoader, Dataset
import numpy as np
from src.utils.logger import logger
from typing import Dict
import torch







class SingleSampleDataset(Dataset):
    """
    Dataset that contains only a single window and its corresponding statistical features.
    Used for testing if the model can overfit to a single sample.
    """
    
    def __init__(
        self,
        lstm_data_dir: str,
        stat_features_dir: str,
        data_type: str = 'train',
        component: str = 'contact',
        sample_idx: int = 0,
        transform=None
    ):
        """
        Initialize the dataset with a single sample.
        
        Args:
            lstm_data_dir: Directory containing LSTM sliding window data
            stat_features_dir: Directory containing statistical features
            data_type: Data type ('train', 'val', or 'test')
            component: Component type ('contact', 'pcb', or 'ring')
            sample_idx: Index of the sample to use
            transform: Optional transform to apply to the data
        """
        self.lstm_data_dir = lstm_data_dir
        self.stat_features_dir = stat_features_dir
        self.data_type = data_type
        self.component = component
        self.sample_idx = sample_idx
        self.transform = transform
        
        # Load a single sample
        self.window, self.label, self.stat_feature = self._load_single_sample()
        
        logger.info(f"Loaded single sample (idx={sample_idx}) for overfitting test")
    
    def __len__(self):
        return 1
    
    def __getitem__(self, idx):
        """
        Get the single sample.
        
        Args:
            idx: Index (ignored since there's only one sample)
            
        Returns:
            Tuple of (window, statistical_features, label)
        """
        window = self.window
        stat_feature = self.stat_feature
        label = self.label
        
        # Convert to torch tensors
        window = torch.FloatTensor(window)
        stat_feature = torch.FloatTensor(stat_feature)
        label = torch.LongTensor([label])[0]
        
        if self.transform:
            window = self.transform(window)
            stat_feature = self.transform(stat_feature)
        
        return window, stat_feature, label
    

    def _load_single_sample(self):
      """
      Load a single sample from the dataset.
      
      Returns:
          Tuple of (window, label, stat_feature)
      """
      # Paths for the specific data type and component
      lstm_component_dir = os.path.join(self.lstm_data_dir, self.data_type, self.component)
      stat_component_dir = os.path.join(self.stat_features_dir, self.data_type, self.component)
      
      # Load LSTM data
      lstm_files = sorted(glob.glob(os.path.join(lstm_component_dir, "*.npz")))
      if not lstm_files:
          raise ValueError(f"No LSTM data files found in {lstm_component_dir}")
      
      # Load statistical features
      stat_files = sorted(glob.glob(os.path.join(stat_component_dir, "*.npz")))
      if not stat_files:
          raise ValueError(f"No statistical feature files found in {stat_component_dir}")
      
      # Load the first file
      lstm_data = np.load(lstm_files[0])
      stat_data = np.load(stat_files[0])
      
      # Get a single sample
      window = lstm_data['windows'][self.sample_idx:self.sample_idx+1][0]
      label = lstm_data['labels'][self.sample_idx]
      stat_feature = stat_data['stat_features'][self.sample_idx:self.sample_idx+1][0]
      
      return window, label, stat_feature




def create_single_sample_data_loader(
    lstm_data_dir: str,
    stat_features_dir: str,
    batch_size: int = 1,
    num_workers: int = 0,
    component: str = 'contact',
    sample_idx: int = 0
) -> Dict[str, DataLoader]:
    """
    Create data loaders with a single sample for overfitting tests.
    
    Args:
        lstm_data_dir: Directory containing LSTM sliding window data
        stat_features_dir: Directory containing statistical features
        batch_size: Batch size for data loaders
        num_workers: Number of worker processes for data loading
        component: Component type ('contact', 'pcb', or 'ring')
        sample_idx: Index of the sample to use
        
    Returns:
        Dictionary of data loaders with a single sample
    """
    data_loaders = {}
    
    # Create a dataset with a single sample for training
    train_dataset = SingleSampleDataset(
        lstm_data_dir=lstm_data_dir,
        stat_features_dir=stat_features_dir,
        data_type='train',
        component=component,
        sample_idx=sample_idx
    )
    
    # Use the same sample for validation and testing
    val_dataset = SingleSampleDataset(
        lstm_data_dir=lstm_data_dir,
        stat_features_dir=stat_features_dir,
        data_type='train',  # Use train data for all splits
        component=component,
        sample_idx=sample_idx
    )
    
    test_dataset = SingleSampleDataset(
        lstm_data_dir=lstm_data_dir,
        stat_features_dir=stat_features_dir,
        data_type='train',  # Use train data for all splits
        component=component,
        sample_idx=sample_idx
    )
    
    # Create data loaders
    data_loaders['train'] = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,  # No need to shuffle with a single sample
        num_workers=num_workers,
        pin_memory=True
    )
    
    data_loaders['val'] = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    data_loaders['test'] = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    logger.info(f"Created data loaders with a single sample (idx={sample_idx}) for overfitting test")
    
    return data_loaders 
