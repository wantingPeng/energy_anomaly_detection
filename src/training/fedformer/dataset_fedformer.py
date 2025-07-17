"""
Data loader for FEDformer model with enhanced time-frequency features.

This module provides dataset class optimized for FEDformer's frequency domain processing,
including time feature engineering and periodic pattern extraction.
"""

import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import glob
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import sys
sys.path.append('src')
from utils.logger import logger

class FEDformerDataset(Dataset):
    """
    Dataset for FEDformer model with enhanced time-frequency features.
    
    This dataset is optimized for FEDformer's Fourier enhanced processing,
    providing both time domain features and time-based auxiliary features.
    """
    
    def __init__(
        self,
        data_dir: str,
        data_type: str = 'train',
        component: str = 'contact',
        seq_len: int = 96,  # FEDformer typical sequence length
        label_len: int = 48,  # Label length for decoder input
        pred_len: int = 24,   # Prediction length
        step_size: int = 1,
        exclude_columns: List[str] = None,
        freq: str = 'h',      # Frequency: 'h' for hourly, 'm' for minutely
        embed: str = 'timeF',  # Time embedding type
        transform=None
    ):
        """
        Initialize the FEDformer dataset.
        
        Args:
            data_dir: Directory containing parquet data files
            data_type: Data type ('train', 'val', or 'test')
            component: Component type ('contact', 'pcb', or 'ring')
            seq_len: Input sequence length
            label_len: Label length for decoder
            pred_len: Prediction length
            step_size: Step size for sliding window
            exclude_columns: Columns to exclude from features
            freq: Data frequency
            features: 'M' for multivariate, 'S' for univariate
            target: Target column name
            embed: Time embedding type
            transform: Optional transform
        """
        self.data_dir = data_dir
        self.data_type = data_type
        self.component = component
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.step_size = step_size
        self.features = 'M' # Always multivariate for FEDformer
        self.target = 'rTotalActivePower' # Default target, can be overridden
        self.freq = freq
        self.embed = embed
        self.transform = transform
        
        # Default columns to exclude
        if exclude_columns is None:
            exclude_columns = ["TimeStamp", "segment_id"]
        self.exclude_columns = exclude_columns
        
        # Path for the specific data type and component
        self.component_dir = os.path.join(data_dir, self.data_type, component)
        
        # Load and process data
        self.raw_data = self._load_raw_data()
        self.data_x, self.data_y, self.data_stamp = self._prepare_data()
        
        logger.info(f"FEDformer Dataset - {data_type}/{component}:")
        logger.info(f"  Total samples: {len(self.data_x)}")
        logger.info(f"  Feature shape: {self.data_x.shape}")
        logger.info(f"  Label shape: {self.data_y.shape}")
        logger.info(f"  Time stamp shape: {self.data_stamp.shape}")
    
    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1
    
    def __getitem__(self, index):
        """
        Get a sample for FEDformer training.
        
        Returns:
            seq_x: [seq_len, features] - encoder input
            seq_y: [label_len + pred_len, features] - decoder input  
            seq_x_mark: [seq_len, time_features] - encoder time features
            seq_y_mark: [label_len + pred_len, time_features] - decoder time features
            labels: [seq_len] - anomaly labels for the sequence
        """
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len 
        r_end = r_begin + self.label_len + self.pred_len

        # Encoder input: [seq_len, features]
        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]
        
        # Anomaly labels for the encoder sequence
        labels = self.raw_data['anomaly_label'].iloc[s_begin:s_end].values
        
        # Convert to tensors
        seq_x = torch.FloatTensor(seq_x)
        seq_y = torch.FloatTensor(seq_y)
        seq_x_mark = torch.FloatTensor(seq_x_mark)
        seq_y_mark = torch.FloatTensor(seq_y_mark)
        labels = torch.LongTensor(labels)
        
        if self.transform:
            seq_x = self.transform(seq_x)
        
        return seq_x, seq_y, seq_x_mark, seq_y_mark, labels
    
    def _load_raw_data(self):
        """Load raw data from parquet files."""
        # Find all parquet files in the component directory
        parquet_files = sorted(glob.glob(os.path.join(self.component_dir, "*.parquet")))
        if not parquet_files:
            raise ValueError(f"No parquet files found in {self.component_dir}")
        
        # Load and concatenate data from all files
        all_data = []
        for parquet_file in parquet_files:
            logger.info(f"Loading {parquet_file}")
            df = pd.read_parquet(parquet_file)
            all_data.append(df)
        
        # Concatenate all data
        combined_data = pd.concat(all_data, ignore_index=True)
        
        # Sort by timestamp to ensure proper temporal order
        if 'TimeStamp' in combined_data.columns:
            combined_data = combined_data.sort_values('TimeStamp').reset_index(drop=True)
        
        logger.info(f"Loaded {len(combined_data)} rows with {len(combined_data.columns)} columns")
        logger.info(f"Columns: {list(combined_data.columns)}")
        
        return combined_data
    
    def _prepare_data(self):
        """Prepare data for FEDformer training."""
        df_raw = self.raw_data.copy()
        
        # Handle missing TimeStamp
        if 'TimeStamp' not in df_raw.columns:
            logger.warning("TimeStamp column not found, creating synthetic timestamps")
            df_raw['TimeStamp'] = pd.date_range(start='2024-01-01', periods=len(df_raw), freq='T')
        
        # Ensure TimeStamp is datetime
        df_raw['TimeStamp'] = pd.to_datetime(df_raw['TimeStamp'])
        
        # Generate time features
        df_stamp = self._generate_time_features(df_raw['TimeStamp'])
        
        # Prepare feature columns - always use multivariate
        feature_columns = [col for col in df_raw.columns 
                          if col not in self.exclude_columns and col != 'anomaly_label']
        
        # Use all feature columns (multivariate approach)
        cols_data = feature_columns
        
        df_data = df_raw[cols_data]
        
        logger.info(f"Selected feature columns ({len(cols_data)}): {cols_data}")
        
        # Convert to numpy arrays
        data_x = df_data.values.astype(np.float32)
        data_y = df_data.values.astype(np.float32)  # For autoregressive prediction
        data_stamp = df_stamp.values.astype(np.float32)
        
        logger.info(f"Data shapes - X: {data_x.shape}, Y: {data_y.shape}, Stamp: {data_stamp.shape}")
        
        return data_x, data_y, data_stamp
    
    def _generate_time_features(self, timestamps: pd.Series) -> pd.DataFrame:
        """
        Generate time features for FEDformer.
        
        Based on your periodicity analysis:
        - Hour of day (0-23) - 24h cycle
        - Day of week (0-6) - 168h cycle  
        - Month (1-12) - Seasonal patterns
        - Day of year (1-365) - Annual patterns
        """
        df_stamp = pd.DataFrame()
        
        if self.embed == 'timeF':
            # Time features based on your periodic analysis
            df_stamp['hour'] = timestamps.dt.hour / 23.0  # Normalize to [0, 1]
            df_stamp['day_of_week'] = timestamps.dt.dayofweek / 6.0  # 0=Monday, 6=Sunday
            df_stamp['day_of_month'] = (timestamps.dt.day - 1) / 30.0  # Approximate normalization
            df_stamp['month'] = (timestamps.dt.month - 1) / 11.0  # 0-based, normalized
            
            # Additional features based on your "5+2" pattern analysis
            df_stamp['is_weekend'] = (timestamps.dt.dayofweek >= 5).astype(float)
            df_stamp['is_workday'] = (timestamps.dt.dayofweek < 5).astype(float)
            
            # Cyclical encoding for better periodicity representation
            df_stamp['hour_sin'] = np.sin(2 * np.pi * timestamps.dt.hour / 24)
            df_stamp['hour_cos'] = np.cos(2 * np.pi * timestamps.dt.hour / 24)
            df_stamp['dow_sin'] = np.sin(2 * np.pi * timestamps.dt.dayofweek / 7)
            df_stamp['dow_cos'] = np.cos(2 * np.pi * timestamps.dt.dayofweek / 7)
            
        else:
            # Simple time encoding
            df_stamp['hour'] = timestamps.dt.hour
            df_stamp['day_of_week'] = timestamps.dt.dayofweek
            df_stamp['month'] = timestamps.dt.month
            df_stamp['day_of_year'] = timestamps.dt.dayofyear
        
        return df_stamp
    
    def inverse_transform(self, data):
        """Inverse transform for denormalization (placeholder)."""
        return data

def create_fedformer_data_loaders(
    data_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
    component: str = 'contact',
    seq_len: int = 96,
    label_len: int = 48,
    pred_len: int = 24,
    step_size: int = 1,
    freq: str = 'h',
    exclude_columns: List[str] = None,
    pin_memory: bool = True
) -> Dict[str, DataLoader]:
    """
    Create FEDformer data loaders for training, validation, and testing.
    
    Args:
        data_dir: Directory containing parquet data files
        batch_size: Batch size for training
        num_workers: Number of worker processes
        component: Component type
        seq_len: Input sequence length
        label_len: Label length for decoder
        pred_len: Prediction length  
        step_size: Step size for sliding window
        freq: Data frequency
        exclude_columns: Columns to exclude
        pin_memory: Whether to use pinned memory
        
    Returns:
        Dictionary containing train, val, test data loaders
    """
    
    datasets = {}
    data_loaders = {}
    
    for data_type in ['train', 'val', 'test']:
        try:
            dataset = FEDformerDataset(
                data_dir=data_dir,
                data_type=data_type,
                component=component,
                seq_len=seq_len,
                label_len=label_len,
                pred_len=pred_len,
                step_size=step_size,
                freq=freq,
                exclude_columns=exclude_columns
            )
            
            datasets[data_type] = dataset
            
            # Create data loader
            shuffle = (data_type == 'train')
            data_loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                pin_memory=pin_memory,
                drop_last=False
            )
            
            data_loaders[data_type] = data_loader
            
            logger.info(f"Created {data_type} dataloader: {len(dataset)} samples, "
                       f"{len(data_loader)} batches")
            
        except Exception as e:
            logger.warning(f"Failed to create {data_type} dataset: {e}")
            continue
    
    return data_loaders

# Utility functions for FEDformer data processing

def get_feature_info(data_dir: str, component: str = 'contact') -> Dict:
    """
    Get feature information from the dataset.
    
    Returns:
        Dictionary with feature names, sizes, and statistics
    """
    # Load a sample to get feature info
    sample_file = glob.glob(os.path.join(data_dir, 'train', component, '*.parquet'))[0]
    df = pd.read_parquet(sample_file)
    
    exclude_columns = ["TimeStamp", "segment_id", "anomaly_label"]
    feature_columns = [col for col in df.columns if col not in exclude_columns]
    
    feature_info = {
        'feature_names': feature_columns,
        'num_features': len(feature_columns),
        'feature_stats': df[feature_columns].describe().to_dict(),
        'anomaly_ratio': df['anomaly_label'].mean() if 'anomaly_label' in df.columns else 0.0,
        'total_samples': len(df)
    }
    
    return feature_info

# create_fedformer_config函数已删除 - 现在直接从YAML读取配置
# 如需获取数据特征信息，请直接调用 get_feature_info() 函数 