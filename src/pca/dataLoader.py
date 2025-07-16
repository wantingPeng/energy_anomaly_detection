#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
DataLoader for Principal Component Analysis data.
This module provides functions to load and prepare PC data for transformer models.
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Dict, Optional, List, Union
from src.utils.logger import logger


class PCDataset(Dataset):
    """Dataset class for PCA features with sliding windows."""
    
    def __init__(
        self, 
        features: np.ndarray, 
        labels: np.ndarray, 
        seq_length: int = 24,
        stride: int = 1,
        transform=None
    ):
        """
        Initialize the PCDataset.
        
        Args:
            features: Feature array with shape [n_samples, n_features]
            labels: Label array with shape [n_samples]
            seq_length: Length of each sequence window
            stride: Step size between windows
            transform: Optional transform to apply to each sample
        """
        self.features = features
        self.labels = labels
        self.seq_length = seq_length
        self.stride = stride
        self.transform = transform
        
        # Create window indices
        self.indices = self._create_windows()
        
    def _create_windows(self) -> List[Tuple[int, int]]:
        """Create sliding windows indices."""
        indices = []
        data_length = len(self.features)
        
        # 正常的滑动窗口
        for i in range(0, data_length - self.seq_length + 1, self.stride):
            indices.append((i, i + self.seq_length))
        
        # 特殊处理最后一个可能不完整的窗口
        if self.stride > 1:  # 只有当步长大于1时才需要考虑边界情况
            last_start = ((data_length - self.seq_length) // self.stride) * self.stride
            last_end = last_start + self.seq_length
            
            # 如果最后一个窗口结束位置超出数据范围，就不再添加
            if last_end <= data_length and (last_start, last_end) not in indices:
                indices.append((last_start, last_end))
        
        logger.info(f"Created {len(indices)} windows from {data_length} samples using seq_length={self.seq_length}, stride={self.stride}")
        return indices
    
    def __len__(self) -> int:
        """Return the number of windows."""
        return len(self.indices)
    
    def __getitem__(self, idx) -> Dict[str, Union[torch.Tensor, np.ndarray]]:
        """Get a window of data."""
        start_idx, end_idx = self.indices[idx]
        
        # Get sequence data
        feature_seq = self.features[start_idx:end_idx]
        label_seq = self.labels[start_idx:end_idx]
        
        # Apply transform if available
        if self.transform:
            feature_seq = self.transform(feature_seq)
        
        # Convert to tensors
        feature_tensor = torch.FloatTensor(feature_seq)
        label_tensor = torch.LongTensor(label_seq)
        
        return {
            'features': feature_tensor,
            'labels': label_tensor,
        }


def load_pc_data(data_path: str = "Data/pc/pc_features_train.parquet") -> pd.DataFrame:
    """
    Load PC data from parquet file.
    
    Args:
        data_path: Path to the PC data parquet file
    
    Returns:
        DataFrame containing PC data
    """
    logger.info(f"Loading PC data from {data_path}")
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"PC data file not found: {data_path}")
    
    df = pd.read_parquet(data_path)
    logger.info(f"Loaded PC data with shape: {df.shape}")
    
    return df


def train_val_split(
    df: pd.DataFrame, 
    val_ratio: float = 0.15, 
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data into training and validation sets.
    
    Args:
        df: Input DataFrame
        val_ratio: Ratio of validation data

    
    Returns:
        Tuple of (train_df, val_df)
    """

    # Time-ordered split (use last val_ratio portion as validation)
    split_idx = int(len(df) * (1 - val_ratio))
    train_df = df.iloc[:split_idx].reset_index(drop=True)
    val_df = df.iloc[split_idx:].reset_index(drop=True)
    
    logger.info(f"Time-ordered split: train={train_df.shape}, val={val_df.shape}")
    
    if 'TimeStamp' in df.columns:
        logger.info(f"Train period: {train_df['TimeStamp'].min()} to {train_df['TimeStamp'].max()}")
        logger.info(f"Val period: {val_df['TimeStamp'].min()} to {val_df['TimeStamp'].max()}")

    return train_df, val_df


def standardize_features(features: np.ndarray) -> Tuple[np.ndarray, dict]:
    """
    Standardize features (zero mean, unit variance) independently for each feature column.
    
    Args:
        features: Feature array with shape [n_samples, n_features]
    
    Returns:
        Tuple of (standardized_features, scaler_params)
        scaler_params contains 'mean' and 'std' for future inverse transformation
    """
    # 计算每个特征（列）的均值和标准差
    mean = np.mean(features, axis=0)
    std = np.std(features, axis=0)
    
    # Avoid division by zero
    std = np.where(std == 0, 1e-7, std)
    
    # Standardize each feature independently
    standardized_features = (features - mean) / std
    
    scaler_params = {'mean': mean, 'std': std}
    
    # 记录每个特征的均值和标准差范围，以便确认是独立标准化
    logger.info(f"Mean shape: {mean.shape}")
    logger.info(f"Std shape: {std.shape}")

    
    return standardized_features, scaler_params


def prepare_features_labels(
    df: pd.DataFrame,
    label_col: str = 'anomaly_label',
    timestamp_col: str = 'TimeStamp',
    standardize: bool = False
) -> Tuple[np.ndarray, np.ndarray, Optional[dict]]:
    """
    Extract features and labels from DataFrame.
    
    Args:
        df: Input DataFrame
        label_col: Column name for labels
        timestamp_col: Column name for timestamps (used to exclude from features)
        standardize: Whether to standardize features
    
    Returns:
        Tuple of (features, labels, scaler_params)
        scaler_params is None if standardize is False
    """
    # Check if label column exists
    if label_col not in df.columns:
        raise ValueError(f"Column {label_col} not found in DataFrame")
    
    # Use all columns except label and timestamp as features
    feature_cols = [col for col in df.columns if col not in [label_col, timestamp_col]]
    
    # Extract arrays
    features = df[feature_cols].values
    labels = df[label_col].values
    
    feature_dim = features.shape[1]
    logger.info(f"Prepared features with shape: {features.shape}, dimension: {feature_dim}")
    
    # Standardize features if requested
    scaler_params = None
    if standardize:
        features, scaler_params = standardize_features(features)
        logger.info("Features standardized to zero mean and unit variance")
    else:
        logger.info("Features kept in original scale (no standardization)")
    
    return features, labels, scaler_params


def create_dataloaders(
    df: pd.DataFrame,
    batch_size: int = 32,
    seq_length: int = 24,
    stride: int = 1,
    val_ratio: float = 0.15,
    label_col: str = 'anomaly_label',
    timestamp_col: str = 'TimeStamp',
    num_workers: int = 4,
    shuffle_train: bool = True,
    standardize: bool = False
) -> Dict[str, Union[DataLoader, dict]]:
    """
    Create PyTorch DataLoaders for training and validation.
    
    Args:
        df: Input DataFrame
        batch_size: Batch size for DataLoader
        seq_length: Sequence length for each window
        stride: Stride between windows
        val_ratio: Ratio of validation data
        label_col: Column name for labels
        timestamp_col: Column name for timestamps (used to exclude from features)
        num_workers: Number of workers for DataLoader
        shuffle_train: If True, shuffle training data
        standardize: Whether to standardize features
    
    Returns:
        Dict with 'train' and 'val' DataLoaders, and 'scaler_params' if standardize is True
    """
    # Split data into train and validation sets
    train_df, val_df = train_val_split(df, val_ratio)
    
    # Prepare features and labels for training set
    train_features, train_labels, scaler_params = prepare_features_labels(
        train_df, label_col, timestamp_col, standardize
    )
    
    # For validation set, always extract raw features first
    val_features, val_labels, _ = prepare_features_labels(
        val_df, label_col, timestamp_col, standardize=False
    )
    
    # If standardization is requested, apply training set's parameters to validation set
    if standardize and scaler_params:
        val_features = (val_features - scaler_params['mean']) / scaler_params['std']
        logger.info("Applied training set standardization parameters to validation features")
    
    # Create datasets
    train_dataset = PCDataset(
        train_features, 
        train_labels,
        seq_length=seq_length, 
        stride=stride
    )
    val_dataset = PCDataset(
        val_features, 
        val_labels,
        seq_length=seq_length, 
        stride=stride
    )
    
    logger.info(f"Created train dataset with {len(train_dataset)} windows")
    logger.info(f"Created validation dataset with {len(val_dataset)} windows")
    
    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        pin_memory=True
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    result = {
        'train': train_dataloader,
        'val': val_dataloader
    }
    
    # Add scaler parameters if standardization was applied
    if standardize:
        result['scaler_params'] = scaler_params
    
    return result


def get_pc_dataloaders(
    data_path: str = "Data/pc/pc_features_train.parquet",
    batch_size: int = 64,
    seq_length: int = 60,
    stride: int = 1,
    val_ratio: float = 0.15,
    num_workers: int = 4,
    standardize: bool = False
) -> Dict[str, Union[DataLoader, dict]]:
    """
    Convenience function to load data and create DataLoaders in one step.
    
    Args:
        data_path: Path to the PC data parquet file
        batch_size: Batch size for DataLoader
        seq_length: Sequence length for each window
        stride: Stride between windows
        val_ratio: Ratio of validation data
        num_workers: Number of workers for DataLoader
        standardize: Whether to standardize features
    
    Returns:
        Dict with 'train' and 'val' DataLoaders, and 'scaler_params' if standardize is True
    """
    # Load data
    df = load_pc_data(data_path)
    
    # Create dataloaders
    dataloaders = create_dataloaders(
        df=df,
        batch_size=batch_size,
        seq_length=seq_length,
        stride=stride,
        val_ratio=val_ratio,
        num_workers=num_workers,
        standardize=standardize
    )
    
    # Get the feature dimension by examining the first batch
    sample_batch = next(iter(dataloaders['train']))
    feature_shape = sample_batch['features'].shape
    
    logger.info(f"PC DataLoaders created successfully")
    logger.info(f"  Train batch shape: [batch_size={feature_shape[0]}, seq_len={feature_shape[1]}, input_dim={feature_shape[2]}]")
    
    return dataloaders


if __name__ == "__main__":
    # Simple test to verify functionality
    dataloaders = get_pc_dataloaders(seq_length=60, stride=60, standardize=True)
    
    # Print sample batch
    for phase, dataloader in dataloaders.items():
        if phase in ['train', 'val']:
            batch = next(iter(dataloader))
            features = batch['features']
            labels = batch['labels']
            logger.info(f"{phase} features shape: {features.shape}")
            logger.info(f"{phase} labels shape: {labels.shape}")
            logger.info(f"Feature dimension: {features.shape[2]}")
            
            # Check standardization (mean should be close to 0, std close to 1)
            if phase == 'train':
                mean = features.mean().item()
                std = features.std().item()
                logger.info(f"Feature statistics - mean: {mean:.4f}, std: {std:.4f}")
            break
    
    # If scaler_params exists in the results, log it
    if 'scaler_params' in dataloaders:
        logger.info("Standardization parameters available for inverse transformation")

