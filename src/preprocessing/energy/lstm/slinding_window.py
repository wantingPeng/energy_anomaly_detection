"""
Sliding window preprocessing for LSTM model training data.

This script processes energy data by creating sliding windows for LSTM training.
It handles segmentation by component type, ensures time continuity within segments,
and properly labels windows based on anomaly overlap percentage.
"""

import os
import gc
import yaml
import pickle
import pandas as pd
import numpy as np
import dask.dataframe as dd
from datetime import datetime, timedelta
from pathlib import Path
from intervaltree import IntervalTree
from tqdm import tqdm
import torch
from typing import Dict, List, Tuple, Union, Optional

from src.utils.logger import logger
from src.utils.memory_left import log_memory
from src.preprocessing.energy.lstm.dataset import LSTMWindowDataset
from src.preprocessing.energy.labeling_slidingWindow import load_anomaly_dict, create_interval_tree


def load_config() -> dict:
    """Load configuration from YAML file."""
    config_path = Path("configs/lsmt_preprocessing.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


import os
import dask.dataframe as dd
import pandas as pd
from src.utils.logger import logger

import os
import dask.dataframe as dd
import pandas as pd
import pyarrow.parquet as pq
from src.utils.logger import logger

def is_valid_parquet(file_path: str) -> bool:
    try:
        pq.ParquetFile(file_path)
        return True
    except Exception:
        return False

def load_data(path: str) -> pd.DataFrame:
    """
    Load data from a directory of part-files or a single .parquet file, with corruption check.
    """
    logger.info(f"Loading data from {path}")
    try:
        if os.path.isdir(path):
            logger.info(f"Path is a directory: {path}")
            part_files = [f for f in os.listdir(path) if f.endswith('.parquet')]
            full_paths = [os.path.join(path, f) for f in part_files]

            # 检查哪些是合法 parquet 文件
            valid_files = [f for f in full_paths if is_valid_parquet(f)]

            if not valid_files:
                raise RuntimeError("No valid .parquet files found in directory.")

            logger.info(f"Found {len(valid_files)} valid parquet files in directory")

            # 用 Dask 读多个文件
            ddf = dd.read_parquet(valid_files, engine="pyarrow")

        else:
            logger.info(f"Path is a file: {path}")
            if not is_valid_parquet(path):
                raise ValueError(f"File {path} is not a valid parquet file.")
            ddf = dd.read_parquet(path, engine="pyarrow")

        logger.info(f"Dask dataframe has {len(ddf.columns)} columns and {len(ddf.divisions)-1} partitions")
        logger.info("Computing Dask dataframe into Pandas...")
        df = ddf.compute()
        logger.info(f"Loaded DataFrame with shape: {df.shape}")
        return df

    except Exception as e:
        logger.error(f"Error loading data from {path}: {e}")
        raise



def split_by_component(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Split data by component type.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Dictionary mapping component type to DataFrame
    """
    logger.info("Splitting data by component type")
    component_dfs = {}
    
    # Check which component type columns exist
    component_cols = [col for col in df.columns if col.startswith('component_type_')]
    
    for col in component_cols:
        # Extract component name from column name
        component = col.replace('component_type_', '')
        
        # Filter rows where this component type is 1
        component_df = df[df[col] == 1].copy()
        
        if not component_df.empty:
            # Sort by timestamp
            component_df = component_df.sort_values('TimeStamp')
            component_dfs[component] = component_df
            logger.info(f"Found {len(component_df)} rows for component {component}")
    
    return component_dfs


def create_sliding_windows(
    df: pd.DataFrame, 
    window_size: int, 
    step_size: int,
    anomaly_trees: Dict[str, IntervalTree],
    anomaly_threshold: float
) -> Tuple[List[np.ndarray], List[int], List[int], List[str], Dict[str, int]]:
    """
    Create sliding windows from DataFrame and label them based on anomaly overlap.
    
    Args:
        df: Input DataFrame sorted by TimeStamp
        window_size: Window size in seconds
        step_size: Step size in seconds
        anomaly_trees: Dictionary mapping station to IntervalTree of anomaly intervals
        anomaly_threshold: Threshold for anomaly labeling (proportion of window that must be anomalous)
        
    Returns:
        Tuple containing:
        - List of window features as numpy arrays
        - List of window labels
        - List of segment IDs
        - List of window start timestamps as strings
        - Statistics dictionary
    """
    logger.info(f"Creating sliding windows (size={window_size}s, step={step_size}s)")
    
    windows = []
    labels = []
    segment_ids = []
    timestamps = []
    
    stats = {
        "total_windows": 0,
        "anomaly_windows": 0,
        "skipped_segments": 0,
        "min_window_length": float('inf'),
        "max_window_length": 0,
    }
    
    # Get list of unique segment IDs
    unique_segments = df['segment_id'].unique()
    logger.info(f"Processing {len(unique_segments)} unique segments")
    
    # Get feature columns (exclude non-feature columns)
    exclude_cols = ['TimeStamp', 'segment_id'] + [col for col in df.columns if col.startswith('component_type_')]
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    # Process each segment
    for segment_id in tqdm(unique_segments, desc="Processing segments"):
        segment_df = df[df['segment_id'] == segment_id].copy()
        
        # Skip very small segments
        if len(segment_df) < window_size:
            stats["skipped_segments"] += 1
            continue
        
        # Sort by timestamp
        segment_df = segment_df.sort_values('TimeStamp')
        
        # Check for time continuity
        segment_df['prev_timestamp'] = segment_df['TimeStamp'].shift(1)
        segment_df['time_diff'] = (segment_df['TimeStamp'] - segment_df['prev_timestamp']).dt.total_seconds()
        
        # If there are gaps, fill them using linear interpolation
        has_gaps = (segment_df['time_diff'].fillna(1) > 1).any()
        if has_gaps:
            # Create a continuous time range with 1-second intervals
            start_time = segment_df['TimeStamp'].min()
            end_time = segment_df['TimeStamp'].max()
            time_range = pd.date_range(start=start_time, end=end_time, freq='1S')
            
            # Create a new DataFrame with the continuous time range
            continuous_df = pd.DataFrame({'TimeStamp': time_range})
            
            # Merge with the original data
            continuous_df = pd.merge_asof(
                continuous_df,
                segment_df[['TimeStamp'] + feature_cols],
                on='TimeStamp',
                direction='nearest'
            )
            
            # Set segment_id
            continuous_df['segment_id'] = segment_id
            
            # Replace original segment DataFrame
            segment_df = continuous_df
        
        # Create sliding windows
        for i in range(0, len(segment_df) - window_size + 1, step_size):
            window_df = segment_df.iloc[i:i + window_size]
            
            # Ensure window has exact length
            if len(window_df) != window_size:
                continue
            
            # Get window features
            window = window_df[feature_cols].values
            
            # Update stats
            stats["min_window_length"] = min(stats["min_window_length"], window.shape[0])
            stats["max_window_length"] = max(stats["max_window_length"], window.shape[0])
            
            # Get window start and end timestamps
            window_start = window_df['TimeStamp'].iloc[0]
            window_end = window_df['TimeStamp'].iloc[-1]
            
            # Check for anomaly overlap
            is_anomaly = 0
            for station, tree in anomaly_trees.items():
                # Calculate window overlap using the same method as in labeling_slidingWindow.py
                window_start_int = int(window_start.timestamp())
                window_end_int = int(window_end.timestamp())
                
                # Find overlapping intervals
                overlapping_intervals = tree[window_start_int:window_end_int]
                
                if overlapping_intervals:
                    # Calculate total overlap duration
                    total_overlap = 0
                    for interval in overlapping_intervals:
                        overlap_start = max(window_start_int, interval.begin)
                        overlap_end = min(window_end_int, interval.end)
                        total_overlap += overlap_end - overlap_start
                    
                    # Calculate overlap ratio
                    window_duration = window_end_int - window_start_int
                    overlap_ratio = total_overlap / window_duration
                    
                    # Mark as anomaly if overlap ratio exceeds threshold
                    if overlap_ratio >= anomaly_threshold:
                        is_anomaly = 1
                        break
            
            # Store window, label, segment_id, and start timestamp
            windows.append(window)
            labels.append(is_anomaly)
            segment_ids.append(segment_id)
            timestamps.append(window_start.strftime('%Y-%m-%d %H:%M:%S'))
            
            # Update statistics
            stats["total_windows"] += 1
            if is_anomaly:
                stats["anomaly_windows"] += 1
    
    return windows, labels, segment_ids, timestamps, stats


def save_results(
    windows: List[np.ndarray],
    labels: List[int],
    segment_ids: List[int],
    timestamps: List[str],
    output_path: str,
    component: str,
    split: str
) -> None:
    """
    Save results as parquet file.
    
    Args:
        windows: List of window features
        labels: List of window labels
        segment_ids: List of segment IDs
        timestamps: List of window start timestamps
        output_path: Output directory path
        component: Component type
        split: Data split (train/val/test)
    """
    logger.info(f"Saving results for {component} ({split})")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    
    # Stack windows to create a 3D array
    windows_array = np.stack(windows)
    
    # Convert to pandas DataFrame for saving
    windows_df = pd.DataFrame({
        'window': [w.tobytes() for w in windows],
        'label': labels,
        'segment_id': segment_ids,
        'start_timestamp': timestamps,
        'window_shape': [str(w.shape) for w in windows]
    })
    
    # Save as parquet
    output_file = os.path.join(output_path, f"window_{component}_{split}.parquet")
    windows_df.to_parquet(output_file)
    logger.info(f"Saved to {output_file}")


def save_stats(stats: Dict[str, int], output_path: str, component: str, split: str) -> None:
    """
    Save statistics as markdown.
    
    Args:
        stats: Statistics dictionary
        output_path: Output directory path
        component: Component type
        split: Data split (train/val/test)
    """
    logger.info(f"Saving statistics for {component} ({split})")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    
    # Create markdown content
    markdown = f"# Sliding Window Statistics - {component.capitalize()} - {split.capitalize()}\n\n"
    markdown += f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    markdown += "## Summary\n\n"
    markdown += f"- Total windows: {stats['total_windows']}\n"
    markdown += f"- Anomaly windows: {stats['anomaly_windows']} ({stats['anomaly_windows']/max(stats['total_windows'], 1)*100:.2f}%)\n"
    markdown += f"- Skipped segments: {stats['skipped_segments']}\n"
    markdown += f"- Min window length: {stats['min_window_length']}\n"
    markdown += f"- Max window length: {stats['max_window_length']}\n"
    
    # Save as markdown
    output_file = os.path.join(output_path, f"{split}.md")
    with open(output_file, "w") as f:
        f.write(markdown)
    logger.info(f"Statistics saved to {output_file}")


def process_split(
    split: str,
    config: dict,
    anomaly_trees: Dict[str, IntervalTree]
) -> None:
    """
    Process a single data split (train/val/test).
    
    Args:
        split: Data split name (train/val/test)
        config: Configuration dictionary
        anomaly_trees: Dictionary of anomaly interval trees
    """
    logger.info(f"Processing {split} data")
    log_memory(f"Before {split}")
    
    # Load data
    input_path = os.path.join(config['paths']['input_dir'], f"{split}.parquet")
    df = load_data(input_path)
    log_memory(f"After loading {split}")
    
    # Split by component type
    component_dfs = split_by_component(df)
    
    # Free memory
    del df
    gc.collect()
    log_memory(f"After splitting {split}")
    
    # Process each component
    for component, component_df in component_dfs.items():
        logger.info(f"Processing component: {component}")
        
        # Create sliding windows
        windows, labels, segment_ids, timestamps, stats = create_sliding_windows(
            component_df,
            config['sliding_window']['window_size'],
            config['sliding_window']['step_size'],
            anomaly_trees,
            config['sliding_window']['anomaly_threshold']
        )
        log_memory(f"After creating windows for {component} ({split})")
        
        # Save results
        save_results(
            windows,
            labels,
            segment_ids,
            timestamps,
            config['paths']['output_dir'],
            component,
            split
        )
        
        # Save statistics
        save_stats(
            stats,
            config['paths']['reports_dir'],
            component,
            split
        )
        
        # Free memory
        del windows, labels, segment_ids, timestamps, stats
        gc.collect()
        log_memory(f"After saving {component} ({split})")
        
        # Test loading a dataset
        output_file = os.path.join(config['paths']['output_dir'], f"window_{component}_{split}.parquet")
        try:
            dataset = LSTMWindowDataset(output_file)
            logger.info(f"Successfully created dataset with {len(dataset)} samples")
        except Exception as e:
            logger.error(f"Error creating dataset: {e}")


def main():
    """Main function to process all data splits."""
    logger.info("Starting sliding window preprocessing")
    
    # Load configuration
    config = load_config()
    logger.info(f"Loaded configuration: {config}")
    
    # Create output directories
    os.makedirs(config['paths']['output_dir'], exist_ok=True)
    os.makedirs(config['paths']['reports_dir'], exist_ok=True)
    
    # Create a temporary config for loading anomaly dict
    anomaly_config = {'paths': {'anomaly_dict': config['paths']['anomaly_dict']}}
    
    # Load anomaly dictionary
    anomaly_dict = load_anomaly_dict(anomaly_config)
    
    # Convert to interval trees
    anomaly_trees = {station: create_interval_tree(periods) for station, periods in anomaly_dict.items()}
    
    # Process each split
    for split in ['train', 'val', 'test']:
        process_split(split, config, anomaly_trees)
    
    logger.info("Sliding window preprocessing completed")


if __name__ == "__main__":
    main()
