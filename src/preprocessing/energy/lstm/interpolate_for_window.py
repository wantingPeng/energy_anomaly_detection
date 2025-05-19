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
import pyarrow.parquet as pq
from datetime import datetime, timedelta
from pathlib import Path
from intervaltree import IntervalTree
from tqdm import tqdm
import torch
import logging
from typing import Dict, List, Tuple, Union, Optional
from joblib import Parallel, delayed
from glob import glob
from src.utils.logger import logger
from src.utils.memory_left import log_memory
from src.preprocessing.energy.lstm.dataset import LSTMWindowDataset
from src.preprocessing.energy.labeling_slidingWindow import (
    load_anomaly_dict, 
    create_interval_tree,
    calculate_window_overlap
)


def load_config() -> dict:
    """Load configuration from YAML file."""
    config_path = Path("configs/interpolat.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def is_valid_parquet(file_path: str) -> bool:
    """Check if a file is a valid parquet file."""
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



def split_by_component(df: pd.DataFrame, dir: str, split: str) -> List[str]:
    """
    Split data by component type and save each component to a temporary parquet file.
    
    Args:
        df: Input DataFrame
        temp_dir: Directory to save temporary component files
        
    Returns:
        List of component names in order they were processed
    """
    logger.info("Splitting data by component type")
    
    # Create temp directory if it doesn't exist
    os.makedirs(dir, exist_ok=True)
    
    component_names = []
    
    # Check which component type columns exist
    component_cols = [col for col in df.columns if col.startswith('component_type_')]
    logger.info(f"Found component columns: {component_cols}")
    
    for col in component_cols:
        # Extract component name from column name
        component = col.replace('component_type_', '')
        
        # Filter rows where this component type is 1
        component_df = df[df[col] == 1].copy()
        
        if not component_df.empty:
            # Sort by timestamp
            component_df = component_df.sort_values('TimeStamp')
            
            # Save to temporary parquet file
            file = os.path.join("Data/processed/lsmt/split_by_component", f"{component}_{split}.parquet")
            component_df.to_parquet(file)
            
            component_names.append(component)
            logger.info(f"Saved {len(component_df)} rows for component {component} to {file}")
            
            # Free memory
            del component_df
            gc.collect()
    
    return component_names


def get_component_df(component: str, dir: str, split: str) -> pd.DataFrame:
    """
    Load a specific component's DataFrame from temporary storage.
    
    Args:
        component: Component name to load
        temp_dir: Directory containing temporary component files
        
    Returns:
        DataFrame for the specified component
    """
    temp_file = os.path.join(dir, f"{component}_{split}.parquet")
    logger.info(f"Loading component data from {temp_file}")
    
    if not os.path.exists(temp_file):
        raise FileNotFoundError(f"Temporary file for component {component} not found")
    
    return pd.read_parquet(temp_file)


def interpolate_segments(df: pd.DataFrame, output_dir: str, split: str, component: str) -> None:
    """
    Interpolate segments using Dask and save the results.
    
    Args:
        df: Input DataFrame
        output_dir: Directory to save interpolated segments
    """
    logger.info("Interpolating segments using Dask")
    
    # Get list of unique segment IDs
    unique_segments = df['segment_id'].unique()
    logger.info(f"Processing {len(unique_segments)} unique segments")
    
    # Get feature columns (exclude non-feature columns)
    exclude_cols = ['TimeStamp', 'segment_id'] + [col for col in df.columns if col.startswith('component_type_')]
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    # Create a list to store processed segments
    batch_size = 50  # Process 50 segments at a time
    
    for batch_start in range(0, len(unique_segments), batch_size):
        batch_segments = unique_segments[batch_start:batch_start + batch_size]
        logger.info(f"Processing batch of segments {batch_start} to {batch_start + len(batch_segments)}")
        
        batch_dfs = []
        for segment_id in tqdm(batch_segments, desc="Processing segments in batch"):
            segment_df = df[df['segment_id'] == segment_id].copy()
            
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
                time_range = pd.date_range(start=start_time, end=end_time, freq='1s')
                
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
            
            # Remove unnecessary columns
            segment_df = segment_df.drop(columns=['prev_timestamp', 'time_diff'], errors='ignore')
            batch_dfs.append(segment_df)
        
        # Combine batch DataFrames
        if batch_dfs:
            batch_df = pd.concat(batch_dfs, ignore_index=True)
            # Convert to Dask DataFrame
            ddf = dd.from_pandas(batch_df, npartitions=min(20, len(batch_segments)))
            dir = os.path.join(output_dir, split, component)
            os.makedirs(dir, exist_ok=True)

            # Save batch using Dask
            batch_output_dir = os.path.join(dir, f'batch_{batch_start}')
            ddf.to_parquet(
                batch_output_dir,
                engine='pyarrow',
                write_index=False,
            )
            
            # Clean up memory
            del batch_dfs, batch_df, ddf
            gc.collect()
            log_memory(f"after interpolating segments batch")

    logger.info("All segments processed and saved")





def process_split(
    split: str,
    config: dict
) -> None:
    """
    Process a single data split (train/val/test).
    
    Args:
        split: Data split name (train/val/test)
        config: Configuration dictionary
    """
    logger.info(f"Processing {split} data")
    log_memory(f"Before {split}")
    
    # Load data
    input_path = os.path.join(config['paths']['input_dir'], f"{split}.parquet")
    df = load_data(input_path)
    log_memory(f"After loading {split}")
    
    # Split by component type and save to temp files
    component_names = split_by_component(df, 'Data/processed/lsmt/split_by_component', split)
    
    # Free memory
    del df
    gc.collect()
    log_memory(f"After splitting {split}")
    

    # Process components in specified order
    for component in component_names:
        logger.info(f"Processing component: {component}")
        # Get component DataFrame
        component_df = get_component_df(component, 'Data/processed/lsmt/split_by_component', split)
        
        # Define paths for interpolated and sliding window data
        interpolated_output_dir = os.path.join(config['paths']['output_dir'], 'interpolated')
        log_memory(f"Before interpolating segments for {component} ({split})")
        # Interpolate segments
        interpolate_segments(component_df, interpolated_output_dir,split,component)
        log_memory(f"After interpolating segments for {component} ({split})")

        del component_df
        gc.collect()
        log_memory(f"After processing {component} ({split})")


def main():
    """Main function to process all data splits."""
    
    # Load configuration
    try:
        config = load_config()
        logger.info("Configuration loaded successfully")
    except FileNotFoundError:
        logger.error("Configuration file not found at configs/lsmt_preprocessing.yaml")
        raise
    except Exception as e:
        logger.error(f"Error loading configuration: {str(e)}")
        raise
 
    # Create output directories
    os.makedirs(config['paths']['output_dir'], exist_ok=True)
    
    try:      
        # Process each split
        for split in ['train', 'val', 'test']:
            process_split(split, config)
        
        logger.info("Sliding window preprocessing completed successfully")
        
    except Exception as e:
        logger.error(f"Error during processing: {str(e)}")
        raise



if __name__ == "__main__":
    main()
