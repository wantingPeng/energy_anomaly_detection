"""
Sliding window feature extraction for XGBoost model using raw energy data.

This script loads interpolated data from contact datasets, creates sliding windows
with 60-second windows, and calculates statistical features for each window.
"""

import os
import gc
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import glob
from datetime import datetime
import dask.dataframe as dd

from src.utils.logger import logger
from src.preprocessing.energy.machine_learning.calculate_window_features import calculate_window_features


def load_dataset(data_dir):
    """
    Load parquet files from specified directory.
    
    Args:
        data_dir: Path to data directory
        
    Returns:
        pandas DataFrame containing loaded data
    """
    if not os.path.exists(data_dir):
        logger.error(f"Directory not found: {data_dir}")
        return pd.DataFrame()
    
    logger.info(f"Loading data from {data_dir}")
    
    # Get all parquet files in the directory (including subdirectories)
    parquet_files = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.parquet'):
                parquet_files.append(os.path.join(root, file))
    
    if not parquet_files:
        logger.warning(f"No parquet files found in {data_dir}")
        return pd.DataFrame()
    
    logger.info(f"Found {len(parquet_files)} parquet files")
    
    # Load data in chunks to avoid memory issues
    dfs = []
    for file in tqdm(parquet_files, desc="Loading parquet files"):
        try:
            df = pd.read_parquet(file)
            dfs.append(df)
        except Exception as e:
            logger.error(f"Error loading {file}: {str(e)}")
    
    if not dfs:
        logger.warning("No data loaded")
        return pd.DataFrame()
    
    # Concatenate all dataframes
    combined_df = pd.concat(dfs, ignore_index=True)
    logger.info(f"Loaded {len(combined_df)} rows of data")
    
    return combined_df


def create_windows_by_segment(df, window_size=60, min_points=5):
    """
    Create fixed-size windows per segment_id.
    
    Args:
        df: DataFrame containing time series data
        window_size: Size of window in seconds
        min_points: Minimum number of data points required in a window
        
    Returns:
        List of window DataFrames
    """
    # Sort by TimeStamp
    df = df.sort_values('TimeStamp')
    
    # Convert TimeStamp to datetime if it's not already
    if not pd.api.types.is_datetime64_any_dtype(df['TimeStamp']):
        df['TimeStamp'] = pd.to_datetime(df['TimeStamp'])
    
    # Group by segment_id
    segments = df.groupby('segment_id')
    
    windows = []
    
    for segment_id, segment_df in segments:
        # Sort by timestamp
        segment_df = segment_df.sort_values('TimeStamp')
        
        # Get first and last timestamp
        first_timestamp = segment_df['TimeStamp'].min()
        last_timestamp = segment_df['TimeStamp'].max()
        
        # Create windows
        start_time = first_timestamp
        
        while start_time + pd.Timedelta(seconds=window_size) <= last_timestamp:
            end_time = start_time + pd.Timedelta(seconds=window_size)
            
            # Get data for this window
            window_mask = (segment_df['TimeStamp'] >= start_time) & (segment_df['TimeStamp'] < end_time)
            window_df = segment_df.loc[window_mask].copy()
            
            # Add window to list if it has sufficient data
            if not window_df.empty and len(window_df) >= min_points:
                windows.append(window_df)
            
            # Move to next window
            start_time += pd.Timedelta(seconds=window_size)
    
    logger.info(f"Created {len(windows)} windows")
    return windows


def process_contact_data():
    """
    Process contact data from the interpolated dataset, create sliding windows,
    and calculate statistical features for each window.
    
    Data is loaded from Data/deepLearning/transform/interpolated/{split}/contact
    and results are saved to Data/row_energyData_subsample_xgboost/slidingWindow/contact
    """
    # Define paths
    base_input_dir = "Data/deepLearning/transform/interpolated"
    output_dir = "Data/row_energyData_subsample_xgboost/slidingWindow/contact"
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each split in order: train, val, test
    splits = ["train", "val", "test"]
    
    for split in splits:
        logger.info(f"Processing {split} split")
        
        # Define input directory for this split
        input_dir = os.path.join(base_input_dir, split, "contact")
        
        # Load dataset
        df = load_dataset(input_dir)
        
        if df.empty:
            logger.warning(f"No data loaded for {split} split")
            continue
        
        # Create sliding windows
        windows = create_windows_by_segment(df, window_size=60, min_points=5)
        
        # Free memory
        del df
        gc.collect()
        
        if not windows:
            logger.warning(f"No windows created for {split} split")
            continue
        
        # Calculate features for each window
        window_features = []
        
        for window_idx, window in enumerate(tqdm(windows, desc=f"Processing {split} windows")):
            try:
                features = calculate_window_features(window)
                window_features.append(features)
            except Exception as e:
                logger.error(f"Error processing window {window_idx}: {str(e)}")
                continue
        
        # Free memory
        del windows
        gc.collect()
        
        if not window_features:
            logger.warning(f"No valid features calculated for {split} split")
            continue
        
        # Create DataFrame from features
        features_df = pd.DataFrame(window_features)
        
        # Save features as parquet
        output_file = os.path.join(output_dir, f"{split}_features.parquet")
        features_df.to_parquet(output_file)
        logger.info(f"Saved features for {split} split: {len(features_df)} rows to {output_file}")


if __name__ == "__main__":
    # Configure logger
    logger.info("Starting contact data processing")
    process_contact_data()
    logger.info("Contact data processing completed")
