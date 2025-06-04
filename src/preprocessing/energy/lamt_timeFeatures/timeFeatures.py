import os
import glob
import numpy as np
import pandas as pd
import dask.dataframe as dd
from datetime import datetime
import math
from pathlib import Path
from src.utils.logger import logger

def get_all_parquet_files(base_dir):
    """
    Get all parquet files from the given directory and its subdirectories.
    
    Args:
        base_dir (str): Base directory to search for parquet files
        
    Returns:
        list: List of parquet file paths
    """
    logger.info(f"Finding all parquet files in {base_dir}")
    return glob.glob(os.path.join(base_dir, "**", "*.parquet"), recursive=True)

def read_and_combine_data(file_paths):
    """
    Read and combine all parquet files into a single Dask DataFrame.
    
    Args:
        file_paths (list): List of parquet file paths
        
    Returns:
        dask.dataframe.DataFrame: Combined Dask DataFrame
    """
    logger.info(f"Reading {len(file_paths)} parquet files")
    ddf = dd.read_parquet(file_paths)
    return ddf

def sort_by_timestamp(ddf):
    """
    Sort the DataFrame by TimeStamp.
    
    Args:
        ddf (dask.dataframe.DataFrame): Input Dask DataFrame
        
    Returns:
        dask.dataframe.DataFrame: Sorted Dask DataFrame
    """
    logger.info("Sorting data by TimeStamp")
    return ddf.sort_values('TimeStamp')

def calculate_time_features(ddf):
    """
    Calculate time features from TimeStamp column.
    
    Args:
        ddf (dask.dataframe.DataFrame): Input Dask DataFrame
        
    Returns:
        dask.dataframe.DataFrame: DataFrame with added time features
    """
    logger.info("Calculating time features")
    
    # Extract month (1-12)
    ddf['month'] = ddf['TimeStamp'].dt.month
    
    # Cyclical encoding for month
    ddf['month_sin'] = np.sin(2 * np.pi * ddf['month'] / 12)
    ddf['month_cos'] = np.cos(2 * np.pi * ddf['month'] / 12)
    
    # Is weekend (1 if Saturday or Sunday, 0 otherwise)
    ddf['is_weekend'] = ddf['TimeStamp'].dt.dayofweek.isin([5, 6]).astype(int)
    
    # Extract hour (0-23)
    ddf['hour'] = ddf['TimeStamp'].dt.hour
    
    # Cyclical encoding for hour
    ddf['hour_sin'] = np.sin(2 * np.pi * ddf['hour'] / 24)
    ddf['hour_cos'] = np.cos(2 * np.pi * ddf['hour'] / 24)
    
    return ddf

def save_time_features(ddf, output_dir):
    """
    Save the DataFrame with time features to parquet files.
    
    Args:
        ddf (dask.dataframe.DataFrame): DataFrame with time features
        output_dir (str): Output directory
        
    Returns:
        None
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info(f"Saving time features to {output_dir}")
    ddf.to_parquet(output_dir, write_index=False)
    logger.info(f"Time features saved successfully to {output_dir}")

def process_component_batch(split, component_type, batch_dir):
    """
    Process a single batch directory for a specific component type.
    
    Args:
        split (str): Split name ('train', 'val', or 'test')
        component_type (str): Component type (e.g., 'contact')
        batch_dir (str): Batch directory name (e.g., 'batch_0')
        
    Returns:
        None
    """
    input_dir = f"Data/processed/lsmt/interpolated/{split}/{component_type}/{batch_dir}"
    output_dir = f"Data/processed/lamt_timeFeatures/add_timeFeatures/{split}/{component_type}/{batch_dir}"
    
    logger.info(f"Processing {split}/{component_type}/{batch_dir}")
    
    # Get all parquet files for this batch
    file_paths = get_all_parquet_files(input_dir)
    logger.info(f"Found {len(file_paths)} parquet files for {split}/{component_type}/{batch_dir}")
    
    if not file_paths:
        logger.warning(f"No parquet files found for {split}/{component_type}/{batch_dir}")
        return
    
    # Read and combine data
    ddf = read_and_combine_data(file_paths)
    
    # Sort by timestamp
    ddf = sort_by_timestamp(ddf)
    
    # Calculate time features
    ddf_with_features = calculate_time_features(ddf)
    
    # Save time features
    save_time_features(ddf_with_features, output_dir)
    
    logger.info(f"Completed processing {split}/{component_type}/{batch_dir}")

def process_component(split, component_type):
    """
    Process all batches for a specific component type.
    
    Args:
        split (str): Split name ('train', 'val', or 'test')
        component_type (str): Component type (e.g., 'contact')
        
    Returns:
        None
    """
    component_dir = f"Data/processed/lsmt/interpolated/{split}/{component_type}"
    
    # Get all batch directories
    batch_dirs = [d for d in os.listdir(component_dir) if os.path.isdir(os.path.join(component_dir, d))]
    logger.info(f"Found {len(batch_dirs)} batch directories for {split}/{component_type}")
    
    for batch_dir in batch_dirs:
        process_component_batch(split, component_type, batch_dir)

def process_split(split):
    """
    Process a single data split (train, val, or test).
    
    Args:
        split (str): Split name ('train', 'val', or 'test')
        
    Returns:
        None
    """
    split_dir = f"Data/processed/lsmt_statisticalFeatures/interpolate/{split}"
    
    if not os.path.exists(split_dir):
        logger.warning(f"Split directory {split_dir} does not exist")
        return
    
    logger.info(f"Processing {split} split")
    
    # Get all component type directories
    component_types = [d for d in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, d))]
    logger.info(f"Found {len(component_types)} component types for {split} split")
    
    for component_type in component_types:
        process_component(split, component_type)
    
    logger.info(f"Completed processing {split} split")

def process_time_features():
    """
    Main function to process time features.
    """
    # Configure logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"experiments/logs/time_features_{timestamp}.log"
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    logger.info("Starting time features processing")
    
    try:
        # Process each split separately
        for split in ['train', 'val', 'test']:
            process_split(split)
        
        logger.info("Time features processing completed successfully")
        
    except Exception as e:
        logger.error(f"Error processing time features: {str(e)}")
        raise

if __name__ == "__main__":
    process_time_features()
