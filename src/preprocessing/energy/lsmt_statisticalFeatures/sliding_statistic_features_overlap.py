"""
Sliding window feature extraction for LSTM model training data with anomaly overlap calculation.

This script loads interpolated data, creates sliding windows with specified 
window size and step size, calculates statistical features for each window,
and adjusts step size based on anomaly overlap ratio.
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
from sklearn.preprocessing import StandardScaler
import joblib
import yaml
import pickle
from intervaltree import IntervalTree
from typing import Dict, List, Tuple

from src.utils.logger import logger
from src.preprocessing.energy.machine_learning.calculate_window_features import calculate_window_features
from src.preprocessing.energy.machine_learning.labeling_slidingWindow import (
    load_anomaly_dict,
    create_interval_tree,
    calculate_window_overlap
)


def load_top_features():
    """
    Load the top 50 features from the saved file.
    
    Returns:
        List of top feature names
    """
    try:
        top_features_df = pd.read_parquet("Data/processed/machinen_learning/top_50_features.parquet")
        # Get all column names except 'segment_id', 'anomaly_label', and component type columns
        feature_cols = [col for col in top_features_df.columns 
                      if col not in ['segment_id', 'anomaly_label'] 
                      and not col.startswith('component_')]
        logger.info(f"Loaded {len(feature_cols)} top features")
        return feature_cols
    except Exception as e:
        logger.error(f"Error loading top features: {str(e)}")
        return []


def filter_features(df, top_features):
    """
    Filter DataFrame to only keep the top features plus metadata columns.
    
    Args:
        df: DataFrame containing all calculated features
        top_features: List of top feature names to keep
        
    Returns:
        Filtered DataFrame
    """
    # Always keep these metadata columns
    metadata_cols = ['window_start', 'window_end', 'segment_id', 'overlap_ratio']
    
    # Create a list of columns to keep
    keep_cols = metadata_cols + [col for col in top_features if col in df.columns]
    
    # Check if all top features are present
    missing_features = [col for col in top_features if col not in df.columns]
    if missing_features:
        logger.warning(f"Missing {len(missing_features)} top features in the DataFrame")
        
    # Filter DataFrame
    filtered_df = df[keep_cols].copy()
    logger.info(f"Filtered features from {df.shape[1]} to {filtered_df.shape[1]} columns")
    
    return filtered_df


def get_batch_dirs(data_type, component):
    """
    Get all batch directories for a specific data type and component.
    
    Args:
        data_type: Data type ('train', 'val', or 'test')
        component: Component type ('contact', 'pcb', or 'ring')
        
    Returns:
        List of batch directory paths
    """
    base_dir = f"Data/processed/lsmt_timeFeatures/add_timeFeatures/{data_type}/{component}"
    batch_dirs = sorted(glob.glob(os.path.join(base_dir, "batch_*")))
    logger.info(f"Found {len(batch_dirs)} batch directories for {data_type}/{component}")
    return batch_dirs


def load_batch_data(batch_dir):
    """
    Load data from a batch directory.
    
    Args:
        batch_dir: Path to batch directory
        
    Returns:
        DataFrame containing batch data
    """
    # List all parquet files in the batch directory
    parquet_files = glob.glob(os.path.join(batch_dir, "*.parquet"))
    
    # Read all parquet files into a list of DataFrames
    dfs = []
    for file in parquet_files:
        try:
            df = pd.read_parquet(file)
            dfs.append(df)
        except Exception as e:
            logger.error(f"Error reading file {file}: {str(e)}")
    
    # Concatenate all DataFrames
    if dfs:
        return pd.concat(dfs, ignore_index=True)
    else:
        return pd.DataFrame()


def create_sliding_windows(df, anomaly_trees, component, window_size=800, step_size=800, 
                          anomaly_step_size=100, overlap_threshold=0.5):
    """
    Create sliding windows from DataFrame with dynamic step size based on anomaly overlap.
    
    Args:
        df: DataFrame containing time series data
        anomaly_trees: Dictionary of interval trees for anomaly detection
        component: Component type ('contact', 'pcb', or 'ring')
        window_size: Size of sliding window in seconds
        step_size: Normal step size for window sliding in seconds
        anomaly_step_size: Step size for windows with anomalies
        overlap_threshold: Threshold for considering a window as having an anomaly
        
    Returns:
        List of tuples containing (window_df, overlap_ratio)
    """
    # Sort by TimeStamp
    df = df.sort_values('TimeStamp')
    
    # Convert TimeStamp to datetime if it's not already
    if not pd.api.types.is_datetime64_any_dtype(df['TimeStamp']):
        df['TimeStamp'] = pd.to_datetime(df['TimeStamp'])
    
    # Map component to station name
    component_to_station = {
        'contact': 'Kontaktieren',
        'ring': 'Ringmontage',
        'pcb': 'Pcb'
    }
    
    # Get station name
    station = component_to_station.get(component)
    if station not in anomaly_trees:
        logger.warning(f"No anomaly data for station {station}")
        interval_tree = IntervalTree()  # Empty tree
    else:
        interval_tree = anomaly_trees[station]
    
    # Group by segment_id
    segments = df.groupby('segment_id')
    
    windows = []
    window_count = 0
    
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
            
            # Calculate overlap with anomalies
            overlap_ratio = calculate_window_overlap(start_time, end_time, interval_tree)
            
            # Determine step size based on overlap ratio
            current_step_size = anomaly_step_size if overlap_ratio > overlap_threshold else step_size
            
            # Get data for this window
            window_mask = (segment_df['TimeStamp'] >= start_time) & (segment_df['TimeStamp'] < end_time)
            window_df = segment_df.loc[window_mask].copy()
            
            # Add window to list if it's not empty
            if not window_df.empty:
                windows.append((window_df, overlap_ratio))
                window_count += 1
            
            # Move to next window
            start_time += pd.Timedelta(seconds=current_step_size)
    
    logger.info(f"Created {window_count} windows from {len(segments)} segments")
    return windows


def process_batch(batch_dir, output_dir, top_features, anomaly_trees, component):
    """
    Process a batch of data: load, create sliding windows, and calculate features.
    
    Args:
        batch_dir: Path to batch directory
        output_dir: Path to output directory
        top_features: List of top feature names to keep
        anomaly_trees: Dictionary of interval trees for anomaly detection
        component: Component type ('contact', 'pcb', or 'ring')
        config: Configuration dictionary
        
    Returns:
        Number of windows processed
    """
    try:

        # Load batch data
        logger.info(f"Loading data from {batch_dir}")
        batch_df = load_batch_data(batch_dir)
        
        if batch_df.empty:
            logger.warning(f"No data loaded from {batch_dir}")
            return 0
        
        # Create sliding windows with overlap calculation
        logger.info(f"Creating sliding windows with window_size=800s, step_size=800s, "
                  f"anomaly_step_size=100s, overlap_threshold=0.7")
        windows = create_sliding_windows(
            batch_df, anomaly_trees, component, 800, 800, 100, 0.7
        )
        
        # Free memory
        del batch_df
        gc.collect()
        
        if not windows:
            logger.warning(f"No windows created from {batch_dir}")
            return 0
        
        # Calculate features for each window
        logger.info(f"Calculating features for {len(windows)} windows")
        window_features = []
        
        for window_df in tqdm(windows, desc="Processing windows"):
            features = calculate_window_features(window_df)
            # Add overlap ratio to features
            window_features.append(features)
        
        # Free memory
        del windows
        gc.collect()
        
        # Create DataFrame from features
        features_df = pd.DataFrame(window_features)
        
        # Filter to keep only top features
        if top_features:
            features_df = filter_features(features_df, top_features)
        
        # Save features as parquet
        batch_name = os.path.basename(batch_dir)

        
        # Extract only numerical features for the statistical features
        metadata_cols = ['window_start', 'window_end', 'segment_id', 'overlap_ratio']
        feature_cols = [col for col in features_df.columns if col not in metadata_cols]
        
        # Apply standardization to numerical features
        # Create a copy to avoid modifying the original DataFrame
        stat_features_df = features_df[feature_cols].copy()
        
        # Apply StandardScaler
        scaler = StandardScaler()
        stat_features = scaler.fit_transform(stat_features_df)
        
        logger.info(f"Applied standardization to {stat_features.shape[1]} features")
        
        # Save as numpy array with standardized features
        np_output_file = os.path.join(output_dir, f"{batch_name}.npz")
        np.savez_compressed(
            np_output_file,
            stat_features=stat_features,
            feature_names=feature_cols  # Save feature names for reference
        )
        logger.info(f"Saved standardized statistical features as numpy array to {np_output_file}")
        
        return len(features_df)
    
    except Exception as e:
        logger.error(f"Error processing batch {batch_dir}: {str(e)}")
        return 0


def main():
    """Main function to process all data."""
    # Load configuration

    
    # Load anomaly dictionary
    try:
        anomaly_dict_path = "Data/interim/Anomaly_Data/anomaly_dict.pkl"
        logger.info(f"Loading anomaly dictionary from {anomaly_dict_path}")
        with open(anomaly_dict_path, 'rb') as f:
            anomaly_dict = pickle.load(f)
        
        # Create interval trees for each station
        anomaly_trees = {station: create_interval_tree(periods) for station, periods in anomaly_dict.items()}
        logger.info(f"Created interval trees for {len(anomaly_trees)} stations")
    except Exception as e:
        logger.error(f"Error loading anomaly dictionary: {str(e)}")
        anomaly_trees = {}
    
    # Load top features
    top_features = load_top_features()
    
    # Initialize counters
    total_windows = 0
    
    # Process each data type and component
    data_types = ['train', 'val', 'test']
    components = ['contact']  # Add 'pcb', 'ring' if needed
    
    logger.info(f"Starting sliding window feature extraction with dynamic step sizes")
    
    for data_type in data_types:
        for component in components:
            # Get batch directories
            batch_dirs = get_batch_dirs(data_type, component)
            
            # Create output directory
            output_dir = f"Data/processed/transform/slidingWindow_noOverlap_800_800_100_0.7_th0.5/statistic_features_standscaler/{data_type}/{component}"
            os.makedirs(output_dir, exist_ok=True)
            
            # Process each batch
            batch_windows = 0
            for batch_idx, batch_dir in enumerate(batch_dirs):
                logger.info(f"Processing batch {batch_idx+1}/{len(batch_dirs)} for {data_type}/{component}")
                
                windows = process_batch(batch_dir, output_dir, top_features, anomaly_trees, component)
                batch_windows += windows
            
            logger.info(f"Processed {batch_windows} windows for {data_type}/{component}")
            total_windows += batch_windows
    
    logger.info(f"Total windows processed: {total_windows}")
    logger.info(f"Processing complete. Results saved to Data/processed/transform/slidingWindow_withOverlap_*/statistic_features_standscaler/")


if __name__ == "__main__":
    main()
