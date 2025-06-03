"""
Sliding window feature extraction for LSTM model training data.

This script loads interpolated data, creates sliding windows with specified 
window size and step size, and calculates statistical features for each window.
It processes data batch by batch to conserve memory.
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

from src.utils.logger import logger
from src.preprocessing.energy.machine_learning.calculate_window_features import calculate_window_features


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
    metadata_cols = ['window_start', 'window_end', 'segment_id']
    
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
    base_dir = f"Data/processed/lsmt_statisticalFeatures/interpolate/{data_type}/{component}"
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


def create_sliding_windows(df, window_size=1200, step_size=300):
    """
    Create sliding windows from DataFrame.
    
    Args:
        df: DataFrame containing time series data
        window_size: Size of sliding window in seconds
        step_size: Step size for window sliding in seconds
        
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
            
            # Get data for this window
            window_mask = (segment_df['TimeStamp'] >= start_time) & (segment_df['TimeStamp'] < end_time)
            window_df = segment_df.loc[window_mask].copy()
            
            # Add window to list if it's not empty
            if not window_df.empty:
                windows.append(window_df)
                window_count += 1
            
            # Move to next window
            start_time += pd.Timedelta(seconds=step_size)
    
    logger.info(f"Created {window_count} windows from {len(segments)} segments")
    return windows


def process_batch(batch_dir, output_dir, top_features, window_size=1200, step_size=300):
    """
    Process a batch of data: load, create sliding windows, and calculate features.
    
    Args:
        batch_dir: Path to batch directory
        output_dir: Path to output directory
        top_features: List of top feature names to keep
        window_size: Size of sliding window in seconds
        step_size: Step size for window sliding in seconds
        
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
        
        # Create sliding windows
        logger.info(f"Creating sliding windows with window_size={window_size}s, step_size={step_size}s")
        windows = create_sliding_windows(batch_df, window_size, step_size)
        
        # Free memory
        del batch_df
        gc.collect()
        
        if not windows:
            logger.warning(f"No windows created from {batch_dir}")
            return 0
        
        # Calculate features for each window
        logger.info(f"Calculating features for {len(windows)} windows")
        window_features = []
        
        for window in tqdm(windows, desc="Processing windows"):
            features = calculate_window_features(window)
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

        
        # Also save as numpy array for easier loading in LSTM model
        # Extract metadata for alignment with LSTM windows
        window_starts = features_df['window_start'].values if 'window_start' in features_df.columns else None
        window_ends = features_df['window_end'].values if 'window_end' in features_df.columns else None
        segment_ids = features_df['segment_id'].values if 'segment_id' in features_df.columns else None
        
        # Extract only numerical features for the statistical features
        metadata_cols = ['window_start', 'window_end', 'segment_id']
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
            window_starts=window_starts,
            window_ends=window_ends,
            segment_ids=segment_ids,
            feature_names=feature_cols  # Save feature names for reference
        )
        logger.info(f"Saved standardized statistical features as numpy array to {np_output_file}")
        
        return len(features_df)
    
    except Exception as e:
        logger.error(f"Error processing batch {batch_dir}: {str(e)}")
        return 0


def main():
    """Main function to process all data."""
    # Set window and step size
    window_size = 1200  # seconds
    step_size = 300     # seconds
    
    # Load top features
    top_features = load_top_features()
    
    # Initialize counters
    total_windows = 0
    
    # Process each data type and component
    data_types = ['train', 'val', 'test']
    components = ['contact']  # Add 'pcb', 'ring' if needed
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"experiments/logs/sliding_window_features_{timestamp}.log"
    
    logger.info(f"Starting sliding window feature extraction with window_size={window_size}s, step_size={step_size}s")
    
    for data_type in data_types:
        for component in components:
            # Get batch directories
            batch_dirs = get_batch_dirs(data_type, component)
            
            # Create output directory
            output_dir = f"Data/processed/lsmt_statisticalFeatures/statistic_features_standscaler/{data_type}/{component}"
            os.makedirs(output_dir, exist_ok=True)
            
            # Process each batch
            batch_windows = 0
            for batch_idx, batch_dir in enumerate(batch_dirs):
                logger.info(f"Processing batch {batch_idx+1}/{len(batch_dirs)} for {data_type}/{component}")
                
                windows = process_batch(batch_dir, output_dir, top_features, window_size, step_size)
                batch_windows += windows
            
            logger.info(f"Processed {batch_windows} windows for {data_type}/{component}")
            total_windows += batch_windows
    
    logger.info(f"Total windows processed: {total_windows}")
    logger.info(f"Processing complete. Results saved to Data/processed/lsmt_statisticalFeatures/statistic_features_standscaler/")
    
    # Create a README file explaining the data format
    readme_path = "Data/processed/lsmt_statisticalFeatures/statistic_features_standscaler/README.md"
    with open(readme_path, "w") as f:
        f.write("# Statistical Features for LSTM Late Fusion\n\n")
        f.write("This directory contains statistical features extracted from sliding windows for use in LSTM Late Fusion models.\n\n")
        f.write("## Data Format\n\n")
        f.write("Each batch is saved in two formats:\n")
        f.write("1. Parquet file: Contains all features and metadata\n")
        f.write("2. NPZ file: Contains numpy arrays for easier loading in PyTorch\n\n")
        f.write("### NPZ File Structure\n\n")
        f.write("- `stat_features`: Standardized statistical features array with shape (n_windows, n_features)\n")
        f.write("- `window_starts`: Start timestamps for each window\n")
        f.write("- `window_ends`: End timestamps for each window\n")
        f.write("- `segment_ids`: Segment IDs for each window\n")
        f.write("- `feature_names`: Names of the features\n\n")
        f.write("These arrays can be used to align statistical features with LSTM sliding window data.\n")
        f.write("\n## Standardization\n\n")
        f.write("All numerical features have been standardized using sklearn's StandardScaler.\n")
        f.write("The scalers for each batch are saved in the `scalers` directory for later use during inference.\n")
    
    logger.info(f"Created README file at {readme_path}")


if __name__ == "__main__":
    main() 