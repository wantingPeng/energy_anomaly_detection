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
    config_path = Path("configs/lsmt_preprocessing.yaml")
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



def split_by_component(df: pd.DataFrame, temp_dir: str) -> List[str]:
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
    os.makedirs(temp_dir, exist_ok=True)
    
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
            temp_file = os.path.join("Data/processed/lsmt/split_by_component", f"{component}_train.parquet")
            component_df.to_parquet(temp_file)
            
            component_names.append(component)
            logger.info(f"Saved {len(component_df)} rows for component {component} to {temp_file}")
            
            # Free memory
            del component_df
            gc.collect()
    
    return component_names


def get_component_df(component: str, temp_dir: str) -> pd.DataFrame:
    """
    Load a specific component's DataFrame from temporary storage.
    
    Args:
        component: Component name to load
        temp_dir: Directory containing temporary component files
        
    Returns:
        DataFrame for the specified component
    """
    temp_file = os.path.join(temp_dir, f"{component}_temp.parquet")
    logger.info(f"Loading component data from {temp_file}")
    
    if not os.path.exists(temp_file):
        raise FileNotFoundError(f"Temporary file for component {component} not found")
    
    return pd.read_parquet(temp_file)


def cleanup_temp_files(temp_dir: str):
    """
    Clean up temporary component files.
    
    Args:
        temp_dir: Directory containing temporary files to clean up
    """
    logger.info(f"Cleaning up temporary files in {temp_dir}")
    for file in os.listdir(temp_dir):
        if file.endswith("_temp.parquet"):
            os.remove(os.path.join(temp_dir, file))
    try:
        os.rmdir(temp_dir)
    except OSError:
        logger.warning(f"Could not remove temp directory {temp_dir} - it may not be empty")


def interpolate_segments(df: pd.DataFrame, output_dir: str) -> None:
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
            
            # Save batch using Dask
            batch_output_dir = os.path.join(output_dir, f'batch_{batch_start}')
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


def sliding_windows(segment_df: pd.DataFrame, anomaly_tree: IntervalTree, config: dict, component: str, split: str, output_dir: str) -> Tuple[pd.DataFrame, dict]:
    """
    Create sliding windows from a segment DataFrame and label them based on anomaly overlap.
    
    Args:
        segment_df: DataFrame for a single segment
        anomaly_tree: IntervalTree of anomaly intervals for the component
        config: Configuration dictionary
        component: Component type
        split: Data split (train/val/test)
        output_dir: Directory to save sliding window results
        
    Returns:
        Tuple[pd.DataFrame, dict]: DataFrame with window data and statistics dictionary
    """
    logger.info(f"Creating sliding windows for segment {segment_df['segment_id'].iloc[0]}")
    
    window_data = []
    
    # Get feature columns (exclude non-feature columns)
    exclude_cols = ['TimeStamp', 'segment_id'] + [col for col in segment_df.columns if col.startswith('component_type_')]
    feature_cols = [col for col in segment_df.columns if col not in exclude_cols]
    
    # Sort by timestamp
    segment_df = segment_df.sort_values('TimeStamp')
    
    # Create sliding windows
    start_time = segment_df['TimeStamp'].min()
    end_time = start_time + timedelta(seconds=config['sliding_window']['window_size'])
    
    stats = {
        "total_windows": 0,
        "anomaly_windows": 0,
        "skipped_segments": 0,
        "min_window_length": float('inf'),
        "max_window_length": 0,
    }
    
    while end_time <= segment_df['TimeStamp'].max():
        # Get window data
        window_mask = (segment_df['TimeStamp'] >= start_time) & (segment_df['TimeStamp'] < end_time)
        window_df = segment_df.loc[window_mask]
        
        if not window_df.empty:
            # Get window features
            window = window_df[feature_cols].values
            
            # Check for anomaly overlap
            overlap_ratio = calculate_window_overlap(
                start_time,
                end_time,
                anomaly_tree
            )
            
            # Label window based on overlap threshold
            is_anomaly = 1 if overlap_ratio >= config['sliding_window']['anomaly_threshold'] else 0
            
            # Store window data as a dictionary
            window_data.append({
                'window': window.tobytes(),
                'window_shape': str(window.shape),
                'label': is_anomaly,
                'segment_id': segment_df['segment_id'].iloc[0],
                'start_timestamp': start_time.strftime('%Y-%m-%d %H:%M:%S')
            })
            
            # Update statistics
            stats["total_windows"] += 1
            if is_anomaly:
                stats["anomaly_windows"] += 1
            stats["min_window_length"] = min(stats["min_window_length"], window.shape[0])
            stats["max_window_length"] = max(stats["max_window_length"], window.shape[0])
        
        # Move to next window
        start_time += timedelta(seconds=config['sliding_window']['step_size'])
        end_time = start_time + timedelta(seconds=config['sliding_window']['window_size'])
    
    # Convert window data to DataFrame
    windows_df = pd.DataFrame(window_data)
    return windows_df, stats


def save_batch_results(
    windows_df: pd.DataFrame,
    temp_dir: str,
    batch_id: int,
    component: str,
    split: str
) -> str:
    """
    Save a single batch of results to a temporary directory.
    
    Args:
        windows_df: DataFrame containing window data for one batch
        temp_dir: Temporary directory path
        batch_id: ID of the current batch
        component: Component type
        split: Data split (train/val/test)
        
    Returns:
        str: Path to the saved temporary file
    """
    logger.info(f"Saving batch {batch_id} for {component} ({split})")
    
    # Create temp directory if it doesn't exist
    os.makedirs(temp_dir, exist_ok=True)
    
    # Add batch_id to DataFrame
    windows_df['batch_id'] = batch_id
    
    # Save as parquet
    temp_file = os.path.join(temp_dir, f"temp_batch_{component}_{split}_{batch_id}.parquet")
    windows_df.to_parquet(temp_file, engine='pyarrow', index=False)
    
    logger.info(f"Saved batch {batch_id} to {temp_file}")
    return temp_file


def save_results(
    windows_batch: List[pd.DataFrame],
    output_path: str,
    component: str,
    split: str
) -> None:
    """
    Combine temporary batch files and save final results.
    
    Args:
        windows_batch: List of paths to temporary batch files
        output_path: Output directory path
        component: Component type
        split: Data split (train/val/test)
    """
    logger.info(f"Combining and saving all results for {component} ({split})")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    
    # Create a Dask DataFrame from all temporary files
    ddf = dd.read_parquet(windows_batch)
    
    # Save combined results using Dask
    output_file = os.path.join(output_path, f"window_{component}_{split}.parquet")
    ddf.to_parquet(
        output_file,
        engine='pyarrow',
        write_index=False,
        partition_on=['batch_id']
    )
    
    logger.info(f"Saved combined results to {output_file}")
    
    # Clean up temporary files
    for temp_file in windows_batch:
        try:
            os.remove(temp_file)
            logger.debug(f"Removed temporary file: {temp_file}")
        except Exception as e:
            logger.warning(f"Failed to remove temporary file {temp_file}: {e}")
    
    # Clean up memory
    del ddf
    gc.collect()


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
    anomaly_dict
) -> None:
    """
    Process a single data split (train/val/test).
    
    Args:
        split: Data split name (train/val/test)
        config: Configuration dictionary
        anomaly_dict: Dictionary of anomaly periods
    """
    logger.info(f"Processing {split} data")
    log_memory(f"Before {split}")
    
    # Load data
    input_path = os.path.join(config['paths']['input_dir'], f"{split}.parquet")
    df = load_data(input_path)
    log_memory(f"After loading {split}")
    
    # Split by component type and save to temp files
    component_names = split_by_component(df, config['paths']['temp_dir'])
    
    # Free memory
    del df
    gc.collect()
    log_memory(f"After splitting {split}")
    
    # Define a mapping from components to stations
    component_to_station = {
        'contact': 'Kontaktieren',
        'ring': 'Ringmontage',
        'pcb': 'Pcb'
    }
    
    # Process components in specified order
    for component in component_names:
        logger.info(f"Processing component: {component}")
        
        # Get the corresponding station for the component
        station = component_to_station.get(component)
        if not station:
            logger.warning(f"No station mapping found for component: {component}")
            continue

        # Convert to interval tree for the specific station
        anomaly_tree = create_interval_tree(anomaly_dict[station])
        
        # Get component DataFrame
        component_df = get_component_df(component, config['paths']['temp_dir'])
        
        # Define paths for interpolated and sliding window data
        interpolated_output_dir = os.path.join(config['paths']['output_dir'], 'segment_fixe')
        sliding_window_output_dir = config['paths']['output_dir']
        temp_batch_dir = os.path.join(config['paths']['temp_dir'], f'batch_results_{component}_{split}')
        log_memory(f"Before interpolating segments for {component} ({split})")
        # Interpolate segments
        #interpolate_segments(component_df, interpolated_output_dir)
        log_memory(f"After interpolating segments for {component} ({split})")

        # Read interpolated data
        base_dir = "Data/processed/lsmt/sliding_window/segment_fixe"
        segment_files = glob(f"{base_dir}/batch_*/*.parquet")
        log_memory(f"After reading interpolated data for {component} ({split})")
        
        # Prepare for batch processing
        batch_size = 30
        n_jobs = 6
        logger.info(f"Using batch processing with {batch_size} segments per batch and {n_jobs} processes per batch")
        
        # Lists to store batch results and stats
        temp_batch_files = []
        all_stats = []
        
        # Process each batch
        for batch_idx in range(0, len(segment_files), batch_size):



            current_batch_files = segment_files[batch_idx:batch_idx + batch_size]
            logger.info(f"Processing batch {batch_idx // batch_size + 1} with {len(current_batch_files)} segments")
            
            # Load batch data
            batch_dfs = [pd.read_parquet(f) for f in current_batch_files]
            batch_df = pd.concat(batch_dfs, ignore_index=True)

            if 'segment_id' not in batch_df.columns:
                raise KeyError("列 'segment_id' 不存在，请检查数据！")
            if batch_df.empty:
                raise ValueError("batch_df 是空的，请检查数据加载！")
            
            # Process batch using joblib
            batch_results = Parallel(n_jobs=n_jobs, prefer='processes')(
                delayed(sliding_windows)(segment_df, anomaly_tree, config, component, split, sliding_window_output_dir)
                for segment_id, segment_df in batch_df.groupby('segment_id')
            )


            
            
            
             

            # Unpack batch results
            batch_windows, batch_stats = zip(*batch_results)
            
            # Combine windows for this batch
            combined_batch_windows = pd.concat(batch_windows, ignore_index=True)
            
            # Save batch results to temporary file
            temp_file = save_batch_results(
                combined_batch_windows,
                temp_batch_dir,
                batch_idx // batch_size,
                component,
                split
            )
            temp_batch_files.append(temp_file)
            
            # Store stats
            all_stats.extend(batch_stats)
            
            # Clean up
            del batch_dfs, batch_df, batch_results, combined_batch_windows
            gc.collect()
            log_memory(f"After processing batch {batch_idx // batch_size + 1} for {component} ({split})")
        # Save all results by combining temporary files
        save_results(temp_batch_files, sliding_window_output_dir, component, split)
        
        # Combine and save statistics
        combined_stats = {
            "total_windows": sum(s["total_windows"] for s in all_stats),
            "anomaly_windows": sum(s["anomaly_windows"] for s in all_stats),
            "skipped_segments": sum(s["skipped_segments"] for s in all_stats),
            "min_window_length": min(s["min_window_length"] for s in all_stats),
            "max_window_length": max(s["max_window_length"] for s in all_stats)
        }
        save_stats(combined_stats, sliding_window_output_dir, component, split)

        # Free memory
        del component_df, all_stats
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
    os.makedirs(config['paths']['reports_dir'], exist_ok=True)
    os.makedirs(config['paths']['temp_dir'], exist_ok=True)
    
    try:
        # Load anomaly dictionary
        anomaly_dict = load_anomaly_dict(config)
        

        
        # Process each split
        for split in ['train', 'val', 'test']:
            process_split(split, config,anomaly_dict)
        
        logger.info("Sliding window preprocessing completed successfully")
        
    except Exception as e:
        logger.error(f"Error during processing: {str(e)}")
        raise
    finally:
        # Cleanup
        try:
            cleanup_temp_files(config['paths']['temp_dir'])
        except Exception as e:
            logger.warning(f"Error during cleanup: {str(e)}")


if __name__ == "__main__":
    main()
