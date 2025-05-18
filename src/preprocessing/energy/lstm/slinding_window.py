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
from pathlib import Path
from intervaltree import IntervalTree
from tqdm import tqdm
import torch
from typing import Dict, List, Tuple, Union, Optional
from datetime import datetime
import glob
from joblib import Parallel, delayed

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


def process_segment(
    segment_data: Tuple[str, pd.DataFrame],
    component: str,
    window_size: int,
    step_size: int,
    anomaly_trees: Dict[str, IntervalTree],
    anomaly_threshold: float = 0.3
) -> Tuple[List, List, List, List, dict]:
    """
    Process a single segment for sliding window creation (for parallel processing).
    
    Args:
        segment_data: Tuple of (segment_id, segment_df)
        component: Component type (e.g., 'contact', 'pcb', 'ring')
        window_size: Size of sliding window in seconds
        step_size: Step size for window sliding in seconds
        anomaly_trees: Dictionary mapping station IDs to IntervalTree objects
        anomaly_threshold: Threshold for anomaly labeling
        
    Returns:
        Tuple of (windows, labels, segment_ids, timestamps, stats)
    """
    segment_id, segment_df = segment_data
    
    windows = []
    labels = []
    segment_ids = []
    timestamps = []
    
    # Statistics tracking
    stats = {
        "total_segments": 1,
        "total_windows": 0,
        "anomaly_windows": 0,
        "normal_windows": 0,
        "segments_with_anomalies": 0,
        "segments_without_anomalies": 0
    }
    
    component_to_station = {
        'contact': 'Kontaktieren',
        'ring': 'Ringmontage',
        'pcb': 'Pcb'
    }
    
    # Exclude non-feature columns
    exclude_cols = ['TimeStamp', 'segment_id'] + [col for col in segment_df.columns if col.startswith('component_type_')]
    feature_cols = [col for col in segment_df.columns if col not in exclude_cols]
    
    # Sort by timestamp
    segment_df = segment_df.sort_values('TimeStamp')
    
    # Get station
    station = component_to_station.get(component)
    interval_tree = anomaly_trees[station]
    segment_has_anomalies = False
    
    # Create sliding windows
    start_idx = 0
    segment_timestamps = pd.to_datetime(segment_df['TimeStamp'])
    segment_features = segment_df[feature_cols].values
    
    while start_idx + window_size <= len(segment_df):
        window_start = segment_timestamps.iloc[start_idx]
        window_end = window_start + pd.Timedelta(seconds=window_size)
        
        # Check if window exceeds segment timestamps
        if window_end > segment_timestamps.iloc[-1]:
            break
        
        # Extract the window data
        window_mask = (segment_timestamps >= window_start) & (segment_timestamps < window_end)
        window_data = segment_features[window_mask]
        
        # Ensure the window has the right number of time steps
        if len(window_data) == window_size:
            # Calculate overlap with anomalies
            overlap_ratio = calculate_window_overlap(window_start, window_end, interval_tree)
            label = 1 if overlap_ratio >= anomaly_threshold else 0
            
            if label == 1:
                segment_has_anomalies = True
                stats["anomaly_windows"] += 1
            else:
                stats["normal_windows"] += 1
            
            windows.append(window_data)
            labels.append(label)
            segment_ids.append(segment_id)
            timestamps.append(window_start)
            
            stats["total_windows"] += 1
        
        # Move to next window position
        start_idx += step_size
    
    if segment_has_anomalies:
        stats["segments_with_anomalies"] = 1
    else:
        stats["segments_without_anomalies"] = 1
    
    return windows, labels, segment_ids, timestamps, stats


def create_sliding_windows(
    df: pd.DataFrame,
    component: str,
    window_size: int,
    step_size: int,
    anomaly_trees: Dict[str, IntervalTree],
    anomaly_threshold: float = 0.3,
    n_jobs: int = 6,
    batch_size: int = 60
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    Create sliding windows from segment data and label them based on anomaly overlap.
    Uses parallel processing for efficiency.
    
    Args:
        df: DataFrame containing the segment data
        component: Component type (e.g., 'contact', 'pcb', 'ring')
        window_size: Size of sliding window in seconds
        step_size: Step size for window sliding in seconds
        anomaly_trees: Dictionary mapping station IDs to IntervalTree objects
        anomaly_threshold: Threshold for anomaly labeling (default: 0.3)
        n_jobs: Number of parallel jobs to run (default: 6)
        batch_size: Number of segments to process in one batch (default: 60)
        
    Returns:
        Tuple containing:
        - windows: Array of sliding windows with shape (n_windows, window_size, n_features)
        - labels: Array of window labels (0 for normal, 1 for anomaly)
        - segment_ids: Array of segment IDs for each window
        - timestamps: Array of window start timestamps
        - stats: Dictionary with statistics about the windowing process
    """
    all_windows = []
    all_labels = []
    all_segment_ids = []
    all_timestamps = []
    
    # Aggregate statistics
    total_stats = {
        "total_segments": 0,
        "total_windows": 0,
        "anomaly_windows": 0,
        "normal_windows": 0,
        "skipped_segments": 0,
        "segments_with_anomalies": 0,
        "segments_without_anomalies": 0
    }
    
    # Group by segment_id
    segments = list(df.groupby('segment_id'))
    total_segments = len(segments)
    logger.info(f"Processing {total_segments} segments in batches of {batch_size}")
    
    # Process segments in batches to manage memory
    for batch_start in range(0, total_segments, batch_size):
        batch_end = min(batch_start + batch_size, total_segments)
        batch_segments = segments[batch_start:batch_end]
        
        logger.info(f"Processing batch {batch_start//batch_size + 1}/{(total_segments-1)//batch_size + 1} with {len(batch_segments)} segments")
        
        # Process segments in parallel
        results = Parallel(n_jobs=n_jobs)(
            delayed(process_segment)(
                segment_data,
                component,
                window_size, 
                step_size,
                anomaly_trees,
                anomaly_threshold
            ) for segment_data in batch_segments
        )
        
        # Collect results
        for windows, labels, segment_ids, timestamps, stats in results:
            all_windows.extend(windows)
            all_labels.extend(labels)
            all_segment_ids.extend(segment_ids)
            all_timestamps.extend(timestamps)
            
            # Update statistics
            for key in total_stats:
                if key in stats:
                    total_stats[key] += stats[key]
        
        # Force garbage collection between batches
        gc.collect()
        log_memory(f"After processing batch {batch_start//batch_size + 1}")
    
    if not all_windows:
        return np.array([]), np.array([]), np.array([]), np.array([]), total_stats
    
    # Convert lists to numpy arrays
    windows_array = np.array(all_windows)
    labels_array = np.array(all_labels)
    segment_ids_array = np.array(all_segment_ids)
    timestamps_array = np.array(all_timestamps)
    
    return windows_array, labels_array, segment_ids_array, timestamps_array, total_stats


def save_results(
    windows: np.ndarray,
    labels: np.ndarray,
    segment_ids: np.ndarray,
    timestamps: np.ndarray,
    output_dir: str,
    component: str,
    data_type: str
) -> str:
    """
    Save the windowed data to parquet format and PyTorch dataset.
    
    Args:
        windows: Array of sliding windows
        labels: Array of window labels
        segment_ids: Array of segment IDs for each window
        timestamps: Array of window start timestamps
        output_dir: Directory to save results
        component: Component type (e.g., 'contact', 'pcb', 'ring')
        data_type: Data type ('train', 'val', or 'test')
        
    Returns:
        Path to the saved PyTorch dataset file
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save as PyTorch dataset
    dataset = LSTMWindowDataset(windows, labels)
    dataset_path = os.path.join(output_dir, f"{component}_{data_type}_dataset.pt")
    dataset.to_file(dataset_path)
    
    # Save as Parquet for future reference
    df = pd.DataFrame({
        'segment_id': segment_ids,
        'timestamp': timestamps,
        'label': labels
    })
    
    # Save windows as serialized numpy arrays
    df['window'] = [w.tobytes() for w in windows]
    
    parquet_path = os.path.join(output_dir, f"window_{component}_{data_type}.parquet")
    df.to_parquet(parquet_path, index=False)
    
    logger.info(f"Saved {len(windows)} windows to {parquet_path} and {dataset_path}")
    return dataset_path


def save_stats(stats: dict, output_dir: str, component: str, data_type: str) -> str:
    """
    Save statistics about the windowing process.
    
    Args:
        stats: Dictionary with statistics
        output_dir: Directory to save results
        component: Component type
        data_type: Data type ('train', 'val', or 'test')
        
    Returns:
        Path to the saved statistics file
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Add percentages
    if stats["total_windows"] > 0:
        stats["anomaly_percentage"] = (stats["anomaly_windows"] / stats["total_windows"]) * 100
        stats["normal_percentage"] = (stats["normal_windows"] / stats["total_windows"]) * 100
    else:
        stats["anomaly_percentage"] = 0
        stats["normal_percentage"] = 0
    
    if stats["total_segments"] > 0:
        stats["segments_with_anomalies_percentage"] = (stats["segments_with_anomalies"] / stats["total_segments"]) * 100
    else:
        stats["segments_with_anomalies_percentage"] = 0
    
    # Save as JSON
    import json
    stats_path = os.path.join(output_dir, f"stats_{component}_{data_type}.json")
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=4)
    
    logger.info(f"Saved statistics to {stats_path}")
    return stats_path


def process_component_data(
    input_dir: str,
    output_dir: str,
    component: str, 
    data_type: str,
    config: dict,
    anomaly_dict: Dict[str, List[Tuple[str, str]]]
) -> None:
    """
    Process data for a specific component and data type.
    
    Args:
        input_dir: Input directory containing segment data
        output_dir: Output directory for processed data
        component: Component type ('contact', 'pcb', or 'ring')
        data_type: Data type ('train', 'val', or 'test')
        config: Configuration dictionary
        anomaly_dict: Dictionary mapping station IDs to lists of anomaly period tuples
    """
    logger.info(f"Processing {component} {data_type} data")
    log_memory(f"Before loading {component} {data_type}")
    
    # Prepare paths
    component_dir = os.path.join(input_dir, data_type, component)
    output_component_dir = os.path.join(output_dir, data_type, component)
    
    # Check if component directory exists
    if not os.path.exists(component_dir):
        logger.warning(f"Component directory {component_dir} does not exist. Skipping.")
        return
        
    os.makedirs(output_component_dir, exist_ok=True)
    
    # Create interval trees for each station
    anomaly_trees = {station: create_interval_tree(periods) for station, periods in anomaly_dict.items()}
    
    # Process data in batches by directory
    batch_dirs = glob.glob(os.path.join(component_dir, "batch_*"))
    
    if not batch_dirs:
        logger.warning(f"No batch directories found for {component} {data_type}. Skipping.")
        return
        
    logger.info(f"Found {len(batch_dirs)} batch directories for {component} {data_type}")
    
    # Create temporary directory for storing intermediate results
    temp_dir = os.path.join(config['paths']['temp_dir'], data_type, component)
    os.makedirs(temp_dir, exist_ok=True)
    
    # Keep track of temporary files
    temp_files = []
    
    combined_stats = {
        "total_segments": 0,
        "total_windows": 0,
        "anomaly_windows": 0,
        "normal_windows": 0,
        "skipped_segments": 0,
        "segments_with_anomalies": 0,
        "segments_without_anomalies": 0
    }
    
    # Process each batch directory
    for batch_idx, batch_dir in enumerate(batch_dirs):
        batch_name = os.path.basename(batch_dir)
        logger.info(f"Processing {batch_name} for {component} {data_type}")
        
        # Load data for this batch using Dask
        ddf = dd.read_parquet(os.path.join(batch_dir, "*.parquet"))
        df = ddf.compute()
        
        log_memory(f"After loading {batch_name} for {component} {data_type}")
        
        if df.empty:
            logger.warning(f"No data found in {batch_name} for {component} {data_type}")
            continue
        
        # Process this batch
        logger.info(f"Creating sliding windows for {batch_name} {component} {data_type}")
        windows, labels, segment_ids, timestamps, stats = create_sliding_windows(
            df,
            component,
            config['sliding_window']['window_size'],
            config['sliding_window']['step_size'],
            anomaly_trees,
            config['sliding_window']['anomaly_threshold'],
            n_jobs=6,
            batch_size=60
        )
        
        # Update combined stats
        for key in combined_stats:
            if key in stats:
                combined_stats[key] += stats[key]
        
        # Save intermediate results to temporary files if there are windows
        if len(windows) > 0:
            temp_batch_file = os.path.join(temp_dir, f"batch_{batch_idx}.npz")
            np.savez_compressed(
                temp_batch_file,
                windows=windows,
                labels=labels,
                segment_ids=segment_ids,
                timestamps=timestamps
            )
            temp_files.append(temp_batch_file)
            logger.info(f"Saved {len(windows)} windows to temporary file {temp_batch_file}")
        
        # Free memory
        del df, windows, labels, segment_ids, timestamps
        gc.collect()
        log_memory(f"After processing {batch_name} for {component} {data_type}")
        
    # Combine results from all batches by loading from temporary files
    if temp_files:
        logger.info(f"Combining results from {len(temp_files)} batch files")
        all_windows = []
        all_labels = []
        all_segment_ids = []
        all_timestamps = []
        
        # Load each temporary file and concatenate data
        for temp_file in temp_files:
            try:
                data = np.load(temp_file)
                all_windows.append(data['windows'])
                all_labels.append(data['labels'])
                all_segment_ids.append(data['segment_ids'])
                all_timestamps.append(data['timestamps'])
                logger.info(f"Loaded data from {temp_file}")
            except Exception as e:
                logger.error(f"Error loading temporary file {temp_file}: {str(e)}")
        
        # Concatenate all data
        try:
            if all_windows:
                windows = np.concatenate(all_windows)
                labels = np.concatenate(all_labels)
                segment_ids = np.concatenate(all_segment_ids)
                timestamps = np.concatenate(all_timestamps)
                
                logger.info(f"Saving {len(windows)} windows for {component} {data_type}")
                
                # Save results
                save_results(
                    windows,
                    labels,
                    segment_ids,
                    timestamps,
                    output_component_dir,
                    component,
                    data_type
                )
                
                # Free memory after saving
                del windows, labels, segment_ids, timestamps, all_windows, all_labels, all_segment_ids, all_timestamps
                gc.collect()
            else:
                logger.warning(f"No windows to combine for {component} {data_type}")
        except Exception as e:
            logger.error(f"Error combining data for {component} {data_type}: {str(e)}")
        
        # Save statistics
        save_stats(
            combined_stats,
            os.path.join(output_dir, "reports"),
            component,
            data_type
        )
        
        # Clean up temporary files
        for temp_file in temp_files:
            try:
                os.remove(temp_file)
                logger.info(f"Removed temporary file {temp_file}")
            except Exception as e:
                logger.warning(f"Could not remove temporary file {temp_file}: {str(e)}")
    else:
        logger.warning(f"No windows created for {component} {data_type}")
    
    log_memory(f"After saving {component} {data_type}")
    
    # Try to remove the temporary directory if it's empty
    try:
        os.rmdir(temp_dir)
        logger.info(f"Removed temporary directory {temp_dir}")
    except:
        pass


def verify_dataset(output_dir: str, component: str, data_type: str) -> bool:
    """
    Verify that the dataset was created properly by loading it and checking basic properties.
    
    Args:
        output_dir: Path to the output directory
        component: Component type
        data_type: Data type (train, val, test)
        
    Returns:
        True if verification passed, False otherwise
    """
    dataset_path = os.path.join(output_dir, data_type, component, f"{component}_{data_type}_dataset.pt")
    parquet_path = os.path.join(output_dir, data_type, component, f"window_{component}_{data_type}.parquet")
    
    if not os.path.exists(dataset_path) or not os.path.exists(parquet_path):
        logger.error(f"Dataset files not found for {component} {data_type}")
        return False
        
    try:
        # Load dataset
        dataset = LSTMWindowDataset.from_file(dataset_path)
        
        # Check dataset properties
        n_samples = len(dataset)
        logger.info(f"Dataset {component}_{data_type} contains {n_samples} samples")
        
        if n_samples == 0:
            logger.warning(f"Dataset {component}_{data_type} is empty")
            return False
            
        # Check if we can retrieve an item
        window, label = dataset[0]
        logger.info(f"First window shape: {window.shape}, label: {label.item()}")
        
        # Check parquet file
        df = pd.read_parquet(parquet_path)
        logger.info(f"Parquet file contains {len(df)} rows")
        
        if len(df) != n_samples:
            logger.error(f"Dataset and parquet file have different sample counts: {n_samples} vs {len(df)}")
            return False
            
        return True
        
    except Exception as e:
        logger.error(f"Error verifying dataset {component}_{data_type}: {str(e)}")
        return False


def main():
    """
    Main function to process all data types and components.
    """
    start_time = datetime.now()
    logger.info(f"Starting sliding window processing at {start_time}")
    
    # Load configuration
    config = load_config()
    
    # Load anomaly dictionary
    anomaly_dict = load_anomaly_dict(config)
    
    # Get paths from config
    input_dir = config['paths']['input_dir']
    output_dir = config['paths']['output_dir']
    
    # Get components from config
    components = config['components']['processing_order']
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each data type and component
    for data_type in ['train', 'val', 'test']:
        for component in components:
            process_component_data(
                input_dir,
                output_dir,
                component,
                data_type,
                config,
                anomaly_dict
            )
            
            # Force garbage collection
            if config['memory']['gc_collect_frequency'] > 0:
                gc.collect()
                log_memory(f"After GC for {component} {data_type}")
    
    # Verify datasets
    logger.info("Verifying created datasets...")
    verification_results = {}
    
    for data_type in ['train', 'val', 'test']:
        for component in components:
            result = verify_dataset(output_dir, component, data_type)
            verification_results[f"{component}_{data_type}"] = result
    
    # Report verification results
    logger.info("Dataset verification results:")
    for dataset_name, result in verification_results.items():
        status = "PASSED" if result else "FAILED"
        logger.info(f"{dataset_name}: {status}")
    
    end_time = datetime.now()
    processing_time = end_time - start_time
    logger.info(f"Completed sliding window processing in {processing_time}")


if __name__ == "__main__":
    main()
