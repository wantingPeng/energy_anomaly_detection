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

from src.preprocessing.energy.machine_learning.labeling_slidingWindow import (
    load_anomaly_dict, 
    create_interval_tree,
    calculate_window_overlap
)




def load_config() -> dict:
    """Load configuration from YAML file."""
    config_path = Path("configs/lsmt_sliding_window.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def process_segment(
    segment_data: Tuple[str, pd.DataFrame],
    component: str,
    window_size: int,
    step_size: int,
    anomaly_step_size: int,
    anomaly_trees: Dict[str, IntervalTree],
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

    

    
    component_to_station = {
        'contact': 'Kontaktieren',
        'ring': 'Ringmontage',
        'pcb': 'Pcb'
    }
    
    # Exclude non-feature columns
    exclude_cols = ['TimeStamp', 'segment_id']
    feature_cols = [col for col in segment_df.columns if col not in exclude_cols]
    
    # Sort by timestamp
    segment_df = segment_df.sort_values('TimeStamp')
    
    # Get station
    station = component_to_station.get(component)
    interval_tree = anomaly_trees[station]

    
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
            current_step_size = anomaly_step_size if overlap_ratio > 0.3 else step_size

            # Only keep windows with overlap_ratio > 0.9 (anomaly) or < 0.1 (normal)
            if overlap_ratio > 0.9:
                label = 1
                windows.append(window_data)
                labels.append(label)
            elif overlap_ratio < 0.1:
                label = 0
                windows.append(window_data)
                labels.append(label)
            # Skip windows with overlap_ratio between 0.1 and 0.9

        # Move to next window position
        start_idx += current_step_size
    return windows, labels


def create_sliding_windows(
    df: pd.DataFrame,
    component: str,
    window_size: int,
    step_size: int,
    anomaly_step_size: int,
    anomaly_trees: Dict[str, IntervalTree],
    n_jobs: int = 6
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


    # Group by segment_id
    segments = list(df.groupby('segment_id'))
    total_segments = len(segments)
    
    logger.info(f"Processing {total_segments} segments for component {component}")
    
    # Process all segments in parallel
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_segment)(
            segment_data,
            component,
            window_size,
            step_size,
            anomaly_step_size,
            anomaly_trees,
        )
        for segment_data in tqdm(segments, desc="Processing segments")
    )
    
    # Combine results
    for windows, labels in results:
        all_windows.extend(windows)
        all_labels.extend(labels)


    
    # Convert to numpy arrays
    all_windows = np.array(all_windows)
    all_labels = np.array(all_labels)
    
    logger.info(f"Created {len(all_windows)} windows for component {component}")
    return all_windows, all_labels

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
    
    # Check if component directory exists
    if not os.path.exists(component_dir):
        logger.warning(f"Component directory {component_dir} does not exist. Skipping.")
        return
        
    #os.makedirs(output_component_dir, exist_ok=True)
    # Create interval trees for each station
    anomaly_trees = {station: create_interval_tree(periods) for station, periods in anomaly_dict.items()}
    
    # Process data in batches by directory
    batch_dirs = glob.glob(os.path.join(component_dir, "batch_*"))
    
    if not batch_dirs:
        logger.warning(f"No batch directories found for {component} {data_type}. Skipping.")
        return
        
    logger.info(f"Found {len(batch_dirs)} batch directories for {component} {data_type}")
    
    output_dir = os.path.join(config['paths']['output_dir'], data_type, component)
    os.makedirs(output_dir, exist_ok=True)
    

    
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
        windows, labels = create_sliding_windows(
            df,
            component,
            config['sliding_window']['window_size'],
            config['sliding_window']['step_size'],
            config['sliding_window']['anomaly_step_size'],
            anomaly_trees,
            n_jobs=6
        )
        

        batch_name = os.path.basename(batch_dir)
        # Save intermediate results to temporary files if there are windows
        if len(windows) > 0:
            batch_file = os.path.join(output_dir, f"{batch_name}.npz")
            np.savez_compressed(
                batch_file,
                windows=windows,
                labels=labels,
            )
            logger.info(f"Saved {len(windows)} windows to temporary file {batch_file}")
        
        # Free memory
        del df, windows, labels
        gc.collect()
        log_memory(f"After processing {batch_name} for {component} {data_type}")
    


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
    #components = ['contact', 'pcb', 'ring']
    components = ['contact']

    # Process each data type and component
    #for data_type in ['train', 'val', 'test']:
    for data_type in [ 'val','train']:

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
    


if __name__ == "__main__":
    main()
