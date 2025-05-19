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




def save_results(
    windows: np.ndarray,
    labels: np.ndarray,
    segment_ids: np.ndarray,
    timestamps: np.ndarray,
    output_dir: str,
    component: str,
    data_type: str,
    append: bool = False
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
        append: Whether to append to existing files (default: False)
        
    Returns:
        Path to the saved PyTorch dataset file
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Define file paths
    dataset_path = os.path.join(output_dir, f"{component}_{data_type}_dataset.pt")
    parquet_path = os.path.join(output_dir, f"window_{component}_{data_type}.parquet")
    
    # Convert numpy arrays to torch tensors
    windows_tensor = torch.FloatTensor(windows)
    labels_tensor = torch.FloatTensor(labels)
    
    if append and os.path.exists(dataset_path):
        # Load existing dataset tensors
        try:
            existing_data = torch.load(dataset_path)
            existing_windows = existing_data['windows']
            existing_labels = existing_data['labels']
            
            # Concatenate with new data
            combined_windows = torch.cat([existing_windows, windows_tensor])
            combined_labels = torch.cat([existing_labels, labels_tensor])
            
            # Save the updated tensors
            torch.save({
                'windows': combined_windows,
                'labels': combined_labels
            }, dataset_path)
            
            logger.info(f"Appended {len(windows)} windows to existing dataset ({len(combined_windows)} total)")
        except Exception as e:
            logger.error(f"Error appending to dataset: {str(e)}. Creating new dataset.")
            # Save as new if there was an error
            torch.save({
                'windows': windows_tensor,
                'labels': labels_tensor
            }, dataset_path)
            logger.info(f"Created new dataset with {len(windows)} windows")
        
        # Append to existing Parquet file
        df = pd.DataFrame({
            'segment_id': segment_ids,
            'timestamp': timestamps,
            'label': labels,
            'window': [w.tobytes() for w in windows]
        })
        
        if os.path.exists(parquet_path):
            # If parquet file already exists, append to it
            existing_df = pd.read_parquet(parquet_path)
            combined_df = pd.concat([existing_df, df], ignore_index=True)
            combined_df.to_parquet(parquet_path, index=False)
            logger.info(f"Appended {len(df)} rows to existing parquet file ({len(combined_df)} total)")
        else:
            # If not, create new file
            df.to_parquet(parquet_path, index=False)
            logger.info(f"Created new parquet file with {len(df)} rows")
    else:
        # Create new dataset file with tensors only (not the entire class)
        torch.save({
            'windows': windows_tensor,
            'labels': labels_tensor
        }, dataset_path)
        
        # Create new Parquet file
        df = pd.DataFrame({
            'segment_id': segment_ids,
            'timestamp': timestamps,
            'label': labels,
            'window': [w.tobytes() for w in windows]
        })
        
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
  
    # Combine results from all batches by loading from temporary files
    if temp_files:
        logger.info(f"Processing {len(temp_files)} batch files one by one")
        
        # Initialize combined stats tracking
        windows_processed = 0
        first_file = True
        
        # Process each temporary file individually to avoid OOM issues
        for temp_idx, temp_file in enumerate(temp_files):
            try:
                logger.info(f"Processing file {temp_idx+1}/{len(temp_files)}: {temp_file}")
                # Load a single temporary file
                data = np.load(temp_file, allow_pickle=True)
                
                # Get data from this file
                windows = data['windows']
                labels = data['labels']
                segment_ids = data['segment_ids']
                timestamps = data['timestamps']
                
                file_windows_count = len(windows)
                windows_processed += file_windows_count
                
                logger.info(f"Loaded {file_windows_count} windows from {temp_file}")
                log_memory(f"After loading file {temp_idx+1}/{len(temp_files)}")
                
                # Save results (append mode for all except the first file)
                save_results(
                    windows,
                    labels,
                    segment_ids,
                    timestamps,
                    output_component_dir,
                    component,
                    data_type,
                    append=(not first_file)  # First file creates new files, subsequent files append
                )
                
                if first_file:
                    first_file = False
                
                # Free memory immediately after saving
                del windows, labels, segment_ids, timestamps, data
                gc.collect()
                log_memory(f"After saving and clearing file {temp_idx+1}/{len(temp_files)}")
                
            except Exception as e:
                logger.error(f"Error processing temporary file {temp_file}: {str(e)}")
        
        logger.info(f"Total windows processed: {windows_processed}")
        
        #Save statistics
        save_stats(
            combined_stats,
            os.path.join(output_dir, "reports"),
            component,
            data_type
        )
    else:
        logger.warning(f"No windows created for {component} {data_type}")
    
    log_memory(f"After saving {component} {data_type}")
    



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
        # Load dataset tensors
        data = torch.load(dataset_path)
        windows = data['windows']
        labels = data['labels']
        
        # Check dataset properties
        n_samples = len(windows)
        logger.info(f"Dataset {component}_{data_type} contains {n_samples} samples")
        
        if n_samples == 0:
            logger.warning(f"Dataset {component}_{data_type} is empty")
            return False
            
        # Check if we can retrieve an item
        first_window = windows[0]
        first_label = labels[0]
        logger.info(f"First window shape: {first_window.shape}, label: {first_label.item()}")
        
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
