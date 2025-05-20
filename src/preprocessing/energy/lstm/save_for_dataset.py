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
    config_path = Path("configs/save_for_dataset.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config




def save_results(
    windows: np.ndarray,
    labels: np.ndarray,
    segment_ids: np.ndarray,
    timestamps: np.ndarray,
    output_dir: str,
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
        data_type: Data type ('train', 'val', or 'test')
        append: Whether to append to existing files (default: False)
        
    Returns:
        Path to the saved PyTorch dataset file
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Define file paths
    dataset_path = os.path.join(output_dir, f"{data_type}_dataset.pt")
    parquet_path = os.path.join(output_dir, f"{data_type}.parquet")
    
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
            
            # Save the updated  Parquet file
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


def process_component_data(
    input_dir: str,
    output_dir: str,
    component: str, 
    data_type: str,
) -> None:
    """
    Process data for a specific component and data type.
    
    Args:
        input_dir: Input directory containing segment data
        output_dir: Output directory for processed data
        component: Component type ('contact', 'pcb', or 'ring')
        data_type: Data type ('train', 'val', or 'test')
    """
    
    logger.info(f"Processing {component} {data_type} data")
    log_memory(f"Before loading {component} {data_type}")
    
    # Prepare paths
    component_dir = os.path.join(input_dir, data_type, component)
    output_component_dir = os.path.join(output_dir, data_type)
    
    # Check if component directory exists
    if not os.path.exists(component_dir):
        logger.warning(f"Component directory {component_dir} does not exist. Skipping.")
        return
        
    os.makedirs(output_component_dir, exist_ok=True)
    
    # Find all NPZ files in the component directory
    npz_files = glob.glob(os.path.join(component_dir, "*.npz"))
    
    if not npz_files:
        logger.warning(f"No NPZ files found in {component_dir}. Skipping.")
        return
        
    logger.info(f"Found {len(npz_files)} NPZ files in {component_dir}")
    
    # Process each NPZ file individually and save immediately
    is_first_file = True
    processed_files = 0
    
    for npz_file in tqdm(npz_files, desc=f"Processing {component} {data_type} files"):
        try:
            # Load NPZ file data
            data = np.load(npz_file, allow_pickle=True)
            
            # Extract data from NPZ file
            if 'windows' in data and 'labels' in data and 'segment_ids' in data and 'timestamps' in data:
                windows = data['windows']
                labels = data['labels']
                segment_ids = data['segment_ids']
                timestamps = data['timestamps']
                
                # Save this file's data immediately
                save_results(
                    windows,
                    labels,
                    segment_ids,
                    timestamps,
                    output_component_dir,
                    data_type,
                    append=(not is_first_file)  # First file creates new files, subsequent files append
                )
                
                logger.info(f"Processed and saved {len(windows)} windows from {os.path.basename(npz_file)}")
                processed_files += 1
                
                if is_first_file:
                    is_first_file = False
                
                # Free memory immediately
                del windows, labels, segment_ids, timestamps
                gc.collect()
                log_memory(f"After processing file {processed_files}/{len(npz_files)}")
            else:
                logger.warning(f"NPZ file {os.path.basename(npz_file)} does not have expected data structure")
                
        except Exception as e:
            logger.error(f"Error processing NPZ file {npz_file}: {str(e)}")
    
    logger.info(f"Completed processing {processed_files} files for {component} {data_type}")
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
    # Dataset files are now stored in data_type directory
    output_data_dir = os.path.join(output_dir, data_type)
    dataset_path = os.path.join(output_data_dir, f"{data_type}_dataset.pt")
    parquet_path = os.path.join(output_data_dir, f"{data_type}.parquet")
    
    if not os.path.exists(dataset_path) or not os.path.exists(parquet_path):
        logger.error(f"Dataset files not found for {data_type} (component: {component})")
        return False
        
    try:
        # Load dataset tensors
        data = torch.load(dataset_path)
        windows = data['windows']
        labels = data['labels']
        
        # Check dataset properties
        n_samples = len(windows)
        logger.info(f"Dataset {data_type} (component: {component}) contains {n_samples} samples")
        
        if n_samples == 0:
            logger.warning(f"Dataset {data_type} (component: {component}) is empty")
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
        logger.error(f"Error verifying dataset {data_type} (component: {component}): {str(e)}")
        return False


def main():
    """
    Main function to process all data types and components.
    """
    start_time = datetime.now()
    logger.info(f"Starting sliding window processing at {start_time}")
    
    # Load configuration
    config = load_config()

    # Get paths from config
    input_dir = config['paths']['input_dir']
    output_dir = config['paths']['output_dir']
    
    # Get components to process
    components = ['contact', 'pcb', 'ring']
    
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
            )
            
            # Force garbage collection
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
