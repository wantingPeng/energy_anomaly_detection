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
from pathlib import Path
from tqdm import tqdm
import torch
from datetime import datetime
import glob
from src.utils.logger import logger
from src.utils.memory_left import log_memory


def save_results(
    windows: np.ndarray,
    labels: np.ndarray,
    output_dir: str,
    batch_name: int
) -> tuple:
    """
    Save the windowed data to separate parquet and PyTorch files for each batch.
    
    Args:
        windows: Array of sliding windows
        labels: Array of window labels
        segment_ids: Array of segment IDs for each window
        timestamps: Array of window start timestamps
        output_dir: Directory to save results
        component: Component type (e.g., 'contact', 'pcb', 'ring')
        data_type: Data type ('train', 'val', or 'test')
        batch_idx: Batch index for file naming
        
    Returns:
        Tuple of paths (dataset_path, parquet_path) to the saved files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Define file paths with batch index and include component and data_type in filename
    dataset_path = os.path.join(output_dir, f"{batch_name}.pt")
    
    # Convert numpy arrays to torch tensors
    windows_tensor = torch.FloatTensor(windows)
    labels_tensor = torch.FloatTensor(labels)
    
    # Create new dataset file with tensors
    torch.save({
        'windows': windows_tensor,
        'labels': labels_tensor
    }, dataset_path)
    

    logger.info(f"Saved {len(windows)} windows to {os.path.basename(dataset_path)}")
    
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
    output_component_dir = os.path.join(output_dir, data_type,component)
    
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
    
    # Save batch file paths for future merging
    batch_files = []
    
    # Process each NPZ file individually and save as separate batch
    for batch_idx, npz_file in enumerate(tqdm(npz_files, desc=f"Processing {component} {data_type} files")):
        try:
            # Load NPZ file data
            data = np.load(npz_file, allow_pickle=True)
            batch_name = os.path.basename(npz_file)
            batch_name = batch_name.replace('.npz', '')
            # Extract data from NPZ file
            if 'windows' in data and 'labels' in data:
                windows = data['windows']
                labels = data['labels']
                
                # Save this file's data as a separate batch
                dataset_path = save_results(
                    windows,
                    labels,
                    output_component_dir,
                    batch_name
                )
                
                # Track the batch files
                batch_files.append(dataset_path)
                
                logger.info(f"Processed and saved batch {batch_idx}: {len(windows)} windows from {os.path.basename(npz_file)}")
                
                # Free memory immediately
                del windows, labels
                gc.collect()
                log_memory(f"After processing batch {batch_idx}/{len(npz_files)}")
            else:
                logger.warning(f"NPZ file {os.path.basename(npz_file)} does not have expected data structure")
                
        except Exception as e:
            logger.error(f"Error processing NPZ file {npz_file}: {str(e)}")
    
    logger.info(f"Completed processing {len(batch_files)} batches for {component} {data_type}")
    
    # Save the list of batch files to a metadata file for later merging
    metadata_path = os.path.join(output_component_dir, f"{data_type}_{component}_batches.pkl")
    with open(metadata_path, 'wb') as f:
        pickle.dump(batch_files, f)
    logger.info(f"Saved batch metadata to {metadata_path}")
    
    log_memory(f"After saving {component} {data_type}")


def main():
    """
    Main function to process all data types and components.
    """
    start_time = datetime.now()
    logger.info(f"Starting saving for dataset at {start_time}")


    # Get paths from config
    input_dir = 'Data/processed/lsmt_statisticalFeatures/sliding_window_1200s'
    output_dir = 'Data/processed/lsmt_statisticalFeatures/dataSet_1200s'
    
    # Get components to process
    #components = ['contact', 'pcb', 'ring']
    #components = ['train', 'val', 'test']
    components = ['contact']

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


if __name__ == "__main__":
    main()
