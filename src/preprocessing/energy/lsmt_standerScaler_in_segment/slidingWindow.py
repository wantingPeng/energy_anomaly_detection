"""
Sliding window preprocessing for LSTM model training data with added time features.

This script processes energy data by creating sliding windows for LSTM training.
It reads data from aligned files, creates sliding windows, and labels them based on anomaly overlap.
"""

import os
import gc
import yaml
import pandas as pd
import numpy as np
import dask.dataframe as dd
from pathlib import Path
from intervaltree import IntervalTree
from tqdm import tqdm
from typing import Dict, List, Tuple, Union, Optional
from datetime import datetime
import glob
from joblib import Parallel, delayed

from src.utils.logger import logger
from src.utils.memory_left import log_memory

# Import functions from the original sliding window module
from src.preprocessing.energy.machine_learning.labeling_slidingWindow import (
    load_anomaly_dict,
    create_interval_tree,
    calculate_window_overlap
)

# Import the process_segment and create_sliding_windows from the original module
from src.preprocessing.energy.lstm.slinding_window import (
    create_sliding_windows
)


def load_config() -> dict:
    """Load configuration from YAML file."""
    config_path = Path("configs/lsmt_preprocessing.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def process_component_data(
    input_dir: str,
    output_dir: str,
    component: str, 
    config: dict,
    anomaly_dict: Dict[str, List[Tuple[str, str]]]
) -> None:
    """
    Process data for a specific component.
    This modified version reads data from batch files and processes each batch separately.
    
    Args:
        input_dir: Input directory containing aligned data
        output_dir: Output directory for processed data
        component: Component type ('contact', 'pcb', or 'ring')
        config: Configuration dictionary
        anomaly_dict: Dictionary mapping station IDs to lists of anomaly period tuples
    """
    
    logger.info(f"Processing {component} data")
    
    # Prepare paths
    component_dir = os.path.join(input_dir, component)
    
    # Check if component directory exists
    if not os.path.exists(component_dir):
        logger.warning(f"Component directory {component_dir} does not exist. Skipping.")
        return
    
    # Create interval trees for each station
    anomaly_trees = {station: create_interval_tree(periods) for station, periods in anomaly_dict.items()}
    
    # Create output directory
    output_component_dir = os.path.join(output_dir, component)
    os.makedirs(output_component_dir, exist_ok=True)
    
    # Process each batch file
    batch_files = glob.glob(os.path.join(component_dir, "batch_*.parquet"))
    
    if not batch_files:
        logger.warning(f"No batch files found for {component}. Skipping.")
        return
    
    for batch_file in batch_files:
        batch_name = os.path.basename(batch_file)
        logger.info(f"Processing {batch_name} for {component}")
        log_memory(f"Before loading {batch_name} for {component}")
        
        # Load data using Dask
        ddf = dd.read_parquet(batch_file)
        df = ddf.compute()
        
        log_memory(f"After loading {batch_name} for {component}")
        
        if df.empty:
            logger.warning(f"No data found in {batch_name} for {component}")
            continue
        
        # Process the data
        logger.info(f"Creating sliding windows for {batch_name} {component}")
        windows, labels, segment_ids, timestamps, stats = create_sliding_windows(
            df,
            component,
            config['sliding_window']['window_size'],
            config['sliding_window']['step_size'],
            anomaly_trees,
            config['sliding_window']['anomaly_threshold'],
            n_jobs=6
        )
        
        log_memory(f"After creating sliding windows for {batch_name} {component}")
        
        # Save results if there are windows
        if len(windows) > 0:
            output_file = os.path.join(output_component_dir, f"{batch_name}_windows.npz")
            np.savez_compressed(
                output_file,
                windows=windows,
                labels=labels,
                segment_ids=segment_ids,
                timestamps=timestamps
            )
            logger.info(f"Saved {len(windows)} windows to {output_file}")
            
            # Save stats
            stats_file = os.path.join(output_component_dir, f"{batch_name}_stats.yaml")
            with open(stats_file, 'w') as f:
                yaml.dump(stats, f)
            logger.info(f"Saved statistics to {stats_file}")
        else:
            logger.warning(f"No windows created for {batch_name} {component}")
        
        # Free memory
        del df, windows, labels, segment_ids, timestamps
        gc.collect()
        log_memory(f"After processing {batch_name} for {component}")


def main():
    """
    Main function to process all data types and components.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger.info(f"Starting sliding window processing with time features at {timestamp}")
    
    # Load configuration
    config = load_config()
    
    # Load anomaly dictionary
    anomaly_dict = load_anomaly_dict(config)
    
    # Define paths
    input_dir = "Data/processed/lsmt/add_time_features/align"
    output_dir = "Data/processed/lsmt/add_time_features/sliding_window"
    
    # Override config paths for this specific process
    config['paths'] = {
        'input_dir': input_dir,
        'output_dir': output_dir
    }
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Get components
    components = ['contact']
    
    # Process for train, val, test (though we might only have one type of data)
    # Assuming we use the same train/val/test split as before
 # Modify if you have 'val' and 'test' as well
    for component in components:
        process_component_data(
            input_dir,
            output_dir,
            component,
            config,
            anomaly_dict
        )
            
        # Force garbage collection
        gc.collect()
        log_memory(f"After GC for {component}")


if __name__ == "__main__":
    main()
