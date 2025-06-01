"""
Split preprocessed LSTM data into train, validation, and test sets.

This script loads sliding window data from processed files and splits them
into train, validation, and test sets in a 70/15/15 ratio, based on window indices.
"""

import os
import gc
import yaml
import numpy as np
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import glob

from src.utils.logger import logger
from src.utils.memory_left import log_memory


def split_data(input_dir: str, output_dir: str, component: str, batch_order: list) -> None:
    """
    Split data for a specific component into train, validation, and test sets.
    
    Args:
        input_dir: Input directory containing sliding window data
        output_dir: Output directory for split data
        component: Component type (e.g., 'contact')
        batch_order: Order of batches to process
    """
    logger.info(f"Splitting data for {component}")
    
    # Create output directories
    output_train_dir = os.path.join(output_dir, component, 'train')
    output_val_dir = os.path.join(output_dir, component, 'val')
    output_test_dir = os.path.join(output_dir, component, 'test')
    
    os.makedirs(output_train_dir, exist_ok=True)
    os.makedirs(output_val_dir, exist_ok=True)
    os.makedirs(output_test_dir, exist_ok=True)
    
    # Process each batch in the specified order
    for batch_num in batch_order:
        batch_name = f"batch_{batch_num}.parquet"
        logger.info(f"Processing {batch_name} for {component}")
        
        # Load data
        data_file = os.path.join(input_dir, component, f"{batch_name}_windows.npz")
        if not os.path.exists(data_file):
            logger.warning(f"File {data_file} does not exist. Skipping.")
            continue
        
        log_memory(f"Before loading {batch_name} for {component}")
        data = np.load(data_file)
        windows = data['windows']
        labels = data['labels']
        
        log_memory(f"After loading {batch_name} for {component}")
        
        # Calculate split indices
        total_windows = len(windows)
        train_size = int(total_windows * 0.7)
        val_size = int(total_windows * 0.15)
        
        # Split the data
        # Train: first 70%
        train_windows = windows[:train_size]
        train_labels = labels[:train_size]

        
        # Validation: next 15%
        val_windows = windows[train_size:train_size+val_size]
        val_labels = labels[train_size:train_size+val_size]

        
        # Test: remaining 15%
        test_windows = windows[train_size+val_size:]
        test_labels = labels[train_size+val_size:]

        # Save train data
        train_file = os.path.join(output_train_dir, f"{batch_name}.npz")
        np.savez_compressed(
            train_file,
            windows=train_windows,
            labels=train_labels,
        )
        logger.info(f"Saved {len(train_windows)} training windows to {train_file}")
        
        # Save validation data
        val_file = os.path.join(output_val_dir, f"{batch_name}.npz")
        np.savez_compressed(
            val_file,
            windows=val_windows,
            labels=val_labels,

        )
        logger.info(f"Saved {len(val_windows)} validation windows to {val_file}")
        
        # Save test data
        test_file = os.path.join(output_test_dir, f"{batch_name}.npz")
        np.savez_compressed(
            test_file,
            windows=test_windows,
            labels=test_labels,
        )
        logger.info(f"Saved {len(test_windows)} test windows to {test_file}")
        
        # Also copy the stats file
        stats_file = os.path.join(input_dir, component, f"{batch_name}_stats.yaml")
        if os.path.exists(stats_file):
            with open(stats_file, 'r') as f:
                stats = yaml.safe_load(f)
            
            # Update stats with split information
            stats['train_size'] = len(train_windows)
            stats['val_size'] = len(val_windows)
            stats['test_size'] = len(test_windows)
            
            # Save updated stats to each directory
            for dir_name, dir_path in [
                ('train', output_train_dir),
                ('val', output_val_dir),
                ('test', output_test_dir)
            ]:
                stats_output = os.path.join(dir_path, f"{batch_name}_stats.yaml")
                with open(stats_output, 'w') as f:
                    yaml.dump(stats, f)
        
        # Free memory
        del windows, labels
        del train_windows, train_labels
        del val_windows, val_labels
        del test_windows, test_labels
        gc.collect()
        log_memory(f"After processing {batch_name} for {component}")


def main():
    """
    Main function to split data for all components.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger.info(f"Starting data splitting process at {timestamp}")
    
    # Define paths
    input_dir = "Data/processed/lsmt/add_time_features/sliding_window"
    output_dir = "Data/processed/lsmt/add_time_features/spilt"
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Define component and batch order
    component = 'contact'
    batch_order = [2, 0, 1, 3]
    
    # Split data
    split_data(input_dir, output_dir, component, batch_order)
    
    logger.info("Data splitting completed successfully")


if __name__ == "__main__":
    main()
