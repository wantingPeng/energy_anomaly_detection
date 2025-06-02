import os
import gc
import numpy as np
import torch
from tqdm import tqdm
from src.utils.logger import logger
import glob
from pathlib import Path

def save_batch(windows, labels, output_dir, batch_idx):
    """
    Save the windowed data to PyTorch files for a batch.
    
    Args:
        windows: Array of sliding windows
        labels: Array of window labels
        output_dir: Directory to save results
        batch_idx: Batch index for file naming
        
    Returns:
        Path to the saved batch file
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Define file path with batch index
    batch_path = os.path.join(output_dir, f"batch_{batch_idx}.pt")
    
    # Convert numpy arrays to torch tensors
    windows_tensor = torch.FloatTensor(windows)
    labels_tensor = torch.FloatTensor(labels)
    
    # Create new dataset file with tensors
    torch.save({
        'windows': windows_tensor,
        'labels': labels_tensor
    }, batch_path)
    
    logger.info(f"Saved batch {batch_idx} with {len(windows)} windows to {batch_path}")
    
    return batch_path

def process_split_data(input_dir, output_dir, split_type, batch_size=10000):
    """
    Process data for a specific split type and save in batches.
    
    Args:
        input_dir: Input directory containing split data
        output_dir: Output directory for processed data
        split_type: Split type ('train', 'val', or 'test')
        batch_size: Number of windows per batch
    """
    logger.info(f"Processing {split_type} data")
    
    # Load data
    X_path = os.path.join(input_dir, split_type, "X_sequences.npy")
    y_path = os.path.join(input_dir, split_type, "y_labels.npy")
    
    if not os.path.exists(X_path) or not os.path.exists(y_path):
        logger.warning(f"Data files not found in {os.path.join(input_dir, split_type)}")
        return
    
    logger.info(f"Loading data from {X_path} and {y_path}")
    
    # Use memory-mapping for large arrays
    X = np.load(X_path, mmap_mode='r')
    y = np.load(y_path)
    
    logger.info(f"Loaded data with shape X: {X.shape}, y: {y.shape}")
    
    # Create output directory
    output_split_dir = os.path.join(output_dir, split_type)
    os.makedirs(output_split_dir, exist_ok=True)
    
    # Process data in batches
    num_samples = len(y)
    num_batches = (num_samples + batch_size - 1) // batch_size  # Ceiling division
    
    batch_paths = []
    
    for batch_idx in tqdm(range(num_batches), desc=f"Processing {split_type} data in batches"):
        # Get batch data
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, num_samples)
        
        # Load batch into memory
        X_batch = X[start_idx:end_idx].copy()
        y_batch = y[start_idx:end_idx].copy()
        
        # Save batch
        batch_path = save_batch(X_batch, y_batch, output_split_dir, batch_idx)
        batch_paths.append(batch_path)
        
        # Clean up memory
        del X_batch, y_batch
        gc.collect()
    
    logger.info(f"Completed processing {split_type} data into {len(batch_paths)} batches")
    
    return batch_paths

def main():
    """
    Main function to process the split data into the format required by the LSTM model.
    """
    try:
        logger.info("Starting conversion of split data to dataset format")
        
        # Input and output paths
        input_dir = "Data/processed/lsmt_base_on_xgboostFeatures/split/contact"
        output_dir = "Data/processed/lsmt_base_on_xgboostFeatures/dataset/contact"
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Process each split type
        for split_type in ['train', 'val', 'test']:
            process_split_data(input_dir, output_dir, split_type)
            
            # Force garbage collection
            gc.collect()
        
        logger.info("Completed converting split data to dataset format")
        
    except Exception as e:
        logger.error(f"Error in data conversion process: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main() 