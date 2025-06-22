import os
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import glob
from datetime import datetime
from src.utils.logger import logger


def create_sliding_windows(data_types=None, components=None, window_size=60, step_size=60, preview=False):
    """
    Create sliding windows from the labeled data.
    
    Args:
        data_types (list): List of data types to process (train, val, test)
        components (list): List of components to process (contact, pcb, ring)
        window_size (int): Number of time points in each window
        step_size (int): Number of time points to slide the window
        preview (bool): Whether to print a preview of the windows
    
    Returns:
        None: Windows are saved to disk as NPZ files
    """
    # Default parameters if not provided
    if data_types is None:
        data_types = ['train', 'val', 'test']
    if components is None:
        components = ['contact', 'pcb', 'ring']
        
    # Input and output directories
    input_base_dir = "Data/row_energyData_subsample_Transform/labeled"
    output_base_dir = "Data/row_energyData_subsample_Transform/slidingWindow"
    
    # Process each data type and component
    for data_type in data_types:
        for component in components:
            input_dir = os.path.join(input_base_dir, data_type, component)
            output_dir = os.path.join(output_base_dir, data_type, component)
            
            # Check if input directory exists
            if not os.path.exists(input_dir):
                logger.warning(f"Input directory not found: {input_dir}")
                continue
                
            # Create output directory if it doesn't exist
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            
            logger.info(f"Processing {data_type} data for {component}...")
            
            # Find all parquet files in the input directory
            parquet_files = glob.glob(os.path.join(input_dir, "*.parquet"))
            
            # Collect all windows and labels for this data_type/component
            all_windows = []
            all_labels = []
            total_segments = 0
            
            for file_idx, parquet_file in enumerate(parquet_files):
                # Load the parquet file
                logger.info(f"Loading {parquet_file}...")
                df = pd.read_parquet(parquet_file)
                logger.info(f"Loaded {len(df)} rows from {parquet_file}")
                
                # Extract required columns
                feature_cols = [col for col in df.columns if col not in ['TimeStamp', 'anomaly_label', 'segment_id']]
                
                # Process data by segment_id to avoid crossing segment boundaries
                segment_groups = df.groupby('segment_id')
                
                for segment_id, segment_df in segment_groups:
                    logger.info(f"Processing segment {segment_id} with {len(segment_df)} rows")
                    total_segments += 1
                    
                    # Sort by timestamp to ensure sequential order
                    segment_df = segment_df.sort_values('TimeStamp')
                    
                    # Extract features and labels
                    features = segment_df[feature_cols].values
                    labels = segment_df['anomaly_label'].values
                    
                    # Create windows for this segment
                    # Slide window through the segment
                    for i in range(0, len(segment_df) - window_size + 1, step_size):
                        # Extract window features
                        window = features[i:i+window_size]
                        
                        # If window is smaller than required, skip it
                        if len(window) < window_size:
                            continue
                            
                        # Extract window labels for each time step
                        window_label = labels[i:i+window_size]
                        
                        all_windows.append(window)
                        all_labels.append(window_label)
                    
            # Convert lists to numpy arrays
            if all_windows:
                windows_array = np.array(all_windows, dtype=np.float32)
                labels_array = np.array(all_labels, dtype=np.int32)
                
                # Save all windows and labels for this data_type/component
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                batch_file = os.path.join(output_dir, f"{data_type}_{component}_{timestamp}.npz")
                
                logger.info(f"Saving {len(all_windows)} windows from {total_segments} segments to {batch_file}")
                logger.info(f"Windows shape: {windows_array.shape}, Labels shape: {labels_array.shape}")
                
                # Preview sample of windows and labels if requested
                if preview and len(all_windows) > 0:
                    logger.info(f"\nPreview of first 5 windows out of {len(all_windows)}:")
                    for i in range(min(5, len(all_windows))):
                        logger.info(f"Window {i+1} shape: {windows_array[i].shape}")
                        logger.info(f"Label {i+1} shape: {labels_array[i].shape}")
                
                # Save all windows and labels to a single NPZ file
                np.savez_compressed(
                    batch_file,
                    windows=windows_array,
                    labels=labels_array,
                )
                logger.info(f"Successfully saved windows to {batch_file}")
            else:
                logger.warning(f"No windows created for {data_type}/{component}")
    
    logger.info("Sliding window creation complete!")


def main():
    """Main function to run the sliding window creation."""
    # Set parameters
    window_size = 60
    step_size = 60
    
    # Create sliding windows
    create_sliding_windows(
        data_types=['train', 'val', 'test'],
        components=['contact'],  # Add 'pcb', 'ring' if needed
        window_size=window_size,
        step_size=step_size,
        preview=True  # Enable preview
    )
    
    
if __name__ == "__main__":
    main()
