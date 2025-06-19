import os
import numpy as np
from pathlib import Path
import glob
from typing import List, Tuple
from tqdm import tqdm
from src.utils.logger import logger

def combine_windows_with_stats(
    sliding_window_dir: str = "Data/deepLearning/transform/slidingWindow_noOverlap_600_600_100_0_0.5",
    stat_features_dir: str = "Data/deepLearning/transform/statistic_features_600_600_100_0_standardized",
    output_dir: str = "Data/deepLearning/transform/window_plus_statistic",
    data_types: List[str] = ["train", "val"],
    component: str = "contact",
    batch_size: int = 1000
) -> None:
    """
    Combines sliding window data with statistical features by broadcasting the statistical 
    features to match window dimensions and concatenating them along the feature axis.
    
    Each window is paired with its corresponding statistical features. The statistical features
    are broadcast to match the window's time dimension, and then concatenated as additional features.
    This preserves the original window structure while adding statistical features.
    
    Args:
        sliding_window_dir: Directory containing sliding window data
        stat_features_dir: Directory containing statistical features
        output_dir: Directory to save combined data
        data_types: List of data types ('train', 'val')
        component: Component type ('contact', 'pcb', 'ring')
        batch_size: Number of samples to process at once to manage memory usage
    """
    logger.info(f"Combining windows with statistical features for component: {component}")
    
    for data_type in data_types:
        logger.info(f"Processing {data_type} data")
        
        # Define directories
        window_component_dir = os.path.join(sliding_window_dir, data_type, component)
        stat_component_dir = os.path.join(stat_features_dir, data_type, component)
        output_component_dir = os.path.join(output_dir, data_type, component)
        
        # Create output directory if it doesn't exist
        os.makedirs(output_component_dir, exist_ok=True)
        
        # Get all files
        window_files = sorted(glob.glob(os.path.join(window_component_dir, "*.npz")))
        stat_files = sorted(glob.glob(os.path.join(stat_component_dir, "*.npz")))
        
        if not window_files:
            raise ValueError(f"No window files found in {window_component_dir}")
        if not stat_files:
            raise ValueError(f"No stat feature files found in {stat_component_dir}")
        
        logger.info(f"Found {len(window_files)} window files and {len(stat_files)} stat feature files")
        
        # Process each pair of files
        for i, (window_file, stat_file) in enumerate(zip(window_files, stat_files)):
            window_basename = os.path.basename(window_file)
            stat_basename = os.path.basename(stat_file)
            logger.info(f"Processing file {i+1}/{len(window_files)}: {window_basename} with {stat_basename}")
            
            # Load data
            window_data = np.load(window_file)
            stat_data = np.load(stat_file)
            
            windows = window_data['windows']  # Shape: [n_samples, window_size, n_features]
            labels = window_data['labels']    # Shape: [n_samples]
            stat_features = stat_data['stat_features']  # Shape: [n_samples, n_stat_features]
            
            # Check if the number of samples match
            if len(windows) != len(stat_features):
                logger.warning(f"Number of window samples ({len(windows)}) does not match "
                              f"number of stat feature samples ({len(stat_features)})")
                logger.error(f"Cannot proceed with mismatched sample counts. Exiting.")
                exit()
            
            # Process in batches to manage memory
            n_samples = len(windows)
            output_file = os.path.join(
                output_component_dir, 
                f"combined_{window_basename}"
            )
            
            # Log shapes for debugging
            n_window_samples, window_size, n_window_features = windows.shape
            n_stat_features = stat_features.shape[1]
            logger.info(f"Window shape: [{n_window_samples}, {window_size}, {n_window_features}]")
            logger.info(f"Stat features shape: [{len(stat_features)}, {n_stat_features}]")
            
            # Process in batches
            combined_windows_list = []
            
            for start_idx in tqdm(range(0, n_samples, batch_size), desc="Combining batches"):
                end_idx = min(start_idx + batch_size, n_samples)
                batch_windows = windows[start_idx:end_idx]  # [batch_size, window_size, n_window_features]
                batch_stat_features = stat_features[start_idx:end_idx]  # [batch_size, n_stat_features]
                
                # Broadcast stat_features to match window dimensions
                # For each window in the batch, repeat the stat_features for each time step
                batch_stat_features_expanded = np.repeat(
                    batch_stat_features[:, np.newaxis, :],  # Shape: [batch_size, 1, n_stat_features]
                    window_size,  # Repeat along time dimension
                    axis=1  # Time dimension
                )  # Result shape: [batch_size, window_size, n_stat_features]
                
                # Concatenate windows and broadcasted stat_features along feature dimension (axis=2)
                # This preserves the batch_size and window_size dimensions
                batch_combined = np.concatenate(
                    [batch_windows, batch_stat_features_expanded], 
                    axis=2  # Feature dimension
                )  # Result shape: [batch_size, window_size, n_window_features + n_stat_features]
                
                combined_windows_list.append(batch_combined)
            
            # Concatenate all batches along the sample dimension (axis=0)
            combined_windows = np.concatenate(combined_windows_list, axis=0)
            
            # Verify the shape is as expected
            expected_shape = (n_window_samples, window_size, n_window_features + n_stat_features)
            if combined_windows.shape != expected_shape:
                logger.error(f"Combined shape {combined_windows.shape} does not match expected shape {expected_shape}")
                exit()
            
            # Save the combined data
            np.savez(
                output_file,
                windows=combined_windows,
                labels=labels
            )
            
            logger.info(f"Saved combined data to {output_file}")
            logger.info(f"Original window shape: {windows.shape}, "
                       f"Stat features shape: {stat_features.shape}, "
                       f"Combined shape: {combined_windows.shape}")

if __name__ == "__main__":
    # Run the function with default parameters
    combine_windows_with_stats() 