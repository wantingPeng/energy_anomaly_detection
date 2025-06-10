import os
import numpy as np
import time
from datetime import datetime
from src.utils.logger import logger

def undersample_normal_windows(sliding_window_path, seed=42):
    """
    Undersample normal windows to ensure anomalies make up 25% of the final dataset.

    Parameters:
    - sliding_window_path: str, path to the sliding window dataset directory
    - seed: int, random seed for reproducibility
    """
    # Set random seed for reproducibility
    np.random.seed(seed)
    
    # Setup logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"undersample_windows_25percent_{timestamp}.log"
    log_filepath = os.path.join("experiments/logs", log_filename)
    os.makedirs("experiments/logs", exist_ok=True)
    
    logger.info("Starting undersampling of normal windows to achieve 25% anomaly ratio")
    
    # Define categories
    categories = ["contact"]
    
    # Create output directories
    sliding_window_output_path = "Data/processed/transform/slidingWindow_noOverlap_0.8_no_stats/train_down_25%"
    
    for category in categories:
        sliding_window_category_path = os.path.join(sliding_window_path, category)
        
        # Create output directories for each category
        sliding_window_output_category_dir = os.path.join(sliding_window_output_path, category)
        
        os.makedirs(sliding_window_output_category_dir, exist_ok=True)
        
        # Get list of batch files
        sliding_window_files = sorted([f for f in os.listdir(sliding_window_category_path) if f.endswith(".npz")])
        
        logger.info(f"Processing {category} category with {len(sliding_window_files)} batch files")
        
        for batch_file in sliding_window_files:
            logger.info(f"Processing batch file: {batch_file}")
            
            # Load data from the dataset
            sliding_window_file_path = os.path.join(sliding_window_category_path, batch_file)
            
            sliding_window_data = np.load(sliding_window_file_path)
            
            # Extract windows and labels
            windows = sliding_window_data['windows']
            labels = sliding_window_data['labels']
            
            # Get indices of normal and anomaly samples
            normal_indices = np.where(labels == 0)[0]
            anomaly_indices = np.where(labels == 1)[0]
            
            # Calculate original ratios
            total_samples = len(labels)
            normal_ratio = len(normal_indices) / total_samples * 100
            anomaly_ratio = len(anomaly_indices) / total_samples * 100
            
            logger.info(
                f"Original {batch_file}: "
                f"Total={total_samples}, "
                f"Normal={normal_ratio:.2f}% ({len(normal_indices)}), "
                f"Anomaly={anomaly_ratio:.2f}% ({len(anomaly_indices)})"
            )
            
            # Calculate how many normal samples to keep to achieve 25% anomaly ratio
            # If anomalies should be 25% of final dataset, normal samples should be 75%
            # So: anomaly_count / (anomaly_count + normal_count) = 0.25
            # Solving for normal_count: normal_count = anomaly_count * 3
            target_normal_count = len(anomaly_indices) * 3
            
            # If we already have fewer normal samples than needed, keep all of them
            if len(normal_indices) <= target_normal_count:
                logger.info(f"Batch {batch_file} already has sufficient anomaly ratio, keeping all samples")
                undersampled_normal_indices = normal_indices
            else:
                # Randomly select normal samples to keep
                np.random.shuffle(normal_indices)
                undersampled_normal_indices = normal_indices[:target_normal_count]
            
            # Combine indices and sort to maintain original order
            selected_indices = np.sort(np.concatenate([undersampled_normal_indices, anomaly_indices]))
            
            # Subset the dataset using the selected indices
            undersampled_windows = windows[selected_indices]
            undersampled_labels = labels[selected_indices]
            
            # Calculate final ratios
            final_normal_count = np.sum(undersampled_labels == 0)
            final_anomaly_count = np.sum(undersampled_labels == 1)
            final_total = len(undersampled_labels)
            final_normal_ratio = final_normal_count / final_total * 100
            final_anomaly_ratio = final_anomaly_count / final_total * 100
            
            logger.info(
                f"Final {batch_file}: "
                f"Total={final_total}, "
                f"Normal={final_normal_ratio:.2f}% ({final_normal_count}), "
                f"Anomaly={final_anomaly_ratio:.2f}% ({final_anomaly_count})"
            )
            
            # Save undersampled dataset
            sliding_window_output_file = os.path.join(sliding_window_output_category_dir, batch_file)
            np.savez(sliding_window_output_file, windows=undersampled_windows, labels=undersampled_labels)
            
            logger.info(f"Saved undersampled data for batch {batch_file}")
    
    logger.info("Completed undersampling of normal windows")


if __name__ == "__main__":
    sliding_window_path = "Data/processed/transform/slidingWindow_noOverlap_0.8_no_stats/train"
    undersample_normal_windows(sliding_window_path) 