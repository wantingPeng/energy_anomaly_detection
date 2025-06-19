import os
import numpy as np
import time
from datetime import datetime
from src.utils.logger import logger

def undersample_normal_windows(sliding_window_path, statistical_features_path, target_anomaly_ratio=0.20, seed=42):
    """
    Undersample normal or anomaly windows to ensure anomalies make up the target percentage of the final dataset.
    Maintains alignment between the two datasets.

    Parameters:
    - sliding_window_path: str, path to the sliding window dataset directory
    - statistical_features_path: str, path to the statistical features dataset directory
    - target_anomaly_ratio: float, target ratio of anomalies in the dataset (default: 0.20 or 20%)
    - seed: int, random seed for reproducibility
    """
    # Set random seed for reproducibility
    np.random.seed(seed)
    
    # Setup logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"undersample_windows_20percent_{timestamp}.log"
    log_filepath = os.path.join("experiments/logs", log_filename)
    os.makedirs("experiments/logs", exist_ok=True)
    
    logger.info(f"Starting resampling to achieve {target_anomaly_ratio*100:.0f}% anomaly ratio")
    
    # Define categories
    categories = ["contact"]
    
    # Create output directories
    sliding_window_output_path = f"Data/deepLearning/transform/slidingWindow_noOverlap_600_600_100_0_0.5_down_{int(target_anomaly_ratio*100)}%/train"
    statistical_features_output_path = f"Data/deepLearning/transform/statistic_features_600_600_100_0_standardized_down_{int(target_anomaly_ratio*100)}%/train"
    
    for category in categories:
        sliding_window_category_path = os.path.join(sliding_window_path, category)
        statistical_features_category_path = os.path.join(statistical_features_path, category)
        
        # Create output directories for each category
        sliding_window_output_category_dir = os.path.join(sliding_window_output_path, category)
        statistical_features_output_category_dir = os.path.join(statistical_features_output_path, category)
        
        os.makedirs(sliding_window_output_category_dir, exist_ok=True)
        os.makedirs(statistical_features_output_category_dir, exist_ok=True)
        
        # Get list of batch files
        sliding_window_files = sorted([f for f in os.listdir(sliding_window_category_path) if f.endswith(".npz")])
        statistical_features_files = sorted([f for f in os.listdir(statistical_features_category_path) if f.endswith(".npz")])
        
        # Verify that both directories have the same batch files
        if set(sliding_window_files) != set(statistical_features_files):
            logger.error("Batch files in the two directories do not match!")
            return
        
        logger.info(f"Processing {category} category with {len(sliding_window_files)} batch files")
        
        for batch_file in sliding_window_files:
            logger.info(f"Processing batch file: {batch_file}")
            
            # Load data from both datasets
            sliding_window_file_path = os.path.join(sliding_window_category_path, batch_file)
            statistical_features_file_path = os.path.join(statistical_features_category_path, batch_file)
            
            sliding_window_data = np.load(sliding_window_file_path)
            statistical_features_data = np.load(statistical_features_file_path)
            
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
            
            current_anomaly_ratio = len(anomaly_indices) / total_samples
            
            # Determine whether we need to undersample normal or anomaly samples
            if current_anomaly_ratio <= target_anomaly_ratio:
                # Current anomaly ratio is lower than target, need to reduce normal samples
                # Calculate how many normal samples to keep
                target_normal_count = int(len(anomaly_indices) * (1 - target_anomaly_ratio) / target_anomaly_ratio)
                
                if len(normal_indices) <= target_normal_count:
                    # We don't have enough normal samples to reduce further
                    logger.info(f"Cannot achieve target ratio - not enough normal samples to reduce. Keeping all.")
                    selected_indices = np.arange(total_samples)
                else:
                    # Undersample normal samples
                    logger.info(f"Undersampling normal samples from {len(normal_indices)} to {target_normal_count}")
                    np.random.shuffle(normal_indices)
                    undersampled_normal_indices = normal_indices[:target_normal_count]
                    selected_indices = np.sort(np.concatenate([undersampled_normal_indices, anomaly_indices]))
            else:
                # Current anomaly ratio is higher than target, need to reduce anomaly samples
                # Calculate how many anomaly samples to keep
                target_anomaly_count = int(len(normal_indices) * target_anomaly_ratio / (1 - target_anomaly_ratio))
                
                logger.info(f"Undersampling anomaly samples from {len(anomaly_indices)} to {target_anomaly_count}")
                np.random.shuffle(anomaly_indices)
                undersampled_anomaly_indices = anomaly_indices[:target_anomaly_count]
                selected_indices = np.sort(np.concatenate([normal_indices, undersampled_anomaly_indices]))
            
            # Subset both datasets using the same indices to maintain alignment
            # For sliding window dataset
            undersampled_windows = windows[selected_indices]
            undersampled_labels = labels[selected_indices]
            
            # For statistical features dataset - only extract and save 'stat_features'
            stat_features = statistical_features_data['stat_features'][selected_indices]
            
            # Verify all necessary keys exist
            if 'stat_features' not in statistical_features_data:
                logger.error(f"'stat_features' key not found in {batch_file}!")
                continue
            
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
            
            # Save undersampled datasets
            sliding_window_output_file = os.path.join(sliding_window_output_category_dir, batch_file)
            np.savez(sliding_window_output_file, windows=undersampled_windows, labels=undersampled_labels)
            
            statistical_features_output_file = os.path.join(statistical_features_output_category_dir, batch_file)
            np.savez(statistical_features_output_file, stat_features=stat_features)
            
            logger.info(f"Saved resampled data for batch {batch_file}")
    
    logger.info(f"Completed resampling to {target_anomaly_ratio*100:.0f}% anomaly ratio")


if __name__ == "__main__":
    sliding_window_path = "Data/deepLearning/transform/slidingWindow_noOverlap_600_600_100_0_0.5/train"
    statistical_features_path = "Data/deepLearning/transform/statistic_features_600_600_100_0_standardized/train"
    undersample_normal_windows(sliding_window_path, statistical_features_path, target_anomaly_ratio=0.20)
