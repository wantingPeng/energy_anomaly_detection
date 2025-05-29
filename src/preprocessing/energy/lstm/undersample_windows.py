import os
import torch
import numpy as np
from src.utils.logger import logger


def undersample_normal_windows(dataset_path, seed=42):
    """
    Undersample normal windows in the dataset, retaining only 30% of them.

    Parameters:
    - dataset_path: str, path to the dataset directory
    - seed: int, random seed for reproducibility
    """
    # Set random seed for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    categories = ["ring", "pcb", "contact"]
    
    logger.info("Starting undersampling of normal windows")
    
    for category in categories:
        category_path = os.path.join(dataset_path, category)
        pt_files = [f for f in os.listdir(category_path) if f.endswith(".pt") and f.startswith("batch_")]
        
        logger.info(f"Processing {category} category with {len(pt_files)} pt files")
        
        # Create output directory for each category
        output_category_dir = os.path.join('Data/processed/lsmt/dataset/train_down_10%', category)
        os.makedirs(output_category_dir, exist_ok=True)
        
        for file in sorted(pt_files):  # Sort to process in order
            file_path = os.path.join(category_path, file)
            data = torch.load(file_path)
            
            labels = data['labels']
            normal_indices = (labels == 0).nonzero(as_tuple=True)[0]
            anomaly_indices = (labels == 1).nonzero(as_tuple=True)[0]
            normal_ratio = len(normal_indices) / (len(normal_indices) + len(anomaly_indices)) * 100
            anomaly_ratio = len(anomaly_indices) / (len(normal_indices) + len(anomaly_indices)) * 100

            logger.info(
                f"Original {file}: "
                f"Normal={normal_ratio:.2f}% ({len(normal_indices)}), "
                f"Anomaly={anomaly_ratio:.2f}% ({len(anomaly_indices)})"
            )
            # Undersample normal indices
            undersampled_normal_indices = normal_indices[torch.randperm(len(normal_indices))[:int(0.1 * len(normal_indices))]]
            
            # Combine undersampled normals with all anomalies
            balanced_indices = torch.cat((undersampled_normal_indices, anomaly_indices))
            
            # Shuffle the combined indices
            balanced_indices = balanced_indices[torch.randperm(len(balanced_indices))]
            
            # Subset the data using balanced indices
            balanced_data = {key: value[balanced_indices] for key, value in data.items()}
            
            # Save with original batch name
            balanced_file_path = os.path.join(output_category_dir, file)
            torch.save(balanced_data, balanced_file_path)
            
            # Log final counts
            final_normal_ratio = len(undersampled_normal_indices) / (len(undersampled_normal_indices) + len(anomaly_indices)) * 100
            final_anomaly_ratio = len(anomaly_indices) / (len(undersampled_normal_indices) + len(anomaly_indices)) * 100
            logger.info(
                f"Final {file}: "
                f"Normal={final_normal_ratio:.2f}% ({len(undersampled_normal_indices)}), "
                f"Anomaly={final_anomaly_ratio:.2f}% ({len(anomaly_indices)})"
            )
    logger.info("Completed undersampling of normal windows")


if __name__ == "__main__":
    dataset_path = "Data/processed/lsmt/dataset/train"
    undersample_normal_windows(dataset_path)
