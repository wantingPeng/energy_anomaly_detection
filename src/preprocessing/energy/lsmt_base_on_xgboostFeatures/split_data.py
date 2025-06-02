import os
import numpy as np
from tqdm import tqdm
from src.utils.logger import logger
import json
import shutil

def split_data(input_dir, output_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    Split the sliding window data into train, validation, and test sets.
    
    Args:
        input_dir (str): Directory containing the data
        output_dir (str): Directory to save the split data
        train_ratio (float, optional): Ratio of training data. Defaults to 0.7.
        val_ratio (float, optional): Ratio of validation data. Defaults to 0.15.
        test_ratio (float, optional): Ratio of test data. Defaults to 0.15.
    """
    logger.info(f"Splitting data with ratios - Train: {train_ratio}, Validation: {val_ratio}, Test: {test_ratio}")
    
    # Load data
    X_file = os.path.join(input_dir, "X_sequences.npy")
    y_file = os.path.join(input_dir, "y_labels.npy")
    
    logger.info(f"Loading data from {X_file} and {y_file}")
    X = np.load(X_file, mmap_mode='r')  # Use memory-mapping for large arrays
    y = np.load(y_file)
    
    # Get total number of samples
    num_samples = len(y)
    logger.info(f"Total samples: {num_samples}")
    
    # Create random indices for shuffling
    indices = np.random.permutation(num_samples)
    
    # Calculate split sizes
    train_size = int(train_ratio * num_samples)
    val_size = int(val_ratio * num_samples)
    
    # Get indices for each split
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    logger.info(f"Split sizes - Train: {len(train_indices)}, Validation: {len(val_indices)}, Test: {len(test_indices)}")
    
    # Create output directories
    train_dir = os.path.join(output_dir, "train")
    val_dir = os.path.join(output_dir , "val")
    test_dir = os.path.join(output_dir , "test")
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    # Split and save data
    logger.info("Saving training data...")
    np.save(os.path.join(train_dir, "X_sequences.npy"), X[train_indices])
    np.save(os.path.join(train_dir, "y_labels.npy"), y[train_indices])
    
    logger.info("Saving validation data...")
    np.save(os.path.join(val_dir, "X_sequences.npy"), X[val_indices])
    np.save(os.path.join(val_dir, "y_labels.npy"), y[val_indices])
    
    logger.info("Saving test data...")
    np.save(os.path.join(test_dir, "X_sequences.npy"), X[test_indices])
    np.save(os.path.join(test_dir, "y_labels.npy"), y[test_indices])
    
    # Create and save metadata for each split
    splits = {
        "train": {
            "size": len(train_indices),
            "ratio": train_ratio,
            "normal_samples": int(np.sum(y[train_indices] == 0)),
            "anomaly_samples": int(np.sum(y[train_indices] == 1)),
            "normal_ratio": float(np.sum(y[train_indices] == 0) / len(train_indices)),
            "anomaly_ratio": float(np.sum(y[train_indices] == 1) / len(train_indices))
        },
        "val": {
            "size": len(val_indices),
            "ratio": val_ratio,
            "normal_samples": int(np.sum(y[val_indices] == 0)),
            "anomaly_samples": int(np.sum(y[val_indices] == 1)),
            "normal_ratio": float(np.sum(y[val_indices] == 0) / len(val_indices)),
            "anomaly_ratio": float(np.sum(y[val_indices] == 1) / len(val_indices))
        },
        "test": {
            "size": len(test_indices),
            "ratio": test_ratio,
            "normal_samples": int(np.sum(y[test_indices] == 0)),
            "anomaly_samples": int(np.sum(y[test_indices] == 1)),
            "normal_ratio": float(np.sum(y[test_indices] == 0) / len(test_indices)),
            "anomaly_ratio": float(np.sum(y[test_indices] == 1) / len(test_indices))
        }
    }
    
    # Load original metadata
    original_metadata_file = os.path.join(input_dir, "metadata.txt")
    original_metadata = {}
    if os.path.exists(original_metadata_file):
        with open(original_metadata_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if ":" in line and not line.startswith("\n"):
                    key, value = line.strip().split(":", 1)
                    original_metadata[key.strip()] = value.strip()
    
    # Save metadata for each split
    for split_name, split_dir in [("train", train_dir), ("val", val_dir), ("test", test_dir)]:
        metadata = {
            "X_shape": tuple([splits[split_name]["size"]] + list(X.shape[1:])),
            "y_shape": (splits[split_name]["size"],),
            "window_size": 6,
            "step_size": 2,
            "feature_count": X.shape[2],
            "normal_samples": splits[split_name]["normal_samples"],
            "anomaly_samples": splits[split_name]["anomaly_samples"],
            "normal_ratio": splits[split_name]["normal_ratio"],
            "anomaly_ratio": splits[split_name]["anomaly_ratio"],
            "original_data": original_metadata
        }
        
        with open(os.path.join(split_dir, "metadata.txt"), 'w') as f:
            for key, value in metadata.items():
                if key != "original_data":
                    f.write(f"{key}: {value}\n")
            
            f.write("\nClass Distribution:\n")
            f.write(f"Normal sequences (0): {metadata['normal_samples']} ({metadata['normal_ratio']*100:.2f}%)\n")
            f.write(f"Anomaly sequences (1): {metadata['anomaly_samples']} ({metadata['anomaly_ratio']*100:.2f}%)\n")
        
        with open(os.path.join(split_dir, "metadata.json"), 'w') as f:
            json.dump(metadata, f, indent=4)
    
    # Save overall split metadata
    split_metadata = {
        "original_data_shape": X.shape,
        "train_samples": len(train_indices),
        "validation_samples": len(val_indices),
        "test_samples": len(test_indices),
        "train_ratio": train_ratio,
        "validation_ratio": val_ratio,
        "test_ratio": test_ratio,
        "splits": splits
    }
    
    with open(os.path.join(output_dir, "split_metadata.json"), 'w') as f:
        json.dump(split_metadata, f, indent=4)
    
    with open(os.path.join(output_dir,  "split_metadata.txt"), 'w') as f:
        f.write(f"Data Split Summary\n")
        f.write(f"=================\n\n")
        f.write(f"Original data shape: {X.shape}\n\n")
        f.write(f"Split Sizes:\n")
        f.write(f"  Train: {len(train_indices)} samples ({train_ratio*100:.2f}%)\n")
        f.write(f"  Validation: {len(val_indices)} samples ({val_ratio*100:.2f}%)\n")
        f.write(f"  Test: {len(test_indices)} samples ({test_ratio*100:.2f}%)\n\n")
        
        # Use consistent keys for splits dictionary
        for split_name, display_name in [("train", "Train"), ("val", "Validation"), ("test", "Test")]:
            f.write(f"{display_name} Set Class Distribution:\n")
            f.write(f"  Normal: {splits[split_name]['normal_samples']} ({splits[split_name]['normal_ratio']*100:.2f}%)\n")
            f.write(f"  Anomaly: {splits[split_name]['anomaly_samples']} ({splits[split_name]['anomaly_ratio']*100:.2f}%)\n\n")
    
    logger.info(f"Data split completed successfully")
    logger.info(f"Data saved to {output_dir}")

def main():
    """
    Main function to split the data.
    """
    try:
        logger.info("Starting data splitting process")
        
        # Input and output paths
        input_dir = "Data/processed/lsmt_base_on_xgboostFeatures/slidingWindow/contact"
        output_dir = "Data/processed/lsmt_base_on_xgboostFeatures/split/contact"
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Split the data
        split_data(input_dir, output_dir)
        
        logger.info("Data splitting completed successfully")
    
    except Exception as e:
        logger.error(f"Error in data splitting process: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main() 