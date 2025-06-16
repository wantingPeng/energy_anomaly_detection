import pandas as pd
import numpy as np
import os
from src.utils.logger import logger

def balance_validation_data(ratio_normal=0.94, random_seed=42, save_dir="Data/processed/machinen_learning"):
    """
    Load validation data from parquet file and balance it to achieve a specified normal-to-anomaly ratio
    
    Args:
        ratio_normal (float): Desired ratio of normal samples (default: 0.94)
        random_seed (int): Random seed for reproducibility (default: 42)
        save_dir (str): Directory to save the balanced data (default: "Data/processed/machinen_learning/val_94:6")
        
    Returns:
        pd.DataFrame: Balanced validation dataset
    """
    logger.info(f"Loading validation data and balancing to {ratio_normal*100}:{(1-ratio_normal)*100} ratio")
    
    # Set random seed for reproducibility
    np.random.seed(random_seed)
    
    # Load the validation dataset
    file_path = "Data/processed/machinen_learning/individual_model/randomly_spilt/val.parquet"
    try:
        df = pd.read_parquet(file_path)
        logger.info(f"Successfully loaded validation data from {file_path}")
        logger.info(f"Original data shape: {df.shape}")
    except Exception as e:
        logger.error(f"Error loading validation data: {e}")
        raise
    
    # Assuming the label column indicates anomaly (1) or normal (0)
    # Adjust the column name if it's different in your dataset
    label_column = "anomaly_label"
    
    # Separate normal and anomaly samples
    normal_samples = df[df[label_column] == 0]
    anomaly_samples = df[df[label_column] == 1]
    
    logger.info(f"Original normal samples: {len(normal_samples)}, anomaly samples: {len(anomaly_samples)}")
    logger.info(f"Original ratio - normal: {len(normal_samples) / len(df):.4f}, "
                f"anomaly: {len(anomaly_samples) / len(df):.4f}")
    
    # Calculate total size to maintain and number of samples for each class
    if len(anomaly_samples) / (len(normal_samples) + len(anomaly_samples)) > (1 - ratio_normal):
        # If we have too many anomalies, we need to downsample anomalies
        # Calculate how many anomaly samples we need
        total_samples = len(anomaly_samples) / (1 - ratio_normal)
        needed_normal_samples = int(total_samples * ratio_normal)
        needed_anomaly_samples = int(total_samples * (1 - ratio_normal))
        
        # If we need more normal samples than we have, we'll need to downsample anomalies further
        if needed_normal_samples > len(normal_samples):
            needed_anomaly_samples = int(len(normal_samples) * (1 - ratio_normal) / ratio_normal)
            normal_samples_balanced = normal_samples
            anomaly_samples_balanced = anomaly_samples.sample(n=needed_anomaly_samples, random_state=random_seed)
        else:
            normal_samples_balanced = normal_samples.sample(n=needed_normal_samples, random_state=random_seed)
            anomaly_samples_balanced = anomaly_samples.sample(n=needed_anomaly_samples, random_state=random_seed)
    else:
        # If we have too many normal samples, we need to downsample normal samples
        # Calculate how many normal samples we need
        total_samples = len(normal_samples) / ratio_normal
        needed_normal_samples = int(total_samples * ratio_normal)
        needed_anomaly_samples = int(total_samples * (1 - ratio_normal))
        
        # If we need more anomaly samples than we have, we'll need to downsample normal samples further
        if needed_anomaly_samples > len(anomaly_samples):
            needed_normal_samples = int(len(anomaly_samples) * ratio_normal / (1 - ratio_normal))
            normal_samples_balanced = normal_samples.sample(n=needed_normal_samples, random_state=random_seed)
            anomaly_samples_balanced = anomaly_samples
        else:
            normal_samples_balanced = normal_samples.sample(n=needed_normal_samples, random_state=random_seed)
            anomaly_samples_balanced = anomaly_samples.sample(n=needed_anomaly_samples, random_state=random_seed)
    
    # Combine the balanced datasets
    df_balanced = pd.concat([normal_samples_balanced, anomaly_samples_balanced])
    
    # Shuffle the data
    df_balanced = df_balanced.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    
    # Verify the ratio
    final_normal = len(df_balanced[df_balanced[label_column] == 0])
    final_anomaly = len(df_balanced[df_balanced[label_column] == 1])
    final_ratio_normal = final_normal / len(df_balanced)
    final_ratio_anomaly = final_anomaly / len(df_balanced)
    
    logger.info(f"Balanced data shape: {df_balanced.shape}")
    logger.info(f"Final normal samples: {final_normal}, anomaly samples: {final_anomaly}")
    logger.info(f"Final ratio - normal: {final_ratio_normal:.4f}, anomaly: {final_ratio_anomaly:.4f}")
    
    # Create directory if it doesn't exist  
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "val_94:6.parquet")
    
    # Save the balanced DataFrame to the specified location
    df_balanced.to_parquet(save_path)
    logger.info(f"Balanced dataset saved to {save_path}")
    
    return df_balanced

if __name__ == "__main__":
    # Example usage
    balanced_df = balance_validation_data(ratio_normal=0.94)
    logger.info("Data balancing completed")
