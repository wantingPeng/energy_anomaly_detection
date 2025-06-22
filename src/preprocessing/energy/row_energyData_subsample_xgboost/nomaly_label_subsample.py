#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This script implements downsampling of normal data (anomaly_label=0) to achieve
a specific ratio between normal and anomalous data.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from src.utils.logger import logger

def downsample_normal_data(
    input_file: str,
    output_dir: str,
    normal_anomaly_ratio: float = 3.0,  # 75:25 ratio means 3:1
    random_seed: int = 42,
    verbose: bool = True
):
    """
    Downsample the normal data (anomaly_label=0) to achieve a specific ratio with anomalous data.
    
    Args:
        input_file: Path to the input parquet file
        output_dir: Directory to save the downsampled data
        normal_anomaly_ratio: Desired ratio of normal to anomalous samples (default: 3.0 for 75:25)
        random_seed: Random seed for reproducibility
        verbose: Whether to print progress information
    
    Returns:
        Path to the saved downsampled file
    """
    # Set up logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"downsample_anomaly_label_{timestamp}.log"
    log_path = os.path.join("experiments/logs", log_filename)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    
    logger.info(f"Loading data from {input_file}")
    df = pd.read_parquet(input_file)
    
    # Get current distribution
    anomaly_count = df[df['anomaly_label'] == 1].shape[0]
    normal_count = df[df['anomaly_label'] == 0].shape[0]
    
    logger.info(f"Original data distribution:")
    logger.info(f"  Total samples: {len(df)}")
    logger.info(f"  Normal samples (label=0): {normal_count} ({normal_count/len(df)*100:.2f}%)")
    logger.info(f"  Anomaly samples (label=1): {anomaly_count} ({anomaly_count/len(df)*100:.2f}%)")
    logger.info(f"  Current ratio normal:anomaly = {normal_count/anomaly_count:.2f}:1")
    
    # Calculate how many normal samples to keep
    target_normal_count = int(anomaly_count * normal_anomaly_ratio)
    
    logger.info(f"Target ratio normal:anomaly = {normal_anomaly_ratio:.2f}:1")
    logger.info(f"Target normal samples: {target_normal_count}")
    logger.info(f"Will remove {normal_count - target_normal_count} normal samples")
    
    # Separate normal and anomaly data
    normal_df = df[df['anomaly_label'] == 0]
    anomaly_df = df[df['anomaly_label'] == 1]
    
    # Random sample from normal data to achieve desired ratio
    np.random.seed(random_seed)
    normal_df_sampled = normal_df.sample(n=target_normal_count, random_state=random_seed)
    
    # Combine sampled normal data with all anomaly data
    balanced_df = pd.concat([normal_df_sampled, anomaly_df], axis=0)
    
    # Shuffle the data
    balanced_df = balanced_df.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    
    # Get final distribution
    final_normal_count = balanced_df[balanced_df['anomaly_label'] == 0].shape[0]
    final_anomaly_count = balanced_df[balanced_df['anomaly_label'] == 1].shape[0]
    final_ratio = final_normal_count / final_anomaly_count
    
    logger.info(f"Final data distribution:")
    logger.info(f"  Total samples: {len(balanced_df)}")
    logger.info(f"  Normal samples (label=0): {final_normal_count} ({final_normal_count/len(balanced_df)*100:.2f}%)")
    logger.info(f"  Anomaly samples (label=1): {final_anomaly_count} ({final_anomaly_count/len(balanced_df)*100:.2f}%)")
    logger.info(f"  Final ratio normal:anomaly = {final_ratio:.2f}:1")
    
    # Create output directory if not exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Save downsampled data
    output_file = os.path.join(output_dir, f"train_downsampled_{int(normal_anomaly_ratio*100)}_{int(100-normal_anomaly_ratio*100)}.parquet")
    balanced_df.to_parquet(output_file, index=False)
    
    logger.info(f"Downsampled data saved to {output_file}")
    
    return output_file

def main():
    # Input file path
    input_file = "Data/row_energyData_subsample_xgboost/ranmdly_REspilt/contact/train.parquet"
    
    # Output directory
    output_dir = "Data/row_energyData_subsample_xgboost/ranmdly_REspilt/contact/train_downsampled"
    
    # Target ratio (75:25 means 3:1 ratio of normal to anomaly)
    normal_anomaly_ratio = 3.0
    
    # Perform downsampling
    output_file = downsample_normal_data(
        input_file=input_file,
        output_dir=output_dir,
        normal_anomaly_ratio=normal_anomaly_ratio,
        random_seed=42,
        verbose=True
    )
    
    logger.info(f"Downsampling completed. Data saved to: {output_file}")

if __name__ == "__main__":
    main() 