#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This script calculates the ratio between normal and anomalous data in a parquet file.
"""

import os
import pandas as pd
from datetime import datetime
from src.utils.logger import logger

def calculate_anomaly_ratio(file_path: str, verbose: bool = True):
    """
    Calculate the ratio between normal and anomalous data in a parquet file.
    
    Args:
        file_path: Path to the parquet file
        verbose: Whether to print progress information
        
    Returns:
        tuple: (normal_count, anomaly_count, normal_percentage, anomaly_percentage, normal_anomaly_ratio)
    """
    # Set up logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"calculate_anomaly_ratio_{timestamp}.log"
    log_path = os.path.join("experiments/logs", log_filename)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    
    logger.info(f"Loading data from {file_path}")
    df = pd.read_parquet(file_path)
    
    # Check if anomaly_label column exists
    if 'anomaly_label' not in df.columns:
        error_msg = f"Column 'anomaly_label' not found in {file_path}. Available columns: {df.columns.tolist()}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    # Get distribution
    anomaly_count = df[df['anomaly_label'] == 1].shape[0]
    normal_count = df[df['anomaly_label'] == 0].shape[0]
    total_count = len(df)
    
    # Calculate percentages
    normal_percentage = normal_count / total_count * 100
    anomaly_percentage = anomaly_count / total_count * 100
    
    # Calculate ratio (normal:anomaly)
    if anomaly_count > 0:
        normal_anomaly_ratio = normal_count / anomaly_count
    else:
        normal_anomaly_ratio = float('inf')  # Avoid division by zero
    
    # Log results
    logger.info(f"Data distribution in {file_path}:")
    logger.info(f"  Total samples: {total_count}")
    logger.info(f"  Normal samples (label=0): {normal_count} ({normal_percentage:.2f}%)")
    logger.info(f"  Anomaly samples (label=1): {anomaly_count} ({anomaly_percentage:.2f}%)")
    logger.info(f"  Ratio normal:anomaly = {normal_percentage:.1f}:{anomaly_percentage:.1f} ({normal_anomaly_ratio:.2f}:1)")
    
    return (normal_count, anomaly_count, normal_percentage, anomaly_percentage, normal_anomaly_ratio)

def main():
    # File path
    file_path = "Data/processed/machinen_learning/individual_model/randomly_spilt/train.parquet"
    
    # Calculate ratio
    calculate_anomaly_ratio(file_path)

if __name__ == "__main__":
    main()
