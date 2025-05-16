"""
Test script for sliding window preprocessing.

This script tests the sliding window implementation with a small sample dataset.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pickle
from pathlib import Path

from src.utils.logger import logger
from src.preprocessing.energy.lstm.slinding_window import (
    load_config,
    create_sliding_windows,
    save_results,
    save_stats
)
from src.preprocessing.energy.labeling_slidingWindow import load_anomaly_dict, create_interval_tree
from src.preprocessing.energy.lstm.dataset import LSTMWindowDataset


def create_test_data(num_samples=1000):
    """
    Create a small test dataset for sliding window testing.
    
    Args:
        num_samples: Number of samples to create
        
    Returns:
        DataFrame with test data
    """
    logger.info(f"Creating test dataset with {num_samples} samples")
    
    # Create timestamps at 1-second intervals
    start_time = datetime(2023, 12, 31, 0, 0, 0)
    timestamps = [start_time + timedelta(seconds=i) for i in range(num_samples)]
    
    # Generate random feature values
    feature_data = np.random.randn(num_samples, 10)  # 10 features
    feature_cols = [f'feature_{i}' for i in range(10)]
    
    # Create segment IDs (5 segments)
    segment_ids = np.repeat(np.arange(1, 6), num_samples // 5 + 1)[:num_samples]
    
    # Create component type flags
    component_type_contact = np.zeros(num_samples)
    component_type_pcb = np.ones(num_samples)
    component_type_ring = np.zeros(num_samples)
    
    # Create DataFrame
    df = pd.DataFrame({
        'TimeStamp': timestamps,
        'segment_id': segment_ids,
        'component_type_contact': component_type_contact,
        'component_type_pcb': component_type_pcb,
        'component_type_ring': component_type_ring,
        **{feature_cols[i]: feature_data[:, i] for i in range(10)}
    })
    
    return df


def create_test_anomaly_dict():
    """
    Create a test anomaly dictionary for testing.
    
    Returns:
        Dictionary mapping station to anomaly intervals
    """
    logger.info("Creating test anomaly dictionary")
    
    # Create anomaly intervals as tuples of timestamps
    start1 = "2023-12-31 00:01:00"
    end1 = "2023-12-31 00:02:00"
    start2 = "2023-12-31 00:03:00"
    end2 = "2023-12-31 00:04:00"
    
    anomaly_dict = {
        "Station: Pcb": [
            (start1, end1),
            (start2, end2)
        ]
    }
    
    return anomaly_dict


def test_sliding_window_processing():
    """Test the sliding window processing pipeline."""
    logger.info("Testing sliding window processing")
    
    # Create test directory
    test_dir = Path("test_output")
    test_dir.mkdir(exist_ok=True)
    
    # Create test data
    df = create_test_data()
    
    # Create test anomaly dictionary
    anomaly_dict = create_test_anomaly_dict()
    
    # Save test anomaly dictionary
    test_anomaly_path = test_dir / "test_anomaly_dict.pkl"
    with open(test_anomaly_path, "wb") as f:
        pickle.dump(anomaly_dict, f)
    
    # Load config
    config = load_config()
    
    # Create a temporary config for loading anomaly dict
    anomaly_config = {'paths': {'anomaly_dict': str(test_anomaly_path)}}
    
    # For the test, we'll create interval trees directly
    anomaly_trees = {station: create_interval_tree(periods) for station, periods in anomaly_dict.items()}
    
    # Create sliding windows
    windows, labels, segment_ids, timestamps, stats = create_sliding_windows(
        df,
        config['sliding_window']['window_size'],
        config['sliding_window']['step_size'],
        anomaly_trees,
        config['sliding_window']['anomaly_threshold']
    )
    
    # Save results
    output_path = test_dir / "windows"
    save_results(
        windows,
        labels,
        segment_ids,
        timestamps,
        str(output_path),
        "pcb",
        "test"
    )
    
    # Save statistics
    report_path = test_dir / "reports"
    save_stats(stats, str(report_path), "pcb", "test")
    
    # Test loading dataset
    dataset = LSTMWindowDataset(str(output_path / "window_pcb_test.parquet"))
    logger.info(f"Successfully created dataset with {len(dataset)} samples")
    
    # Test getting an item
    if len(dataset) > 0:
        window_tensor, label_tensor = dataset[0]
        logger.info(f"Window shape: {window_tensor.shape}, Label: {label_tensor}")
    
    logger.info("Test completed successfully")


if __name__ == "__main__":
    test_sliding_window_processing() 