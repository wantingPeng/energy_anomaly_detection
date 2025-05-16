import os
import pandas as pd
import numpy as np
from intervaltree import IntervalTree
from src.utils.logger import logger
import torch
from torch.utils.data import Dataset
import yaml
import pickle

class EnergyDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def load_config() -> dict:
    """Load preprocessing configuration from YAML file."""
    config_path = "configs/lsmt_preprocessing.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_interval_tree(anomaly_periods):
    """Create an interval tree from anomaly periods."""
    tree = IntervalTree()
    for start, end in anomaly_periods:
        start_int = int(pd.Timestamp(start).timestamp())
        end_int = int(pd.Timestamp(end).timestamp())
        if start_int == end_int:
            end_int += 1
        tree[start_int:end_int] = True
    return tree


def process_segment(segment, interval_tree, config):
    """Process a single segment to create sliding windows."""
    window_size = config['sliding_window']['window_size']
    step_size = config['sliding_window']['step_size']
    anomaly_threshold = config['sliding_window']['anomaly_threshold']

    segment['TimeStamp'] = pd.to_datetime(segment['TimeStamp'])
    segment = segment.sort_values('TimeStamp')
    segment = segment.set_index('TimeStamp').resample('1S').interpolate().reset_index()

    windows = []
    labels = []

    for start in range(0, len(segment) - window_size + 1, step_size):
        window = segment.iloc[start:start + window_size]
        window_start = window['TimeStamp'].iloc[0]
        window_end = window['TimeStamp'].iloc[-1]

        # Check for anomalies
        overlap = calculate_window_overlap(window_start, window_end, interval_tree)
        label = 1 if overlap >= anomaly_threshold else 0

        windows.append(window.drop(columns=['TimeStamp', 'IsOutlier']).values)
        labels.append(label)

    return windows, labels


def calculate_window_overlap(window_start, window_end, interval_tree):
    """Calculate the overlap ratio between a window and anomaly periods."""
    window_start_int = int(window_start.timestamp())
    window_end_int = int(window_end.timestamp())
    overlapping_intervals = interval_tree[window_start_int:window_end_int]
    if not overlapping_intervals:
        return 0.0
    total_overlap = sum(min(window_end_int, interval.end) - max(window_start_int, interval.begin) for interval in overlapping_intervals)
    window_duration = window_end_int - window_start_int
    return total_overlap / window_duration


def main():
    config = load_config()
    data_path = "Data/interim/Energy_time_series/contact_20250509_093729/part.0.parquet"
    logger.info(f"Loading data from {data_path}")
    df = pd.read_parquet(data_path)
    logger.info(f"Data loaded. Shape: {df.shape}")
    logger.info(f"First 10 rows:\n{df.iloc[:10, :2]}")

    df = df.drop(columns=['IsOutlier'])

    anomaly_dict_path = config['paths']['anomaly_dict']
    with open(anomaly_dict_path, 'rb') as f:
        anomaly_dict = pickle.load(f)

    interval_tree = create_interval_tree(anomaly_dict['contact'])

    segments = df.groupby('segment_id')
    all_windows = []
    all_labels = []

    for segment_id, segment in segments:
        windows, labels = process_segment(segment, interval_tree, config)
        all_windows.extend(windows)
        all_labels.extend(labels)

    dataset = EnergyDataset(np.array(all_windows), np.array(all_labels))
    output_path = "Data/interim/Energy_time_series/contact_sliding_windows.parquet"
    pd.DataFrame({'data': all_windows, 'labels': all_labels}).to_parquet(output_path)
    logger.info(f"Sliding windows saved to {output_path}")

if __name__ == "__main__":
    main()
