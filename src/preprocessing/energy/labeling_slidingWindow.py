import os
import pickle
import yaml
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from datetime import datetime
from intervaltree import IntervalTree
from src.utils.logger import logger
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from src.preprocessing.energy.calculate_window_features import calculate_window_features
from multiprocessing import Pool, cpu_count
from functools import partial
from joblib import Parallel, delayed
import gc  # Add garbage collection to release memory


def load_config() -> dict:
    """Load preprocessing configuration from YAML file."""
    config_path = "configs/preprocessing.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_anomaly_dict(config: dict) -> Dict[str, List[Tuple[str, str]]]:
    """Load anomaly dictionary from pickle file."""
    try:
        with open(config['paths']['anomaly_dict'], 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        logger.error(f"Error loading anomaly dictionary: {str(e)}")
        raise


def create_interval_tree(anomaly_periods: List[Tuple[str, str]]) -> IntervalTree:
    """Create an interval tree from anomaly periods."""
    tree = IntervalTree()
    for start, end in anomaly_periods:
        # Convert string timestamps to integers for interval tree
        start_int = int(pd.Timestamp(start).timestamp())
        end_int = int(pd.Timestamp(end).timestamp())
        
        # 如果开始时间和结束时间相同，将结束时间加1秒
        if start_int == end_int:
            end_int += 1
            
        tree[start_int:end_int] = True
    return tree


def calculate_window_overlap(window_start: pd.Timestamp, 
                           window_end: pd.Timestamp,
                           interval_tree: IntervalTree) -> float:
    """Calculate the overlap ratio between a window and anomaly periods."""
    window_start_int = int(window_start.timestamp())
    window_end_int = int(window_end.timestamp())
    
    # Find overlapping intervals
    overlapping_intervals = interval_tree[window_start_int:window_end_int]
    
    if not overlapping_intervals:
        return 0.0
    
    # Calculate total overlap duration
    total_overlap = 0
    for interval in overlapping_intervals:
        overlap_start = max(window_start_int, interval.begin)
        overlap_end = min(window_end_int, interval.end)
        total_overlap += overlap_end - overlap_start
    
    # Calculate overlap ratio
    window_duration = window_end_int - window_start_int
    return total_overlap / window_duration


def generate_report(df: pd.DataFrame, 
                   config: dict,
                   station_name: str) -> str:
    """Generate a markdown report with analysis results."""
    total_windows = len(df)
    anomaly_windows = df['anomaly_label'].sum()
    anomaly_ratio = anomaly_windows / total_windows
    
    report = f"""# Energy Data Labeling Report
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Station: {station_name}

### Label Distribution
- Total Windows: {total_windows}
- Anomaly Windows: {anomaly_windows}
- Anomaly Ratio: {anomaly_ratio:.2%}

"""

    
    return report


def process_segment(segment_data: Tuple[str, pd.DataFrame], 
                   interval_tree: IntervalTree,
                   config: dict) -> List[dict]:
    """
    处理单个segment的函数，用于并行处理
    
    Args:
        segment_data: (segment_id, segment_df) 元组
        interval_tree: 用于检查异常重叠的区间树
        config: 配置字典
    
    Returns:
        List[dict]: 该segment的所有窗口特征列表
    """
    segment_id, segment = segment_data
    logger.info(f"Processing segment {segment_id}")
    
    try:
        window_features = []
        window_size = pd.Timedelta(seconds=config['sliding_window']['window_size'])
        normal_step_size = pd.Timedelta(seconds=60)
        anomaly_step_size = pd.Timedelta(seconds=10)
        
        segment['TimeStamp'] = pd.to_datetime(segment['TimeStamp'])
        segment = segment.sort_values('TimeStamp')
        
        timestamps = segment['TimeStamp']
        start_time = timestamps.iloc[0]
        end_time = start_time + window_size
        
        while end_time <= timestamps.iloc[-1]:
            current_overlap = calculate_window_overlap(
                start_time,
                end_time,
                interval_tree
            )
            
            step_size = anomaly_step_size if current_overlap > 0 else normal_step_size
            
            window_mask = (segment['TimeStamp'] >= start_time) & (segment['TimeStamp'] < end_time)
            window = segment.loc[window_mask]
            
            if not window.empty:
                features = calculate_window_features(window).to_dict()
                features['anomaly_label'] = current_overlap >= config['sliding_window']['overlap_threshold']
                features['overlap_ratio'] = current_overlap
                features['step_size'] = step_size.total_seconds()
                
                window_features.append(features)
            
            start_time += step_size
            end_time += step_size
        
        return window_features

    except Exception as e:
        logger.error(f"Error processing segment {segment_id}: {str(e)}")
        return []


def label(energy_data_path: str, station_name: str) -> None:
    """
    Main function to label energy data and analyze correlations.
    
    Args:
        energy_data_path: Path to the energy data directory
        station_name: Name of the station to process
    """
    try:
        # Load configuration
        config = load_config()
        
        # Load anomaly dictionary
        anomaly_dict = load_anomaly_dict(config)
        if station_name not in anomaly_dict:
            raise ValueError(f"Station {station_name} not found in anomaly dictionary")
        
        # Create interval tree for efficient overlap checking
        interval_tree = create_interval_tree(anomaly_dict[station_name])
        
        # Load energy data
        logger.info(f"Loading energy data from {energy_data_path}")
        df = pd.read_parquet(energy_data_path)
        
        # 准备并行处理
        n_jobs = 20  # Set to use 20 processes
        logger.info(f"Using {n_jobs} threads for parallel processing")
        
        # 使用 joblib 进行并行处理
        results = Parallel(n_jobs=n_jobs, prefer='threads')(  # Use threads to reduce data copying
            delayed(process_segment)(segment_data, interval_tree, config)
            for segment_data in df.groupby('segment_id')
        )
        
        # 合并所有结果
        window_features = []
        for segment_features in results:
            window_features.extend(segment_features)
        
        # Explicitly release memory
        del results
        gc.collect()
        
        # Create DataFrame from window features
        labeled_df = pd.DataFrame(window_features)
        
        # Generate and save report
        report = generate_report(labeled_df, config, station_name)
        report_path = config['paths']['report_file']
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        with open(report_path, 'w') as f:
            f.write(report)
        
        # Save labeled data
        output_path = os.path.join(config['paths']['output_dir'], 
                                 f"{station_name}_labeled.parquet")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        labeled_df.to_parquet(output_path)
        
        logger.info(f"Successfully processed and saved labeled data for {station_name}")
        
        # Additional statistics
        logger.info("\nAdditional Statistics:")
        logger.info(f"Total windows processed: {len(labeled_df)}")
        logger.info(f"Anomaly windows: {labeled_df['anomaly_label'].sum()}")
        logger.info(f"Anomaly ratio: {labeled_df['anomaly_label'].mean():.2%}")
        logger.info(f"Average overlap ratio: {labeled_df['overlap_ratio'].mean():.2%}")

    except Exception as e:
        logger.error(f"Error in label_and_analyze_correlations: {str(e)}")
        raise


if __name__ == "__main__":
    # Example usage
    energy_data_path = "Data/interim/Energy_time_series/contact_20250509_093729/part.0.parquet"
    station_name = "Kontaktieren"
    label(energy_data_path, station_name) 