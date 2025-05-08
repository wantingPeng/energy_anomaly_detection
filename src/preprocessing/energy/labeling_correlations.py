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


def analyze_correlations(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Analyze correlations between features and anomaly labels."""
    # Calculate correlations
    correlations = df.corr(method=config['correlation']['method'])['anomaly_label'].abs()
    correlations = correlations.sort_values(ascending=False)
    
    # Filter correlations above threshold
    correlations = correlations[correlations > config['correlation']['min_correlation']]
    
    return correlations


def generate_report(df: pd.DataFrame, 
                   correlations: pd.Series,
                   config: dict,
                   station_name: str) -> str:
    """Generate a markdown report with analysis results."""
    total_windows = len(df)
    anomaly_windows = df['anomaly_label'].sum()
    anomaly_ratio = anomaly_windows / total_windows
    
    report = f"""# Energy Data Labeling and Correlation Analysis Report
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Station: {station_name}

### Label Distribution
- Total Windows: {total_windows}
- Anomaly Windows: {anomaly_windows}
- Anomaly Ratio: {anomaly_ratio:.2%}

### Top Correlated Features
"""
    
    # Add top correlated features
    top_features = correlations.head(config['correlation']['top_n_features'])
    for feature, corr in top_features.items():
        report += f"- {feature}: {corr:.3f}\n"
    
    return report


def calculate_window_features(window: pd.DataFrame) -> pd.Series:
    """
    Calculate statistical features for a given window.
    
    Args:
        window: DataFrame containing the window data
        
    Returns:
        pd.Series: Statistical features for the window
    """
    features = {}
    
    # 需要排除的元数据列
    metadata_cols = ['IsOutlier', 'time_diff', 'segment_id', 'ID', 'TimeStamp', 'Station']
    
    # 获取数值类型的列，排除元数据列
    numeric_cols = window.select_dtypes(include=[np.number]).columns
    feature_cols = [col for col in numeric_cols if col not in metadata_cols]
    
    # 对每个特征列计算统计量
    for col in feature_cols:
        values = window[col].values
        if len(values) > 0:  # 确保窗口内有数据
            # 基本统计量
            mean_val = np.mean(values)
            std_val = np.std(values)
            min_val = np.min(values)
            max_val = np.max(values)
            
            features.update({
                f"{col}_mean": mean_val,
                f"{col}_std": std_val,
                f"{col}_min": min_val,
                f"{col}_max": max_val,
                f"{col}_range": max_val - min_val,
            })
            
            # 安全计算z-score
            if std_val > 1e-10:  # 避免除以接近零的标准差
                features[f"{col}_z_score"] = np.mean((values - mean_val) / std_val)
            else:
                features[f"{col}_z_score"] = 0.0
            
            # 安全计算偏度
            try:
                # 只在数据有足够变化时计算偏度
                if std_val > 1e-10:
                    skew_val = stats.skew(values)
                    features[f"{col}_skew"] = skew_val if not np.isnan(skew_val) else 0.0
                else:
                    features[f"{col}_skew"] = 0.0
            except Exception as e:
                logger.warning(f"Skewness calculation failed for {col}: {str(e)}")
                features[f"{col}_skew"] = 0.0
            
            # 计算趋势斜率
            try:
                x = np.arange(len(values))
                if len(values) > 1 and np.any(np.diff(values) != 0):  # 只在数据有变化时计算斜率
                    slope, _ = np.polyfit(x, values, 1)
                    features[f"{col}_trend_slope"] = slope
                else:
                    features[f"{col}_trend_slope"] = 0.0
            except Exception as e:
                logger.warning(f"Trend slope calculation failed for {col}: {str(e)}")
                features[f"{col}_trend_slope"] = 0.0
            
            # 计算差分特征
            if len(values) > 1:
                diffs = np.diff(values)
                features[f"{col}_diff_mean"] = np.mean(diffs) if len(diffs) > 0 else 0.0
                features[f"{col}_diff_std"] = np.std(diffs) if len(diffs) > 0 else 0.0
            else:
                features[f"{col}_diff_mean"] = 0.0
                features[f"{col}_diff_std"] = 0.0
    
    # 添加时间相关的元数据
    features['window_start'] = window['TimeStamp'].min()
    features['window_end'] = window['TimeStamp'].max()
    features['segment_id'] = window['segment_id'].iloc[0]
    
    return pd.Series(features)


def label_and_analyze_correlations(energy_data_path: str, station_name: str) -> None:
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
        
        # Initialize results storage
        window_features = []
        
        window_size = pd.Timedelta(seconds=config['sliding_window']['window_size'])
        step_size = pd.Timedelta(seconds=config['sliding_window']['step_size'])

        # Process each segment
        for segment_id, segment in df.groupby('segment_id'):
            logger.info(f"Processing segment {segment_id}")
            
            # Ensure timestamp column is datetime type and sorted
            segment['TimeStamp'] = pd.to_datetime(segment['TimeStamp'])
            segment = segment.sort_values('TimeStamp')
            
            # Get timestamps array
            timestamps = segment['TimeStamp']
            start_time = timestamps.iloc[0]
            end_time = start_time + window_size
            
            # Sliding window processing
            while end_time <= timestamps.iloc[-1]:
                # Get current window data
                window_mask = (segment['TimeStamp'] >= start_time) & (segment['TimeStamp'] < end_time)
                window = segment.loc[window_mask]
                
                if not window.empty:
                    # Calculate overlap ratio with anomaly periods
                    overlap_ratio = calculate_window_overlap(
                        start_time,
                        end_time,
                        interval_tree
                    )
                    
                    # Label window as anomaly if overlap threshold is met
                    is_anomaly = overlap_ratio >= config['sliding_window']['overlap_threshold']
                    
                    # Calculate window features
                    features = calculate_window_features(window)
                    features['anomaly_label'] = is_anomaly
                    features['overlap_ratio'] = overlap_ratio
                    
                    window_features.append(features)
                
                # Move to next window
                start_time += step_size
                end_time += step_size
        
        # Create DataFrame from window features
        labeled_df = pd.DataFrame(window_features)
        
        # Analyze correlations
        correlations = analyze_correlations(labeled_df, config)
        
        # Generate and save report
        report = generate_report(labeled_df, correlations, config, station_name)
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
        
        # Plot correlation heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(labeled_df.corr()[['anomaly_label']].sort_values('anomaly_label', ascending=False),
                    annot=True, cmap='coolwarm')
        plt.title(f'Feature Correlations with Anomaly Label - {station_name}')
        plt.tight_layout()
        
        # Save correlation plot
        plot_path = os.path.join(config['paths']['output_dir'], 
                                f"{station_name}_correlations.png")
        plt.savefig(plot_path)
        plt.close()
        
    except Exception as e:
        logger.error(f"Error in label_and_analyze_correlations: {str(e)}")
        raise


if __name__ == "__main__":
    # Example usage
    energy_data_path = "Data/interim/Energy_time_series/ring_20250508_132202"
    station_name = "Ringmontage"
    label_and_analyze_correlations(energy_data_path, station_name) 