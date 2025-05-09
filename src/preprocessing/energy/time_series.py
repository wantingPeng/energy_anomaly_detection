import os
import pandas as pd
from venv import logger
from src.preprocessing.data_loader import data_loader
import matplotlib.pyplot as plt
from src.utils.logger import logger
import yaml
import dask.dataframe as dd
from src.preprocessing.data_save import data_save

def load_config():
    """Load preprocessing configuration."""
    with open('configs/preprocessing.yaml', 'r') as file:
        return yaml.safe_load(file)

def analyze_time_series(df, timestamp_col):
    """
    Analyze time series characteristics and add time-based features.
    
    Args:
        df: pandas DataFrame
        timestamp_col: name of timestamp column
    
    Returns:
        DataFrame with added features, time series statistics
    """
    config = load_config()
    gap_threshold = config['time']['gap_threshold']
    window_size = config['sliding_window']['window_size']

    # Sort by timestamp
    df = df.sort_values(timestamp_col)

    # Calculate time difference
    df['time_diff'] = df[timestamp_col].diff().dt.total_seconds()
    
    # Calculate time series statistics
    time_stats = {
        'min_interval': df['time_diff'].min(),
        'max_interval': df['time_diff'].max(),
        'mean_interval': df['time_diff'].mean(),
        'std_interval': df['time_diff'].std()
    }
    
    # Identify segments based on time gap threshold
    segment_changes = (df['time_diff'] > gap_threshold).astype(int).cumsum()
    df['segment_id'] = segment_changes
    
    # Calculate segment lengths
    segment_lengths = df.groupby("segment_id").size()
    
    # Filter segments based on window size
    valid_segments = segment_lengths[segment_lengths >= window_size].index
    filtered_df = df[df['segment_id'].isin(valid_segments)]
    
    # Calculate filtering statistics
    total_segments = len(segment_lengths)
    valid_segments_count = len(valid_segments)
    filtered_percentage = ((total_segments - valid_segments_count) / total_segments) * 100
    
    segment_stats = {
        'total_segments': total_segments,
        'valid_segments': valid_segments_count,
        'filtered_percentage': filtered_percentage,
        'segment_lengths': segment_lengths[segment_lengths >= window_size].describe()
    }
    
    return filtered_df, time_stats, segment_stats



def main():
    df = data_loader(
        "Data/interim/Energy_Data_cleaned/PCB_20250509_085245/part.0.parquet"
    )
    logger.info(f"Data type: {type(df)}")
    print("验证加载后数据:")
    print(df.head())
    config = load_config()
    # Convert Dask DataFrame to pandas for time series operations
    df = df.compute()
    # 3. Time series analysis
    timestamp_col = config['columns']['timestamp_col']
    if timestamp_col in df.columns:
        # Standardize timestamp
        df[timestamp_col] = pd.to_datetime(df[timestamp_col], utc=True)
        
        # Perform time series analysis
        filtered_df, time_stats, segment_stats = analyze_time_series(df, timestamp_col)

        
      # 直方图可视化 segment 长度分布（限最大1000显示）
        lengths = segment_stats['segment_lengths']
        lengths.plot.hist(bins=50)
        plt.xlabel("Segment Length (seconds)")
        plt.ylabel("Frequency")
        plt.title("Distribution of Segment Lengths (<1000s)")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        plt.tight_layout()
        plt.savefig("segment_length_hist.png", dpi=300)
        report_content = ["# time series Report\n\n"]

        report_content.append("\n## Time Series Analysis\n")
        report_content.append("### Time Interval Statistics\n")
        report_content.append(f"- Minimum interval: {time_stats['min_interval']:.3f} seconds\n")
        report_content.append(f"- Maximum interval: {time_stats['max_interval']:.3f} seconds\n")
        report_content.append(f"- Mean interval: {time_stats['mean_interval']:.3f} seconds\n")
        report_content.append(f"- Standard deviation: {time_stats['std_interval']:.3f} seconds\n")
        
        report_content.append("\n### Segmentation Statistics\n")
        report_content.append(f"- Total segments: {segment_stats['total_segments']}\n")
        report_content.append(f"- Valid segments (length >= {config['sliding_window']['window_size']}s): {segment_stats['valid_segments']}\n")
        report_content.append(f"- Filtered segments percentage: {segment_stats['filtered_percentage']:.2f}%\n")
        report_content.append(f"- Segment details: {segment_stats['segment_lengths']}\n")
    
    report_path = "experiments/reports/Energy_Data_pcb_time_series.md"
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, 'w') as f:
         f.write('\n'.join(report_content))
    logger.info("Verifying data before saving:")
    logger.info(filtered_df.head())

    # Convert back to Dask DataFrame for saving
    df = dd.from_pandas(filtered_df)
    # Save the cleaned and sorted dataset with year-month partitioning
    data_save(
        df=df,
        filename="PCB",
        save_dir="Data/interim/Energy_time_series",
        timestamp_col=timestamp_col
    )
    return df

if __name__ == "__main__":
    main()