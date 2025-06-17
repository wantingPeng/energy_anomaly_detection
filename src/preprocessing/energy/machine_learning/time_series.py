import os
import pandas as pd
import matplotlib.pyplot as plt
import dask.dataframe as dd
from src.utils.logger import logger

def analyze_time_series(df):
    """
    Analyze time series characteristics and add time-based features.
    
    Args:
        df: pandas DataFrame
    
    Returns:
        DataFrame with added features, time series statistics
    """
    gap_threshold = 15
    window_size = 600

    # Sort by timestamp
    df = df.sort_values("TimeStamp")

    # Calculate time difference
    df['time_diff'] = df["TimeStamp"].diff().dt.total_seconds()
    
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
        'segment_lengths': segment_lengths[segment_lengths >= window_size].describe().apply(lambda x: f"{int(x):,}")
    }
    
    return filtered_df, time_stats, segment_stats

def main():
    # Manually load data
    input_file = "Data/machine/cleaning_utc/Contacting_cleaned.parquet"
    logger.info(f"Loading data from: {input_file}")
    df = pd.read_parquet(input_file)
    
    logger.info(f"Data type: {type(df)}")
    print("验证加载后数据:")
    print(df.head())
    
    # Standardize timestamp
    df["TimeStamp"] = pd.to_datetime(df["TimeStamp"], utc=True)
    
    # Perform time series analysis
    filtered_df, time_stats, segment_stats = analyze_time_series(df)

    report_content = ["# time series Report\n\n"]

    report_content.append("\n## Time Series Analysis\n")
    report_content.append("### Time Interval Statistics\n")
    report_content.append(f"- Minimum interval: {time_stats['min_interval']:.3f} seconds\n")
    report_content.append(f"- Maximum interval: {time_stats['max_interval']:.3f} seconds\n")
    report_content.append(f"- Mean interval: {time_stats['mean_interval']:.3f} seconds\n")
    report_content.append(f"- Standard deviation: {time_stats['std_interval']:.3f} seconds\n")
    
    report_content.append("\n### Segmentation Statistics\n")
    report_content.append(f"- Total segments: {segment_stats['total_segments']}\n")
    report_content.append(f"- Valid segments (length >= 600): {segment_stats['valid_segments']}\n")
    report_content.append(f"- Filtered segments percentage: {segment_stats['filtered_percentage']:.2f}%\n")
    report_content.append(f"- Segment details: {segment_stats['segment_lengths']}\n")
    
    report_path = "Data/machine/Energy_time_series/Energy_Data_contact_time_series.md"
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_content))
    
    logger.info("Verifying data before saving:")
    logger.info(filtered_df.head())

    # Save the cleaned and sorted dataset
    output_dir = "Data/machine/Energy_time_series"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "Contacting_time_series.parquet")
    filtered_df.to_parquet(output_file)
    logger.info(f"Saved time series data to: {output_file}")
    
    return filtered_df

if __name__ == "__main__":
    main()