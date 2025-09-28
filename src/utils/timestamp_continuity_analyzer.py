import pandas as pd
import numpy as np
import os
from datetime import timedelta
import matplotlib.pyplot as plt
from src.utils.logger import logger
import time

def analyze_timestamp_continuity(file_path, time_column='TimeStamp', min_gap_to_report=timedelta(minutes=5)):
    """
    Analyze the continuity of timestamp values in a dataset.
    
    Args:
        file_path (str): Path to the parquet file
        time_column (str): Name of the timestamp column
        min_gap_to_report (timedelta): Minimum gap duration to report
        
    Returns:
        dict: Analysis results
    """
    start_time = time.time()
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_filename = f"timestamp_continuity_analysis_{timestamp}.log"
    
    logger.info(f"Loading dataset from {file_path}")
    try:
        df = pd.read_parquet(file_path)
    except Exception as e:
        logger.error(f"Failed to load the dataset: {e}")
        return None
    
    logger.info(f"Dataset shape: {df.shape}")
    
    # Ensure the timestamp column exists
    if time_column not in df.columns:
        logger.error(f"Timestamp column '{time_column}' not found in the dataset")
        return None
    
    # Ensure the timestamp column is in datetime format
    if not pd.api.types.is_datetime64_any_dtype(df[time_column]):
        logger.info(f"Converting {time_column} to datetime")
        df[time_column] = pd.to_datetime(df[time_column], utc=True)
    
    # Sort by timestamp
    df = df.sort_values(by=time_column)
    
    # Calculate time differences
    logger.info("Calculating time differences between consecutive timestamps")
    df['time_diff'] = df[time_column].diff()
    
    # Summary statistics
    stats = {
        'total_points': len(df),
        'start_time': df[time_column].min(),
        'end_time': df[time_column].max(),
        'time_span': df[time_column].max() - df[time_column].min(),
        'median_interval': df['time_diff'].median(),
        'mean_interval': df['time_diff'].mean(),
        'min_interval': df['time_diff'].min(),
        'max_interval': df['time_diff'].max(),
        'std_interval': df['time_diff'].std()
    }
    
    # Identify common intervals
    interval_counts = df['time_diff'].value_counts().sort_values(ascending=False).head(5)
    stats['common_intervals'] = interval_counts.to_dict()
    
    # Find significant gaps
    significant_gaps = df[df['time_diff'] > min_gap_to_report]
    gaps = []
    
    if len(significant_gaps) > 0:
        logger.info(f"Found {len(significant_gaps)} significant gaps (>{min_gap_to_report})")
        
        for idx, row in significant_gaps.iterrows():
            gap_info = {
                'gap_start': row[time_column] - row['time_diff'],
                'gap_end': row[time_column],
                'gap_duration': row['time_diff']
            }
            gaps.append(gap_info)
    else:
        logger.info(f"No significant gaps (>{min_gap_to_report}) found")
    
    stats['significant_gaps'] = gaps
    stats['gap_count'] = len(gaps)
    
    # Calculate continuity score (percentage of expected data points that are present)
    if stats['median_interval'].total_seconds() > 0:
        expected_points = stats['time_span'].total_seconds() / stats['median_interval'].total_seconds()
        stats['continuity_score'] = min(100.0, (stats['total_points'] / expected_points) * 100)
    else:
        stats['continuity_score'] = None
    
    # Generate visualization
    output_dir = "experiments/timestamp_analysis"
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot histogram of intervals
    plt.figure(figsize=(12, 6))
    
    # Convert to seconds for better readability
    interval_seconds = df['time_diff'].dt.total_seconds()
    
    # Exclude NaN values
    interval_seconds = interval_seconds.dropna()
    
    plt.hist(interval_seconds, bins=100)
    plt.xlabel('Interval (seconds)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Time Intervals')
    
    # Save the figure
    plot_path = os.path.join(output_dir, f"interval_distribution_{timestamp}.png")
    plt.savefig(plot_path)
    logger.info(f"Saved interval distribution plot to {plot_path}")
    
    # Generate report
    report_content = ["# Timestamp Continuity Analysis Report\n\n"]
    report_content.append(f"## Dataset Information\n")
    report_content.append(f"- File: {file_path}\n")
    report_content.append(f"- Total data points: {stats['total_points']}\n")
    report_content.append(f"- Time span: {stats['time_span']}\n")
    report_content.append(f"- Start time: {stats['start_time']}\n")
    report_content.append(f"- End time: {stats['end_time']}\n\n")
    
    report_content.append(f"## Interval Statistics\n")
    report_content.append(f"- Median interval: {stats['median_interval']}\n")
    report_content.append(f"- Mean interval: {stats['mean_interval']}\n")
    report_content.append(f"- Min interval: {stats['min_interval']}\n")
    report_content.append(f"- Max interval: {stats['max_interval']}\n")
    report_content.append(f"- Standard deviation: {stats['std_interval']}\n\n")
    
    report_content.append(f"## Most Common Intervals\n")
    for interval, count in stats['common_intervals'].items():
        report_content.append(f"- {interval}: {count} occurrences\n")
    
    report_content.append(f"\n## Significant Gaps\n")
    if len(gaps) > 0:
        for i, gap in enumerate(gaps[:20]):  # Limit to first 20 gaps to keep report manageable
            report_content.append(f"- Gap {i+1}: {gap['gap_duration']} from {gap['gap_start']} to {gap['gap_end']}\n")
        
        if len(gaps) > 20:
            report_content.append(f"\n... and {len(gaps) - 20} more gaps\n")
    else:
        report_content.append("No significant gaps found.\n")
    
    if stats['continuity_score'] is not None:
        report_content.append(f"\n## Continuity Score\n")
        report_content.append(f"- Score: {stats['continuity_score']:.2f}% (percentage of expected data points that are present)\n")
    
    # Save the report
    report_path = os.path.join(output_dir, f"timestamp_continuity_report_{timestamp}.md")
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_content))
    
    logger.info(f"Analysis completed and saved to {report_path}")
    logger.info(f"Time taken: {time.time() - start_time:.2f} seconds")
    
    return stats

if __name__ == "__main__":
    file_path = "Data/downsampleData_scratch_1minut/Contacting_cleaned_1minut_20250802_170647.parquet"
    
    logger.info(f"Starting timestamp continuity analysis for {file_path}")
    
    # Analyze with default parameters (report gaps >= 5 minutes)
    results = analyze_timestamp_continuity(file_path)
    
    if results:
        logger.info("Analysis summary:")
        logger.info(f"- Time span: {results['time_span']}")
        logger.info(f"- Median interval: {results['median_interval']}")
        logger.info(f"- Found {results['gap_count']} significant gaps")
        if results['continuity_score'] is not None:
            logger.info(f"- Continuity score: {results['continuity_score']:.2f}%")
