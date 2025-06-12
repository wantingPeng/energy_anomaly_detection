import pandas as pd
import numpy as np
import dask.dataframe as dd
from datetime import datetime
from dask.diagnostics import ProgressBar
import os
from pathlib import Path

from src.utils.logger import logger
from src.preprocessing.data_loader import data_loader
from src.preprocessing.data_save import data_save

def interpolate_within_segments(df, timestamp_col='TimeStamp'):
    """
    Perform interpolation within segments where timestamps are not continuous.
    
    Args:
        df: pandas DataFrame with time series data and segment_id column
        timestamp_col: name of the timestamp column
        
    Returns:
        DataFrame with interpolated values for gaps within segments
    """
    logger.info("Starting interpolation within segments...")
    
    # Convert to pandas for time series operations
    if isinstance(df, dd.DataFrame):
        logger.info("Converting Dask DataFrame to pandas for interpolation...")
        df = df.compute()
    
    # Ensure the timestamp column is datetime type
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    
    # Sort by segment_id and timestamp
    df = df.sort_values(['segment_id', timestamp_col])
    
    # Initialize tracking variables
    total_segments = df['segment_id'].nunique()
    total_gaps = 0
    total_interpolated_points = 0
    
    # Process each segment separately
    segments = []
    for segment_id, group in df.groupby('segment_id'):
        # Sort by timestamp
        group = group.sort_values(timestamp_col)
        
        # Calculate time differences in seconds
        group['time_diff'] = group[timestamp_col].diff().dt.total_seconds()
        
        # Find gaps (time differences > 1 second)
        gaps = group[group['time_diff'] > 1].copy()
        segment_gaps = len(gaps)
        total_gaps += segment_gaps
        
        if segment_gaps > 0:
            # Create a new DataFrame with a complete time series (1-second intervals)
            start_time = group[timestamp_col].min()
            end_time = group[timestamp_col].max()
            
            # Create a complete time range with 1-second intervals
            complete_range = pd.date_range(start=start_time, end=end_time, freq='1S')
            
            # Create a DataFrame with the complete range
            complete_df = pd.DataFrame({timestamp_col: complete_range})
            
            # Merge with original data
            merged = pd.merge(complete_df, group, on=timestamp_col, how='left')
            
            # Fill segment_id for all rows
            merged['segment_id'] = segment_id
            
            # Interpolate numeric columns within the segment
            numeric_cols = merged.select_dtypes(include=['number']).columns
            merged[numeric_cols] = merged[numeric_cols].interpolate(method='linear')
            
            # Count interpolated points
            interpolated_points = len(merged) - len(group)
            total_interpolated_points += interpolated_points
            
            segments.append(merged)
            
            logger.info(f"Segment {segment_id}: {segment_gaps} gaps found, {interpolated_points} points interpolated")
        else:
            segments.append(group)
    
    # Combine all segments back into a single DataFrame
    result_df = pd.concat(segments, ignore_index=True)
    
    # Drop the time_diff column which was used for calculations
    if 'time_diff' in result_df.columns:
        result_df = result_df.drop(columns=['time_diff'])
    
    logger.info(f"Interpolation complete. Processed {total_segments} segments, found {total_gaps} gaps, added {total_interpolated_points} interpolated points.")
    
    return result_df

def main():
    # Load data
    input_path = "Data/interim/Energy_time_series/contact_20250509_093729/part.0.parquet"
    output_dir = "Data/interim/interpolate"
    
    logger.info(f"Loading data from {input_path}")
    df = data_loader(input_path)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Log initial data information
    logger.info(f"Initial data shape: {df.shape}")
    logger.info(f"Initial columns: {df.columns}")
    
    # Drop unnecessary columns
    columns_to_drop = ['ID', 'Station', 'IsOutlier', 'time_diff']
    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])
    logger.info(f"Columns after dropping: {df.columns}")
    
    # Perform interpolation
    interpolated_df = interpolate_within_segments(df)
    
    # Convert back to Dask DataFrame for saving
    dask_df = dd.from_pandas(interpolated_df, npartitions=10)
    
    # Save the interpolated data
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = data_save(
        df=dask_df,
        filename="interpolated_timeseries",
        save_dir=output_dir,
        timestamp_col='TimeStamp'
    )
    
    logger.info(f"Interpolated data saved to {output_path}")
    return interpolated_df

if __name__ == "__main__":
    main() 