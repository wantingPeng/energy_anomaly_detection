"""
Sliding window preprocessing for LSTM model training data.

This script processes energy data by creating sliding windows for LSTM training.
It handles segmentation by component type, ensures time continuity within segments,
and properly labels windows based on anomaly overlap percentage.
"""

import os
import gc
import yaml
import pickle
import pandas as pd
import numpy as np
import dask.dataframe as dd
import pyarrow.parquet as pq
from datetime import datetime, timedelta
from pathlib import Path
from intervaltree import IntervalTree
from tqdm import tqdm
import torch
import logging
from typing import Dict, List, Tuple, Union, Optional
from joblib import Parallel, delayed
import glob
from src.utils.logger import logger
from src.utils.memory_left import log_memory



def interpolate_segments(df: pd.DataFrame, output_dir: str, split: str, component: str) -> None:
    """
    Interpolate segments using Dask and save the results.
    
    Args:
        df: Input DataFrame
        output_dir: Directory to save interpolated segments
    """
    logger.info("Interpolating segments using Dask")
    assert pd.api.types.is_datetime64_ns_dtype(df['TimeStamp']), "TimeStamp must be datetime64[ns] dtype"

    # Get list of unique segment IDs
    unique_segments = df['segment_id'].unique()
    logger.info(f"Processing {len(unique_segments)} unique segments")
    
    # Get feature columns (exclude non-feature columns)
    exclude_cols = ['TimeStamp', 'segment_id']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    # Create a list to store processed segments
    batch_size = 50  # Process 50 segments at a time
    
    for batch_start in range(0, len(unique_segments), batch_size):
        batch_segments = unique_segments[batch_start:batch_start + batch_size]
        logger.info(f"Processing batch of segments {batch_start} to {batch_start + len(batch_segments)}")
        
        batch_dfs = []
        for segment_id in tqdm(batch_segments, desc="Processing segments in batch"):
            segment_df = df[df['segment_id'] == segment_id].copy()
            
            # Sort by timestamp
            segment_df = segment_df.sort_values('TimeStamp')
            
            # Check for time continuity
            segment_df['prev_timestamp'] = segment_df['TimeStamp'].shift(1)
            segment_df['time_diff'] = (segment_df['TimeStamp'] - segment_df['prev_timestamp']).dt.total_seconds()
            
            # If there are gaps, fill them using linear interpolation
            has_gaps = (segment_df['time_diff'].fillna(1) > 1).any()
            if has_gaps:
                # Create a continuous time range with 1-second intervals
                start_time = segment_df['TimeStamp'].min()
                end_time = segment_df['TimeStamp'].max()
                time_range = pd.date_range(start=start_time, end=end_time, freq='1s')
                
                # Create a new DataFrame with the continuous time range
                continuous_df = pd.DataFrame({'TimeStamp': time_range})
                
                # Merge with the original data
                continuous_df = pd.merge_asof(
                    continuous_df,
                    segment_df[['TimeStamp'] + feature_cols],
                    on='TimeStamp',
                    direction='backward'
                )
                
                # Set segment_id
                continuous_df['segment_id'] = segment_id
                
                # Replace original segment DataFrame
                segment_df = continuous_df
            
            # Remove unnecessary columns
            segment_df = segment_df.drop(columns=['prev_timestamp', 'time_diff'], errors='ignore')
            batch_dfs.append(segment_df)
        
        # Combine batch DataFrames
        if batch_dfs:
            batch_df = pd.concat(batch_dfs, ignore_index=True)
            # Convert to Dask DataFrame
            ddf = dd.from_pandas(batch_df, npartitions=min(20, len(batch_segments)))
            dir = os.path.join(output_dir, split, component)
            os.makedirs(dir, exist_ok=True)

            # Save batch using Dask
            batch_output_dir = os.path.join(dir, f'batch_{batch_start}')
            ddf.to_parquet(
                batch_output_dir,
                engine='pyarrow',
                write_index=False,
            )
            
            # Clean up memory
            del batch_dfs, batch_df, ddf
            gc.collect()
            log_memory(f"after interpolating segments batch")

    logger.info("All segments processed and saved")


def main():
    """Main function to process all data splits."""
    input_dir = 'Data/processed/lsmt/spilt'
    output_dir = 'Data/processed/lsmt_statisticalFeatures/interpolate'
    
    # Define components to process
    component_names = ['contact']   # Full list would be ['contact', 'pcb', 'ring']
 
    try:      
        # Process each split and component
        for split in ['train', 'val', 'test']:
            for component in component_names:
                logger.info(f"Processing component: {split}/{component}")
                
                # Get component directory path
                component_dir = os.path.join(input_dir,component)
                
                # Find all parquet files in the component directory
                file_paths = glob.glob(os.path.join(component_dir, f"{split}.parquet"))
                logger.info(f"File paths: {file_paths}")
                if not file_paths:
                    logger.warning(f"No parquet files found for {split}/{component}. Skipping.")
                    continue
                
                # Use Dask to read multiple parquet files
                logger.info(f"Found {len(file_paths)} parquet files for {split}/{component}")
                ddf = dd.read_parquet(file_paths, engine="pyarrow")
                component_df = ddf.compute()
                logger.info(f"Loaded DataFrame with shape: {component_df.shape}")
                
                # Define paths for interpolated data
                log_memory(f"Before interpolating segments for {component} ({split})")
                
                # Interpolate segments
                interpolate_segments(component_df, output_dir, split, component)
                log_memory(f"After interpolating segments for {split}/{component})")

                del component_df
                gc.collect()
                log_memory(f"After processing {component} ({split})")
        
        logger.info("Sliding window preprocessing completed successfully")
        
    except Exception as e:
        logger.error(f"Error during processing: {str(e)}")
        raise



if __name__ == "__main__":
    main()
