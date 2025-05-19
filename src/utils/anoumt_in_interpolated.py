import os
import glob
import pandas as pd
import numpy as np
from pathlib import Path
import time
from tqdm import tqdm
from src.utils.logger import logger
from typing import Dict, List, Tuple

def check_timestamp_continuity(df: pd.DataFrame) -> Dict[str, Dict]:
    """
    Check if timestamps within each segment are continuous.
    
    Args:
        df: DataFrame with 'segment_id' and 'TimeStamp' columns
        
    Returns:
        Dictionary with segment_id as keys and continuity info as values
    """
    if 'segment_id' not in df.columns or 'TimeStamp' not in df.columns:
        return {}
        
    continuity_info = {}
    
    # Ensure TimeStamp is datetime
    if not pd.api.types.is_datetime64_any_dtype(df['TimeStamp']):
        df['TimeStamp'] = pd.to_datetime(df['TimeStamp'])
    
    # Group by segment_id
    for segment_id, group in df.groupby('segment_id'):
        # Sort by timestamp
        group = group.sort_values('TimeStamp')
        
        # Calculate time differences
        time_diffs = group['TimeStamp'].diff().dropna()
        
        # Get statistics
        continuity_info[segment_id] = {
            'min_diff': time_diffs.min().total_seconds(),
            'max_diff': time_diffs.max().total_seconds(),
            'mean_diff': time_diffs.mean().total_seconds(),
            'is_continuous': (time_diffs.max() == time_diffs.min())
        }
        
    return continuity_info

def count_records_in_train_folders():
    """
    Count the number of records in each subfolder of Data/processed/lsmt/segment_fixe/train.
    Also counts unique segments and checks timestamp continuity within segments.
    
    This function counts records in the ring, pcb, and contact subfolders by reading
    all parquet files in each folder structure.
    
    Returns:
        dict: Dictionary with subfolder names as keys and analysis results as values
    """
    # Set up paths
    base_path = "Data/processed/lsmt/segment_fixe/train"
    subfolders = ["ring", "pcb", "contact"]
    
    # Dictionary to store results
    results = {}
    
    # Process each subfolder
    for subfolder in subfolders:
        folder_path = os.path.join(base_path, subfolder)
        logger.info(f"Processing {subfolder} folder at {folder_path}")
        
        # Find all parquet files in this subfolder (including nested directories)
        parquet_files = glob.glob(os.path.join(folder_path, "**/*.parquet"), recursive=True)
        
        if not parquet_files:
            logger.warning(f"No parquet files found in {folder_path}")
            results[subfolder] = {
                'total_records': 0,
                'segment_count': 0,
                'continuity_info': {}
            }
            continue
            
        logger.info(f"Found {len(parquet_files)} parquet files in {subfolder}")
        
        # Count records and collect all segment IDs
        total_records = 0
        all_segment_ids = set()
        continuity_info = {}
        
        for file_path in tqdm(parquet_files, desc=f"Analyzing {subfolder}"):
            try:
                # Read parquet file
                df = pd.read_parquet(file_path)
                file_records = len(df)
                total_records += file_records
                logger.debug(f"File {file_path}: {file_records} records")
                
                # Extract segment IDs if available
                if 'segment_id' in df.columns:
                    segment_ids_in_file = set(df['segment_id'].unique())
                    all_segment_ids.update(segment_ids_in_file)
                    
                    # Check continuity of timestamps within segments
                    file_continuity = check_timestamp_continuity(df)
                    
                    # Merge with overall continuity info
                    for segment_id, info in file_continuity.items():
                        if segment_id in continuity_info:
                            # Update existing segment info
                            continuity_info[segment_id]['min_diff'] = min(
                                continuity_info[segment_id]['min_diff'],
                                info['min_diff']
                            )
                            continuity_info[segment_id]['max_diff'] = max(
                                continuity_info[segment_id]['max_diff'],
                                info['max_diff']
                            )
                            # If any file shows non-continuity, mark the segment as non-continuous
                            if not info['is_continuous']:
                                continuity_info[segment_id]['is_continuous'] = False
                        else:
                            # Add new segment info
                            continuity_info[segment_id] = info
                
            except Exception as e:
                logger.error(f"Error reading {file_path}: {str(e)}")
                
        # Store results
        segment_count = len(all_segment_ids)
        
        # Count how many segments have continuous timestamps
        continuous_segments = sum(1 for info in continuity_info.values() if info.get('is_continuous', False))
        
        results[subfolder] = {
            'total_records': total_records,
            'segment_count': segment_count,
            'continuous_segments': continuous_segments,
            'discontinuous_segments': segment_count - continuous_segments,
            'continuity_percentage': (continuous_segments / segment_count * 100) if segment_count > 0 else 0
        }
        
        logger.info(f"Total records in {subfolder}: {total_records:,}")
        logger.info(f"Total segments in {subfolder}: {segment_count:,}")
        logger.info(f"Continuous segments in {subfolder}: {continuous_segments:,} ({results[subfolder]['continuity_percentage']:.2f}%)")
    
    # Display summary
    logger.info("===== Summary of records and segments =====")
    for subfolder, result in results.items():
        logger.info(f"{subfolder}: {result['total_records']:,} records, {result['segment_count']:,} segments, " +
                    f"{result['continuous_segments']} continuous segments ({result['continuity_percentage']:.2f}%)")
    
    return results

if __name__ == "__main__":
    # When run as script, execute the count and log results
    start_time = time.time()
    results = count_records_in_train_folders()
    elapsed_time = time.time() - start_time
    
    logger.info(f"Completed analysis in {elapsed_time:.2f} seconds")
    
    # Display total across all folders
    total_records = sum(result['total_records'] for result in results.values())
    total_segments = sum(result['segment_count'] for result in results.values())
    logger.info(f"Total records across all folders: {total_records:,}")
    logger.info(f"Total segments across all folders: {total_segments:,}")
