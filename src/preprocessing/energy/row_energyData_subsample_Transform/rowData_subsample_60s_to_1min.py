#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import pandas as pd
import numpy as np
from pathlib import Path
import time
from tqdm import tqdm
import dask.dataframe as dd
import logging
from src.utils.logger import logger

def downsample_to_1min(input_dir, output_dir, component):
    """
    Downsample data from per-second sampling to per-minute sampling.
    
    Args:
        input_dir (str): Directory containing input data
        output_dir (str): Directory to save output data
        component (str): Component name (e.g., 'contact')
        
    Returns:
        None
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all batch directories
    batch_dirs = glob.glob(os.path.join(input_dir, component, "batch_*"))
    
    for batch_dir in tqdm(batch_dirs, desc=f"Processing {component} batches"):
        batch_name = os.path.basename(batch_dir)
        output_batch_dir = os.path.join(output_dir, component, batch_name)
        os.makedirs(output_batch_dir, exist_ok=True)
        
        # Get all parquet files in the batch directory
        parquet_files = glob.glob(os.path.join(batch_dir, "*.parquet"))
        
        for file_path in tqdm(parquet_files, desc=f"Processing {batch_name} files", leave=False):
            file_name = os.path.basename(file_path)
            output_file_path = os.path.join(output_batch_dir, file_name)
            
            # Skip if output file already exists
            if os.path.exists(output_file_path):
                logger.info(f"Skipping existing file: {output_file_path}")
                continue
            
            try:
                # Read parquet file
                logger.info(f"Reading file: {file_path}")
                df = pd.read_parquet(file_path)
                
                # Drop ID and Station columns immediately
                if 'ID' in df.columns:
                    df = df.drop('ID', axis=1)
                if 'Station' in df.columns:
                    df = df.drop('Station', axis=1)
                
                # Ensure TimeStamp column exists and is datetime type
                if 'TimeStamp' in df.columns:
                    df['TimeStamp'] = pd.to_datetime(df['TimeStamp'])
                    
                    # Sort by TimeStamp to ensure chronological order
                    df = df.sort_values('TimeStamp')
                    
                    # Process each segment_id separately
                    if 'segment_id' in df.columns:
                        # Group by segment_id
                        grouped_dfs = []
                        for segment_id, segment_df in df.groupby('segment_id'):
                            # Sort by TimeStamp within each segment (in case groupby changes the order)
                            segment_df = segment_df.sort_values('TimeStamp')
                            
                            # Set TimeStamp as index
                            segment_df.set_index('TimeStamp', inplace=True)
                            
                            # Resample to 1 minute within this segment
                            resampled_segment = segment_df.resample('1min', origin='start').mean()
                            
                            # Reset index to make TimeStamp a column again
                            resampled_segment.reset_index(inplace=True)
                            
                            # Add to the list of processed segments
                            grouped_dfs.append(resampled_segment)
                        
                        # Combine all resampled segments
                        if grouped_dfs:
                            df_resampled = pd.concat(grouped_dfs, ignore_index=False)
                            
                            # Sort the combined dataframe by TimeStamp to ensure chronological order
                            df_resampled = df_resampled.sort_values('TimeStamp')
                            
                            # Save downsampled data
                            logger.info(f"Saving downsampled data to: {output_file_path}")
                            df_resampled.to_parquet(output_file_path, index=False)
                        else:
                            logger.warning(f"No valid segments found in {file_path}")
                    else:
                        logger.warning(f"segment_id column not found in {file_path}")
                else:
                    logger.warning(f"TimeStamp column not found in {file_path}")
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")

def process_all_components():
    """
    Process all components (train, val, test) from the input directory
    and save the downsampled data to the output directory.
    """
    base_input_dir = "Data/deepLearning/transform/interpolated"
    base_output_dir = "Data/row_energyData_subsample_Transform/downsampled_1min"
    
    # Create output directory if it doesn't exist
    os.makedirs(base_output_dir, exist_ok=True)
    
    # Process for each split (train, val, test)
    for split in ["train", "val", "test"]:
        input_dir = os.path.join(base_input_dir, split)
        output_dir = os.path.join(base_output_dir, split)
        
        # Get all component directories
        components = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]
        
        for component in components:
            logger.info(f"Processing {split} - {component}")
            downsample_to_1min(input_dir, output_dir, component)

if __name__ == "__main__":
    # Configure the logger
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_filename = f"experiments/logs/downsample_data_to_1min_{timestamp}.log"
    os.makedirs(os.path.dirname(log_filename), exist_ok=True)
    
    # Start processing
    logger.info("Starting data downsampling from 1s to 1min")
    start_time = time.time()
    
    process_all_components()
    
    elapsed_time = time.time() - start_time
    logger.info(f"Data downsampling completed in {elapsed_time:.2f} seconds")
