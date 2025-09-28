#!/usr/bin/env python3
"""
Energy Data Downsampling and Anomaly Labeling Script
====================================================

This script loads energy data from second-level measurements, downsamples
it to minute-level by averaging every 60 data points, and adds anomaly labels
based on predefined anomaly time periods for the Kontaktieren station.

Input: 
  - Data/machine/cleaning_utc/Contacting_cleaned.parquet (second-level data)
  - Data/machine/Anomaly_Data/anomaly_dict_merged.pkl (anomaly periods)
  
Output: 
  - Data/downsampleData_scratch_1minut/ (minute-level data with anomaly labels)

Features:
  - Remove ID and Station columns
  - Downsample from second to minute-level (60:1 ratio)
  - Add anomaly_label column (1 for anomaly, 0 for normal)
  - Comprehensive logging and data preview

Author: Assistant
Date: 2025-01-27
"""

import os
import sys
import pandas as pd
import dask.dataframe as dd
import pickle
from datetime import datetime
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from utils.logger import logger


def load_energy_data(input_file_path: str) -> dd.DataFrame:
    """
    Load energy data from parquet file using Dask for memory efficiency.
    
    Args:
        input_file_path (str): Path to the input parquet file
        
    Returns:
        dd.DataFrame: Loaded data as Dask DataFrame
    """
    logger.info(f"Loading energy data from: {input_file_path}")
    
    if not os.path.exists(input_file_path):
        raise FileNotFoundError(f"Input file not found: {input_file_path}")
    
    # Load data using Dask for memory efficiency
    df = dd.read_parquet(input_file_path)
    
    logger.info(f"Data loaded successfully. Shape: {df.shape[0].compute()} rows, {df.shape[1]} columns")
    logger.info(f"Columns: {list(df.columns)}")
    
    return df


def remove_unnecessary_columns(df: dd.DataFrame) -> dd.DataFrame:
    """
    Remove ID and Station columns from the dataset.
    
    Args:
        df (dd.DataFrame): Input dataframe
        
    Returns:
        dd.DataFrame: Dataframe with unnecessary columns removed
    """
    logger.info("Removing ID and Station columns")
    
    columns_to_remove = ['ID', 'Station']
    
    # Check which columns exist
    existing_columns = [col for col in columns_to_remove if col in df.columns]
    missing_columns = [col for col in columns_to_remove if col not in df.columns]
    
    if missing_columns:
        logger.warning(f"Columns not found in data: {missing_columns}")
    
    if existing_columns:
        df = df.drop(columns=existing_columns)
        logger.info(f"Removed columns: {existing_columns}")
    
    logger.info(f"Remaining columns after removal: {len(df.columns)}")
    
    return df


def downsample_to_minute(df: dd.DataFrame) -> dd.DataFrame:
    """
    Downsample data from second-level to minute-level by averaging every 60 data points.
    
    Args:
        df (dd.DataFrame): Input dataframe with second-level data
        
    Returns:
        dd.DataFrame: Downsampled dataframe with minute-level data
    """
    logger.info("Starting downsampling to minute-level data")
    
    # Ensure TimeStamp column exists and is datetime
    if 'TimeStamp' not in df.columns:
        raise ValueError("TimeStamp column not found in the data")
    
    # Convert to pandas for resampling (Dask doesn't have full resample support)
    logger.info("Converting to pandas for resampling operation")
    df_pandas = df.compute()
    
    logger.info(f"Data converted to pandas. Shape: {df_pandas.shape}")
    
    # Ensure TimeStamp is datetime and set as index
    df_pandas['TimeStamp'] = pd.to_datetime(df_pandas['TimeStamp'])
    df_pandas = df_pandas.set_index('TimeStamp')
    
    # Sort by timestamp to ensure proper order
    df_pandas = df_pandas.sort_index()
    
    logger.info(f"Time range: {df_pandas.index.min()} to {df_pandas.index.max()}")
    
    # Resample to 1-minute intervals and take the mean
    logger.info("Resampling data to 1-minute intervals using mean aggregation")
    df_resampled = df_pandas.resample('1min').mean()
    
    # Remove rows with all NaN values (if any)
    df_resampled = df_resampled.dropna(how='all')
    
    logger.info(f"Downsampled data shape: {df_resampled.shape}")
    logger.info(f"Data reduction: {df_pandas.shape[0]} -> {df_resampled.shape[0]} rows")
    logger.info(f"Compression ratio: {df_pandas.shape[0] / df_resampled.shape[0]:.1f}x")
    
    # Reset index to make TimeStamp a column again
    df_resampled = df_resampled.reset_index()
    
    return df_resampled


def save_downsampled_data(df: pd.DataFrame, output_dir: str) -> str:
    """
    Save downsampled data to parquet format.
    
    Args:
        df (pd.DataFrame): Downsampled dataframe
        output_dir (str): Output directory path
        
    Returns:
        str: Path to the saved file
    """
    logger.info(f"Saving downsampled data to: {output_dir}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate output filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"Ring_cleaned_1minut_{timestamp}.parquet"
    output_path = os.path.join(output_dir, output_filename)
    
    # Save to parquet format
    df.to_parquet(output_path, index=False)
    
    logger.info(f"Data saved successfully to: {output_path}")
    logger.info(f"File size: {os.path.getsize(output_path) / (1024*1024):.2f} MB")
    
    return output_path


def preview_data(df: pd.DataFrame, name: str, num_rows: int = 5):
    """
    Preview the first and last few rows of the dataframe.
    
    Args:
        df (pd.DataFrame): Dataframe to preview
        name (str): Name/description for the data
        num_rows (int): Number of rows to display
    """
    logger.info(f"\n=== {name} - Preview (first {num_rows} rows) ===")
    logger.info(f"Shape: {df.shape}")
    logger.info(f"Columns: {list(df.columns)}")
    logger.info(f"\nFirst {num_rows} rows:")
    print(df.head(num_rows))
    
    logger.info(f"\nLast {num_rows} rows:")
    print(df.tail(num_rows))
    
    if 'TimeStamp' in df.columns:
        logger.info(f"\nTime range: {df['TimeStamp'].min()} to {df['TimeStamp'].max()}")


def load_anomaly_dict(anomaly_file_path: str) -> dict:
    """
    Load anomaly dictionary from pickle file.
    
    Args:
        anomaly_file_path (str): Path to the anomaly dictionary pickle file
        
    Returns:
        dict: Dictionary containing anomaly periods for different stations
    """
    logger.info(f"Loading anomaly dictionary from: {anomaly_file_path}")
    
    if not os.path.exists(anomaly_file_path):
        raise FileNotFoundError(f"Anomaly file not found: {anomaly_file_path}")
    
    try:
        with open(anomaly_file_path, 'rb') as f:
            anomaly_dict = pickle.load(f)
        
        logger.info(f"Keys in anomaly dict: {list(anomaly_dict.keys())}")
            
        return anomaly_dict
        
    except Exception as e:
        logger.error(f"Error loading anomaly dictionary: {e}")
        raise


def is_anomaly(timestamp: pd.Timestamp, anomaly_periods: list) -> int:
    """
    Check if timestamp falls within any anomaly period.
    
    Args:
        timestamp (pd.Timestamp): Timestamp to check
        anomaly_periods (list): List of (start_time, end_time) tuples
        
    Returns:
        int: 1 if timestamp is within any anomaly period, 0 otherwise
    """
    for start_time, end_time in anomaly_periods:
        if start_time <= timestamp <= end_time:
            return 1
    return 0


def add_anomaly_labels(df: pd.DataFrame, anomaly_dict: dict, station_name: str = 'Kontaktieren') -> pd.DataFrame:
    """
    Add anomaly labels to the dataframe based on anomaly periods.
    
    Args:
        df (pd.DataFrame): Dataframe with TimeStamp column
        anomaly_dict (dict): Dictionary containing anomaly periods
        station_name (str): Name of the station to get anomaly periods for
        
    Returns:
        pd.DataFrame: Dataframe with added anomaly_label column
    """
    logger.info(f"Adding anomaly labels based on {station_name} station anomaly periods")
    
    if station_name not in anomaly_dict:
        logger.warning(f"Station '{station_name}' not found in anomaly dictionary")
        # Add all zeros if station not found
        df['anomaly_label'] = 0
        return df
    
    # Get anomaly periods for the specified station
    anomaly_periods = anomaly_dict[station_name]
    logger.info(f"Processing {len(anomaly_periods)} anomaly periods")
    
    # Ensure TimeStamp is datetime
    if df['TimeStamp'].dtype != 'datetime64[ns, UTC]':
        df['TimeStamp'] = pd.to_datetime(df['TimeStamp'], utc=True)
    
    # Apply anomaly labeling
    logger.info("Applying anomaly labels to timestamps...")
    df['anomaly_label'] = df['TimeStamp'].apply(lambda ts: is_anomaly(ts, anomaly_periods))
    
    # Log statistics
    total_rows = len(df)
    anomaly_count = df['anomaly_label'].sum()
    anomaly_percentage = (anomaly_count / total_rows) * 100 if total_rows > 0 else 0
    
    logger.info(f"Anomaly labeling completed:")
    logger.info(f"  Total rows: {total_rows}")
    logger.info(f"  Anomaly rows: {anomaly_count}")
    logger.info(f"  Anomaly percentage: {anomaly_percentage:.2f}%")
    
    # Show sample of labeled data
    if total_rows > 0:
        logger.info("\nSample of labeled data:")
        sample_size = min(5, total_rows)
        sample = df.sample(sample_size) if total_rows > sample_size else df
        for _, row in sample.iterrows():
            logger.info(f"  TimeStamp: {row['TimeStamp']}, Label: {row['anomaly_label']}")
    
    return df


def main():
    """
    Main function to orchestrate the downsampling pipeline.
    """ 
    # Define file paths
    input_file = "Data/machine/cleaning_utc/Contacting_cleaned_1.parquet"
    output_dir = "Data/downsampleData_scratch_1minut_contact"
    anomaly_file = "Data/machine/Anomaly_Data/anomaly_dict_merged.pkl"
    
    try:
        logger.info("Starting energy data downsampling and anomaly labeling pipeline")
        logger.info("=" * 60)
        
        # Step 1: Load data
        df = load_energy_data(input_file)
        
        # Step 2: Remove unnecessary columns
        df_cleaned = remove_unnecessary_columns(df)
        
        # Step 3: Downsample to minute-level
        df_downsampled = downsample_to_minute(df_cleaned)
        
        # Step 4: Load anomaly dictionary
        anomaly_dict = load_anomaly_dict(anomaly_file)
        
        # Step 5: Add anomaly labels
        df_labeled = add_anomaly_labels(df_downsampled, anomaly_dict, 'Kontaktieren')
        
        # Step 6: Preview the final result
        preview_data(df_labeled, "Downsampled and Labeled Data")
        
        # Step 7: Save final data with anomaly labels
        output_path = save_downsampled_data(df_labeled, output_dir)
        
        logger.info("=" * 60)
        logger.info("Energy data downsampling and anomaly labeling completed successfully!")
        logger.info(f"Output saved to: {output_path}")
        
        # Final summary
        total_rows = len(df_labeled)
        anomaly_count = df_labeled['anomaly_label'].sum()
        logger.info("\nFinal Summary:")
        logger.info(f"  Total minute-level data points: {total_rows}")
        logger.info(f"  Anomalous data points: {anomaly_count}")
        logger.info(f"  Normal data points: {total_rows - anomaly_count}")
        logger.info(f"  Anomaly rate: {(anomaly_count/total_rows)*100:.3f}%")
        
    except Exception as e:
        logger.error(f"Error in downsampling and labeling pipeline: {str(e)}")
        raise


if __name__ == "__main__":
    main()
