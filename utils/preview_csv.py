#!/usr/bin/env python
# -*- coding: utf-8 -*-

from utils.logger import logger
import pandas as pd
import os
import time

def preview_csv(file_path, n_rows=5):
    """
    Preview the first and last n rows of a CSV file
    
    Args:
        file_path (str): Path to the CSV file
        n_rows (int): Number of rows to preview from top and bottom
        
    Returns:
        None
    """
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_file = f"experiments/logs/preview_csv_{timestamp}.log"
    
    # Ensure log directory exists
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    logger.info(f"Previewing CSV file: {file_path}")
    
    try:
        # Get file size
        file_size_bytes = os.path.getsize(file_path)
        file_size_mb = file_size_bytes / (1024 * 1024)
        logger.info(f"File size: {file_size_mb:.2f} MB")
        
        # Read the first few rows to get column information
        df_head = pd.read_csv(file_path, nrows=n_rows)
        
        logger.info(f"CSV has {len(df_head.columns)} columns: {', '.join(df_head.columns)}")
        logger.info(f"Data types:\n{df_head.dtypes}")
        
        # Display first n rows
        logger.info(f"First {n_rows} rows:")
        for i, row in df_head.iterrows():
            logger.info(f"Row {i+1}: {dict(row)}")
        
        # Try to read last n rows (this might be memory intensive for large files)
        try:
            # For large files, we'll use a more efficient approach
            if file_size_mb > 500:  # If file is larger than 500MB
                logger.info(f"File is large. Skipping reading last rows to avoid memory issues.")
            else:
                # Read the last few rows
                df_tail = pd.read_csv(file_path).tail(n_rows)
                logger.info(f"Last {n_rows} rows:")
                for i, row in df_tail.iterrows():
                    logger.info(f"Row {i+1}: {dict(row)}")
        except Exception as e:
            logger.warning(f"Could not read last rows: {str(e)}")
        
        # Show basic statistics for numeric columns
        logger.info("Basic statistics for numeric columns:")
        numeric_stats = df_head.describe()
        logger.info(f"{numeric_stats}")
        
        # Check for missing values in the preview
        missing_values = df_head.isnull().sum()
        logger.info(f"Missing values in preview:\n{missing_values}")
        
    except Exception as e:
        logger.error(f"Error previewing CSV: {str(e)}")

if __name__ == "__main__":
    file_path = "Data/row/Energy_Data/Contacting/Dezember_2024.csv"
    preview_csv(file_path, n_rows=10)