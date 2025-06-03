import numpy as np
import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from src.utils.logger import logger

def preview_npz_data(npz_file_path, num_rows=100):
    """
    Preview the first num_rows of data from an NPZ file.
    
    Args:
        npz_file_path (str): Path to the NPZ file
        num_rows (int): Number of rows to preview (default: 100)
    """
    try:
        # Check if file exists
        if not os.path.exists(npz_file_path):
            logger.error(f"File not found: {npz_file_path}")
            return
            
        # Load the NPZ file
        logger.info(f"Loading NPZ file: {npz_file_path}")
        data = np.load(npz_file_path, allow_pickle=True)
        
        # Print the keys in the NPZ file
        logger.info(f"Keys in NPZ file: {data.files}")
        
        # Preview each array in the NPZ file
        for key in data.files:
            array = data[key]
            
            logger.info(f"\nPreview of '{key}':")
            logger.info(f"Shape: {array.shape}")
            logger.info(f"Data type: {array.dtype}")
            
            # Check for NaN values if array is numeric
            if np.issubdtype(array.dtype, np.number):
                nan_count = np.isnan(array).sum()
                total_elements = array.size
                nan_percentage = (nan_count / total_elements) * 100 if total_elements > 0 else 0
                logger.info(f"NaN values: {nan_count} out of {total_elements} ({nan_percentage:.2f}%)")
            else:
                logger.info("NaN check skipped (non-numeric data type)")
            
            # If array is multi-dimensional, reshape it for preview
            if array.ndim > 1:
                # For 2D+ arrays, display shape information for each dimension
                for i, dim in enumerate(array.shape):
                    logger.info(f"Dimension {i}: {dim} elements")
                
                # If 2D, try to show as a table
                if array.ndim == 2 and array.shape[0] > 0:
                    # Preview first num_rows rows (or all if fewer)
                    preview_rows = min(num_rows, array.shape[0])
                    logger.info(f"\nFirst {preview_rows} rows:")
                    
                    # Create a DataFrame for prettier display
                    preview_df = pd.DataFrame(
                        array[:preview_rows], 
                        columns=[f"Col_{i}" for i in range(array.shape[1])]
                    )
                    
                    # Convert to string with a fixed width for each column
                    preview_str = preview_df.to_string(index=True, max_cols=20)
                    logger.info(preview_str)
                    
                # For 3D or higher, show first slice
                elif array.ndim >= 3 and array.shape[0] > 0:
                    logger.info(f"\nFirst slice of the first {min(num_rows, array.shape[0])} items:")
                    for i in range(min(num_rows, array.shape[0])):
                        logger.info(f"Item {i}, first slice shape: {array[i].shape}")
                        if i < 3:  # Only show details for first few items
                            logger.info(f"Preview: {array[i][0][:5]}...")  # First 5 elements of first slice
            else:
                # For 1D arrays, show the values directly
                preview_count = min(num_rows, array.shape[0])
                logger.info(f"\nFirst {preview_count} values:")
                
                # Specially handle object dtype (e.g., datetime, strings, etc.)
                if array.dtype == np.dtype('O'):
                    for i in range(preview_count):
                        logger.info(f"[{i}] {array[i]}")
                else:
                    logger.info(array[:preview_count])
                
            # Check for infinite values if array is numeric
            if np.issubdtype(array.dtype, np.number):
                inf_count = np.isinf(array).sum()
                if inf_count > 0:
                    logger.warning(f"Warning: {inf_count} infinite values detected in '{key}'")
                
    except Exception as e:
        logger.error(f"Error previewing NPZ file: {str(e)}")

def check_parquet_for_nans(parquet_file_path):
    """
    Check a parquet file for NaN values.
    
    Args:
        parquet_file_path (str): Path to the parquet file
    """
    try:
        # Check if file exists
        if not os.path.exists(parquet_file_path):
            logger.error(f"File not found: {parquet_file_path}")
            return
        
        # Load the parquet file
        logger.info(f"Loading parquet file: {parquet_file_path}")
        df = pd.read_parquet(parquet_file_path)
        
        # Print basic info about the dataframe
        logger.info(f"DataFrame shape: {df.shape}")
        logger.info(f"DataFrame columns: {df.columns.tolist()}")
        
        # Check for NaN values in the entire dataframe
        total_elements = df.size
        nan_count = df.isna().sum().sum()
        nan_percentage = (nan_count / total_elements) * 100 if total_elements > 0 else 0
        
        logger.info(f"Total NaN values: {nan_count} out of {total_elements} ({nan_percentage:.2f}%)")
        
        # Check for NaN values in each column
        logger.info("\nNaN values by column:")
        for column in df.columns:
            col_nan_count = df[column].isna().sum()
            col_elements = len(df[column])
            col_nan_percentage = (col_nan_count / col_elements) * 100 if col_elements > 0 else 0
            logger.info(f"{column}: {col_nan_count} out of {col_elements} ({col_nan_percentage:.2f}%)")
        
        # Show a preview of the first few rows
        logger.info("\nPreview of the first 5 rows:")
        logger.info(df.head().to_string())
        
        return nan_count > 0
        
    except Exception as e:
        logger.error(f"Error checking parquet file: {str(e)}")
        return False

def check_directory_for_nans(directory_path):
    """
    Check all parquet files in a directory for NaN values.
    
    Args:
        directory_path (str): Path to the directory containing parquet files
    """
    try:
        # Check if directory exists
        if not os.path.exists(directory_path):
            logger.error(f"Directory not found: {directory_path}")
            return
        
        # Get all parquet files in the directory
        parquet_files = []
        
        # Check if directory_path is a file or directory
        if os.path.isfile(directory_path) and directory_path.endswith('.npz'):
            parquet_files = [directory_path]
            logger.info(f"Found 1 parquet file: {directory_path}")
        else:
            # Walk through the directory to find all parquet files
            for root, _, files in os.walk(directory_path):
                for file in files:
                    if file.endswith('.parquet'):
                        parquet_files.append(os.path.join(root, file))
            
            logger.info(f"Found {len(parquet_files)} parquet files in {directory_path}")
        
        if not parquet_files:
            logger.warning(f"No parquet files found in {directory_path}")
            return
        
        # Check each parquet file for NaN values
        files_with_nans = 0
        for parquet_file in parquet_files:
            logger.info(f"\n{'='*50}")
            logger.info(f"Checking file: {parquet_file}")
            has_nans = check_parquet_for_nans(parquet_file)
            if has_nans:
                files_with_nans += 1
        
        logger.info(f"\n{'='*50}")
        logger.info(f"Summary: {files_with_nans} out of {len(parquet_files)} files contain NaN values")
        
    except Exception as e:
        logger.error(f"Error checking directory: {str(e)}")

def main():
    """Main function to run the preview."""
    #NPZ file preview
    # npz_file_path = "Data/processed/lsmt/sliding_window_800s/test/contact/batch_0.npz"
    # preview_npz_data(npz_file_path)
    
    directory_path = "Data/processed/lsmt_statisticalFeatures/sliding_window_1200s/train/contact/batch_0.npz"
    preview_npz_data(directory_path)

if __name__ == "__main__":
    main()
