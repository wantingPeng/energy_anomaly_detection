import numpy as np
import os
from src.utils.logger import logger

def preview_npz_file(file_path, max_rows=100):
    """
    Preview the contents of an NPZ file by displaying the first 'max_rows' rows of each array.
    
    Args:
        file_path (str): Path to the NPZ file
        max_rows (int): Maximum number of rows to display for each array, defaults to 100
    
    Returns:
        dict: Dictionary containing array names and their previews
    """
    # Check if file exists
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return None
    
    try:
        # Load the NPZ file
        logger.info(f"Loading NPZ file: {file_path}")
        data = np.load(file_path, allow_pickle=True)
        
        # Log the keys in the file
        logger.info(f"Keys in NPZ file: {data.files}")
        
        # Dictionary to store previews
        previews = {}
        
        # Iterate through all arrays in the NPZ file
        for key in data.files:
            arr = data[key]
            logger.info(f"\nPreview of '{key}':")
            logger.info(f"Shape: {arr.shape}")
            logger.info(f"Data type: {arr.dtype}")
            
            # Get the preview of the array
            if arr.size > 0:
                # Check for NaN values if array is numeric
                if np.issubdtype(arr.dtype, np.number):
                    nan_count = np.isnan(arr).sum()
                    logger.info(f"NaN count: {nan_count} ({nan_count/arr.size:.2%} of values)")
                    
                    # Log basic statistics
                    if nan_count < arr.size:  # If not all values are NaN
                        logger.info(f"Min: {np.nanmin(arr)}")
                        logger.info(f"Max: {np.nanmax(arr)}")
                        logger.info(f"Mean: {np.nanmean(arr)}")
                        logger.info(f"Std: {np.nanstd(arr)}")
                else:
                    logger.info(f"NaN check skipped (non-numeric data type)")
                
                # Get the first max_rows or fewer rows
                if arr.ndim >= 2:
                    preview_rows = min(arr.shape[0], max_rows)
                    preview = arr[:preview_rows]
                    logger.info(f"\nFirst {preview_rows} rows:")
                    
                    # For 2D arrays, display in a more readable format
                    if arr.ndim == 2:
                        if arr.shape[1] > 20:  # If too many columns, show summary
                            for i, row in enumerate(preview):
                                logger.info(f"Row {i}: shape={row.shape}, first 5 values: {row[:5]}, "
                                          f"last 5 values: {row[-5:] if row.shape[0] >= 5 else row}")
                        else:  # Otherwise show full rows
                            for i, row in enumerate(preview):
                                logger.info(f"Row {i}: {row}")
                    else:
                        logger.info(str(preview))
                else:
                    # Handle 1D arrays
                    preview_elements = min(arr.size, max_rows)
                    preview = arr[:preview_elements]
                    logger.info(f"\nFirst {preview_elements} values:")
                    logger.info(str(preview))
                
                previews[key] = preview
            else:
                logger.info(f"Array '{key}' is empty")
                previews[key] = arr
        
        return previews
    
    except Exception as e:
        logger.error(f"Error loading NPZ file: {e}")
        return None

def display_preview(file_path, max_rows=10):
    """
    Display preview of the NPZ file with nicely formatted output.
    
    Args:
        file_path (str): Path to the NPZ file
        max_rows (int): Maximum number of rows to display for each array, defaults to 100
    """
    previews = preview_npz_file(file_path, max_rows)
    
    if previews:
        logger.info(f"=== NPZ File Preview: {file_path} ===")
        logger.info("Preview complete. See details above.")
    else:
        logger.error(f"Failed to preview NPZ file: {file_path}")

if __name__ == "__main__":
    # Example usage
    npz_file = "Data/deepLearning/transform/statistic_features_600_600_100_0_standardized/train/contact/batch_0.npz"
    display_preview(npz_file, max_rows=100)
