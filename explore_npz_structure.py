import numpy as np
import os
from src.utils.logger import logger

def explore_npz_structure(npz_file_path):
    """
    Explore the structure of an NPZ file to understand its format.
    
    Args:
        npz_file_path: Path to the NPZ file to explore
    """
    try:
        logger.info(f"Exploring structure of: {npz_file_path}")
        
        # Load the NPZ file
        npz_data = np.load(npz_file_path, allow_pickle=True)
        
        # Get the keys in the NPZ file
        keys = list(npz_data.keys())
        logger.info(f"Keys in NPZ file: {keys}")
        
        # For each key, get information about the corresponding array
        for key in keys:
            array = npz_data[key]
            logger.info(f"Key: {key}, Shape: {array.shape}, Type: {type(array)}")
            
            # If it's a structured array, examine its structure
            if hasattr(array, 'dtype') and array.dtype.names is not None:
                logger.info(f"Fields in structured array: {array.dtype.names}")
                
                # Examine the first element to understand the structure
                if len(array) > 0:
                    first_element = array[0]
                    logger.info(f"First element type: {type(first_element)}")
                    
                    # If the first element is a structured array, examine its fields
                    if hasattr(first_element, 'dtype') and first_element.dtype.names is not None:
                        logger.info(f"Fields in first element: {first_element.dtype.names}")
                        
                        # Check if there's a timestamp field and examine it
                        if 'timestamp' in first_element.dtype.names:
                            timestamps = timestamps['timestamp']
                            logger.info(f"Timestamp data type: {timestamps.dtype}")
                            logger.info(f"First few timestamps: {timestamps[:5]}")
                            logger.info(f"Timestamp differences: {np.diff(timestamps[:5])}")
                            
                            # Check ordering and continuity
                            is_ordered = np.all(np.diff(timestamps) >= 0)
                            logger.info(f"Timestamps are ordered: {is_ordered}")
                            
                            unique_diffs = np.unique(np.diff(timestamps))
                            logger.info(f"Unique timestamp differences: {unique_diffs}")
                    else:
                        # If it's not a structured array, show a sample
                        logger.info(f"Sample of first element: {first_element}")
    
    except Exception as e:
        logger.error(f"Error exploring {npz_file_path}: {str(e)}")

if __name__ == "__main__":
    # Set the path to the NPZ file to explore
    npz_dir = "Data/processed/lsmt/standerScaler_in_segment/spilt_after_sliding/sliding_window/contact"
    npz_files = [os.path.join(npz_dir, f) for f in os.listdir(npz_dir) if f.endswith('_windows.npz')]
    
    if npz_files:
        # Just examine the first file for now
        explore_npz_structure(npz_files[0])
    else:
        logger.error("No NPZ files found in the specified directory") 