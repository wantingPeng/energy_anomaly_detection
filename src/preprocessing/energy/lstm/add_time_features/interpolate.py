"""
Interpolation script for energy time series data.

This script loads energy time series data from interim directory, 
interpolates segments for each component (contact, ring, pcb),
and saves the results to the processed directory.
"""

import os
import gc
import pandas as pd
import numpy as np
import dask.dataframe as dd
import glob
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
from typing import Dict, List, Tuple, Union, Optional

from src.utils.logger import logger
from src.utils.memory_left import log_memory
from src.preprocessing.energy.lstm.interpolate_in_segment import interpolate_segments

def main():
    """Main function to interpolate data for all components."""
    # Setup timestamp for logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Set up input and output directories
    input_dir = 'Data/interim/Energy_time_series'
    output_dir = 'Data/processed/lsmt/add_time_features'
    
    # Ensure output directories exist
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Find all component directories in the input_dir
        component_dirs = glob.glob(os.path.join(input_dir, "*"))
        
        for component_dir in component_dirs:
            # Extract component name from directory path
            component_name = os.path.basename(component_dir).split('_')[0].lower()
            logger.info(f"Processing {component_name} data from {component_dir}")
            
            # Check if component directory exists
            if not os.path.exists(component_dir):
                logger.warning(f"Directory not found: {component_dir}. Skipping.")
                continue
                
            # Find parquet files in the component directory
            parquet_files = glob.glob(os.path.join(component_dir, "*.parquet"))
            
            if not parquet_files:
                logger.warning(f"No parquet files found in {component_dir}. Skipping.")
                continue
                
            # Load data using dask
            ddf = dd.read_parquet(parquet_files, engine="pyarrow")
            df = ddf.compute()
            
            logger.info(f"Loaded {component_name} data with shape: {df.shape}")
            log_memory(f"After loading {component_name} data")
            
            # Remove specified columns before interpolation
            columns_to_delete = ['IsOutlier', 'ID', 'Station', 'time_diff', 'component_type']
            df = df.drop(columns=columns_to_delete, errors='ignore')
            
            logger.info(f"Columns {columns_to_delete} removed from {component_name} data")
            
            # Use "all" as a placeholder for split
            split = "interpolate"
            
            # Process the data using the imported function
            interpolate_segments(df, output_dir, split, component_name)
            
            # Clean up
            del df, ddf
            gc.collect()
            log_memory(f"After processing {component_name}")
            
        logger.info("Interpolation completed successfully for all components")
            
    except Exception as e:
        logger.error(f"Error during interpolation: {str(e)}")
        raise


if __name__ == "__main__":
    main() 