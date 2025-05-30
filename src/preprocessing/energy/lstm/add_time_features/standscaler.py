import glob
import os
import pandas as pd
import dask.dataframe as dd
from pathlib import Path
from datetime import datetime
from src.utils.logger import logger
import gc

def z_score_normalize(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """
    Apply z-score normalization to numeric columns of the dataframe grouped by a specific column.

    Parameters:
    - df: DataFrame containing the data to normalize.
    - group_col: Column name to group by for normalization.

    Returns:
    - DataFrame with normalized numeric columns.
    """
    numeric_cols = df.select_dtypes(include=['number']).columns
    numeric_cols = numeric_cols.difference(['TimeStamp', 'segment_id'])
    df[numeric_cols] = df.groupby(group_col)[numeric_cols].transform(
        lambda x: (x - x.mean()) / x.std()
    )
    return df


def main():
    """Main function to normalize data for all components."""
    # Setup timestamp for logging
    
    # Set up input and output directories
    input_dir = 'Data/processed/lsmt/add_time_features/interpolate'
    output_dir = 'Data/processed/lsmt/add_time_features/standscaler'
    
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
                
            # Find parquet files in the component directory recursively
            parquet_files = glob.glob(os.path.join(component_dir, "**", "*.parquet"), recursive=True)
            
            if not parquet_files:
                logger.warning(f"No parquet files found in {component_dir}. Skipping.")
                continue
                
            # Load data using dask
            ddf = dd.read_parquet(parquet_files, engine="pyarrow")
            df = ddf.compute()
            
            logger.info(f"Loaded {component_name} data with shape: {df.shape}")
            
            # Normalize the data
            df_normalized = z_score_normalize(df, 'segment_id')
            
            # Save the normalized data
            component_output_dir = os.path.join(output_dir, component_name)
            os.makedirs(component_output_dir, exist_ok=True)
            output_file = os.path.join(component_output_dir, f"normalized.parquet")
            df_normalized.to_parquet(output_file, engine='pyarrow')
            
            logger.info(f"Normalized data saved to {output_file}")
            
            # Clean up
            del df, ddf, df_normalized
            gc.collect()
            
        logger.info("Normalization completed successfully for all components")
        
    except Exception as e:
        logger.error(f"Error during normalization: {str(e)}")
        raise


if __name__ == "__main__":
    main() 