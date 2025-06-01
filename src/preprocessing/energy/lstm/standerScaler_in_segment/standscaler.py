import glob
import os
import pandas as pd
import numpy as np
import dask.dataframe as dd
from pathlib import Path
from datetime import datetime
from src.utils.logger import logger
import gc

def z_score_normalize(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """
    Apply z-score normalization to numeric columns of the dataframe grouped by a specific column.
    Handles the case where standard deviation is zero by returning zeros instead of NaN.

    Parameters:
    - df: DataFrame containing the data to normalize.
    - group_col: Column name to group by for normalization.

    Returns:
    - DataFrame with normalized numeric columns.
    """
    numeric_cols = df.select_dtypes(include=['number']).columns
    numeric_cols = numeric_cols.difference(['TimeStamp', 'segment_id'])
    
    # Create a copy of the dataframe to avoid SettingWithCopyWarning
    df_normalized = df.copy()
    
    # Apply normalization with handling for zero standard deviation
    for col in numeric_cols:
        # Calculate mean and standard deviation for each group
        group_stats = df.groupby(group_col)[col].agg(['mean', 'std'])
        
        # Replace zero standard deviations with 1 to avoid division by zero
        # This will effectively keep the original values where std=0 (after subtracting the mean)
        group_stats['std'] = group_stats['std'].replace(0, 1)
        
        # Map the stats back to the original dataframe
        means = df[group_col].map(group_stats['mean'])
        stds = df[group_col].map(group_stats['std'])
        
        # Apply the normalization
        df_normalized[col] = (df[col] - means) / stds
        
        # Check for any remaining NaN values (should not happen with this approach)
        nan_count = df_normalized[col].isna().sum()
        if nan_count > 0:
            logger.warning(f"Column {col} still has {nan_count} NaN values after normalization")
    
    return df_normalized


def main():
    """Main function to normalize data for all components."""
    # Setup timestamp for logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"experiments/logs/standscaler_{timestamp}.log"
    
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
            
            # Check for NaN values in input data
            nan_count_before = df.isna().sum().sum()
            if nan_count_before > 0:
                logger.warning(f"Input data contains {nan_count_before} NaN values before normalization")
                for col in df.columns:
                    col_nan_count = df[col].isna().sum()
                    if col_nan_count > 0:
                        logger.warning(f"Column '{col}' has {col_nan_count} NaN values before normalization")
            
            # Check for columns with zero standard deviation
            numeric_cols = df.select_dtypes(include=['number']).columns
            numeric_cols = numeric_cols.difference(['TimeStamp', 'segment_id'])
            
            logger.info(f"Checking for zero standard deviation in {len(numeric_cols)} numeric columns")
            zero_std_features = []
            
            for col in numeric_cols:
                # Count groups with zero standard deviation
                group_stats = df.groupby('segment_id')[col].agg(['std'])
                zero_std_groups = (group_stats['std'] == 0).sum()
                if zero_std_groups > 0:
                    zero_std_features.append(col)
                    total_groups = len(group_stats)
                    logger.warning(f"Feature '{col}' has zero standard deviation in {zero_std_groups} out of {total_groups} groups ({zero_std_groups/total_groups*100:.2f}%)")
            
            if zero_std_features:
                logger.warning(f"Found {len(zero_std_features)} features with zero standard deviation in some groups")
                logger.warning(f"These features are: {', '.join(zero_std_features)}")
            
            # Normalize the data
            logger.info(f"Applying z-score normalization to {component_name} data")
            df_normalized = z_score_normalize(df, 'segment_id')
            
            # Check for NaN values after normalization
            nan_count_after = df_normalized.isna().sum().sum()
            if nan_count_after > 0:
                logger.error(f"Normalized data contains {nan_count_after} NaN values after normalization")
                for col in df_normalized.columns:
                    col_nan_count = df_normalized[col].isna().sum()
                    if col_nan_count > 0:
                        logger.error(f"Column '{col}' has {col_nan_count} NaN values after normalization")
            else:
                logger.info("No NaN values detected after normalization")
            
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