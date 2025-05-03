import os
from datetime import datetime
import pandas as pd
import dask.dataframe as dd
from dask.diagnostics import ProgressBar
from src.utils.logger import logger
import sys

def data_save(df, filename, save_dir=None, timestamp_col='timestamp'):
    """
    Save large DataFrame to parquet format using Dask.
    Implements best practices for big data storage:
    - Optimized chunk sizes for large datasets
    - Data sorting within chunks
    - Progress monitoring
    - Efficient compression
    
    Args:
        df: Dask or pandas DataFrame
        filename (str): Base name for the output file
        save_dir (str, optional): Path to save directory
        timestamp_col (str): Name of timestamp column for sorting
    
    Returns:
        str: Path to the saved parquet directory
    """
    logger.info(f"Data type: {type(df)}")
    
    try:
        # Load save directory from config if not provided
        if save_dir is None:
            import yaml
            with open('configs/preprocessing.yaml', 'r') as file:
                config = yaml.safe_load(file)
                save_dir = config['data']['interim_dir']

        os.makedirs(save_dir, exist_ok=True)
        
        # Convert to Dask DataFrame if needed
        if isinstance(df, pd.DataFrame):
            # For large DataFrames, estimate good partition size
            partition_size = max(int(len(df) / 10), 1000000)  # At least 1M rows per partition
            df = dd.from_pandas(df, npartitions=partition_size)
        
        if not isinstance(df, dd.DataFrame):
            raise TypeError(f"Unsupported DataFrame type: {type(df)}")

        # Ensure timestamp column is datetime
        df[timestamp_col] = dd.to_datetime(df[timestamp_col])
        
        # Generate output directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(save_dir, f"{filename}_{timestamp}")
        
        logger.info(f"Starting data save to: {output_dir}")
        logger.info(f"Number of partitions: {df.npartitions}")
        logger.info("Verifying data right before saving:")
        logger.info(df.head())
        
        # Save with progress bar
        with ProgressBar():
            df.map_partitions(lambda x: x.sort_values(timestamp_col))\
              .to_parquet(
                  output_dir,
                  engine='pyarrow',
                  compression='snappy',
                  write_index=False,
                  write_metadata_file=True
              )

        logger.info("✅ Data saved successfully")
        logger.info(f"📁 Data location: {output_dir}")
        return output_dir

    except Exception as e:
        logger.error(f"❌ Error saving data: {str(e)}")
        raise
