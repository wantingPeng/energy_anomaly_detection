import pandas as pd
import glob
import os
from src.utils.logger import logger

def check_duplicate_timestamps(data_dir: str):
    """
    Check for duplicate timestamps in parquet files.
    
    Args:
        data_dir (str): Directory containing parquet files or single parquet file
        
    Returns:
        None: Prints information about duplicate timestamps
    """
    total_duplicates = 0
    total_rows = 0
    
    # If data_dir is a directory, get all parquet files, else treat as single file
    if os.path.isdir(data_dir):
        parquet_files = glob.glob(os.path.join(data_dir, "**/*.parquet"), recursive=True)
    else:
        parquet_files = [data_dir]
        
    logger.info(f"Found {len(parquet_files)} parquet files to process")
    
    for file in parquet_files:
        logger.info(f"Processing file: {file}")
        df = pd.read_parquet(file)
        total_rows += len(df)
        
        # Convert TimeStamp to datetime if it's not already
        df['TimeStamp'] = pd.to_datetime(df['TimeStamp'])
        
        # Check duplicates
        duplicates = df[df.duplicated(['TimeStamp'], keep=False)]
        n_duplicates = len(duplicates)
        total_duplicates += n_duplicates
        
        if n_duplicates > 0:
            logger.info(f"Found {n_duplicates} duplicate timestamps in {file}")
            # Group duplicates and show their counts
            duplicate_counts = duplicates.groupby('TimeStamp').size()
            logger.info("\nDuplicate timestamp occurrences:")
            logger.info(f"Number of unique timestamps that are duplicated: {len(duplicate_counts)}")
            logger.info("\nSample of duplicated timestamps:")
            logger.info(duplicate_counts.head())
            
    logger.info(f"\nSummary:")
    logger.info(f"Total rows processed: {total_rows}")
    logger.info(f"Total duplicate timestamps found: {total_duplicates}")
    logger.info(f"Percentage of duplicates: {(total_duplicates/total_rows)*100:.2f}%")

if __name__ == "__main__":
    data_dir = "Data/processed/lsmt/segment_fixe/train/pcb"
    check_duplicate_timestamps(data_dir)
