import pandas as pd
import glob
import os
from datetime import timedelta
from src.utils.logger import logger

def check_segment_time_continuity(data_dir: str):
    """
    Check time series continuity within segments from parquet files.
    
    Args:
        data_dir (str): Directory containing parquet files
        
    Returns:
        None: Prints information about discontinuous segments
    """
    # Get all parquet files in the directory
    parquet_files = glob.glob(os.path.join(data_dir, "*.parquet"))
    logger.info(f"Found {len(parquet_files)} parquet files in {data_dir}")
    
    discontinuous_segments = {}
    
    for file in parquet_files:
        logger.info(f"Processing file: {file}")
        df = pd.read_parquet(file)
        
        # Convert TimeStamp to datetime if it's not already
        df['TimeStamp'] = pd.to_datetime(df['TimeStamp'])
        
        # Group by segment_id
        for segment_id, segment_data in df.groupby('segment_id'):
            # Sort by timestamp
            segment_data = segment_data.sort_values('TimeStamp')
            
            # Calculate time differences between consecutive rows
            time_diff = segment_data['TimeStamp'].diff()
            
            # Expected time difference (1 second)
            expected_diff = timedelta(seconds=1)
            
            # Find gaps larger than 1 second
            gaps = time_diff[time_diff > expected_diff]
            
            if not gaps.empty:
                if segment_id not in discontinuous_segments:
                    discontinuous_segments[segment_id] = []
              
    count = 0
    # Print results
    if discontinuous_segments:
        logger.info("\nFound discontinuous segments:")
        for segment_id, gaps in discontinuous_segments.items():          
                count += 1
        logger.info(f"Total discontinuous segments: {count}")
    else:
        logger.info("\nNo discontinuous segments found.")

if __name__ == "__main__":
    data_dir = "Data/processed/lsmt/standerScaler/test.parquet"
    check_segment_time_continuity(data_dir)
