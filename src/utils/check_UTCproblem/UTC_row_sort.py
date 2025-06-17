import pandas as pd
import os
from src.utils.logger import logger


def sort_data_by_timestamp():
    """
    Load data from Januar_2024.parquet, sort by TimeStamp column and print first 100 rows.
    """
    # File path
    file_path = "Data/interim/Energy_Data/Contacting/Januar_2024.parquet"
    
    # Check if file exists
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return
    
    try:
        # Load parquet file
        logger.info(f"Loading data from {file_path}")
        df = pd.read_parquet(file_path)
        
        # Sort by TimeStamp
        logger.info("Sorting data by TimeStamp")
        df_sorted = df.sort_values(by="TimeStamp")
        
        # Print info about the dataframe
        logger.info(f"DataFrame shape: {df.shape}")
        logger.info(f"DataFrame columns: {df.columns.tolist()}")
        
        # Print first 100 rows
        logger.info("First 100 rows after sorting:")
        logger.info("\n" + str(df_sorted.head(100)))
        
        return df_sorted
    
    except Exception as e:
        logger.error(f"Error processing data: {str(e)}")
        return None


if __name__ == "__main__":
    sort_data_by_timestamp()
