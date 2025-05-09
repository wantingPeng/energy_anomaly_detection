# count_segments.py
import pandas as pd
from src.utils.logger import logger
from datetime import datetime
import os

def count_segment_ids():
    # 设置日志文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = "experiments/logs"
    os.makedirs(log_dir, exist_ok=True)
    
    try:
        # 读取parquet文件
        file_path = "Data/interim/Energy_time_series/contact_20250509_093729/part.0.parquet"
        logger.info(f"Reading parquet file from: {file_path}")
        
        df = pd.read_parquet(file_path)
        
        # 统计唯一的segment_id数量
        unique_segments = df['segment_id'].nunique()
        total_rows = len(df)
        
        logger.info(f"Total number of rows in the dataset: {total_rows}")
        logger.info(f"Number of unique segment_ids: {unique_segments}")
        
    
        return unique_segments
        
    except FileNotFoundError:
        logger.error(f"Parquet file not found at the specified path")
        raise
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        num_segments = count_segment_ids()
        logger.info("Script completed successfully")
    except Exception as e:
        logger.error("Script failed to complete")