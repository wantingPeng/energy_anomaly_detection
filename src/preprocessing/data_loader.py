import os
import glob
import pandas as pd
import yaml
from src.utils.logger import logger
import dask.dataframe as dd

def load_config():
    """Load preprocessing configuration."""
    with open('configs/preprocessing.yaml', 'r') as file:
        return yaml.safe_load(file)
def get_parquet_paths(base_dir):
    """
    获取 base_dir 下所有可用的 parquet 路径（包括目录式和单文件式）

    Args:
        base_dir (str): parquet 文件或目录的根目录（例如 "Data/interim/Energy_Data/Contacting"）

    Returns:
        List[str]: 所有 parquet 文件路径（文件和以 .parquet 结尾的目录）
    """
    parquet_paths = []
    
    for item in os.listdir(base_dir):
        full_path = os.path.join(base_dir, item)
        # 如果是文件，且以 .parquet 结尾
        if os.path.isfile(full_path) and item.endswith(".parquet"):
            parquet_paths.append(full_path)
        # 如果是目录，且目录名以 .parquet 结尾（说明是 parquet dataset）
        elif os.path.isdir(full_path) and item.endswith(".parquet"):
            parquet_paths.append(full_path)

    return parquet_paths

def data_loader(file_pattern=None):
    """
    Load energy data from parquet files using Dask.
    
    Args:
        file_pattern (str, optional): Specific file pattern to load. If None, loads all parquet files.
    
    Returns:
        dask.dataframe: Loaded dataset
    """

    logger.info(f"Loading data from pattern: {file_pattern}")
    try:
        # # 使用glob递归查找所有parquet文件
        # all_parquet_files = glob.glob(file_pattern, recursive=True)
        
        # if not all_parquet_files:
        #     logger.error(f"No parquet files found matching pattern: {file_pattern}")
        #     raise FileNotFoundError(f"No parquet files found in {file_pattern}")
            
        # logger.info(f"Found {len(all_parquet_files)} parquet files")
        
        # 使用Dask读取所有找到的parquet文件
        df = dd.read_parquet(file_pattern)
        
        # 获取数据集的基本信息
        logger.info(f"Data type: {type(df)}")
        logger.info(f"Loaded dataset with {len(df.columns)} columns")
        logger.info(f"Approximate number of rows: {len(df)}")
        
        return df
        
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise
if __name__ == "__main__":
  data_loader("Data/interim/Energy_Data/Contacting/*/*.parquet") 