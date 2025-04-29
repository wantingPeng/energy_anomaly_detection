import os
from pathlib import Path
import dask.dataframe as dd
from src.utils.logger import logger
from datetime import datetime

def convert_csv_to_parquet(
    input_dir: str = "/home/wanting/energy_anomaly_detection/Data/row",
    output_dir: str = "/home/wanting/energy_anomaly_detection/Data/interim",
    chunk_size: str = "100MB"
) -> None:
    """
    Convert CSV files to Parquet format using Dask, maintaining the same directory structure.
    
    Args:
        input_dir (str): Root directory containing CSV files
        output_dir (str): Root directory where Parquet files will be saved
        chunk_size (str): Size of chunks for Dask to process the data
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Ensure output directory exists
    output_path.mkdir(parents=True, exist_ok=True)
    
    def process_directory(current_dir: Path, relative_path: Path = Path("")):
        logger.info(f"Processing directory: {current_dir}")
        
        # Create corresponding output directory
        current_output_dir = output_path / relative_path
        current_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Process all CSV files in current directory
        csv_files = list(current_dir.glob("*.csv"))
        if not csv_files:
            logger.warning(f"No CSV files found in directory: {current_dir}")
            
        for file_path in csv_files:
            try:
                # Construct output path with same structure but .parquet extension
                relative_output_path = relative_path / file_path.stem
                output_file = str(output_path / relative_output_path) + ".parquet"
                
                logger.info(f"Converting {file_path} to {output_file}")
                
                # Read CSV using Dask with explicit data types
                df = dd.read_csv(
                    str(file_path),
                    blocksize=chunk_size,
                    assume_missing=True,
                    low_memory=False
                )
                
                # Log DataFrame information
         
                
                # Write to Parquet format
                df.to_parquet(
                    output_file,
                    engine='pyarrow',
                    compression='snappy',
                    write_index=True  # 保存索引
                )
                
                logger.info(f"Successfully converted {file_path.name} to Parquet format")
                
            except Exception as e:
                logger.error(f"Error processing {file_path}: {str(e)}")
                continue
        
        # Recursively process subdirectories
        for dir_path in current_dir.glob("*/"):
            if dir_path.is_dir():
                process_directory(dir_path, relative_path / dir_path.name)
    
    # Start processing from root directory
    logger.info(f"Starting CSV to Parquet conversion from {input_dir} to {output_dir}")
    process_directory(input_path)
    logger.info("Completed converting all CSV files to Parquet format")

if __name__ == "__main__":
    # 检查输入目录是否存在
    input_dir = "/home/wanting/energy_anomaly_detection/Data/row"
    if not Path(input_dir).exists():
        logger.error(f"Input directory does not exist: {input_dir}")
        exit(1)
        
    convert_csv_to_parquet()