import os
import pandas as pd
from pathlib import Path
import sys

# Add the src directory to the path to import modules from src
src_dir = Path(__file__).resolve().parents[4]
sys.path.append(str(src_dir))

from preprocessing.energy.lstm.interpolate_in_segment import interpolate_segments
from utils.logger import logger

def interpolate_datasets():
    """
    Load energy datasets from Data/interim/Energy_time_series,
    interpolate them by segment, and save the results back to the same location.
    """
    data_dir = src_dir / "Data" / "interim" / "Energy_time_series"
    
    # List of datasets to process
    datasets = [
        "building_1_energy_consumption.parquet",
        "building_2_energy_consumption.parquet",
        "building_3_energy_consumption.parquet"
    ]
    
    for dataset in datasets:
        input_path = data_dir / dataset
        output_path = data_dir / f"interpolated_{dataset}"
        
        if not input_path.exists():
            logger.warning(f"Dataset {dataset} not found at {input_path}")
            continue
        
        logger.info(f"Loading dataset: {dataset}")
        df = pd.read_parquet(input_path)
        
        logger.info(f"Interpolating dataset: {dataset}")
        interpolated_df = interpolate_segments(df)
        
        logger.info(f"Saving interpolated dataset to: {output_path}")
        interpolated_df.to_parquet(output_path)
        
        logger.info(f"Successfully processed dataset: {dataset}")

if __name__ == "__main__":
    logger.info("Starting interpolation process for energy datasets")
    interpolate_datasets()
    logger.info("Interpolation process completed") 