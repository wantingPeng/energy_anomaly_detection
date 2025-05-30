import os
import pandas as pd
import numpy as np
from pathlib import Path
from src.utils.logger import logger

def align_features():
    """
    Align features across contact, ring, and pcb datasets:
    1. Load datasets from Data/processed/lsmt/add_time_features/standscaler
    2. Find the union of all features across datasets
    3. Fill missing features with zeros
    4. Save aligned datasets to Data/processed/lsmt/add_time_features/align
    """
    
    logger.info("Starting feature alignment process")
    
    # Define input and output paths
    input_base_path = Path("Data/processed/lsmt/add_time_features/standscaler")
    output_base_path = Path("Data/processed/lsmt/add_time_features/align")
    
    # Create output directories if they don't exist
    for folder in ["contact", "ring", "pcb"]:
        os.makedirs(output_base_path / folder, exist_ok=True)
    
    # Load the three datasets
    datasets = {}
    all_columns = set()
    
    for dataset_name in ["contact", "ring", "pcb"]:
        file_path = input_base_path / dataset_name / "normalized.parquet"
        logger.info(f"Loading dataset from {file_path}")
        
        try:
            datasets[dataset_name] = pd.read_parquet(file_path)
            # Keep track of all unique columns across datasets
            all_columns.update(datasets[dataset_name].columns)
            logger.info(f"Loaded {dataset_name} dataset with {len(datasets[dataset_name])} rows and {len(datasets[dataset_name].columns)} columns")
        except Exception as e:
            logger.error(f"Error loading {dataset_name} dataset: {e}")
            raise
    
    # Convert the columns set to a sorted list for consistent order
    all_columns = sorted(list(all_columns))
    logger.info(f"Union of all features contains {len(all_columns)} columns")
    
    # Align datasets by adding missing columns filled with zeros
    aligned_datasets = {}
    for dataset_name, df in datasets.items():
        logger.info(f"Aligning features for {dataset_name} dataset")
        
        # Find missing columns for this dataset
        missing_columns = set(all_columns) - set(df.columns)
        logger.info(f"{dataset_name} dataset is missing {len(missing_columns)} columns")
        
        # Add missing columns filled with zeros
        for col in missing_columns:
            df[col] = 0
            
        # Ensure all columns are in the same order
        aligned_datasets[dataset_name] = df[all_columns]
        
        # Save the aligned dataset
        output_path = output_base_path / dataset_name / "aligned.parquet"
        logger.info(f"Saving aligned {dataset_name} dataset to {output_path}")
        aligned_datasets[dataset_name].to_parquet(output_path)
            
    return aligned_datasets

if __name__ == "__main__":
    align_features()
