import os
import pandas as pd
import numpy as np
from pathlib import Path
from src.utils.logger import logger
from src.utils.memory_left import log_memory

def align_features():
    """
    Align features across contact, ring, and pcb datasets:
    1. Load datasets from Data/processed/lsmt/add_time_features/standscaler
    2. Find the union of all features across datasets
    3. Fill missing features with zeros
    4. Save aligned datasets to Data/processed/lsmt/add_time_features/align
    """
    
    logger.info("Starting feature alignment process")
    log_memory("Start")
    
    # Define input and output paths
    input_base_path = Path("Data/processed/lsmt/add_time_features/standscaler")
    output_base_path = Path("Data/processed/lsmt/add_time_features/align")
    
    # Create output directories if they don't exist
    for folder in ["contact", "ring", "pcb"]:
        os.makedirs(output_base_path / folder, exist_ok=True)
    
    # Load the three datasets to determine all columns
    all_columns = set()
    
    for dataset_name in ["contact", "ring", "pcb"]:
        file_path = input_base_path / dataset_name / "normalized.parquet"
        logger.info(f"Loading dataset from {file_path} to determine columns")
        log_memory(f"Loading {dataset_name} for columns")
        
        try:
            df = pd.read_parquet(file_path)
            # Keep track of all unique columns across datasets
            all_columns.update(df.columns)
            logger.info(f"Loaded {dataset_name} dataset with {len(df)} rows and {len(df.columns)} columns")
        except Exception as e:
            logger.error(f"Error loading {dataset_name} dataset: {e}")
            raise
    
    # Convert the columns set to a sorted list for consistent order
    all_columns = sorted(list(all_columns))
    print(all_columns)
    logger.info(f"Union of all features contains {len(all_columns)} columns")
    log_memory("After union")
    
    # Clear datasets to free memory
    del df
    
    # Align datasets by reloading and processing each one
    for dataset_name in ["contact", "ring", "pcb"]:
        file_path = input_base_path / dataset_name / "normalized.parquet"
        logger.info(f"Reloading {dataset_name} dataset for alignment")
        log_memory(f"Reloading {dataset_name}")
        
        try:
            df = pd.read_parquet(file_path)
            
            # Find missing columns for this dataset
            missing_columns = set(all_columns) - set(df.columns)
            logger.info(f"{dataset_name} dataset is missing {len(missing_columns)} columns")
            
            # Add missing columns filled with zeros
            for col in missing_columns:
                df[col] = 0
                
            # Ensure all columns are in the same order
            aligned_df = df[all_columns]
            
            # Split the aligned dataset into four batches
            split_size = len(aligned_df) // 4
            batches = [aligned_df.iloc[i*split_size:(i+1)*split_size] for i in range(4)]
            
            # Handle any remaining rows
            if len(aligned_df) % 4 != 0:
                batches[-1] = pd.concat([batches[-1], aligned_df.iloc[4*split_size:]])
            
            logger.info(f"Split {dataset_name} dataset into 4 batches")
            log_memory(f"After splitting {dataset_name}")
            
            # Save the aligned dataset batches
            for batch_idx, batch_df in enumerate(batches):
                output_path = output_base_path / dataset_name / f"batch_{batch_idx}.parquet"
                logger.info(f"Saving aligned {dataset_name} batch_{batch_idx} to {output_path}")
                batch_df.to_parquet(output_path)
                log_memory(f"After saving {dataset_name} batch_{batch_idx}")
        except Exception as e:
            logger.error(f"Error processing {dataset_name} dataset: {e}")
            raise

if __name__ == "__main__":
    align_features()
