"""
Merge batch dataset files into consolidated datasets.

This script merges batch files created by save_for_dataset.py into
consolidated datasets organized by data type (train, val, test).
"""

import os
import gc
import yaml
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import torch
from typing import Dict, List, Tuple, Union
from datetime import datetime
import glob

from src.utils.logger import logger
from src.utils.memory_left import log_memory


def load_config() -> dict:
    """Load configuration from YAML file."""
    config_path = Path("configs/merge_for_dataset.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def merge_datasets(
    input_dir: str,
    output_dir: str,
    data_type: str,
    component: str,
) -> bool:
    """
    Merge batch dataset files for a specific data type across all components.
    
    Args:
        input_dir: Input directory containing batch files
        output_dir: Output directory for merged datasets
        data_type: Data type ('train', 'val', or 'test')
        component: Component type ('contact', 'pcb', 'ring')
        
    Returns:
        True if successful, False otherwise
    """
    logger.info(f"Merging datasets for {data_type}")
    log_memory(f"Before merging {data_type}")
    
    # Prepare paths
    input_data_dir = os.path.join(input_dir, data_type, component)
    os.makedirs(output_dir, exist_ok=True)
    
    # Define output files
    output_dataset_path = os.path.join(output_dir, f"{data_type}.pt")
    output_parquet_path = os.path.join(output_dir, f"{data_type}.parquet")
    
    # Check if output files already exist and delete them
    for path in [output_dataset_path, output_parquet_path]:
        if os.path.exists(path):
            os.remove(path)
            logger.info(f"Removed existing file: {path}")
    
    # Initialize merged data structures
    all_windows, all_labels, df_chunks = [], [], []
    
    # Process each component    
    logger.info(f"Processing component: {component}")
    
    # Load batch metadata
    metadata_path = glob.glob(os.path.join(input_data_dir, "*.pkl"))
    if not metadata_path:
        logger.warning(f"Metadata file not found: {metadata_path}. Skipping component.")
        return False
          
    try:
        with open(metadata_path[0], 'rb') as f:
            batch_files = pickle.load(f)
            
        logger.info(f"Found {len(batch_files)} batch files for {component}")
        
        # Process each batch individually
        for batch_idx, (dataset_path, parquet_path) in enumerate(batch_files):
            logger.info(f"Processing batch {batch_idx + 1}/{len(batch_files)}")
            
            if not os.path.exists(dataset_path) or not os.path.exists(parquet_path):
                logger.warning(f"Batch files not found: {dataset_path} or {parquet_path}. Skipping.")
                continue
                
            # Load PyTorch dataset
            try:
                data = torch.load(dataset_path)
                windows, labels = data['windows'], data['labels']
                
                # Add to collection
                all_windows.append(windows)
                all_labels.append(labels)
                
                # Load parquet file
                df = pd.read_parquet(parquet_path)
                df_chunks.append(df)
                
                logger.info(f"Loaded batch with {len(windows)} windows from {os.path.basename(dataset_path)}")
            except Exception as e:
                logger.error(f"Error loading batch files: {str(e)}. Skipping.")
                continue
            
            # Save intermediate results periodically to avoid OOM
            if batch_idx > 0 and batch_idx % 2 == 0:  # Save every 5 batches as a checkpoint
                try:
                    # Concatenate windows and labels
                    combined_windows = torch.cat(all_windows)
                    combined_labels = torch.cat(all_labels)
                    
                    # Save intermediate PyTorch dataset
                    intermediate_path = os.path.join(output_dir, f"{data_type}_intermediate.pt")
                    torch.save({
                        'windows': combined_windows,
                        'labels': combined_labels
                    }, intermediate_path)
                    
                    # Clear memory
                    del all_windows, all_labels, combined_windows, combined_labels
                    all_windows, all_labels = [], []
                    gc.collect()
                    
                    # Reload from intermediate file
                    data = torch.load(intermediate_path)
                    all_windows = [data['windows']]
                    all_labels = [data['labels']]
                    
                    # Concatenate and save intermediate parquet
                    if df_chunks:
                        combined_df = pd.concat(df_chunks, ignore_index=True)
                        intermediate_parquet = os.path.join(output_dir, f"{data_type}_intermediate.parquet")
                        combined_df.to_parquet(intermediate_parquet, index=False)
                        
                        # Clear memory
                        del df_chunks, combined_df
                        df_chunks = []
                        gc.collect()
                        
                        # Reload from intermediate file
                        df_chunks = [pd.read_parquet(intermediate_parquet)]
                    
                    logger.info(f"Saved and reloaded intermediate results after batch {batch_idx}")
                    log_memory(f"After intermediate save for {data_type}")
                except Exception as e:
                    logger.error(f"Error saving intermediate results: {str(e)}")
                    return False
                    
    except Exception as e:
        logger.error(f"Error processing component {component}: {str(e)}")
        return False

    # Final merge and save
    if not all_windows:
        logger.warning(f"No data found for {data_type}. Nothing to merge.")
        return False
      
    try:
        # Concatenate windows and labels
        final_windows = torch.cat(all_windows)
        final_labels = torch.cat(all_labels)
        
        # Save final PyTorch dataset
        torch.save({
            'windows': final_windows,
            'labels': final_labels
        }, output_dataset_path)
        
        logger.info(f"Saved merged PyTorch dataset with {len(final_windows)} windows to {output_dataset_path}")
        
        # Clear memory
        del all_windows, all_labels, final_windows, final_labels
        gc.collect()
        
        # Concatenate and save final parquet
        if df_chunks:
            final_df = pd.concat(df_chunks, ignore_index=True)
            final_df.to_parquet(output_parquet_path, index=False)
            logger.info(f"Saved merged parquet file with {len(final_df)} rows to {output_parquet_path}")
            
            # Clear memory
            del df_chunks, final_df
            gc.collect()
        
        log_memory(f"After merging {data_type}")
        return True
        
    except Exception as e:
        logger.error(f"Error in final merge for {data_type}: {str(e)}")
        return False


def verify_merged_dataset(output_dir: str, data_type: str) -> bool:
    """
    Verify that the merged dataset was created properly.
    
    Args:
        output_dir: Path to the output directory
        data_type: Data type ('train', 'val', or 'test')
        
    Returns:
        True if verification passed, False otherwise
    """
    dataset_path = os.path.join(output_dir, f"{data_type}.pt")
    parquet_path = os.path.join(output_dir, f"{data_type}.parquet")
    
    if not os.path.exists(dataset_path) or not os.path.exists(parquet_path):
        logger.error(f"Merged dataset files not found for {data_type}")
        return False
        
    try:
        # Load dataset tensors
        data = torch.load(dataset_path)
        windows = data['windows']
        labels = data['labels']
        
        # Check dataset properties
        n_samples = len(windows)
        logger.info(f"Merged dataset for {data_type} contains {n_samples} samples")
        
        if n_samples == 0:
            logger.warning(f"Merged dataset for {data_type} is empty")
            return False
            
        # Check parquet file
        df = pd.read_parquet(parquet_path)
        logger.info(f"Merged parquet file contains {len(df)} rows")
        
        if len(df) != n_samples:
            logger.error(f"Merged dataset and parquet file have different sample counts: {n_samples} vs {len(df)}")
            return False
            
        return True
        
    except Exception as e:
        logger.error(f"Error verifying merged dataset {data_type}: {str(e)}")
        return False


def main():
    """
    Main function to merge all batch datasets into consolidated datasets.
    """
    start_time = datetime.now()
    logger.info(f"Starting dataset merging at {start_time}")
    
    # Get paths from config
    input_dir = "Data/processed/lsmt/dataset"  # Should point to the output of save_for_dataset.py
    output_dir = "Data/processed/lsmt/mergend_dataset"  # Directory to save merged datasets
    
    # Get components to process
    components = ['contact', 'pcb', 'ring']
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each data type
    results = {}
    for data_type in ['train', 'val', 'test']:
        results[data_type] = {}
        for component in components:
            logger.info(f"Processing {data_type} data for component {component}")
            results[data_type][component] = merge_datasets(
                input_dir,
                output_dir,
                data_type,
                component
            )
    
    # Verify merged datasets
    logger.info("Verifying merged datasets...")
    verification_results = {}
    
    for data_type in ['train', 'val', 'test']:
        # Check if at least one component was successfully processed
        if any(results[data_type].values()):
            verification_results[data_type] = verify_merged_dataset(output_dir, data_type)
        else:
            verification_results[data_type] = False
            logger.warning(f"All components failed for {data_type}, skipping verification")
    
    # Report verification results
    logger.info("Merged dataset verification results:")
    for data_type, result in verification_results.items():
        status = "PASSED" if result else "FAILED"
        logger.info(f"{data_type}: {status}")
    
    end_time = datetime.now()
    processing_time = end_time - start_time
    logger.info(f"Completed dataset merging in {processing_time}")


if __name__ == "__main__":
    main() 