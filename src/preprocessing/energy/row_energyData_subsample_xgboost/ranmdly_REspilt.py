import os
import pandas as pd
import numpy as np
from datetime import datetime
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split
from src.utils.logger import logger

def resplit_contact_data():
    """
    Load data from labeled contact directories, combine them, randomly resplit into
    new train (70%), val (15%), and test (15%) sets, and save to new directory structure.
    """
    # Setup paths
    input_dir = "Data/row_energyData_subsample_xgboost/labeled/contact"
    output_dir = "Data/row_energyData_subsample_xgboost/ranmdly_REspilt/contact"
    
    # Create timestamp for logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Ensure output directory exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Log start of processing
    logger.info(f"Starting to resplit contact data")
    
    # Load train, val, test datasets
    logger.info("Loading train dataset")
    train_df = pd.read_parquet(os.path.join(input_dir, "train.parquet"))
    logger.info(f"Loaded train data with shape: {train_df.shape}")
    
    logger.info("Loading validation dataset")
    val_df = pd.read_parquet(os.path.join(input_dir, "val.parquet"))
    logger.info(f"Loaded validation data with shape: {val_df.shape}")
    
    logger.info("Loading test dataset")
    test_df = pd.read_parquet(os.path.join(input_dir, "test.parquet"))
    logger.info(f"Loaded test data with shape: {test_df.shape}")
    
    # Combine all datasets
    logger.info("Combining all datasets")
    combined_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    logger.info(f"Combined data shape: {combined_df.shape}")
    
    # Remove window_start, window_end, segment_id columns
    logger.info("Removing window_start, window_end, segment_id columns")
    columns_to_drop = ['window_start', 'window_end', 'segment_id']
    # Check if columns exist before dropping
    existing_columns = [col for col in columns_to_drop if col in combined_df.columns]
    if existing_columns:
        combined_df = combined_df.drop(columns=existing_columns)
        logger.info(f"Dropped columns: {existing_columns}")
    else:
        logger.info("No columns to drop found in the dataframe")
    
    # First split data into train and temp (val+test)
    logger.info("Performing random split: 70% train, 30% temp")
    train_df, temp_df = train_test_split(
        combined_df, 
        test_size=0.3, 
        random_state=42, 
        shuffle=True
    )
    
    # Split temp into val and test (50% each of temp, which is 15% each of original)
    logger.info("Splitting temp data into val and test (50% each)")
    val_df, test_df = train_test_split(
        temp_df, 
        test_size=0.5, 
        random_state=42, 
        shuffle=True
    )
    
    # Log split results
    logger.info(f"New train data shape: {train_df.shape} ({len(train_df) / len(combined_df) * 100:.1f}%)")
    logger.info(f"New val data shape: {val_df.shape} ({len(val_df) / len(combined_df) * 100:.1f}%)")
    logger.info(f"New test data shape: {test_df.shape} ({len(test_df) / len(combined_df) * 100:.1f}%)")
    
    # Save to new locations
    logger.info("Saving train dataset")
    train_df.to_parquet(os.path.join(output_dir, "train.parquet"), index=False)
    
    logger.info("Saving validation dataset")
    val_df.to_parquet(os.path.join(output_dir, "val.parquet"), index=False)
    
    logger.info("Saving test dataset")
    test_df.to_parquet(os.path.join(output_dir, "test.parquet"), index=False)
    
    # Create a record file with information about the split
    record_path = os.path.join(output_dir, "record.md")
    with open(record_path, 'w') as f:
        f.write("# Random Resplit of Contact Data\n\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("## Data Sources\n")
        f.write(f"- Original train data: {train_df.shape}\n")
        f.write(f"- Original validation data: {val_df.shape}\n")
        f.write(f"- Original test data: {test_df.shape}\n")
        f.write(f"- Combined data: {combined_df.shape}\n\n")
        f.write("## New Split\n")
        f.write(f"- New train data: {train_df.shape} ({len(train_df) / len(combined_df) * 100:.1f}%)\n")
        f.write(f"- New validation data: {val_df.shape} ({len(val_df) / len(combined_df) * 100:.1f}%)\n")
        f.write(f"- New test data: {test_df.shape} ({len(test_df) / len(combined_df) * 100:.1f}%)\n\n")
        f.write("## Removed Columns\n")
        f.write(f"- Columns removed: {existing_columns}\n")
        
    logger.info(f"Resplit complete. Results saved to {output_dir}")
    
if __name__ == "__main__":
    resplit_contact_data()
