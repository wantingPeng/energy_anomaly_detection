import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path
from src.utils.logger import logger

def randomly_split_dataset():
    """
    Load dataset, remove specified columns, and split into train, validation, and test sets.
    
    The function loads the Kontaktieren_labeled.parquet file, removes window_start, window_end,
    segment_id, overlap_ratio, and step_size columns, then splits the data randomly:
    - 70% training
    - 15% validation
    - 15% testing
    
    The split datasets are saved to Data/processed/machinen_learning/individual_model/randomly_spilt
    """
    # Define input and output paths
    input_path = "Data/processed/machinen_learning/merged_features_no_correlation.parquet"
    output_dir = "Data/processed/machinen_learning/randomly_spilt"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info(f"Loading dataset from {input_path}")
    # Load the dataset
    try:
        df = pd.read_parquet(input_path)
        logger.info(f"Dataset loaded successfully with shape: {df.shape}")
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        raise
    
    # Log the original columns
    logger.info(f"Original columns: {df.columns.tolist()}")
    
    # Columns to remove
    columns_to_remove = ['window_start', 'window_end', 'segment_id', 'overlap_ratio', 'step_size']
    
    # Check if all columns to be removed exist in the dataset
    missing_columns = [col for col in columns_to_remove if col not in df.columns]
    if missing_columns:
        logger.warning(f"The following columns to be removed are not in the dataset: {missing_columns}")
        # Filter out missing columns
        columns_to_remove = [col for col in columns_to_remove if col in df.columns]
    
    # Remove specified columns
    df = df.drop(columns=columns_to_remove, errors='ignore')
    logger.info(f"Removed columns: {columns_to_remove}")
    
    # Get features and target (assuming 'label' is the target column)
    # If 'label' is not the target column, you'll need to replace it with the correct one
    if 'anomaly_label' in df.columns:
        X = df.drop(columns=['anomaly_label'])
        y = df['anomaly_label']
        logger.info(f"Target column 'label' found. X shape: {X.shape}, y shape: {y.shape}")
    else:
        logger.warning("No 'anomaly_label' column found. Using all columns as features.")
        X = df
        y = None
    
    # Split the data into train and temporary test sets (70% / 30%)
    if y is not None:
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
        
        # Split the temporary test set into validation and test sets (50% / 50%, which is 15% / 15% of the total)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
        
        # Reconstruct the dataframes
        train_df = pd.concat([X_train, y_train], axis=1)
        val_df = pd.concat([X_val, y_val], axis=1)
        test_df = pd.concat([X_test, y_test], axis=1)
    else:
        # If no target column, split without stratification
        train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42)
        val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
    
    logger.info(f"Train set: {train_df.shape} ({train_df.shape[0]/df.shape[0]:.2%})")
    logger.info(f"Validation set: {val_df.shape} ({val_df.shape[0]/df.shape[0]:.2%})")
    logger.info(f"Test set: {test_df.shape} ({test_df.shape[0]/df.shape[0]:.2%})")
    
    # Save the datasets to parquet files
    train_path = os.path.join(output_dir, "train.parquet")
    val_path = os.path.join(output_dir, "val.parquet")
    test_path = os.path.join(output_dir, "test.parquet")
    
    logger.info(f"Saving train set to {train_path}")
    train_df.to_parquet(train_path, index=False)
    
    logger.info(f"Saving validation set to {val_path}")
    val_df.to_parquet(val_path, index=False)
    
    logger.info(f"Saving test set to {test_path}")
    test_df.to_parquet(test_path, index=False)
    
    logger.info("Dataset splitting completed successfully")
    
    # Return the paths to the saved files
    return {
        'train': train_path,
        'val': val_path,
        'test': test_path
    }

if __name__ == "__main__":
    logger.info("Starting random split of Kontaktieren dataset")
    split_paths = randomly_split_dataset()
    logger.info(f"Split completed. Files saved to: {split_paths}")
