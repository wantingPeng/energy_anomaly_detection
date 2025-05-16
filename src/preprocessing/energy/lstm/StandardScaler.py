import os
import gc
import numpy as np
import dask.dataframe as dd
from dask_ml.preprocessing import StandardScaler
import psutil
from src.utils.logger import logger
from src.utils.memory_left import log_memory
def standardize_dataframes():
    """
    Load parquet files from Data/processed/lsmt/merged directory,
    apply standard scaling to numerical columns, and save to
    Data/processed/lsmt/standerScaler directory.
    
    Columns to delete: 'IsOutlier', 'ID', 'Station', 'time_diff'
    Columns to keep but not scale: 'segment_id', 'TimeStamp', 
                                 'component_type_contact', 
                                 'component_type_pcb', 
                                 'component_type_ring'
    """
    # Set input and output directories
    input_dir = 'Data/processed/lsmt/merged'
    output_dir = 'Data/processed/lsmt/standerScaler'
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info("Starting data standardization process")
    
    # List of files to process
    files = ['train.parquet', 'test.parquet', 'val.parquet']
    
    # Columns to delete
    columns_to_delete = ['IsOutlier', 'ID', 'Station', 'time_diff']
    
    # Columns to keep but not standardize
    columns_to_skip = [
        'segment_id', 'TimeStamp', 
        'component_type_contact', 'component_type_pcb', 'component_type_ring'
    ]
    
    # Create a StandardScaler object
    scaler = StandardScaler()
    
    # Fit scaler on training data first
    logger.info("Loading training data for fitting the scaler")
    train_df = dd.read_parquet(os.path.join(input_dir, 'train.parquet'))
    
    # Remove columns to delete
    for col in columns_to_delete:
        if col in train_df.columns:
            train_df = train_df.drop(col, axis=1)
    
    # Identify numerical columns for scaling (excluding columns_to_skip)
    all_columns = train_df.columns
    numeric_columns = []
    
    # Explicitly compute dtypes to identify numeric columns
    dtypes = train_df.dtypes
    
    for col in all_columns:
        if col not in columns_to_skip:
            dtype = dtypes[col]
            if np.issubdtype(dtype, np.number):
                numeric_columns.append(col)
    
    logger.info(f"Identified {len(numeric_columns)} numeric columns for standardization")
    
    # Fit the scaler on the numeric columns of the training data
    logger.info("Fitting StandardScaler on training data")
    numeric_data = train_df[numeric_columns]
    scaler.fit(numeric_data)
    
    # Free memory after fitting
    del train_df, numeric_data
    gc.collect()
    logger.info("Freed memory after fitting the scaler")
    
    # Process each file
    for file in files:
        logger.info(f"Processing {file}")
        df = dd.read_parquet(os.path.join(input_dir, file))
        
        # Remove columns to delete
        for col in columns_to_delete:
            if col in df.columns:
                df = df.drop(col, axis=1)
        
        # Apply scaling to numeric columns
        logger.info(f"Applying standardization to {file}")
        df_numeric = df[numeric_columns]
        log_memory('before scaling')

        df_numeric_scaled = scaler.transform(df_numeric)
        log_memory('after scaling')

        # Replace original values with scaled values using assign to maintain lazy evaluation
        logger.info("Replacing columns with standardized values")
        df = df.assign(**{col: df_numeric_scaled[col] for col in numeric_columns})
        log_memory('after assign')

        # Save the processed DataFrame
        output_path = os.path.join(output_dir, file)
        logger.info(f"Saving standardized {file} to {output_path}")
        log_memory('before repartition')
        df = df.repartition(partition_size="600MB")
        log_memory('after repartition')
        df.to_parquet(
              output_path,
              engine="pyarrow",
              write_index=False,
              write_metadata_file=False,  
              schema="infer"
          )
        log_memory('after to_parquet')
        # Free memory after processing each file
        del df, df_numeric, df_numeric_scaled
        gc.collect()
        logger.info(f"Freed memory after processing {file}")
    
    logger.info("Data standardization completed successfully")

if __name__ == '__main__':
    standardize_dataframes()
