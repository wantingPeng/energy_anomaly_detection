import os
import gc
import numpy as np
import dask.dataframe as dd
from dask_ml.preprocessing import StandardScaler
import psutil
from src.utils.logger import logger
from src.utils.memory_left import log_memory
from pandas.api.types import is_numeric_dtype

def standardize_dataframes():
    """
    Load parquet files from Data/processed/lsmt/merged directory,
    apply standard scaling to numerical columns for each component type separately,
    and save to Data/processed/lsmt/standerScaler directory.
    
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
    
    logger.info("Starting data standardization process with separate scalers for each component type")
    
    # Component types
    component_types = ['contact', 'pcb', 'ring']
    
    # Columns to delete
    columns_to_delete = ['IsOutlier', 'ID', 'Station', 'time_diff','component_type']
    
    # Columns to keep but not standardize
    columns_to_skip = [
        'segment_id', 'TimeStamp' ]
    
    # Create a dictionary to store scalers for each component type
    scalers = {}
    
    # For each component type, fit a scaler on the training data
    for component_type in component_types:
        logger.info(f"Processing component type: {component_type}")
        
        # Create output directories for this component
        for split in ['train', 'test', 'val']:
            component_output_dir = os.path.join(output_dir, split, component_type)
            os.makedirs(component_output_dir, exist_ok=True)
        
        # Load training data for this component type
        component_train_path = os.path.join(input_dir, 'train', component_type)
        logger.info(f"Loading training data from {component_train_path}")
        train_df = dd.read_parquet(component_train_path)
        
        # Remove columns to delete
        for col in columns_to_delete:
            if col in train_df.columns:
                train_df = train_df.drop(col, axis=1)
        
        # Identify numerical columns for scaling (excluding columns_to_skip)
        all_columns = train_df.columns
        numeric_columns = []
        
        # Explicitly compute dtypes to identify numeric columns        
        for col in all_columns:
            if col not in columns_to_skip:
                if is_numeric_dtype(train_df[col]):
                    numeric_columns.append(col)      

        
        logger.info(f"Identified {len(numeric_columns)} numeric columns for standardization")
        
        # Create and fit the scaler for this component
        scaler = StandardScaler()
        logger.info(f"Fitting StandardScaler for {component_type}")
        numeric_data = train_df[numeric_columns]
        scaler.fit(numeric_data)
        
        # Store the scaler in the dictionary
        scalers[component_type] = {
            'scaler': scaler,
            'numeric_columns': numeric_columns
        }
        
        # Free memory after fitting
        del train_df, numeric_data
        gc.collect()
        logger.info(f"Freed memory after fitting the scaler for {component_type}")
    
    # Now apply the scalers to each dataset
    for split in ['train', 'test', 'val']:
        logger.info(f"Processing {split} split")
        
        for component_type in component_types:
            logger.info(f"Applying standardization for {component_type} in {split}")
            
            # Get the appropriate scaler and numeric columns
            scaler = scalers[component_type]['scaler']
            numeric_columns = scalers[component_type]['numeric_columns']
            
            # Load the data
            input_path = os.path.join(input_dir, split, component_type)
            df = dd.read_parquet(input_path)
            logger.info(f"Loaded {split}/{component_type} data from {input_path}")
            # Remove columns to delete
            for col in columns_to_delete:
                if col in df.columns:
                    df = df.drop(col, axis=1)
            
            # Apply scaling to numeric columns
            logger.info(f"Applying {component_type} standardization to {split}")
            df_numeric = df[numeric_columns]

            df_numeric_scaled = scaler.transform(df_numeric)

            # Replace original values with scaled values using assign to maintain lazy evaluation
            logger.info("Replacing columns with standardized values")
            df = df.assign(**{col: df_numeric_scaled[col] for col in numeric_columns})

            # Save the processed DataFrame
            output_path = os.path.join(output_dir, split, component_type)
            logger.info(f"Saving standardized {split}/{component_type} to {output_path}")
            df = df.repartition(partition_size="600MB")
            df.to_parquet(
                output_path,
                engine="pyarrow",
                write_index=False,
                write_metadata_file=False,  
                schema="infer"
            )
            
            # Free memory after processing
            del df, df_numeric, df_numeric_scaled
            gc.collect()
            logger.info(f"Freed memory after processing {split}/{component_type}")
    
    logger.info("Component-specific data standardization completed successfully")

if __name__ == '__main__':
    standardize_dataframes()
