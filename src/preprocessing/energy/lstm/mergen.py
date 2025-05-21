import os
import dask.dataframe as dd

from src.utils.logger import logger

# Function to load and merge datasets using Dask

def load_and_merge_datasets(base_dir, component_types):
    logger.info('Loading and merging datasets from %s', base_dir)
    train_dfs, val_dfs, test_dfs = [], [], []
    for component in component_types:
        component_dir = os.path.join(base_dir, component)
        train_path = os.path.join(component_dir, 'train.parquet')
        val_path = os.path.join(component_dir, 'val.parquet')
        test_path = os.path.join(component_dir, 'test.parquet')
        if os.path.exists(train_path):
            df = dd.read_parquet(train_path)
            train_dfs.append(df)
        if os.path.exists(val_path):
            df = dd.read_parquet(val_path)
            val_dfs.append(df)
        if os.path.exists(test_path):
            df = dd.read_parquet(test_path)
            test_dfs.append(df)
    train_dfs = align_columns(train_dfs)
    val_dfs = align_columns(val_dfs)
    test_dfs = align_columns(test_dfs)
    return train_dfs, val_dfs, test_dfs

# Function to align DataFrame columns

def align_columns(dfs):
    all_columns = set().union(*(df.columns for df in dfs))
    aligned_dfs = []
    for df in dfs:
        missing_cols = all_columns - set(df.columns)
        for col in missing_cols:
            df[col] = 0
        aligned_dfs.append(df)
    return aligned_dfs

# Function to save DataFrames using Dask

def save_dataframes(train_dfs, val_dfs, test_dfs, component_types, output_dir):
    for split, dfs in [('train', train_dfs), ('val', val_dfs), ('test', test_dfs)]:
        for df, component in zip(dfs, component_types):
            # Create directory path
            component_dir = os.path.join(output_dir, split, component)
            os.makedirs(component_dir, exist_ok=True)
            
            # Save data
            output_path = component_dir
            df.to_parquet(output_path)
            logger.info(f'DataFrame for {component} saved to {output_path}')

# Main function to execute the merging process

def main():
    base_dir = 'Data/processed/lsmt/spilt'
    output_dir = 'Data/processed/lsmt/merged'
    component_types = ['contact', 'pcb', 'ring']
    train_dfs, val_dfs, test_dfs = load_and_merge_datasets(base_dir, component_types)       
    save_dataframes(train_dfs, val_dfs, test_dfs, component_types, output_dir)

# Execute the main function
if __name__ == '__main__':
    main()
