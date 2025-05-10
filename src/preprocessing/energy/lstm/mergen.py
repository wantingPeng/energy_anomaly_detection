import os
import dask.dataframe as dd
from sklearn.preprocessing import OneHotEncoder

from src.utils.logger import logger

# Function to load and merge datasets using Dask

def load_and_merge_datasets(base_dir):
    logger.info('Loading and merging datasets from %s', base_dir)
    train_dfs, val_dfs, test_dfs = [], [], []
    for component in ['contact', 'pcb', 'ring']:
        component_dir = os.path.join(base_dir, component)
        train_path = os.path.join(component_dir, 'train.parquet')
        val_path = os.path.join(component_dir, 'val.parquet')
        test_path = os.path.join(component_dir, 'test.parquet')
        if os.path.exists(train_path):
            df = dd.read_parquet(train_path)
            df['component_type'] = component
            train_dfs.append(df)
        if os.path.exists(val_path):
            df = dd.read_parquet(val_path)
            df['component_type'] = component
            val_dfs.append(df)
        if os.path.exists(test_path):
            df = dd.read_parquet(test_path)
            df['component_type'] = component
            test_dfs.append(df)
    train_dfs = align_columns(train_dfs)
    val_dfs = align_columns(val_dfs)
    test_dfs = align_columns(test_dfs)
    train_df = dd.concat(train_dfs)
    val_df = dd.concat(val_dfs)
    test_df = dd.concat(test_dfs)
    return train_df, val_df, test_df

# Function to one-hot encode component type using Dask

def one_hot_encode(df):
    df = df.categorize(columns=['component_type'])
    encoded_df = dd.get_dummies(df, columns=['component_type'], dtype=int)
    return encoded_df

# Function to save DataFrames using Dask

def save_dataframes(train_df, val_df, test_df, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    train_df.to_parquet(os.path.join(output_dir, 'train.parquet'))
    val_df.to_parquet(os.path.join(output_dir, 'val.parquet'))
    test_df.to_parquet(os.path.join(output_dir, 'test.parquet'))
    logger.info('Merged DataFrames saved to %s', output_dir)

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

# Main function to execute the merging process

def main():
    base_dir = 'Data/processed/lsmt/spilt'
    output_dir = 'Data/processed/lsmt/merged'
    train_df, val_df, test_df = load_and_merge_datasets(base_dir)
    train_df = one_hot_encode(train_df)
    val_df = one_hot_encode(val_df)
    test_df = one_hot_encode(test_df)
    save_dataframes(train_df, val_df, test_df, output_dir)

# Execute the main function
if __name__ == '__main__':
    main()
