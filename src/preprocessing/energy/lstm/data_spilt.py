import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from src.utils.logger import logger

# Define the directory paths
input_dir = 'Data/interim/Energy_time_series'
output_dir = 'Data/processed/lsmt/spilt'
os.makedirs(output_dir, exist_ok=True)

# Function to load and sort datasets

def load_and_sort_datasets():
    logger.info('Loading datasets from %s', input_dir)
    if not os.path.exists(input_dir):
        logger.error('Input directory does not exist: %s', input_dir)
        return []
    dataframes = []
    for root, _, files in os.walk(input_dir):
        for filename in files:
            if filename.endswith('.parquet'):
                file_path = os.path.join(root, filename)
                df = pd.read_parquet(file_path)
                if not df['TimeStamp'].is_monotonic_increasing:
                    logger.warning('DataFrame %s is not monotonic increasing, sorting...', file_path)
                    df = df.sort_values(by='TimeStamp')
                dataframes.append(df)
                logger.info('Loaded and sorted file: %s', file_path)
    if not dataframes:
        logger.error('No dataframes were loaded.')
    return dataframes



# Function to split data into train, val, test

def split_data(dataframes):
    train_dfs, val_dfs, test_dfs = [], [], []
    for i, df in enumerate(dataframes):
        logger.debug('Processing DataFrame %d with shape %s', i, df.shape)
        if df.empty:
            logger.warning('DataFrame %d is empty and will be skipped', i)
            continue
        train, temp = train_test_split(df, test_size=0.3, shuffle=False)
        val, test = train_test_split(temp, test_size=0.5, shuffle=False)
        train_dfs.append(train)
        val_dfs.append(val)
        test_dfs.append(test)
        logger.debug('Split DataFrame %d into train (%s), val (%s), test (%s)', i, train.shape, val.shape, test.shape)
    if not train_dfs:
        logger.error('No training data was created.')
    if not val_dfs:
        logger.error('No validation data was created.')
    if not test_dfs:
        logger.error('No test data was created.')
    return pd.concat(train_dfs), pd.concat(val_dfs), pd.concat(test_dfs)


# Function to save DataFrames

def save_dataframes(train_df, val_df, test_df, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    train_df.to_parquet(os.path.join(output_dir, 'train.parquet'))
    val_df.to_parquet(os.path.join(output_dir, 'val.parquet'))
    test_df.to_parquet(os.path.join(output_dir, 'test.parquet'))
    logger.info('DataFrames saved to %s', output_dir)

# Function to process a single dataset

def process_single_dataset(file_path, output_dir):
    logger.info('Processing single dataset: %s', file_path)
    df = pd.read_parquet(file_path)
    if not df['TimeStamp'].is_monotonic_increasing:
        df = df.sort_values(by='TimeStamp')
    train_df, val_df, test_df = split_data([df])
    save_dataframes(train_df, val_df, test_df, output_dir)

# Main function to execute the pipeline

def main():
    pcb_file = 'Data/interim/Energy_Data_cleaned/PCB_20250509_085245/part.0.parquet'
    pcb_output_dir = 'Data/processed/lsmt/spilt/pcb_cleaned'
    process_single_dataset(pcb_file, pcb_output_dir)
    # You can manually process the remaining datasets and merge them later

# Execute the main function
if __name__ == '__main__':
    main()
