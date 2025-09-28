from src.utils.logger import logger
import dask.dataframe as dd
import pandas as pd
import os
import numpy as np

def identify_zero_variance_columns(df, threshold=0.01):
    """Identify columns with zero or near-zero variance."""
    numeric_df = df.select_dtypes(include=["int", "float", "number"])
    variances = numeric_df.var()
    zero_var_cols = variances[variances <= threshold].index.tolist()
    return zero_var_cols

def get_column_stats(df):
    """Get basic statistics about the columns."""
    stats = {
        'total_rows': len(df),
        'missing_values': df.isnull().sum().compute(),
        'dtypes': df.dtypes,
    }
    return stats

def cleaning(df):
    """
    Clean the energy dataset with enhanced time series analysis.
    
    Args:
        df: Input DataFrame
    """
    logger.info(f"Data type: {type(df)}")

    # Explicitly define parameters
    zero_variance_threshold = 0.01
    remove_duplicates = True
    
    initial_stats = get_column_stats(df)
    
    # Convert Dask DataFrame to pandas for time series operations
    df = df.compute()

    # Create report content
    report_content = ["# Energy Data Cleaning Report\n\n"]
    report_content.append("## Initial Dataset Statistics\n")
    report_content.append(f"- Total rows: {initial_stats['total_rows']}\n")
    report_content.append(f"- Total columns: {len(df.columns)}\n")
    report_content.append("- Missing values per column:\n")
    for col, missing in initial_stats['missing_values'].items():
        report_content.append(f"  - {col}: {missing}\n")
    
    # 1. Remove zero variance columns
    zero_var_cols = identify_zero_variance_columns(df, zero_variance_threshold)
    if zero_var_cols:
        logger.info(f"Removing zero variance columns: {zero_var_cols}")
        df = df.drop(columns=zero_var_cols)
        report_content.append("\n## Removed Zero Variance Columns\n")
        report_content.append(f"- Columns removed: {', '.join(zero_var_cols)}\n")
    
    # 2. Remove duplicates
    if remove_duplicates:
        n_rows_before = len(df)
        df = df.drop_duplicates()
        n_rows_after = len(df)
        n_duplicates = n_rows_before - n_rows_after
        logger.info(f"Removed {n_duplicates} duplicate rows")
        report_content.append("\n## Duplicate Removal\n")
        report_content.append(f"- Initial rows: {n_rows_before}\n")
        report_content.append(f"- Rows after deduplication: {n_rows_after}\n")
        report_content.append(f"- Duplicates removed: {n_duplicates}\n")
    
    # 3. Convert timestamps to datetime64[ns, UTC]
    if 'TimeStamp' in df.columns:
        df['TimeStamp'] = pd.to_datetime(df['TimeStamp'], utc=True)
        logger.info(f"Converted TimeStamp to datetime64[ns, UTC]")
        report_content.append("\n## Timestamp Conversion\n")
        report_content.append(f"- Converted TimeStamp to datetime64[ns, UTC]\n")
    
    # 4. Sort data by TimeStamp
    if 'TimeStamp' in df.columns:
        df = df.sort_values(by='TimeStamp')
        logger.info("Sorted data by TimeStamp")
        report_content.append("\n## Data Sorting\n")
        report_content.append("- Sorted data by TimeStamp in ascending order\n")
    
    # Save the report
    report_path = "Data/machine/cleaning_utc/ring_cleaned_1.md"
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_content))

    # Manually save the data
    output_dir = "Data/machine/cleaning_utc"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save as parquet
    output_file = os.path.join(output_dir, "Energy_Data_ring_cleaned_1.parquet")
    df.to_parquet(output_file, index=False)
    logger.info(f"Saved cleaned data to {output_file}")
    
    return df

if __name__ == "__main__":
    # Load data from specified path
    input_path = "Data/row/Energy_Data/Ring"
    logger.info(f"Loading data from {input_path}")
    
    # Use Dask to read all CSV files in the directory
    df = dd.read_csv(os.path.join(input_path, "*.csv"))
    
    # Process the data
    cleaned_df = cleaning(df)
    
    # Print the first 5 rows of the processed data
    print("\nPreview of processed data:")
    print(cleaned_df.head(5))

