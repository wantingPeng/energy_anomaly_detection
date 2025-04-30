import pandas as pd
import numpy as np
from src.utils.logger import logger
from src.preprocessing.data_save import data_save
import pytz
import yaml
import dask.dataframe as dd
from dask.diagnostics import ProgressBar
import os

def load_config():
    """Load preprocessing configuration."""
    with open('configs/preprocessing.yaml', 'r') as file:
        return yaml.safe_load(file)

def get_column_stats(df):
    """Get basic statistics about the columns."""
    stats = {
        'total_rows': len(df),
        'missing_values': df.isnull().sum().compute(),
        'dtypes': df.dtypes,
    }
    return stats

def identify_zero_variance_columns(df, threshold=0.01):
    """Identify columns with zero or near-zero variance."""
    numeric_df = df.select_dtypes(include=["int", "float", "number"])
    variances = numeric_df.var().compute()
    zero_var_cols = variances[variances <= threshold].index.tolist()
    return zero_var_cols

def detect_outliers(df, numeric_cols):
    """
    Detect outliers using IQR method and mark them.
    Returns DataFrame with new IsOutlier column and outlier statistics.
    """
    config = load_config()
    outlier_method = config['outlier']['method']
    
    if outlier_method == 'iqr':
        multiplier = config['outlier']['iqr_multiplier']
        
        outlier_stats = {}
        df['IsOutlier'] = 0
        
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25).compute()
            Q3 = df[col].quantile(0.75).compute()   
            IQR = Q3 - Q1
            lower_bound = Q1 - multiplier * IQR
            upper_bound = Q3 + multiplier * IQR
            
            outliers = (df[col] < lower_bound) | (df[col] > upper_bound)
            df['IsOutlier'] = df['IsOutlier'] | outliers
            
            # Store statistics
            outlier_stats[col] = {
                'lower_bound': float(lower_bound),
                'upper_bound': float(upper_bound),
                'n_outliers': int(outliers.sum().compute())
            }
            
    return df, outlier_stats

def cleaning(df):
    """
    Clean the energy dataset.
    
    Args:
        df: Input Dask DataFrame
    """
    config = load_config()
    initial_stats = get_column_stats(df)
    
    # Create report content
    report_content = ["# Energy Data Cleaning Report\n\n"]
    report_content.append("## Initial Dataset Statistics\n")
    report_content.append(f"- Total rows: {initial_stats['total_rows']}\n")
    report_content.append(f"- Total columns: {len(df.columns)}\n")
    report_content.append("- Missing values per column:\n")
    for col, missing in initial_stats['missing_values'].items():
        report_content.append(f"  - {col}: {missing}\n")
    
    # 1. Remove zero variance columns
    zero_var_cols = identify_zero_variance_columns(df, config['cleaning']['zero_variance_threshold'])
    if zero_var_cols:
        logger.info(f"Removing zero variance columns: {zero_var_cols}")
        df = df.drop(columns=zero_var_cols)
        report_content.append("\n## Removed Zero Variance Columns\n")
        report_content.append(f"- Columns removed: {', '.join(zero_var_cols)}\n")
    
    # 2. Remove duplicates
    if config['cleaning']['remove_duplicates']:
        n_rows_before = len(df)
        df = df.drop_duplicates()
        n_rows_after = len(df)
        n_duplicates = n_rows_before - n_rows_after
        logger.info(f"Removed {n_duplicates} duplicate rows")
        report_content.append("\n## Duplicate Removal\n")
        report_content.append(f"- Initial rows: {n_rows_before}\n")
        report_content.append(f"- Rows after deduplication: {n_rows_after}\n")
        report_content.append(f"- Duplicates removed: {n_duplicates}\n")
    
    # 3. Standardize timestamp
    timestamp_col = config['columns']['timestamp_col']
    if timestamp_col in df.columns:
        df = df.compute()
        df[timestamp_col] = pd.to_datetime(df[timestamp_col], utc=True)  
        df = dd.from_pandas(df, npartitions=1)
        
        report_content.append("\n## Timestamp Standardization\n")
    
    # 4. Standardize Station column
    station_col = config['columns']['station_col']
    if station_col in df.columns:
        df[station_col] = df[station_col].astype(str).str.strip()
        
    # 5. Handle outliers
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    df, outlier_stats = detect_outliers(df, numeric_cols)
    
    report_content.append("\n## Outlier Detection\n")
    for col, stats in outlier_stats.items():
        report_content.append(f"\n### {col}\n")
        report_content.append(f"- Lower bound: {stats['lower_bound']:.2f}\n")
        report_content.append(f"- Upper bound: {stats['upper_bound']:.2f}\n")
        report_content.append(f"- Number of outliers: {stats['n_outliers']}\n")
    
    # Save the report
    report_path = "experiments/reports/Energy_Data_cleaned.md"
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_content))
    
    # Save the cleaned dataset
    data_save(df, "Energy_Data_cleaned", save_dir="Data/interim")
    
    return df
