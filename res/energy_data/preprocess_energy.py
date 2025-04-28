import pandas as pd
import numpy as np
from datetime import datetime
import pytz
import os
from pathlib import Path

def load_data(file_path: str) -> pd.DataFrame:
    """Load the CSV file and return a DataFrame."""
    print("\nLoading data...")
    df = pd.read_csv(file_path)
    print(f"Initial data shape: {df.shape}")
    return df

def remove_useless_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    """Remove columns that are completely empty or contain only one unique value."""
    print("\nRemoving useless columns...")
    
    # Get initial column count
    initial_cols = df.columns.tolist()
    initial_count = len(initial_cols)
    
    # Find columns with zero variance
    zero_var_cols = [col for col in df.columns if df[col].nunique() <= 1]
    
    # Remove these columns
    df = df.drop(columns=zero_var_cols)
    
    print(f"Removed {initial_count - len(df.columns)} columns")
    return df, zero_var_cols

def remove_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Remove completely duplicated rows."""
    print("\nRemoving duplicate rows...")
    
    initial_count = len(df)
    df = df.drop_duplicates()
    duplicates_removed = initial_count - len(df)
    
    print(f"Removed {duplicates_removed} duplicate rows")
    return df, duplicates_removed

def standardize_timestamp(df: pd.DataFrame, timestamp_col: str = 'TimeStamp') -> pd.DataFrame:
    """Standardize timestamp format and convert to UTC."""
    print("\nStandardizing timestamps...")
    
    # Convert to datetime without timezone first
    df[timestamp_col] = pd.to_datetime(df[timestamp_col], utc=False)
    
    # Check if the timestamp is already tz-aware
    if df[timestamp_col].dt.tz is None:
        # If not tz-aware, assume it's in Berlin time and convert to UTC
        berlin_tz = pytz.timezone('Europe/Berlin')
        df[timestamp_col] = df[timestamp_col].dt.tz_localize(berlin_tz).dt.tz_convert('UTC')
    else:
        # If already tz-aware, just convert to UTC
        df[timestamp_col] = df[timestamp_col].dt.tz_convert('UTC')
    
    print("Timestamps standardized to UTC")
    print(f"Time range: {df[timestamp_col].min()} to {df[timestamp_col].max()}")
    return df

def normalize_station_column(df: pd.DataFrame, station_col: str = 'Station') -> pd.DataFrame:
    """Standardize text in the Station column."""
    if station_col not in df.columns:
        print(f"Warning: {station_col} column not found")
        return df
        
    print("\nNormalizing Station column...")
    
    # Store original unique values
    original_values = df[station_col].unique()
    
    # Standardize text
    df[station_col] = df[station_col].str.strip().str.upper()
    
    # Store new unique values
    new_values = df[station_col].unique()
    
    print(f"Standardized {len(original_values)} unique station values to {len(new_values)} values")
    return df

def analyze_outliers(df: pd.DataFrame) -> dict:
    """Analyze potential outliers using the IQR method."""
    print("\nAnalyzing outliers...")
    
    outlier_stats = {}
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        
        outlier_stats[col] = {
            'min': df[col].min(),
            'max': df[col].max(),
            'Q1': Q1,
            'Q3': Q3,
            'IQR': IQR,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'n_outliers': len(outliers),
            'outlier_percentage': (len(outliers) / len(df)) * 100
        }
    
    print(f"Analyzed outliers for {len(numeric_cols)} numeric columns")
    return outlier_stats

def generate_markdown_report(original_df: pd.DataFrame, 
                           cleaned_df: pd.DataFrame,
                           zero_var_cols: list,
                           duplicates_removed: int,
                           outlier_stats: dict,
                           report_path: str):
    """Generate a detailed Markdown report of the cleaning process."""
    
    report = f"""# Data Cleaning Report
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Initial Data Overview
- Total rows: {len(original_df)}
- Total columns: {len(original_df.columns)}
- Memory usage: {original_df.memory_usage().sum() / 1024**2:.2f} MB

## Cleaning Steps and Results

### 1. Removed Useless Columns
- Initial column count: {len(original_df.columns)}
- Final column count: {len(cleaned_df.columns)}
- Removed columns ({len(zero_var_cols)}):
```
{chr(10).join(['- ' + col for col in zero_var_cols])}
```

### 2. Duplicate Rows Removal
- Initial row count: {len(original_df)}
- Duplicates removed: {duplicates_removed}
- Final row count: {len(cleaned_df)}
- Percentage reduced: {(duplicates_removed/len(original_df))*100:.2f}%

### 3. Timestamp Standardization
- Format: Converted to datetime64[ns, UTC]
- Timezone: Standardized to UTC
- Time range: {cleaned_df['TimeStamp'].min()} to {cleaned_df['TimeStamp'].max()}

### 4. Station Column Normalization
- Standardized text formatting (uppercase, stripped whitespace)
- Unique values: {cleaned_df['Station'].nunique() if 'Station' in cleaned_df.columns else 'Column not found'}

### 5. Outlier Analysis
"""
    
    # Add outlier analysis for each numeric column
    for col, stats in outlier_stats.items():
        report += f"""
#### {col}
- Range: {stats['min']:.2f} to {stats['max']:.2f}
- Q1: {stats['Q1']:.2f}
- Q3: {stats['Q3']:.2f}
- IQR: {stats['IQR']:.2f}
- Outlier bounds: [{stats['lower_bound']:.2f}, {stats['upper_bound']:.2f}]
- Number of outliers: {stats['n_outliers']} ({stats['outlier_percentage']:.2f}%)
"""

    report += f"""
## Final Data Overview
- Total rows: {len(cleaned_df)}
- Total columns: {len(cleaned_df.columns)}
- Memory usage: {cleaned_df.memory_usage().sum() / 1024**2:.2f} MB

## Remaining Columns
```
{chr(10).join(['- ' + col for col in cleaned_df.columns])}
```
"""

    # Write report to file
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)

def main():
    # File paths
    data_file = "Data/Energy_Data/Contacting/April_2024.csv"
    report_file = "src/data_preprocessing/energy_data/data_cleaning_report.md"
    
    # Load data
    original_df = load_data(data_file)
    df = original_df.copy()
    
    # Clean data
    df, zero_var_cols = remove_useless_columns(df)
    df, duplicates_removed = remove_duplicates(df)
    df = standardize_timestamp(df)
    df = normalize_station_column(df)
    outlier_stats = analyze_outliers(df)
    
    # Generate report
    generate_markdown_report(
        original_df=original_df,
        cleaned_df=df,
        zero_var_cols=zero_var_cols,
        duplicates_removed=duplicates_removed,
        outlier_stats=outlier_stats,
        report_path=report_file
    )
    
    print(f"\nCleaning complete. Report generated at {report_file}")
    return df

if __name__ == "__main__":
    main()
