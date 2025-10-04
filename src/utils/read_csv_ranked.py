import pandas as pd
from pathlib import Path
from typing import Optional, Union
from src.utils.logger import logger


def read_first_n_rows(
    csv_path: Union[str, Path],
    nrows: int = 1000,
    output_path: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """
    Read the first N rows of a CSV file (for quick inspection).
    
    Args:
        csv_path: Path to the CSV file.
        nrows: Number of rows to read (default: 1000).
        output_path: Optional path to save the sample (as CSV).
    
    Returns:
        DataFrame with the first N rows.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    logger.info(f"Reading first {nrows} rows from: {csv_path}")
    
    # Read only the first N rows
    df = pd.read_csv(csv_path, nrows=nrows)
    
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info(f"Saved sample to: {output_path}")
    
    return df


if __name__ == "__main__":
    # Load data from CSV and sort by TimeStamp
    input_csv = "Data/row/Energy_Data/Contacting/Januar_2024.csv"
    
    try:
        logger.info(f"Loading data from: {input_csv}")
        
        # Read the entire CSV file
        df = pd.read_csv(input_csv)
        logger.info(f"Original data shape: {df.shape}")
        
        # Check if TimeStamp column exists
        if 'TimeStamp' not in df.columns:
            logger.error(f"TimeStamp column not found. Available columns: {df.columns.tolist()}")
            raise ValueError("TimeStamp column not found in the CSV file")
        
        # Sort by TimeStamp
        df_sorted = df.sort_values(by='TimeStamp', ascending=True)
        logger.info(f"Data sorted by TimeStamp")
        
        # Print first 10 rows
        print("\n" + "="*80)
        print("前10行数据 (按 TimeStamp 排序):")
        print("="*80)
        print(df_sorted.head(10).to_string())
        print("="*80 + "\n")
        
        logger.info(f"Successfully displayed first 10 rows sorted by TimeStamp")
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise