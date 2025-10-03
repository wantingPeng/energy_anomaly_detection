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
    # Example usage (modify paths as needed)
    input_csv = "dataset/ALLcontact_noSegment/test_processed.csv"
    #output_csv = "Data/data_preview/row_anomaly_data.csv"  # Optional
    
    try:
        df = read_first_n_rows(
            csv_path=input_csv,
            nrows=1000,
            #output_path=output_csv,  # Set to None if you don't want to save
        )
        logger.info(f"Data preview (first 5 rows):\n{df.head()}")
        logger.info(f"Data shape: {df.shape}")
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise