import os
from pathlib import Path
import dask.dataframe as dd
from src.utils.logger import logger
import pandas as pd

def anomaly_data_explore(
    input_dir: str = "/home/wanting/energy_anomaly_detection/Data/interim/Anomaly_Data",
    output_file: str = "/home/wanting/energy_anomaly_detection/experiments/reports/anomaly_data_explore.md",
    target_text_columns: list = ["Condition", "Comment", "Station", "Line"]
) -> None:
    """
    Explore specific text-based columns in anomaly data and save unique values to markdown report.
    
    Args:
        input_dir (str): Directory containing parquet files
        output_file (str): Path to output markdown file
        target_text_columns (list): Columns to analyze for unique text values
    """
    input_path = Path(input_dir)
    output_path = Path(output_file)
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Starting anomaly data exploration from {input_dir}")
    
    text_content = {col: set() for col in target_text_columns}
    
    for file_path in input_path.glob("**/*.parquet"):
        try:
            logger.info(f"Processing file: {file_path}")
            df = dd.read_parquet(str(file_path))
            
            for col in target_text_columns:
                if col not in df.columns:
                    logger.warning(f"Column '{col}' not found in {file_path.name}. Skipping.")
                    continue
                
                try:
                    unique_values = df[col].dropna().unique().compute()
                    text_content[col].update(unique_values)
                    logger.info(f"Column '{col}': {len(unique_values)} unique values found.")
                except Exception as e:
                    logger.error(f"Error processing column '{col}' in {file_path.name}: {e}")
        
        except Exception as e:
            logger.error(f"Failed to read {file_path}: {e}")
    
    # Generate markdown report
    logger.info("Generating markdown report")
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("# Anomaly Data Text Content Exploration\n\n")
        
        for col, values in text_content.items():
            f.write(f"## Column: {col}\n\n")
            if values:
                f.write("Unique values (including rare ones):\n\n")
                for value in sorted(values):
                    if pd.notna(value):
                        f.write(f"- {value}\n")
            else:
                f.write("_No non-null values found._\n")
            f.write("\n")
    
    logger.info(f"Report saved to {output_file}")
if __name__ == "__main__":
    anomaly_data_explore()