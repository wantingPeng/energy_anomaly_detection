import pandas as pd
from pathlib import Path
from src.utils.logger import logger
import os
from datetime import datetime

def analyze_parquet_basic_info():
    """
    Analyze basic information of the parquet file and save results to markdown.
    """
    # Get current timestamp for logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    try:
        # Define input and output paths
        input_path = Path("Data/interim/Energy_labeling_correlations/Pcb_labeled.parquet")
        output_dir = Path("experiments/reports")
        output_file = output_dir / "Window_Feature.md"

        # Create output directory if it doesn't exist
        output_dir.mkdir(parents=True, exist_ok=True)

        # Read parquet file
        logger.info(f"Reading parquet file from: {input_path}")
        df = pd.read_parquet(input_path)

        # Get basic information
        n_rows = len(df)
        n_cols = len(df.columns)
        columns = list(df.columns)

        # Create markdown content
        markdown_content = f"""# Parquet File Analysis Report
Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Basic Information
- **File Path**: {input_path}
- **Number of Rows**: {n_rows:,}
- **Number of Columns**: {n_cols}

## Column Names
{', '.join(columns)}
"""

        # Save markdown content to file
        with open(output_file, 'w') as f:
            f.write(markdown_content)

        logger.info(f"Analysis completed. Results saved to: {output_file}")
    except Exception as e:
        logger.error(f"Error analyzing parquet file: {e}")

if __name__ == "__main__":
    analyze_parquet_basic_info()
