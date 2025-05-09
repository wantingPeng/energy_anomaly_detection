import pandas as pd
import dask.dataframe as dd
from pathlib import Path
from src.utils.logger import logger
import numpy as np
from datetime import datetime

def calculate_variance_stats(data_path: str) -> dict:
    """Calculate variance statistics for each column in the energy data."""
    logger.info(f"Starting variance calculation for data in {data_path}")
    results = {}
    data_dir = Path(data_path)
    
    for subdir in data_dir.iterdir():
        if subdir.is_dir():
            logger.info(f"Processing directory: {subdir.name}")
            
            # Get all parquet files in the month directories
            parquet_files = []
            for month_dir in subdir.iterdir():
                if month_dir.is_dir():
                    parquet_files.extend(list(month_dir.glob("*.parquet")))
            
            if not parquet_files:
                logger.warning(f"No parquet files found in {subdir.name}")
                continue
                
            logger.info(f"Found {len(parquet_files)} parquet files in {subdir.name}")
            
            # Read all parquet files using dask
            df = dd.read_parquet([str(f) for f in parquet_files])
            
            # Select numeric columns
            numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
            if not numeric_cols:
                logger.warning(f"No numeric columns in {subdir.name}")
                continue
            
            logger.info(f"Processing {len(numeric_cols)} numeric columns")
            
            # Compute stats for numeric columns
            stats = {
                'mean': df[numeric_cols].mean().compute(),
                'std': df[numeric_cols].std().compute(),
                'var': df[numeric_cols].var().compute()
            }
            
            # Calculate summary statistics
     
            
            results[subdir.name] = {
                'statistics': stats,
            }
            
            logger.info(f"Completed variance calculation for {subdir.name}")
    
    return results

def save_variance_report(results: dict, output_path: str):
    """Save variance statistics to a markdown file."""
    logger.info(f"Saving variance report to {output_path}")
    
    # Create markdown content
    md_content = "# Energy Data Variance Analysis Report\n\n"
    md_content += f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    
    for dataset_name, stats in results.items():
        md_content += f"## {dataset_name}\n\n"
        
        # Column-wise Statistics (sorted by variance)
        md_content += "### Column-wise Statistics (Sorted by Variance)\n"
        md_content += "| Column | Mean | Standard Deviation | Variance |\n"
        md_content += "|--------|------|-------------------|----------|\n"
        
        # Sort columns by variance
        sorted_columns = sorted(
            stats['statistics']['var'].items(),
            key=lambda x: x[1]
        )
        
        for col, var in sorted_columns:
            mean = stats['statistics']['mean'][col]
            std = stats['statistics']['std'][col]
            md_content += f"| {col} | {mean:.4f} | {std:.4f} | {var:.4f} |\n"
        
        md_content += "\n"
    
    # Save to file
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write(md_content)
    
    logger.info("Variance report saved successfully")

def main():
    """Main function to run the variance analysis."""
    data_path = "Data/interim/Energy_Data"
    output_path = "experiments/reports/energy_variance.md"
    
    try:
        results = calculate_variance_stats(data_path)
        save_variance_report(results, output_path)
        logger.info("Variance analysis completed successfully")
    except Exception as e:
        logger.error(f"Error during variance analysis: {str(e)}")
        raise

if __name__ == "__main__":
    main() 