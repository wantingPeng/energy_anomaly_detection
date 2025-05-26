import os
import pandas as pd
from src.utils.logger import logger

def calculate_anomaly_ratio():
    """
    Calculate the ratio of anomalies to normal samples in the LSMT train dataset
    """
    dataset_path = "Data/processed/lsmt/dataset/train"
    categories = ["ring", "pcb", "contact"]
    
    total_samples = 0
    total_anomalies = 0
    
    logger.info("Calculating anomaly ratio for LSMT train dataset")
    
    # Process each category
    for category in categories:
        category_path = os.path.join(dataset_path, category)
        category_samples = 0
        category_anomalies = 0
        
        # Find all parquet files
        parquet_files = [f for f in os.listdir(category_path) if f.endswith(".parquet")]
        
        logger.info(f"Processing {category} category with {len(parquet_files)} parquet files")
        
        # Process each parquet file
        for file in parquet_files:
            file_path = os.path.join(category_path, file)
            df = pd.read_parquet(file_path, engine='pyarrow')
            
            # Count samples and anomalies
            file_samples = len(df)
            file_anomalies = df['label'].sum()
            
            category_samples += file_samples
            category_anomalies += file_anomalies
            
            logger.info(f"  {file}: {file_samples} samples, {file_anomalies} anomalies, ratio: {file_anomalies/file_samples:.6f}")
        
        # Calculate category ratio
        category_ratio = category_anomalies / category_samples if category_samples > 0 else 0
        logger.info(f"{category} category: {category_samples} samples, {category_anomalies} anomalies, ratio: {category_ratio:.6f}")
        
        total_samples += category_samples
        total_anomalies += category_anomalies
    
    # Calculate overall ratio
    overall_ratio = total_anomalies / total_samples if total_samples > 0 else 0
    logger.info(f"Overall dataset: {total_samples} samples, {total_anomalies} anomalies, ratio: {overall_ratio:.6f}")
    logger.info(f"Anomaly percentage: {overall_ratio * 100:.4f}%")
    
    return {
        "total_samples": total_samples,
        "total_anomalies": total_anomalies,
        "anomaly_ratio": overall_ratio
    }

if __name__ == "__main__":
    calculate_anomaly_ratio()
