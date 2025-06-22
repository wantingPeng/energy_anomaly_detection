import pickle
import pandas as pd
from datetime import datetime
import os
from pathlib import Path
import numpy as np
from src.utils.logger import logger

# Function to check if a timestamp is within any anomaly periods
def is_anomaly(timestamp, anomaly_periods):
    """
    Check if timestamp falls within any anomaly period.
    
    Args:
        timestamp: Pandas timestamp to check
        anomaly_periods: List of (start_time, end_time) tuples
        
    Returns:
        1 if timestamp is within any anomaly period, 0 otherwise
    """
    # Extract just the year, month, day, hour, minute (ignore seconds)
    # Note: We are working with timezone-aware timestamps
    for start_time, end_time in anomaly_periods:
        # Check if timestamp falls between start and end
        if start_time <= timestamp <= end_time:
            return 1
    return 0

# Load the anomaly dictionary
try:
    logger.info("Loading anomaly dictionary...")
    with open('Data/machine/Anomaly_Data/anomaly_dict_merged.pkl', 'rb') as f:
        anomaly_dict = pickle.load(f)
    
    logger.info(f"Keys in anomaly dict: {list(anomaly_dict.keys())}")
    
    # Print example anomaly periods for Kontaktieren
    if 'Kontaktieren' in anomaly_dict:
        logger.info("\nExample anomaly periods for Kontaktieren:")
        for i, (start, end) in enumerate(anomaly_dict['Kontaktieren'][:3]):
            logger.info(f"  {i+1}. {start} to {end}")
except Exception as e:
    logger.error(f"Error loading anomaly dictionary: {e}")
    exit(1)

# Component to station mapping
component_to_station = {
    'contact': 'Kontaktieren',
    'ring': 'Ringmontage',
    'pcb': 'Pcb'
}

# Define the directories to process
splits = ['train', 'val', 'test']
components = ['contact']  # Add others if needed

# Process each file and add anomaly labels
for split in splits:
    for component in components:
        station = component_to_station.get(component)
        if not station or station not in anomaly_dict:
            logger.warning(f"Station not found for component {component} or no anomaly data available")
            continue
            
        # Get anomaly periods for this station
        anomaly_periods = anomaly_dict[station]
        
        # Load the data
        input_path = f"Data/row_energyData_subsample_Transform/standscaler/{split}/{component}/part.0.parquet"
        output_path = f"Data/row_energyData_subsample_Transform/labeled/{split}/{component}/part.0.parquet"
        
        if not os.path.exists(input_path):
            logger.warning(f"File not found: {input_path}")
            continue
            
        logger.info(f"\nProcessing {input_path}...")
        
        # Create output directory if it doesn't exist
        Path(os.path.dirname(output_path)).mkdir(parents=True, exist_ok=True)
        
        # Load the data
        df = pd.read_parquet(input_path)
        logger.info(f"Loaded {len(df)} rows from {input_path}")
        
        # Add anomaly label column
        logger.info("Adding anomaly labels...")
        df['anomaly_label'] = df['TimeStamp'].apply(lambda ts: is_anomaly(ts, anomaly_periods))
        
        # Count anomalies
        anomaly_count = df['anomaly_label'].sum()
        logger.info(f"Found {anomaly_count} anomalies out of {len(df)} rows ({anomaly_count/len(df)*100:.2f}%)")
        
        # Sample of labeled data
        logger.info("\nSample of labeled data:")
        sample = df.sample(min(5, len(df)))
        for _, row in sample.iterrows():
            logger.info(f"Timestamp: {row['TimeStamp']}, Label: {row['anomaly_label']}")
        
        # Save the labeled data
        logger.info(f"Saving labeled data to {output_path}...")
        df.to_parquet(output_path, index=False)
        
logger.info("\nProcessing complete!") 