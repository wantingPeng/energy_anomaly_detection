import os
import pickle
from typing import Dict, List, Tuple
import dask.dataframe as dd
import pandas as pd
from datetime import datetime
from src.utils.logger import logger
from pathlib import Path


def merge_overlapping_periods(periods: List[Tuple[datetime, datetime]]) -> List[Tuple[datetime, datetime]]:
    """
    Merge overlapping time periods into their union.
    
    Args:
        periods (List[Tuple[datetime, datetime]]): List of (start_time, end_time) tuples
        
    Returns:
        List[Tuple[datetime, datetime]]: List of merged (start_time, end_time) tuples
    """
    if not periods:
        return []
    
    # Sort periods by start time
    sorted_periods = sorted(periods, key=lambda x: x[0])
    
    merged = []
    current_start, current_end = sorted_periods[0]
    
    for start, end in sorted_periods[1:]:
        # If current period overlaps with the next one, extend the current period
        if start <= current_end:
            current_end = max(current_end, end)
        else:
            # No overlap, add the current period to results and start a new one
            merged.append((current_start, current_end))
            current_start, current_end = start, end
    
    # Add the last period
    merged.append((current_start, current_end))
    
    return merged


def generate_anomaly_dict(merge_overlaps: bool = True) -> Dict[str, List[Tuple[datetime, datetime]]]:
    """
    Generate a dictionary of anomaly time periods organized by station.
    
    Args:
        merge_overlaps (bool): Whether to merge overlapping time periods
    
    Returns:
        Dict[str, List[Tuple[datetime, datetime]]]: Dictionary where keys are station names and values are
        lists of (start_time, end_time) tuples sorted by start_time.
    """
    # Define input and output paths
    input_path = "Data/interim/Anomaly_Data/Duration_of_Anomalies_cleaned.parquet"
    output_path = "Data/interim/Anomaly_Data/anomaly_dict_merged.pkl"
    
    try:
        # Check if input file exists
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input file not found: {input_path}")
            
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_path)
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info(f"Reading anomaly data from {input_path}")
        
        # Read parquet file using dask
        ddf = dd.read_parquet(input_path)
        
        # Ensure required columns exist and rename if necessary
        required_columns = {'Station', 'StartTime', 'EndTime'}
        actual_columns = set(ddf.columns)
        
        if not required_columns.issubset(actual_columns):
            logger.warning("Column names don't match expected names. Attempting to map columns...")
            # Add column mapping logic here if needed
            raise ValueError("Required columns not found in the dataset")
        
        # Convert to pandas for easier processing
        logger.info("Converting to pandas DataFrame for processing")
        df = ddf.compute()
        
        # Initialize the anomaly dictionary
        anomaly_dict = {}
        
        # Group by station and process each group
        logger.info("Processing anomaly periods by station")
        for station, group in df.groupby('Station'):
            # Sort by starttime and create list of tuples
            periods = list(zip(group['StartTime'], group['EndTime']))
            
            if merge_overlaps:
                logger.info(f"Merging overlapping periods for station: {station}")
                periods = merge_overlapping_periods(periods)
                logger.info(f"Station {station}: {len(group)} original periods merged into {len(periods)} non-overlapping periods")
            else:
                periods.sort(key=lambda x: x[0])  # Sort by starttime
                
            anomaly_dict[station] = periods
            
        # Save the dictionary
        logger.info(f"Saving anomaly dictionary to {output_path}")
        with open(output_path, 'wb') as f:
            pickle.dump(anomaly_dict, f)
            
        logger.info(f"Successfully processed {len(anomaly_dict)} stations")
        return anomaly_dict
        
    except Exception as e:
        logger.error(f"Error generating anomaly dictionary: {str(e)}")
        raise

if __name__ == "__main__":
    # Test the function
    print("Generating anomaly dictionary with merged overlapping periods")
    # Generate anomaly dictionary with merged overlaps
    anomaly_dict = generate_anomaly_dict(merge_overlaps=True)
    logger.info("Successfully generated anomaly dictionary with merged overlapping periods")

  