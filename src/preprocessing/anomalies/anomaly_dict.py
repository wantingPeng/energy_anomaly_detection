import os
import pickle
from typing import Dict, List, Tuple
import dask.dataframe as dd
from src.utils.logger import logger
from pathlib import Path


def generate_anomaly_dict() -> Dict[str, List[Tuple[str, str]]]:
    """
    Generate a dictionary of anomaly time periods organized by station.
    
    Returns:
        Dict[str, List[Tuple[str, str]]]: Dictionary where keys are station names and values are
        lists of (start_time, end_time) tuples sorted by start_time.
    """
    # Define input and output paths
    input_path = "Data/interim/Anomaly_Data/Duration_of_Anomalies_cleaned.parquet"
    output_path = "Data/interim/Anomaly_Data/anomaly_dict.pkl"
    
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
    try:
        anomaly_dict = generate_anomaly_dict()
        logger.info("Successfully generated anomaly dictionary")
    except Exception as e:
        logger.error(f"Failed to generate anomaly dictionary: {str(e)}") 