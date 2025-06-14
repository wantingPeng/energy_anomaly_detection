import pandas as pd
import pickle
import json
from datetime import datetime
from typing import List, Tuple, Dict
from src.utils.logger import logger
from pathlib import Path

def check_time_overlaps(anomaly_dict_path: str = "Data/interim/Anomaly_Data/anomaly_dict_merged.pkl") -> Dict:
    """
    Check for time overlaps in anomaly data for the Kontaktieren station.
    
    Args:
        anomaly_dict_path (str): Path to the anomaly dictionary pickle file
        
    Returns:
        Dict: Dictionary containing overlap information
    """
    try:
        # Load the anomaly dictionary
        with open(anomaly_dict_path, 'rb') as f:
            anomaly_dict = pickle.load(f)
        
        # Get data for Kontaktieren station
        if 'Kontaktieren' not in anomaly_dict:
            logger.error("Kontaktieren station not found in anomaly dictionary")
            return {"error": "Station not found"}
        
        kontak_data = anomaly_dict['Kontaktieren']
        
        # Log the type and structure of the data for debugging
        logger.info(f"Data type: {type(kontak_data)}")
        if len(kontak_data) > 0:
            logger.info(f"First element type: {type(kontak_data[0])}")
            logger.info(f"First element: {kontak_data[0]}")
        
        # Convert time ranges to datetime objects and sort them
        time_ranges = []
        for idx, time_range in enumerate(kontak_data):
            # Handle different possible data structures
            if isinstance(time_range, tuple):
                start_time = pd.to_datetime(time_range[0])
                end_time = pd.to_datetime(time_range[1])
            elif isinstance(time_range, dict):
                start_time = pd.to_datetime(time_range['start_time'])
                end_time = pd.to_datetime(time_range['end_time'])
            elif isinstance(time_range, list):
                start_time = pd.to_datetime(time_range[0])
                end_time = pd.to_datetime(time_range[1])
            else:
                logger.error(f"Unexpected data type for time range: {type(time_range)}")
                continue
                
            time_ranges.append((start_time, end_time, idx))
        
        # Sort time ranges by start time
        time_ranges.sort(key=lambda x: x[0])
        
        # Check for overlaps
        overlaps = []
        for i in range(len(time_ranges) - 1):
            current_end = time_ranges[i][1]
            next_start = time_ranges[i + 1][0]
            
            if current_end >= next_start:
                overlap_info = {
                    "overlap_id": len(overlaps) + 1,
                    "first_range": {
                        "index": time_ranges[i][2],
                        "start": time_ranges[i][0],
                        "end": time_ranges[i][1]
                    },
                    "second_range": {
                        "index": time_ranges[i + 1][2],
                        "start": time_ranges[i + 1][0],
                        "end": time_ranges[i + 1][1]
                    },
                    "overlap_duration": current_end - next_start
                }
                overlaps.append(overlap_info)
                logger.warning(f"Found overlap between ranges {time_ranges[i][2]} and {time_ranges[i + 1][2]}")
        
        result = {
            "total_ranges": len(time_ranges),
            "total_overlaps": len(overlaps),
            "overlaps": overlaps
        }
        
        if len(overlaps) == 0:
            logger.info("No time overlaps found in Kontaktieren station data")
        else:
            logger.warning(f"Found {len(overlaps)} time overlaps in Kontaktieren station data")
        
        return result
        
    except Exception as e:
        logger.error(f"Error checking time overlaps: {str(e)}")
        return {"error": str(e)}

def convert_timestamps_to_str(obj):
    """
    Recursively convert all Timestamp and Timedelta objects in a dictionary to string format.
    
    Args:
        obj: Any object that might contain Timestamp or Timedelta objects
        
    Returns:
        Object with all Timestamps and Timedeltas converted to strings
    """
    if isinstance(obj, pd.Timestamp):
        return obj.strftime('%Y-%m-%d %H:%M:%S')
    elif isinstance(obj, pd.Timedelta):
        return str(obj)  # Convert Timedelta to string representation
    elif isinstance(obj, dict):
        return {key: convert_timestamps_to_str(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_timestamps_to_str(item) for item in obj]
    return obj

def save_overlap_results(result, output_dir="Data/data_preview/soft_label"):
    """
    Save overlap results to a JSON file with timestamp.
    
    Args:
        result (dict): Dictionary containing overlap results
        output_dir (str): Directory to save the results
    """
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"tree_overlap_results_{timestamp}.json"
    filepath = output_path / filename
    
    # Convert Timestamps to strings before saving
    serializable_result = convert_timestamps_to_str(result)
    
    # Save results to file
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(serializable_result, f, indent=4, ensure_ascii=False)
    
    return filepath


if __name__ == "__main__":
    # Example usage
  result = check_time_overlaps()
  if result['total_overlaps'] > 0:
      # Save results to file
      output_file = save_overlap_results(result)
      print(f"\nResults have been saved to: {output_file}")
      
      # Print summary to console
      print(f"\nTotal overlaps found: {result['total_overlaps']}")
      
      # Calculate and print total overlap duration
      total_duration = pd.Timedelta(0)
      for overlap in result['overlaps']:
          total_duration += pd.Timedelta(overlap['overlap_duration'])
      print(f"Total overlap duration: {total_duration}")
      
      print(f"Detailed results can be found in: {output_file}")
  else:
      print("\nNo overlaps found in the data.")

