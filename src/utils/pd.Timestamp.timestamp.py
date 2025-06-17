import pandas as pd
import pytz
from src.utils.logger import logger

def compare_timestamps(time_str):
    """
    Compare integer timestamps from datetime64[ns, UTC] and datetime64[ns] formats
    for the same datetime string.
    
    Args:
        time_str (str): A datetime string in format 'YYYY-MM-DD HH:MM:SS'
    
    Returns:
        dict: Dictionary containing comparison result and both integer timestamps
    """
    # Create timestamp with UTC timezone
    utc_timestamp = pd.Timestamp(time_str, tz='UTC')
    
    # Create timestamp without timezone
    naive_timestamp = pd.Timestamp(time_str)
    
    # Convert both to integer timestamps
    utc_int_timestamp = int(utc_timestamp.timestamp())
    naive_int_timestamp = int(naive_timestamp.timestamp())
    
    # Check if they are equal
    are_equal = utc_int_timestamp == naive_int_timestamp
    
    result = {
        "are_equal": are_equal,
        "utc_timestamp_int": utc_int_timestamp,
        "naive_timestamp_int": naive_int_timestamp,
        "difference": utc_int_timestamp - naive_int_timestamp
    }
    
    logger.info(f"Comparing timestamps for {time_str}")
    logger.info(f"UTC timestamp ({utc_timestamp}): {utc_int_timestamp}")
    logger.info(f"Naive timestamp ({naive_timestamp}): {naive_int_timestamp}")
    logger.info(f"Are equal: {are_equal}, Difference: {result['difference']} seconds")
    
    return result

def validate_specific_timestamps():
    """
    Validate the specific timestamps mentioned in the requirements:
    - datetime64[ns, UTC] 2023-12-31 00:00:03+00:00
    - datetime64[ns] 2023-12-31 00:00:03
    """
    time_str = "2023-12-31 00:00:03"
    
    # Compare timestamps
    result = compare_timestamps(time_str)
    
    # Additional explanation of the result
    if result["are_equal"]:
        logger.info("The timestamps are EQUAL after conversion to integer timestamp.")
    else:
        logger.info(f"The timestamps are NOT EQUAL after conversion. UTC timestamp is {result['difference']} seconds different.")
        
        # If there's a difference, explain potential timezone effect
        local_tz = pd.Timestamp.now().tz
        logger.info(f"Note: The difference likely corresponds to the local timezone offset.")
        logger.info(f"Current system timezone appears to be {local_tz}")
        
        # Show an example with local timezone
        local_ts = pd.Timestamp(time_str, tz=local_tz)
        local_int_ts = int(local_ts.timestamp())
        logger.info(f"For reference, local timezone timestamp ({local_ts}): {local_int_ts}")
    
    return result

if __name__ == "__main__":
    validate_specific_timestamps()
