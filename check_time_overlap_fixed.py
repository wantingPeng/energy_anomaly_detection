#!/usr/bin/env python
# -*- coding: utf-8 -*-

from utils.logger import logger
import pandas as pd
import os
import time
from collections import Counter

def check_time_overlap(file_path1, file_path2):
    """
    Check if two CSV files have overlapping timestamps
    
    Args:
        file_path1 (str): Path to the first CSV file
        file_path2 (str): Path to the second CSV file
        
    Returns:
        bool: True if there is overlap, False otherwise
    """
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_file = f"experiments/logs/check_time_overlap_{timestamp}.log"
    
    # Ensure log directory exists
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    logger.info(f"Checking time overlap between:")
    logger.info(f"File 1: {file_path1}")
    logger.info(f"File 2: {file_path2}")
    
    try:
        # Read only the timestamp column to save memory
        logger.info("Reading timestamps from first file...")
        df1 = pd.read_csv(file_path1, usecols=['TimeStamp'])
        logger.info(f"File 1 has {len(df1)} rows")
        
        logger.info("Reading timestamps from second file...")
        df2 = pd.read_csv(file_path2, usecols=['TimeStamp'])
        logger.info(f"File 2 has {len(df2)} rows")
        
        # Check for duplicate timestamps within each file
        logger.info("Checking for duplicate timestamps within each file...")
        df1_duplicates = df1['TimeStamp'].duplicated().sum()
        df2_duplicates = df2['TimeStamp'].duplicated().sum()
        
        logger.info(f"File 1 has {df1_duplicates} duplicate timestamps")
        logger.info(f"File 2 has {df2_duplicates} duplicate timestamps")
        
        # Convert timestamps to datetime
        logger.info("Converting timestamps to datetime...")
        df1['TimeStamp'] = pd.to_datetime(df1['TimeStamp'])
        df2['TimeStamp'] = pd.to_datetime(df2['TimeStamp'])
        
        # Sort by timestamp
        logger.info("Sorting timestamps...")
        df1 = df1.sort_values('TimeStamp')
        df2 = df2.sort_values('TimeStamp')
        
        # Get min and max timestamps for each file
        min_time1 = df1['TimeStamp'].min()
        max_time1 = df1['TimeStamp'].max()
        min_time2 = df2['TimeStamp'].min()
        max_time2 = df2['TimeStamp'].max()
        
        logger.info(f"File 1 time range: {min_time1} to {max_time1}")
        logger.info(f"File 2 time range: {min_time2} to {max_time2}")
        
        # Get unique timestamps in each file
        unique_timestamps1 = set(df1['TimeStamp'])
        unique_timestamps2 = set(df2['TimeStamp'])
        logger.info(f"File 1 has {len(unique_timestamps1)} unique timestamps")
        logger.info(f"File 2 has {len(unique_timestamps2)} unique timestamps")
        
        # Check for overlap
        has_overlap = (min_time1 <= max_time2) and (max_time1 >= min_time2)
        
        if has_overlap:
            logger.info("The two files HAVE overlapping timestamps")
            
            # Find the overlapping time range
            overlap_start = max(min_time1, min_time2)
            overlap_end = min(max_time1, max_time2)
            
            logger.info(f"Overlap period: {overlap_start} to {overlap_end}")
            
            # Count records in the overlapping time range
            overlap_df1 = df1[(df1['TimeStamp'] >= overlap_start) & (df1['TimeStamp'] <= overlap_end)]
            overlap_df2 = df2[(df2['TimeStamp'] >= overlap_start) & (df2['TimeStamp'] <= overlap_end)]
            
            overlap_count1 = len(overlap_df1)
            overlap_count2 = len(overlap_df2)
            
            logger.info(f"Number of records in overlapping time range in file 1: {overlap_count1}")
            logger.info(f"Number of records in overlapping time range in file 2: {overlap_count2}")
            
            # Count unique timestamps in the overlapping time range
            unique_overlap1 = set(overlap_df1['TimeStamp'])
            unique_overlap2 = set(overlap_df2['TimeStamp'])
            
            logger.info(f"Number of unique timestamps in overlapping time range in file 1: {len(unique_overlap1)}")
            logger.info(f"Number of unique timestamps in overlapping time range in file 2: {len(unique_overlap2)}")
            
            # Find common timestamps (exact matches)
            common_timestamps = unique_timestamps1.intersection(unique_timestamps2)
            exact_matches = len(common_timestamps)
            
            logger.info(f"Number of exact timestamp matches (unique timestamps): {exact_matches}")
            
            # Calculate percentage of overlap
            if len(unique_timestamps1) > 0 and len(unique_timestamps2) > 0:
                overlap_percent1 = (exact_matches / len(unique_timestamps1)) * 100
                overlap_percent2 = (exact_matches / len(unique_timestamps2)) * 100
                logger.info(f"Percentage of file 1 timestamps that overlap with file 2: {overlap_percent1:.2f}%")
                logger.info(f"Percentage of file 2 timestamps that overlap with file 1: {overlap_percent2:.2f}%")
            
            # Check if the files are completely overlapping
            complete_overlap = False
            if exact_matches == len(unique_timestamps1) == len(unique_timestamps2):
                complete_overlap = True
                logger.info("The two files have COMPLETELY OVERLAPPING timestamps (all timestamps match)")
            elif exact_matches == len(unique_timestamps1):
                logger.info("File 1 is completely contained within File 2")
            elif exact_matches == len(unique_timestamps2):
                logger.info("File 2 is completely contained within File 1")
            else:
                logger.info("The files have PARTIALLY OVERLAPPING timestamps")
                
            # Find the first few and last few matching timestamps
            common_timestamps_list = sorted(list(common_timestamps))
            if len(common_timestamps_list) > 0:
                logger.info("First 5 matching timestamps:")
                for ts in common_timestamps_list[:5]:
                    logger.info(f"  {ts}")
                    
                logger.info("Last 5 matching timestamps:")
                for ts in common_timestamps_list[-5:]:
                    logger.info(f"  {ts}")
            
        else:
            logger.info("The two files DO NOT have overlapping timestamps")
            
            # Calculate the time gap between the files
            if max_time1 < min_time2:
                gap = min_time2 - max_time1
                logger.info(f"Gap between file 1 end and file 2 start: {gap}")
            else:
                gap = min_time1 - max_time2
                logger.info(f"Gap between file 2 end and file 1 start: {gap}")
        
        return has_overlap
        
    except Exception as e:
        logger.error(f"Error checking time overlap: {str(e)}")
        return None

if __name__ == "__main__":
    file_path1 = "Data/row/Energy_Data/Contacting/Dezember_2024.csv"
    file_path2 = "Data/row/Energy_Data/Contacting/Januar_2025.csv"
    
    has_overlap = check_time_overlap(file_path1, file_path2)
    
    if has_overlap is not None:
        logger.info(f"Overlap check result: {'Overlap exists' if has_overlap else 'No overlap'}")