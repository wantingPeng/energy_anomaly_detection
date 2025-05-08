import pandas as pd
import os
from pathlib import Path
from datetime import datetime
from src.utils.logger import logger

def get_month_name(dt: datetime) -> str:
    """
    Convert datetime to German month name as used in the directory structure.
    
    Args:
        dt (datetime): Datetime object
        
    Returns:
        str: German month name
    """
    month_map = {
        1: 'Januar',
        2: 'Februar',
        3: 'März',
        4: 'April',
        5: 'Mai',
        6: 'Juni',
        7: 'Juli',
        8: 'August',
        9: 'September',
        10: 'Oktober',
        11: 'November',
        12: 'Dezember'
    }
    return month_map[dt.month]

def anormaly_preview(output_path: str = None) -> None:
    """
    Preview anomaly data by matching energy data with anomaly time periods.
    
    Args:
        output_path (str, optional): Path to save the output CSV file. 
            If None, saves to project root directory.
    """
    try:
        # Read anomaly data
        anomaly_path = "Data/interim/Anomaly_Data/Duration_of_Anomalies_cleaned.parquet"
        logger.info(f"Reading anomaly data from {anomaly_path}")
        anomaly_df = pd.read_parquet(anomaly_path)
        anomaly_df = anomaly_df.head(3)  # Get first 10 records

        # Initialize list to store results
        all_anomaly_data = []
        
        # Process each anomaly record
        for idx, row in anomaly_df.iterrows():
            station = row['Station'].lower()
            start_time = pd.to_datetime(row['StartTime'])
            end_time = pd.to_datetime(row['EndTime'])
            
            # Get month name and year for file matching
            month_name = get_month_name(start_time)
            year = start_time.year

            # Map station names to directory names
            station_map = {
                'pcb': 'PCB',
                'ringmontage': 'Ring',
                'kontaktieren': 'Contacting'
            }
            
            station_dir = station_map.get(station)
            if not station_dir:
                logger.warning(f"Unknown station type: {station}")
                continue

            energy_data_path = f"Data/interim/Energy_Data/{station_dir}"
            
            # Create the expected file pattern for the specific month and year
            file_pattern = f"{month_name}_{year}.parquet"
            target_file = Path(energy_data_path) / file_pattern
            
            logger.info(f"Looking for energy data in {target_file}")
            
            if not target_file.exists():
                logger.warning(f"Energy data file not found for {month_name} {year} in {station_dir}")
                continue
                
            try:
                energy_df = pd.read_parquet(target_file)
                
                # Convert timestamp to datetime if needed
                if not pd.api.types.is_datetime64_any_dtype(energy_df['TimeStamp']):
                    energy_df['TimeStamp'] = pd.to_datetime(energy_df['TimeStamp'])
                
                # Filter data within anomaly period
                mask = (energy_df['TimeStamp'] >= start_time) & (energy_df['TimeStamp'] <= end_time)
                anomaly_period_data = energy_df[mask].copy()
                
                if not anomaly_period_data.empty:
                    anomaly_period_data['anomaly'] = 1
                    anomaly_period_data['station'] = station
                    all_anomaly_data.append(anomaly_period_data)
                    logger.info(f"Found {len(anomaly_period_data)} anomaly records for {month_name} {year}")
                else:
                    logger.warning(f"No anomaly data found in the specified period for {month_name} {year}")
            
            except Exception as e:
                logger.error(f"Error processing file {target_file}: {str(e)}")
                continue

        # Combine all anomaly data
        if all_anomaly_data:
            final_df = pd.concat(all_anomaly_data, ignore_index=True)
            
            # Determine output path
            if output_path is None:
                output_path = "anormaly_preview.csv"
            
            # Save to CSV
            final_df.to_csv(output_path, index=False)
            logger.info(f"Anomaly preview data saved to {output_path}")
            logger.info(f"Total anomaly records: {len(final_df)}")
        else:
            logger.warning("No anomaly data found in the specified periods")

    except Exception as e:
        logger.error(f"Error in anormaly_preview: {str(e)}")
        raise

if __name__ == "__main__":
    anormaly_preview()
