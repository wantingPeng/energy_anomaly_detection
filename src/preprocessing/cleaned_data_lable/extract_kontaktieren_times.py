#!/usr/bin/env python3
"""
Extract Kontaktieren station StartTime and EndTime from Duration_of_Anomalies data.

This script processes the cleaned anomaly data to extract time periods for the Kontaktieren station.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Tuple, Optional
from tqdm import tqdm

import sys
from pathlib import Path
# Add src directory to path
src_dir = Path(__file__).parent.parent.parent
sys.path.append(str(src_dir))

from utils.logger import logger


def extract_kontaktieren_times(
    input_file: str = "Data/machine/Anomaly_Data/Duration_of_Anomalies_cleaned.parquet",
    output_dir: str = "Data/cleaned_data_lable_t-sne",
    output_filename: str = "kontaktieren_times.parquet"
) -> Tuple[pd.DataFrame, str]:
    """
    Extract StartTime and EndTime for Station=Kontaktieren from anomaly data.
    
    Args:
        input_file (str): Path to input parquet file
        output_dir (str): Directory to save output
        output_filename (str): Name of output file
        
    Returns:
        Tuple[pd.DataFrame, str]: Processed dataframe and output file path
    """
    try:
        logger.info("Starting Kontaktieren station time extraction process")
        
        # Construct full paths
        input_path = Path(input_file)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        full_output_path = output_path / output_filename
        
        logger.info(f"Input file: {input_path}")
        logger.info(f"Output directory: {output_path}")
        logger.info(f"Output file: {full_output_path}")
        
        # Load the original data
        logger.info("Loading anomaly data from parquet file...")
        df = pd.read_parquet(input_path)
        logger.info(f"Loaded {len(df):,} total records")
        print(f"Original data shape: {df.shape}")
        
        # Display data overview
        logger.info("Data columns: " + ", ".join(df.columns.tolist()))
        logger.info("Station distribution:")
        station_counts = df['Station'].value_counts()
        for station, count in station_counts.items():
            logger.info(f"  - {station}: {count:,} records")
            print(f"  - {station}: {count:,} records")
        
        # Filter for Kontaktieren station
        logger.info("Filtering data for Station=Kontaktieren...")
        kontaktieren_df = df[df['Station'] == 'Kontaktieren'].copy()
        
        if len(kontaktieren_df) == 0:
            raise ValueError("No data found for Station=Kontaktieren")
            
        logger.info(f"Found {len(kontaktieren_df):,} records for Kontaktieren station")
        print(f"Kontaktieren data shape: {kontaktieren_df.shape}")
        
        # Extract only StartTime and EndTime columns
        logger.info("Extracting StartTime and EndTime columns...")
        result_df = kontaktieren_df[['StartTime', 'EndTime']].copy()
        
        # Add some useful metadata
        result_df['Station'] = 'Kontaktieren'
        result_df['Duration_minutes'] = (
            result_df['EndTime'] - result_df['StartTime']
        ).dt.total_seconds() / 60
        
        # Sort by StartTime
        result_df = result_df.sort_values('StartTime').reset_index(drop=True)
        
        # Display statistics
        logger.info("Time period statistics:")
        logger.info(f"  - Time range: {result_df['StartTime'].min()} to {result_df['EndTime'].max()}")
        logger.info(f"  - Average duration: {result_df['Duration_minutes'].mean():.2f} minutes")
        logger.info(f"  - Min duration: {result_df['Duration_minutes'].min():.2f} minutes")
        logger.info(f"  - Max duration: {result_df['Duration_minutes'].max():.2f} minutes")
        
        print("\nTime period statistics:")
        print(f"  - Time range: {result_df['StartTime'].min()} to {result_df['EndTime'].max()}")
        print(f"  - Average duration: {result_df['Duration_minutes'].mean():.2f} minutes")
        print(f"  - Min duration: {result_df['Duration_minutes'].min():.2f} minutes")
        print(f"  - Max duration: {result_df['Duration_minutes'].max():.2f} minutes")
        
        # Display sample data
        logger.info("Sample extracted data:")
        sample_df = result_df.head()
        for idx, row in sample_df.iterrows():
            logger.info(f"  Row {idx}: {row['StartTime']} -> {row['EndTime']} ({row['Duration_minutes']:.2f} min)")
        
        print("\nFirst 5 extracted records:")
        print(result_df.head())
        
        # Save to parquet
        logger.info(f"Saving extracted data to {full_output_path}...")
        with tqdm(total=1, desc="Saving parquet file") as pbar:
            result_df.to_parquet(full_output_path, index=False)
            pbar.update(1)
        
        # Verify saved file
        logger.info("Verifying saved file...")
        verification_df = pd.read_parquet(full_output_path)
        if len(verification_df) == len(result_df):
            logger.info(f"✅ Successfully saved {len(verification_df):,} records")
            print(f"✅ Successfully saved {len(verification_df):,} records to {full_output_path}")
        else:
            raise ValueError(f"Verification failed: expected {len(result_df)} records, got {len(verification_df)}")
        
        logger.info("Kontaktieren time extraction completed successfully")
        
        return result_df, str(full_output_path)
        
    except Exception as e:
        logger.error(f"Error in extract_kontaktieren_times: {str(e)}")
        print(f"❌ Error: {str(e)}")
        raise


def main():
    """Main function to run the extraction process."""
    try:
        # Run the extraction
        result_df, output_path = extract_kontaktieren_times()
        
        print(f"\n{'='*60}")
        print("EXTRACTION SUMMARY")
        print(f"{'='*60}")
        print(f"✅ Extracted {len(result_df):,} Kontaktieren station time records")
        print(f"📁 Saved to: {output_path}")
        print(f"📊 Data columns: {', '.join(result_df.columns.tolist())}")
        print(f"⏰ Time span: {result_df['StartTime'].min()} to {result_df['EndTime'].max()}")
        print(f"{'='*60}")
        
    except Exception as e:
        logger.error(f"Main execution failed: {str(e)}")
        print(f"❌ Execution failed: {str(e)}")
        return 1
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main()) 