import pandas as pd
from pathlib import Path
import sys
import os

# Add the project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils.logger import logger

def analyze_overlap():
    # Read the parquet file
    input_path = Path("Data/interim/Anomaly_Data/Duration_of_Anomalies_cleaned.parquet")
    output_dir = Path("Data/data_preview/Duration_of_Anomalies_cleaned_soft_label")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "overlap_analysis.csv"
    logger.info(f"Reading parquet file from {input_path}")
    
    df = pd.read_parquet(input_path)
    
    # Filter for Station='Kontaktieren'
    kontak_df = df[df['Station'] == 'Kontaktieren'].copy()
    
    # Sort by StartTime
    kontak_df = kontak_df.sort_values('StartTime')
    
    # Calculate duration for each anomaly
    kontak_df['Duration'] = (kontak_df['EndTime'] - kontak_df['StartTime']).dt.total_seconds()
    
    # Find overlapping anomalies
    overlap_records = []
    for i in range(len(kontak_df) - 1):
        current_end = kontak_df.iloc[i]['EndTime']
        next_start = kontak_df.iloc[i + 1]['StartTime']
        
        if current_end >= next_start:
            overlap_duration = (current_end - next_start).total_seconds()
            overlap_records.append({
                'first_index': kontak_df.index[i],
                'first_start': kontak_df.iloc[i]['StartTime'],
                'first_end': kontak_df.iloc[i]['EndTime'],
                'second_index': kontak_df.index[i + 1],
                'second_start': kontak_df.iloc[i + 1]['StartTime'],
                'second_end': kontak_df.iloc[i + 1]['EndTime'],
                'overlap_duration': overlap_duration
            })
    
    # Save to CSV
    overlap_df = pd.DataFrame(overlap_records)
    overlap_df.to_csv(output_path, index=False)
    logger.info(f"Saved overlap analysis to {output_path}, total overlaps: {len(overlap_df)}")

if __name__ == "__main__":
    analyze_overlap() 