import pandas as pd
from pathlib import Path
from src.utils.logger import logger

def preview_kontaktieren_data():
    # Define input and output paths
    input_path = Path("Data/interim/Anomaly_Data/Duration_of_Anomalies_cleaned.parquet")
    output_dir = Path("Data/data_preview/soft_label")
    output_path = output_dir / "kontaktieren_preview.csv"
    
    try:
        # Read the parquet file
        logger.info(f"Reading parquet file from {input_path}")
        df = pd.read_parquet(input_path)
        
        # Filter for Station='Kontaktieren' and sort by StartTime
        logger.info("Filtering for Station='Kontaktieren' and sorting by StartTime")
        filtered_df = df[df['Station'] == 'Kontaktieren'].sort_values('StartTime')
        
        # Take first 1000 rows
        preview_df = filtered_df.head(1000)
        
        # Save to CSV
        logger.info(f"Saving preview to {output_path}")
        preview_df.to_csv(output_path, index=False)
        
        logger.info(f"Successfully saved {len(preview_df)} rows to {output_path}")
        
    except Exception as e:
        logger.error(f"Error processing data: {str(e)}")
        raise

if __name__ == "__main__":
    preview_kontaktieren_data() 