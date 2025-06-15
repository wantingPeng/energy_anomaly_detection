import pandas as pd
from pathlib import Path
from src.utils.logger import logger

def load_and_sort_parquet_by_timestamp(file_path: str, timestamp_column: str = "TimeStamp"):
    """
    Load a parquet file and sort it by timestamp column
    
    Args:
        file_path: Path to the parquet file
        timestamp_column: Name of the timestamp column (default: "TimeStamp")
        
    Returns:
        Sorted DataFrame
    """
    logger.info(f"Loading parquet file from: {file_path}")
    
    try:
        # Load the parquet file
        df = pd.read_parquet(file_path)
        logger.info(f"Successfully loaded parquet file with {len(df)} rows")
        
        # Display the columns
        logger.info(f"Columns in the dataframe: {df.columns.tolist()}")
        
        # Sort by timestamp
        logger.info(f"Sorting by '{timestamp_column}' column")
        sorted_df = df.sort_values(by=timestamp_column)
        
        return sorted_df
        
    except Exception as e:
        logger.error(f"Error loading or sorting parquet file: {e}")
        raise

def load_and_sort_csv_by_timestamp(file_path: str, timestamp_column: str = "Date"):
    """
    Load a CSV file and sort it by timestamp column
    
    Args:
        file_path: Path to the CSV file
        timestamp_column: Name of the timestamp column (default: "TimeStamp")
        
    Returns:
        Sorted DataFrame
    """
    logger.info(f"Loading CSV file from: {file_path}")
    
    try:
        # Load the CSV file
        df = pd.read_csv(file_path, sep=';', encoding='utf-8')
        logger.info(f"Successfully loaded CSV file with {len(df)} rows")
        
        # Strip whitespace from column names
        df.columns = df.columns.str.strip()
            # Filter for Station containing "kontaktieren" (case-insensitive)
        filtered_data = df[df['Station'].str.contains('kontaktieren', case=False)]
        logger.info(f"Filtered data for Station containing 'kontaktieren': {len(filtered_data)} rows")
        df = filtered_data    
        # Display the columns
        logger.info(f"Columns in the dataframe: {df.columns.tolist()}")
        
        # Sort by timestamp
        logger.info(f"Sorting by '{timestamp_column}' column")
        sorted_df = df.sort_values(by=timestamp_column)
         # Create output directory if it doesn't exist
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        
        # Save as CSV (easily viewable in Excel or text editors)
        csv_path = out_path / "kontaktieren_anomalies_row.csv"
        sorted_df.to_csv(csv_path, index=False)
        logger.info(f"Saved CSV output to {csv_path}")
        
       
        return sorted_df
        
    except Exception as e:
        logger.error(f"Error loading or sorting CSV file: {e}")
        raise

def process_kontaktieren_data(file_path: str, output_dir: str, timestamp_column: str = "Date"):
    """
    Load the cleaned parquet file, filter for Station=Kontaktieren, sort by timestamp,
    and save the results to the output directory.
    
    Args:
        file_path: Path to the parquet file
        output_dir: Directory to save outputs
        timestamp_column: Name of the timestamp column (default: "StartTime")
        
    Returns:
        Filtered and sorted DataFrame
    """
    logger.info(f"Loading cleaned anomaly data from: {file_path}")
    
    try:
        # Load the parquet file
        df = pd.read_parquet(file_path)
        logger.info(f"Successfully loaded parquet file with {len(df)} rows")
        
        # Display the columns
        logger.info(f"Columns in the dataframe: {df.columns.tolist()}")
        
        # Display unique Station values for verification
        logger.info(f"Unique Station values: {df['Station'].unique().tolist()}")
        
        # Filter for Station="Kontaktieren"
        filtered_df = df[df['Station'].str.contains('Kontaktieren', case=False)]
        logger.info(f"Filtered data for Station=Kontaktieren: {len(filtered_df)} rows")
        
        # Sort by timestamp
        logger.info(f"Sorting by '{timestamp_column}' column")
        sorted_df = filtered_df.sort_values(by=timestamp_column)
        
        # Display the first 10 rows
        logger.info("First 10 rows of filtered and sorted Kontaktieren data:")
        logger.info("\n" + str(sorted_df.head(10)))
        
        # Create output directory if it doesn't exist
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        
        # Save as CSV (easily viewable in Excel or text editors)
        csv_path = out_path / "kontaktieren_anomalies.csv"
        sorted_df.to_csv(csv_path, index=False)
        logger.info(f"Saved CSV output to {csv_path}")
        
        # Save as HTML (nicely formatted for browser viewing)
        html_path = out_path / "kontaktieren_anomalies.html"
        sorted_df.to_html(html_path, index=False)
        logger.info(f"Saved HTML output to {html_path}")
        
        return sorted_df
        
    except Exception as e:
        logger.error(f"Error processing Kontaktieren data: {e}")
        raise

if __name__ == "__main__":
    # Path to the CSV file
    csv_file = "Data/row/Anomaly_Data/Duration_of_Anomalies.csv"
    output_dir = "Data/data_preview/soft_label/anomaly_dict"
    # Load and sort the data
    sorted_data = load_and_sort_csv_by_timestamp(csv_file)
    
    # Display unique Station values
    logger.info(f"Unique Station values: {sorted_data['Station'].unique().tolist()}")
    

    
    '''# Process cleaned parquet data for Kontaktieren
    cleaned_file = "Data/interim/Anomaly_Data/Duration_of_Anomalies_cleaned.parquet"
    output_dir = "Data/data_preview/soft_label/anomaly_dict"
    kontaktieren_data = process_kontaktieren_data(cleaned_file, output_dir)'''
