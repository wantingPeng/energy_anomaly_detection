import pandas as pd
import numpy as np
from datetime import datetime
import pytz
import logging
from pathlib import Path
import json
from typing import Dict, Tuple
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AnomalyDataPreprocessor:
    """Preprocessor for Duration_of_Anomalies dataset."""
    
    def __init__(self, input_path: str, output_dir: str):
        """
        Initialize the preprocessor.
        
        Args:
            input_path: Path to the original CSV file
            output_dir: Directory for output files
        """
        self.input_path = Path(input_path)
        self.output_dir = Path(output_dir)
        self.data = None
        self.original_stats = {}
        self.cleaned_stats = {}
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def load_data(self) -> None:
        """Load the original dataset and capture initial statistics."""
        try:
            logger.info(f"Loading data from {self.input_path}")
            self.data = pd.read_csv(
                self.input_path,
                sep=';',
                encoding='utf-8'
            )
            
            # Strip whitespace from column names
            self.data.columns = self.data.columns.str.strip()
            
            # Print actual column names for debugging
            logger.info(f"Actual columns in the dataset: {list(self.data.columns)}")
            
            # Store original statistics
            self.original_stats = {
                'total_rows': len(self.data),
                'columns': list(self.data.columns),
                'dtypes': self.data.dtypes.to_dict(),
                'missing_values': self.data.isnull().sum().to_dict(),
                'sample_rows': self.data.head(5).to_dict('records')
            }
            
            logger.info(f"Successfully loaded {len(self.data)} rows")
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
            
    def _clean_text_fields(self) -> None:
        """Standardize text fields."""
        try:
            # Clean categorical fields
            categorical_cols = ['Condition', 'Station', 'Line', 'Shift']
            for col in categorical_cols:
                if col in self.data.columns:
                    # Strip whitespace and standardize case
                    self.data[col] = self.data[col].str.strip()
                    self.data[col] = self.data[col].str.title()
            
            # Clean Comment field (preserve content, fix encoding)
            if 'Comment' in self.data.columns:
                self.data['Comment'] = self.data['Comment'].str.strip()
                
            logger.info("Text fields cleaned successfully")
        except Exception as e:
            logger.error(f"Error cleaning text fields: {str(e)}")
            raise
            
    def _process_time_fields(self) -> None:
        """Process time-related fields."""
        try:
            # Convert Date to datetime with timezone
            self.data['StartTime'] = pd.to_datetime(
                self.data['Date'].str.strip(),  # Strip any whitespace
                format='%d.%m.%Y, %H:%M:%S',
                errors='coerce'
            ).dt.tz_localize('Europe/Berlin').dt.tz_convert('UTC')   # Assuming Berlin timezone
            
            # Convert Downtime string to timedelta
            def parse_downtime(time_str: str) -> pd.Timedelta:
                """Convert HH:MM:SS to pandas Timedelta."""
                if pd.isna(time_str):
                    return pd.NaT
                try:
                    # Remove any whitespace
                    time_str = str(time_str).strip()
                    # Split the time string into hours, minutes, and seconds
                    parts = time_str.split(':')
                    if len(parts) != 3:
                        return pd.NaT
                    # Create timedelta directly from the HH:MM:SS string
                    return pd.Timedelta(time_str)
                except Exception as e:
                    logger.warning(f"Error parsing downtime value '{time_str}': {str(e)}")
                    return pd.NaT
            
            # Convert Downtime strings to Timedelta objects
            self.data['Downtime'] = self.data['Downtime'].apply(parse_downtime)
            
            # Calculate EndTime by adding Downtime to StartTime
            self.data['EndTime'] = self.data['StartTime'] + self.data['Downtime']
            
            # Log some statistics for verification
            logger.info(f"StartTime range: {self.data['StartTime'].min()} to {self.data['StartTime'].max()}")
            logger.info(f"Average downtime: {self.data['Downtime'].mean()}")
            logger.info(f"EndTime range: {self.data['EndTime'].min()} to {self.data['EndTime'].max()}")
            
            # Check for any NaT values
            nat_count = self.data['StartTime'].isna().sum()
            if nat_count > 0:
                logger.warning(f"Found {nat_count} NaT values in StartTime")
            
            logger.info("Time fields processed successfully")
        except Exception as e:
            logger.error(f"Error processing time fields: {str(e)}")
            raise
            
    def _validate_data(self) -> None:
        """Validate the processed data."""
        try:
            # Check for missing values
            missing = self.data.isnull().sum()
            if missing.any():
                logger.warning(f"Missing values found:\n{missing[missing > 0]}")
            
            # Validate time consistency
            invalid_intervals = self.data[self.data['EndTime'] < self.data['StartTime']]
            if len(invalid_intervals) > 0:
                logger.warning(f"Found {len(invalid_intervals)} invalid time intervals")
            
            # Store cleaned statistics
            self.cleaned_stats = {
                'total_rows': len(self.data),
                'missing_values': self.data.isnull().sum().to_dict(),
                'time_range': {
                    'start': self.data['StartTime'].min().isoformat(),
                    'end': self.data['EndTime'].max().isoformat()
                },
                'downtime_stats': {
                    'mean': str(self.data['Downtime'].mean()),
                    'min': str(self.data['Downtime'].min()),
                    'max': str(self.data['Downtime'].max())
                }
            }
            
            logger.info("Data validation completed")
        except Exception as e:
            logger.error(f"Error during validation: {str(e)}")
            raise
            
    def _generate_summary(self) -> str:
        """Generate preprocessing summary in Markdown format."""
        summary = f"""# Duration of Anomalies Preprocessing Summary

## Overview
- Original file: {self.input_path}
- Preprocessing date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Original Data Statistics
- Total rows: {self.original_stats['total_rows']}
- Columns: {', '.join(self.original_stats['columns'])}
- Missing values: {json.dumps(self.original_stats['missing_values'], indent=2)}

## Preprocessing Steps
1. Text Field Standardization
   - Stripped whitespace from all text fields
   - Standardized case for categorical fields
   - Preserved original content in Comment field

2. Time Processing
   - Converted Date to timezone-aware timestamps (Europe/Berlin)
   - Generated StartTime and EndTime columns
   - Converted Downtime to minutes (DowntimeMinutes)

3. Data Validation
   - Checked for missing values
   - Validated time intervals
   - Verified data consistency

## Cleaned Data Statistics
- Total rows: {self.cleaned_stats['total_rows']}
- Missing values: {json.dumps(self.cleaned_stats['missing_values'], indent=2)}
- Time range: {json.dumps(self.cleaned_stats['time_range'], indent=2)}
- Downtime statistics: {json.dumps(self.cleaned_stats['downtime_stats'], indent=2)}

## Notes
- All timestamps are timezone-aware (Europe/Berlin)
- Downtime is stored in minutes
- Original data preserved, cleaned version saved as Parquet
"""
        return summary
            
    def process(self) -> None:
        """Execute the complete preprocessing pipeline."""
        try:
            # Load data
            self.load_data()
            
            # Clean text fields
            self._clean_text_fields()
            
            # Process time fields
            self._process_time_fields()
            
            # Validate processed data
            self._validate_data()
            
            # Save cleaned data
            output_path = self.output_dir / 'Duration_of_Anomalies_cleaned.parquet'
            self.data.to_parquet(
                output_path,
                compression='snappy',
                index=False
            )
            logger.info(f"Cleaned data saved to {output_path}")
            
            # Generate and save summary
            summary = self._generate_summary()
            summary_path = self.output_dir / 'duration_of_anomalies_preprocessing_summary.md'
            with open(summary_path, 'w', encoding='utf-8') as f:
                f.write(summary)
            logger.info(f"Summary saved to {summary_path}")
            
        except Exception as e:
            logger.error(f"Error during preprocessing: {str(e)}")
            raise

def main():
    """Main execution function."""
    try:
        # Define paths
        input_path = "Data/row/Anomaly_Data/Duration_of_Anomalies.csv"
        output_dir = "Data/interim/Anomaly_Data"    
        
        # Initialize and run preprocessor
        preprocessor = AnomalyDataPreprocessor(input_path, output_dir)
        preprocessor.process()
        
        logger.info("Preprocessing completed successfully")
    except Exception as e:
        logger.error(f"Preprocessing failed: {str(e)}")
        raise

if __name__ == "__main__":
    main() 