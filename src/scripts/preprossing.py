import os
import sys
from pathlib import Path

# Add the project root directory to Python path
project_root = str(Path(__file__).resolve().parents[2])
if project_root not in sys.path:
    sys.path.append(project_root)

from src.utils.logger import logger
from src.preprocessing.data_loader import data_loader
from src.preprocessing.energy.cleaning import cleaning
import yaml

def load_config():
    """Load preprocessing configuration."""
    with open('configs/preprocessing.yaml', 'r') as file:
        return yaml.safe_load(file)

def main():
    """Main preprocessing pipeline."""
    try:
        logger.info("Starting preprocessing pipeline")
        
        # Load configuration
        config = load_config()
        
        # Load all energy data files
        input_pattern = config['data']['interim_dir']
        logger.info(f"Loading data from: {input_pattern}")
        
        df = data_loader(input_pattern)
        logger.info("Data loading completed")
        logger.info(f"Data first 5 lines: {df.head()}")
        logger.info(f"Data type: {type(df)}")

        # Clean the data
        logger.info("Starting data cleaning process")
        cleaned_df = cleaning(df)
        logger.info("Data cleaning completed")
        
        logger.info("Preprocessing pipeline completed successfully")
        
    except Exception as e:
        logger.error(f"Error in preprocessing pipeline: {str(e)}")
        raise

if __name__ == "__main__":
    main()  