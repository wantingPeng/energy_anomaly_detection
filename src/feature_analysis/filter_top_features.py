import os
import sys
import pandas as pd
from pathlib import Path
import re

# Add the project root to the path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from utils.logger import logger

def extract_feature_names_from_file(file_path):
    """Extract feature names from top_features.txt file."""
    logger.info(f"Extracting feature names from {file_path}")
    
    feature_names = []
    try:
        with open(file_path, 'r') as f:
            for line in f:
                # Use regex to extract the feature name before the colon
                match = re.match(r'.*?([a-zA-Z0-9_]+):', line)
                if match:
                    feature_name = match.group(1)
                    feature_names.append(feature_name)
    except Exception as e:
        logger.error(f"Error reading feature file: {e}")
        raise
    
    logger.info(f"Extracted {len(feature_names)} feature names: {feature_names}")
    return feature_names

def filter_dataset(input_file, output_file, feature_names):
    """Filter dataset to keep only specified features + anomaly_label and TimeStamp."""
    logger.info(f"Filtering dataset: {input_file}")
    
    # Add required columns
    required_columns = ["TimeStamp", "anomaly_label"] 
    columns_to_keep = list(set(required_columns + feature_names))
    
    # Load dataset
    try:
        df = pd.read_parquet(input_file)
        logger.info(f"Original dataset shape: {df.shape}")
        
        # Check if all columns exist
        missing_columns = [col for col in columns_to_keep if col not in df.columns]
        if missing_columns:
            logger.warning(f"These columns don't exist in the dataset: {missing_columns}")
            # Keep only columns that exist in the dataset
            columns_to_keep = [col for col in columns_to_keep if col in df.columns]
        
        # Filter the dataset
        filtered_df = df[columns_to_keep]
        logger.info(f"Filtered dataset shape: {filtered_df.shape}")
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Save the filtered dataset
        filtered_df.to_parquet(output_file)
        logger.info(f"Filtered dataset saved to {output_file}")
        
        return True
    except Exception as e:
        logger.error(f"Error filtering dataset: {e}")
        raise

def main():
    # Define paths
    features_file = "experiments/feature_analysis/contact_cleaned_1minut/top_features.txt"
    input_file = "Data/downsampleData_scratch_1minut/contact/contact_cleaned_1minut_20250928_172122.parquet"
    
    # Extract filename for the output
    input_filename = Path(input_file).name
    output_dir = "Data/filtered_feature"
    output_file = f"{output_dir}/top_features_{input_filename}"
    
    # Extract feature names
    feature_names = extract_feature_names_from_file(features_file)
    
    # Filter dataset
    success = filter_dataset(input_file, output_file, feature_names)
    
    if success:
        logger.info("Feature filtering completed successfully")
    else:
        logger.error("Feature filtering failed")

if __name__ == "__main__":
    main()
