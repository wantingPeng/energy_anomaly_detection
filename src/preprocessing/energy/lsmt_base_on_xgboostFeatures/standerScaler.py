import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import joblib
from src.utils.logger import logger

def standardize_features():
    """
    Standardize features from the parquet file, excluding specific columns.
    
    This function:
    1. Loads data from the specified parquet file
    2. Identifies columns to standardize (excluding specified ones)
    3. Applies StandardScaler to these columns
    4. Saves the standardized data to the output directory
    5. Saves the scaler model for future use
    """
    # Configure logger
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"experiments/logs/standardize_features_{timestamp}.log"
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # Input and output paths
    input_path = "Data/interim/Energy_labeling_windowFeatures/Kontaktieren_labeled.parquet"
    output_dir = "Data/processed/lsmt_base_on_xgboostFeatures/standerlizes/contact"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Columns to exclude from standardization
    excluded_columns = ['window_start', 'window_end', 'segment_id', 'anomaly_label', 'overlap_ratio', 'step_size']
    
    logger.info(f"Loading data from {input_path}")
    try:
        # Load data
        df = pd.read_parquet(input_path)
        logger.info(f"Loaded data with shape: {df.shape}")
        
        # Identify columns to standardize
        feature_columns = [col for col in df.columns if col not in excluded_columns]
        logger.info(f"Found {len(feature_columns)} feature columns to standardize")
        
        # Initialize and fit the scaler
        scaler = StandardScaler()
        df_features = df[feature_columns]
        
        # Fit and transform the features
        logger.info("Applying StandardScaler to features")
        standardized_features = scaler.fit_transform(df_features)
        
        # Create a new DataFrame with standardized features
        standardized_df = pd.DataFrame(standardized_features, columns=feature_columns)
        
        # Add back the excluded columns
        for col in excluded_columns:
            standardized_df[col] = df[col].values
        
        # Save the standardized data
        output_path = os.path.join(output_dir, "standardized_data.parquet")
        logger.info(f"Saving standardized data to {output_path}")
        standardized_df.to_parquet(output_path)
        
        # Save the scaler for future use
        scaler_path = os.path.join(output_dir, "standard_scaler.joblib")
        logger.info(f"Saving scaler model to {scaler_path}")
        joblib.dump(scaler, scaler_path)
        
        logger.info("Standardization completed successfully")
        return standardized_df
        
    except Exception as e:
        logger.error(f"Error during standardization: {str(e)}")
        raise
        
if __name__ == "__main__":
    standardize_features()
