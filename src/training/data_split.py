import pandas as pd
import numpy as np
from pathlib import Path
from src.utils.logger import logger

def split_data():
    """
    Split the merged features data into train, validation and test sets.
    The split is done by component type (contact, pcb, ring) to ensure each component's data is properly distributed.
    
    Returns:
        tuple: (train_df, val_df, test_df) - Three DataFrames containing the split data
    """
    # Create output directory if it doesn't exist
    output_dir = Path("Data/processed/splits_top_50_features")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load the merged features data
    logger.info("Loading merged features data...")
    df = pd.read_parquet("Data/processed/top_50_features.parquet")
    
    # Print first 10 rows
    logger.info("First 10 rows of the data:")
    logger.info("\n" + str(df.iloc[:10, -6:]))
    
    # Remove unnecessary columns
    logger.info("Removing unnecessary columns...")
    #df = df.drop(['window_start', 'window_end',"overlap_ratio","step_size"], axis=1)
    
    # Get feature columns (all columns except anomaly_label and component columns)
    component_cols = ['component_contact', 'component_pcb', 'component_ring']
    feature_cols = [col for col in df.columns if col != 'anomaly_label' and col not in component_cols]
    logger.info(f"Feature columns: {feature_cols}")
    
    # Initialize empty DataFrames for train, validation and test
    train_dfs = []
    val_dfs = []
    test_dfs = []
    
    # Split data by component type
    logger.info("Splitting data by component type...")
    for component_col in component_cols:
        # Get data for this component type
        component_data = df[df[component_col] == 1].copy()
        
        if len(component_data) > 0:
            logger.info(f"Processing {component_col} with {len(component_data)} samples")
            
            # Calculate split indices
            n_samples = len(component_data)
            train_end = int(n_samples * 0.7)
            val_end = train_end + int(n_samples * 0.15)
            
            # Split the data
            train_dfs.append(component_data.iloc[:train_end])
            val_dfs.append(component_data.iloc[train_end:val_end])
            test_dfs.append(component_data.iloc[val_end:])
    
    # Combine the splits
    train_df = pd.concat(train_dfs, axis=0)
    val_df = pd.concat(val_dfs, axis=0)
    test_df = pd.concat(test_dfs, axis=0)
    
    # Save the splits
    logger.info("Saving split datasets...")
    train_df.to_parquet(output_dir / "train.parquet")
    val_df.to_parquet(output_dir / "val.parquet")
    test_df.to_parquet(output_dir / "test.parquet")
    
    # Log the split sizes and component distribution
    logger.info(f"Train set size: {len(train_df)}")
    logger.info(f"Validation set size: {len(val_df)}")
    logger.info(f"Test set size: {len(test_df)}")
    
    # Log component distribution in each split
    for split_name, split_df in [("Train", train_df), ("Validation", val_df), ("Test", test_df)]:
        logger.info(f"\n{split_name} set component distribution:")
        for component_col in component_cols:
            count = split_df[component_col].sum()
            logger.info(f"{component_col}: {count} samples")
    
    return train_df, val_df, test_df

if __name__ == "__main__":
    train_df, val_df, test_df = split_data()
