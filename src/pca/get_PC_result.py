#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Get PCA Results - Extract first 4 principal components from energy data.
Saves the results as a parquet file with timestamps and labels.
"""

import os
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from datetime import datetime
from src.utils.logger import logger

def create_output_dir():
    """Create output directory if not exists."""
    os.makedirs("Data/pc", exist_ok=True)
    logger.info("Output directory Data/pc checked/created")

def load_data():
    """Load the energy data from parquet file."""
    data_path = "Data/row_energyData_subsample_Transform/labeled/train/contact/part.0.parquet"
    
    logger.info(f"Loading data from: {data_path}")
    df = pd.read_parquet(data_path)
    logger.info(f"Data loaded with shape: {df.shape}")
    
    return df

def process_data(df):
    """Process data and extract feature columns."""
    # Get feature columns (exclude metadata columns)
    feature_columns = [col for col in df.columns if col not in ['TimeStamp', 'segment_id', 'anomaly_label']]
    logger.info(f"Number of feature columns: {len(feature_columns)}")
    
    # Prepare feature matrix X and metadata
    X = df[feature_columns].values
    timestamps = df['TimeStamp'].values
    labels = df['anomaly_label'].values if 'anomaly_label' in df.columns else None
    
    # Handle missing or infinite values if present
    if np.isnan(X).sum() > 0 or np.isinf(X).sum() > 0:
        logger.info("Handling missing and infinite values...")
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    return X, feature_columns, timestamps, labels

def perform_pca(X, n_components=20):
    """Perform PCA and extract the first n_components."""
    logger.info(f"Performing PCA to extract {n_components} components...")
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    
    # Log explained variance
    explained_variance = pca.explained_variance_ratio_
    cumulative_explained_variance = np.sum(explained_variance)
    
    logger.info(f"Explained variance ratios for first {n_components} principal components:")
    for i, ratio in enumerate(explained_variance):
        logger.info(f"PC{i+1}: {ratio:.4f} ({ratio*100:.2f}%)")
    logger.info(f"Cumulative explained variance: {cumulative_explained_variance:.4f} ({cumulative_explained_variance*100:.2f}%)")
    
    return X_pca, pca

def create_pc_dataframe(X_pca, timestamps, labels):
    """Create a DataFrame with PCs, timestamps, and labels."""
    # Create DataFrame with PCs
    pc_columns = [f'PC{i+1}' for i in range(X_pca.shape[1])]
    pc_df = pd.DataFrame(X_pca, columns=pc_columns)
    
    # Add metadata columns
    pc_df['TimeStamp'] = timestamps
    
    if labels is not None:
        pc_df['anomaly_label'] = labels
    

    return pc_df

def save_results(pc_df):
    """Save the PC results to parquet file."""
    output_path = f"Data/pc/pc_features_train_20.parquet"
    
    pc_df.to_parquet(output_path, index=False)
    logger.info(f"PC features saved to: {output_path}")
    
    return output_path

def main():
    """Main function to extract and save PCs."""
    logger.info("Starting PCA feature extraction process")
    
    # Create output directory
    create_output_dir()
    
    # Load data
    df = load_data()
    
    # Process data
    X, feature_columns, timestamps, labels = process_data(df)
    
    # Perform PCA
    X_pca, pca_model = perform_pca(X, n_components=20)
    
    # Create DataFrame with PCs and metadata
    pc_df = create_pc_dataframe(X_pca, timestamps, labels)
    
    # Preview the results
    logger.info(f"PC DataFrame shape: {pc_df.shape}")
    logger.info(f"PC DataFrame columns: {list(pc_df.columns)}")
    logger.info(f"First 5 rows of PC DataFrame:\n{pc_df.head()}")
    
    # Save results
    output_path = save_results(pc_df)
    
    logger.info("PCA feature extraction completed successfully")
    return output_path

if __name__ == "__main__":
    main()
