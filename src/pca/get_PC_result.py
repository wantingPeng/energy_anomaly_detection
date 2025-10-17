#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Get PCA Results - Extract first 4 principal components from energy data.
Saves the results as a parquet file with timestamps and labels.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from datetime import datetime
from src.utils.logger import logger

def load_data():
    """Load the energy data from parquet file."""
    data_path = "Data/downsampleData_scratch_1minut/contact/contact_cleaned_1minut_20250928_172122.parquet"
    
    logger.info(f"Loading data from: {data_path}")
    df = pd.read_parquet(data_path)
    logger.info(f"Data loaded with shape: {df.shape}")
    
    return df

def process_data(df):
    """Process data and extract feature columns."""
    # Get feature columns (exclude metadata columns)
    feature_columns = [col for col in df.columns if col not in ['TimeStamp', 'anomaly_label']]
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
    # Standardize the features before PCA
    logger.info("Standardizing features before PCA...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Log scaling information
    logger.info(f"Feature scaling completed:")
    logger.info(f"  Original feature means range: [{X.mean(axis=0).min():.3f}, {X.mean(axis=0).max():.3f}]")
    logger.info(f"  Original feature stds range: [{X.std(axis=0).min():.3f}, {X.std(axis=0).max():.3f}]")
    logger.info(f"  Scaled feature means range: [{X_scaled.mean(axis=0).min():.3f}, {X_scaled.mean(axis=0).max():.3f}]")
    logger.info(f"  Scaled feature stds range: [{X_scaled.std(axis=0).min():.3f}, {X_scaled.std(axis=0).max():.3f}]")
    
    logger.info(f"Performing PCA to extract {n_components} components...")
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    
    # Log explained variance
    explained_variance = pca.explained_variance_ratio_
    cumulative_explained_variance = np.sum(explained_variance)
    
    logger.info(f"Explained variance ratios for first {n_components} principal components:")
    for i, ratio in enumerate(explained_variance):
        logger.info(f"PC{i+1}: {ratio:.4f} ({ratio*100:.2f}%)")
    logger.info(f"Cumulative explained variance: {cumulative_explained_variance:.4f} ({cumulative_explained_variance*100:.2f}%)")
    
    return X_pca, pca, scaler

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

def save_results(pc_df, output_dir="Data/pca_analysis_and_result"):
    """Save the PC results to parquet file.
    
    Parameters:
    -----------
    pc_df : pd.DataFrame
        DataFrame containing PC features
    output_dir : str
        Output directory path
        
    Returns:
    --------
    str
        Path to the saved parquet file
    """
    output_path = os.path.join(output_dir, "pca_features_ring.parquet")
    
    pc_df.to_parquet(output_path, index=False)
    logger.info(f"PC features saved to: {output_path}")
    
    return output_path

def plot_pc1_pc2_scatter(pc_df, output_dir="Data/pca_analysis_and_result", max_points=50000):
    """
    Plot PC1 vs PC2 scatter plot with color coding for anomaly labels.
    
    Parameters:
    -----------
    pc_df : pd.DataFrame
        DataFrame containing PC1, PC2, and anomaly_label columns
    output_dir : str
        Directory to save the plot
    max_points : int
        Maximum number of points to plot (default: 50000)
    """
    logger.info(f"Creating PC1 vs PC2 scatter plot (using first {max_points} points)...")
    
    # Sample data if necessary
    if len(pc_df) > max_points:
        plot_df = pc_df.iloc[:max_points].copy()
        logger.info(f"Using first {max_points} data points for plotting")
    else:
        plot_df = pc_df.copy()
        logger.info(f"Using all {len(plot_df)} data points for plotting")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Check if anomaly_label exists
    if 'anomaly_label' in plot_df.columns:
        # Separate normal and anomaly points
        normal_mask = plot_df['anomaly_label'] == 0
        anomaly_mask = plot_df['anomaly_label'] == 1
        
        # Plot normal points
        ax.scatter(plot_df.loc[normal_mask, 'PC1'], 
                  plot_df.loc[normal_mask, 'PC2'],
                  c='blue', alpha=0.5, s=10, label='Normal', edgecolors='none')
        
        # Plot anomaly points
        ax.scatter(plot_df.loc[anomaly_mask, 'PC1'], 
                  plot_df.loc[anomaly_mask, 'PC2'],
                  c='red', alpha=0.7, s=10, label='Anomaly', edgecolors='none')
        
        ax.legend(loc='best')
    else:
        # Plot all points without color coding
        ax.scatter(plot_df['PC1'], plot_df['PC2'], 
                  c='blue', alpha=0.5, s=10, edgecolors='none')
    
    ax.set_xlabel('PC1 (First Principal Component)', fontsize=12)
    ax.set_ylabel('PC2 (Second Principal Component)', fontsize=12)
    ax.set_title('PCA: PC1 vs PC2 Scatter Plot', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Save plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"pca_pc1_pc2_scatter_{timestamp}.png")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"PC1 vs PC2 scatter plot saved to: {output_path}")
    return output_path

def plot_loading_vectors(pca_model, feature_columns, output_dir="Data/pca_analysis_and_result"):
    """
    Plot loading vectors showing the contribution of each feature to PC1 and PC2.
    
    Parameters:
    -----------
    pca_model : sklearn.decomposition.PCA
        Fitted PCA model
    feature_columns : list
        List of feature column names
    output_dir : str
        Directory to save the plot
    """
    logger.info("Creating loading plot (PC1 vs PC2)...")
    
    # Get loadings for PC1 and PC2
    loadings = pca_model.components_[:2, :].T  # Shape: (n_features, 2)
    pc1_loadings = loadings[:, 0]
    pc2_loadings = loadings[:, 1]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Plot loading vectors as arrows
    for i, feature in enumerate(feature_columns):
        ax.arrow(0, 0, pc1_loadings[i], pc2_loadings[i], 
                head_width=0.01, head_length=0.01, fc='blue', ec='blue', alpha=0.6)
        
        # Add feature labels
        ax.text(pc1_loadings[i] * 1.1, pc2_loadings[i] * 1.1, feature, 
               fontsize=8, ha='center', va='center')
    
    ax.set_xlabel('PC1 Loadings', fontsize=12)
    ax.set_ylabel('PC2 Loadings', fontsize=12)
    ax.set_title('PCA Loading Plot: Feature Contributions to PC1 and PC2', 
                fontsize=14, fontweight='bold')
    ax.axhline(y=0, color='k', linestyle='--', linewidth=0.5)
    ax.axvline(x=0, color='k', linestyle='--', linewidth=0.5)
    ax.grid(True, alpha=0.3)
    
    # Set equal aspect ratio
    ax.set_aspect('equal', adjustable='box')
    
    # Save plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"pca_loading_plot_{timestamp}.png")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Loading plot saved to: {output_path}")
    logger.info(f"Top 5 features contributing to PC1: {np.array(feature_columns)[np.argsort(np.abs(pc1_loadings))[-5:][::-1]].tolist()}")
    logger.info(f"Top 5 features contributing to PC2: {np.array(feature_columns)[np.argsort(np.abs(pc2_loadings))[-5:][::-1]].tolist()}")
    
    return output_path

def plot_scree_plot(pca_model, output_dir="Data/pca_analysis_and_result"):
    """
    Plot scree plot showing explained variance ratio for each principal component.
    
    Parameters:
    -----------
    pca_model : sklearn.decomposition.PCA
        Fitted PCA model
    output_dir : str
        Directory to save the plot
    """
    logger.info("Creating scree plot...")
    
    # Get explained variance ratios
    explained_variance_ratio = pca_model.explained_variance_ratio_
    n_components = len(explained_variance_ratio)
    cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Bar plot of explained variance ratio
    pc_labels = [f'PC{i+1}' for i in range(n_components)]
    ax1.bar(range(1, n_components + 1), explained_variance_ratio, 
           color='steelblue', alpha=0.8, edgecolor='black')
    ax1.set_xlabel('Principal Component', fontsize=12)
    ax1.set_ylabel('Explained Variance Ratio', fontsize=12)
    ax1.set_title('Scree Plot: Explained Variance by Component', 
                 fontsize=14, fontweight='bold')
    ax1.set_xticks(range(1, n_components + 1))
    ax1.set_xticklabels(pc_labels, rotation=45 if n_components > 10 else 0)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add percentage labels on bars
    for i, (x, y) in enumerate(zip(range(1, n_components + 1), explained_variance_ratio)):
        ax1.text(x, y, f'{y*100:.1f}%', ha='center', va='bottom', fontsize=9)
    
    # Plot 2: Cumulative explained variance
    ax2.plot(range(1, n_components + 1), cumulative_variance_ratio, 
            marker='o', linestyle='-', color='steelblue', linewidth=2, markersize=8)
    ax2.fill_between(range(1, n_components + 1), cumulative_variance_ratio, 
                     alpha=0.3, color='steelblue')
    ax2.axhline(y=0.95, color='r', linestyle='--', linewidth=1, label='95% Variance')
    ax2.axhline(y=0.90, color='orange', linestyle='--', linewidth=1, label='90% Variance')
    ax2.set_xlabel('Principal Component', fontsize=12)
    ax2.set_ylabel('Cumulative Explained Variance Ratio', fontsize=12)
    ax2.set_title('Cumulative Explained Variance', fontsize=14, fontweight='bold')
    ax2.set_xticks(range(1, n_components + 1))
    ax2.set_xticklabels(pc_labels, rotation=45 if n_components > 10 else 0)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='lower right')
    ax2.set_ylim([0, 1.05])
    
    # Add percentage labels on line
    for i, (x, y) in enumerate(zip(range(1, n_components + 1), cumulative_variance_ratio)):
        if i % max(1, n_components // 10) == 0 or i == n_components - 1:  # Show every nth label
            ax2.text(x, y + 0.02, f'{y*100:.1f}%', ha='center', va='bottom', fontsize=9)
    
    # Save plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"pca_scree_plot_{timestamp}.png")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Scree plot saved to: {output_path}")
    logger.info(f"Cumulative variance explained by all {n_components} components: {cumulative_variance_ratio[-1]*100:.2f}%")
    
    return output_path

def main(output_dir="Data/pca_analysis_and_result"):
    """Main function to extract and save PCs.
    
    Parameters:
    -----------
    output_dir : str
        Output directory path for all results and plots
    """
    logger.info("Starting PCA feature extraction process")
    output_dir = "Data/pca_analysis_and_result/ring"
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Output directory {output_dir} created/verified")
    # Create output directory and get the path
    
    logger.info(f"All outputs will be saved to: {output_dir}")
    
    # Load data
    df = load_data()
    
    # Process data
    X, feature_columns, timestamps, labels = process_data(df)
    
    # Perform PCA
    X_pca, pca_model, scaler = perform_pca(X, n_components=4)
    
    # Create DataFrame with PCs and metadata
    pc_df = create_pc_dataframe(X_pca, timestamps, labels)
    
    # Preview the results
    logger.info(f"PC DataFrame shape: {pc_df.shape}")
    logger.info(f"PC DataFrame columns: {list(pc_df.columns)}")
    logger.info(f"First 5 rows of PC DataFrame:\n{pc_df.head()}")
    
    # Save results
    output_path = save_results(pc_df, output_dir)
    
    # Generate visualization plots
    logger.info("=" * 60)
    logger.info("Generating PCA visualization plots...")
    logger.info("=" * 60)
    
    # Plot 1: PC1 vs PC2 scatter plot
    scatter_path = plot_pc1_pc2_scatter(pc_df, output_dir, max_points=50000)
    
    # Plot 2: Loading vectors plot
    loading_path = plot_loading_vectors(pca_model, feature_columns, output_dir)
    
    # Plot 3: Scree plot
    scree_path = plot_scree_plot(pca_model, output_dir)
    
    logger.info("=" * 60)
    logger.info("All visualization plots generated successfully!")
    logger.info(f"  - PC1 vs PC2 scatter plot: {scatter_path}")
    logger.info(f"  - Loading vectors plot: {loading_path}")
    logger.info(f"  - Scree plot: {scree_path}")
    logger.info("=" * 60)
    
    logger.info("PCA feature extraction completed successfully")
    return output_path

if __name__ == "__main__":
    main()
