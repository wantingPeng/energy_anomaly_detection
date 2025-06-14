"""
Visualization of soft label distributions in training and validation sets.

This script visualizes the distributions of soft labels in the training and validation datasets,
providing insights into the label balance and distribution characteristics.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import glob
from pathlib import Path
import pandas as pd
import seaborn as sns
from src.utils.logger import logger
from datetime import datetime

def load_soft_labels(data_dir, component='contact'):
    """
    Load all soft labels from the specified directory for a given component.
    
    Args:
        data_dir (str): Directory containing the npz files with soft labels
        component (str): Component type (default: 'contact')
        
    Returns:
        np.ndarray: Array of soft labels
    """
    component_dir = os.path.join(data_dir, component)
    if not os.path.exists(component_dir):
        logger.warning(f"Component directory {component_dir} does not exist.")
        return np.array([])
        
    # Find all batch files
    batch_files = glob.glob(os.path.join(component_dir, "batch_*.npz"))
    
    if not batch_files:
        logger.warning(f"No batch files found in {component_dir}")
        return np.array([])
        
    logger.info(f"Found {len(batch_files)} batch files in {component_dir}")
    
    # Load soft labels from all batch files
    all_soft_labels = []
    for batch_file in batch_files:
        try:
            data = np.load(batch_file)
            if 'soft_labels' in data:
                soft_labels = data['soft_labels']
                all_soft_labels.append(soft_labels)
                logger.info(f"Loaded {len(soft_labels)} soft labels from {os.path.basename(batch_file)}")
            else:
                logger.warning(f"'soft_labels' not found in {batch_file}")
        except Exception as e:
            logger.error(f"Error loading {batch_file}: {str(e)}")
    
    if not all_soft_labels:
        logger.warning("No soft labels loaded")
        return np.array([])
        
    # Concatenate all soft labels
    return np.concatenate(all_soft_labels)

def visualize_soft_label_distribution(train_labels, val_labels, output_dir=None):
    """
    Visualize the distribution of soft labels for training and validation sets.
    
    Args:
        train_labels (np.ndarray): Array of soft labels from the training set
        val_labels (np.ndarray): Array of soft labels from the validation set
        output_dir (str, optional): Directory to save the visualization plots
    """
    # Create a timestamp for the filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Set up the figure for distribution plots
    plt.figure(figsize=(14, 10))
    
    # Create a single plot with both distributions for comparison
    plt.subplot(2, 2, 1)
    sns.histplot(train_labels, kde=True, bins=50, color='blue', alpha=0.5, label='Train')
    sns.histplot(val_labels, kde=True, bins=50, color='red', alpha=0.5, label='Validation')
    plt.title('Soft Label Distribution Comparison')
    plt.xlabel('Soft Label Value')
    plt.ylabel('Count')
    plt.legend()
    
    # Create separate plots for better detail
    plt.subplot(2, 2, 2)
    sns.histplot(train_labels, kde=True, bins=50, color='blue')
    plt.title('Training Set Soft Label Distribution')
    plt.xlabel('Soft Label Value')
    plt.ylabel('Count')
    
    plt.subplot(2, 2, 3)
    sns.histplot(val_labels, kde=True, bins=50, color='red')
    plt.title('Validation Set Soft Label Distribution')
    plt.xlabel('Soft Label Value')
    plt.ylabel('Count')
    
    # Create boxplots for another view of the distribution
    plt.subplot(2, 2, 4)
    boxplot_data = [train_labels, val_labels]
    plt.boxplot(boxplot_data)
    plt.xticks([1, 2], ['Train', 'Validation'])
    plt.title('Soft Label Distribution Boxplot')
    plt.ylabel('Soft Label Value')
    
    plt.tight_layout()
    
    # Save the figure if output_dir is provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, f"soft_label_distribution_{timestamp}.png"), dpi=300)
        logger.info(f"Saved plot to {os.path.join(output_dir, f'soft_label_distribution_{timestamp}.png')}")
    
    plt.show()
    
    # Print statistics
    train_stats = {
        'mean': np.mean(train_labels),
        'std': np.std(train_labels),
        'min': np.min(train_labels),
        'max': np.max(train_labels),
        'median': np.median(train_labels),
        '25%': np.percentile(train_labels, 25),
        '75%': np.percentile(train_labels, 75),
        'count': len(train_labels)
    }
    
    val_stats = {
        'mean': np.mean(val_labels),
        'std': np.std(val_labels),
        'min': np.min(val_labels),
        'max': np.max(val_labels),
        'median': np.median(val_labels),
        '25%': np.percentile(val_labels, 25),
        '75%': np.percentile(val_labels, 75),
        'count': len(val_labels)
    }
    
    # Create a DataFrame for prettier display of statistics
    stats_df = pd.DataFrame({
        'Training': train_stats,
        'Validation': val_stats
    })
    
    logger.info("Soft Label Distribution Statistics:")
    logger.info("\n" + str(stats_df))
    
    # If output_dir is provided, save statistics to a CSV file
    if output_dir:
        stats_file = os.path.join(output_dir, f"soft_label_stats_{timestamp}.csv")
        stats_df.to_csv(stats_file)
        logger.info(f"Saved statistics to {stats_file}")
    
    return stats_df

def visualize_soft_labels(train_dir, val_dir, component='contact', output_dir=None):
    """
    Load and visualize soft labels from training and validation directories.
    
    Args:
        train_dir (str): Directory containing training data
        val_dir (str): Directory containing validation data
        component (str, optional): Component type to analyze (default: 'contact')
        output_dir (str, optional): Directory to save the visualization outputs
    """
    logger.info(f"Visualizing soft label distributions for component: {component}")
    
    # Load soft labels from training and validation sets
    logger.info(f"Loading training soft labels from {train_dir}")
    train_labels = load_soft_labels(train_dir, component)
    
    logger.info(f"Loading validation soft labels from {val_dir}")
    val_labels = load_soft_labels(val_dir, component)
    
    if len(train_labels) == 0:
        logger.error("No training labels loaded, visualization aborted")
        return
        
    if len(val_labels) == 0:
        logger.error("No validation labels loaded, visualization aborted")
        return
    
    logger.info(f"Loaded {len(train_labels)} training labels and {len(val_labels)} validation labels")
    
    # Create output directory if specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Visualize the distributions
    stats = visualize_soft_label_distribution(train_labels, val_labels, output_dir)
    
    return stats

def main():
    """
    Main function to visualize soft label distributions.
    """
    # Define paths to training and validation data
    train_dir = "Data/processed/soft_label/slidingWindow_600_600_200/train"
    val_dir = "Data/processed/soft_label/slidingWindow_600_600_200/val"
    
    # Set output directory for visualization
    output_dir = "experiments/logs/soft_label_visualization"
    
    # Components to visualize
    components = ['contact']  # Could be expanded to include 'pcb', 'ring' if needed
    
    # Visualize for each component
    for component in components:
        visualize_soft_labels(train_dir, val_dir, component, output_dir)

if __name__ == "__main__":
    main() 