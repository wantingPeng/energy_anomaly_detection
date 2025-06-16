"""
Analyze the distribution of soft labels in the sliding window dataset.

This script provides functionality to analyze the distribution of soft labels
across different value ranges, helping to understand class balance.
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import pandas as pd
from typing import Dict, List, Tuple
from tqdm import tqdm

from src.utils.logger import logger


def analyze_soft_label_distribution(
    data_dir: str,
    ranges: List[Tuple[float, float]] = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 1.0)],
    plot: bool = True,
    save_plot: bool = True,
    output_dir: str = "experiments/data_analysis"
) -> Dict[str, float]:
    """
    Analyze the distribution of soft labels in the sliding window dataset.
    
    Args:
        data_dir: Directory containing the NPZ files with soft labels
        ranges: List of tuples defining the ranges to analyze
        plot: Whether to create a visualization of the distribution
        save_plot: Whether to save the visualization to disk
        output_dir: Directory to save the plot (if save_plot is True)
        
    Returns:
        Dictionary with the percentage of labels in each range
    """
    logger.info(f"Analyzing soft label distribution in {data_dir}")
    
    # Get all NPZ files in the directory
    npz_files = glob.glob(os.path.join(data_dir, "*.npz"))
    
    if not npz_files:
        logger.warning(f"No NPZ files found in {data_dir}")
        return {}
    
    # Initialize counters for each range
    range_counts = {f"[{start},{end})": 0 for start, end in ranges[:-1]}
    # Handle the last range separately to include the upper bound
    start, end = ranges[-1]
    range_counts[f"[{start},{end}]"] = 0
    
    # Initialize total counter
    total_labels = 0
    
    # Process each NPZ file
    for npz_file in tqdm(npz_files, desc="Processing files"):
        try:
            # Load data
            data = np.load(npz_file)
            soft_labels = data['soft_labels']
            
            # Update total count
            total_labels += len(soft_labels)
            
            # Count labels in each range
            for i, (start, end) in enumerate(ranges):
                if i == len(ranges) - 1:  # Last range, include the upper bound
                    mask = (soft_labels >= start) & (soft_labels <= end)
                    range_key = f"[{start},{end}]"
                else:
                    mask = (soft_labels >= start) & (soft_labels < end)
                    range_key = f"[{start},{end})"
                
                range_counts[range_key] += np.sum(mask)
                
        except Exception as e:
            logger.error(f"Error processing {npz_file}: {str(e)}")
    
    # Calculate percentages
    percentages = {}
    for range_key, count in range_counts.items():
        percentage = (count / total_labels) * 100 if total_labels > 0 else 0
        percentages[range_key] = percentage
        logger.info(f"Range {range_key}: {count} samples ({percentage:.2f}%)")
    
    # Plot distribution if requested
    if plot and total_labels > 0:
        plt.figure(figsize=(10, 6))
        plt.bar(percentages.keys(), percentages.values(), color='skyblue')
        plt.title('Soft Label Distribution')
        plt.xlabel('Soft Label Range')
        plt.ylabel('Percentage (%)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save_plot:
            os.makedirs(output_dir, exist_ok=True)
            timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
            plot_path = os.path.join(output_dir, f"soft_label_distribution_{timestamp}.png")
            plt.savefig(plot_path)
            logger.info(f"Plot saved to {plot_path}")
        else:
            plt.show()
    
    return percentages


def main():
    """
    Main function to analyze soft label distribution from command line.
    """
    data_dir = "Data/processed/soft_label/slidingWindow_600_600_100/train/contact"
    
    logger.info(f"Analyzing soft label distribution in {data_dir}")
    
    # Define ranges
    ranges = [(0.0, 0.5), (0.5, 1.0)]
    
    # Analyze distribution
    distribution = analyze_soft_label_distribution(
        data_dir=data_dir,
        ranges=ranges,
        plot=True,
        save_plot=True
    )
    
    # Print summary
    logger.info("Soft Label Distribution Summary:")
    for range_key, percentage in distribution.items():
        logger.info(f"{range_key}: {percentage:.2f}%")


if __name__ == "__main__":
    main()