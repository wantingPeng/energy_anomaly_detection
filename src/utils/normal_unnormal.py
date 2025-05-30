import numpy as np
import matplotlib.pyplot as plt
from src.utils.logger import logger
import os
import torch

def analyze_normal_anomaly_distribution(file_path):
    """
    Analyze the distribution of normal and anomaly samples in the dataset.
    
    Args:
        file_path (str): Path to the .npz file
    """
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return

    try:
        data = np.load(file_path)
        labels = data['labels']
        
        # Count normal and anomaly samples
        unique_labels, counts = np.unique(labels, return_counts=True)
        total_samples = len(labels)
        
        # Calculate percentages
        percentages = (counts / total_samples) * 100
        
        # Log the distribution
        logger.info(f"Total number of samples: {total_samples}")
        for label, count, percentage in zip(unique_labels, counts, percentages):
            label_type = "Normal" if label == 0 else "Anomaly"
            logger.info(f"{label_type}: {count} samples ({percentage:.2f}%)")
        
        # Create visualization
        plt.figure(figsize=(10, 6))
        plt.bar(['Normal', 'Anomaly'], counts)
        plt.title('Distribution of Normal vs Anomaly Samples')
        plt.ylabel('Number of Samples')
        plt.grid(True, alpha=0.3)
        
        # Add value labels on top of each bar
        for i, count in enumerate(counts):
            plt.text(i, count, f'{count}\n({percentages[i]:.1f}%)', 
                    ha='center', va='bottom')
        
        plt.tight_layout()
        save_path = 'experiments/figures/normal_anomaly_distribution.png'
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        logger.info(f"Distribution plot saved to {save_path}")
        plt.close()

    except Exception as e:
        logger.error(f"Error analyzing file: {str(e)}")

def analyze_normal_anomaly_distribution_pt(directory_path):
    """
    Analyze the distribution of normal and anomaly samples in all .pt files within a directory.
    
    Args:
        directory_path (str): Path to the directory containing .pt files
    """
    if not os.path.exists(directory_path):
        logger.error(f"Directory not found: {directory_path}")
        return

    try:
        all_labels = []
        for file_name in os.listdir(directory_path):
            if file_name.endswith('.pt'):
                file_path = os.path.join(directory_path, file_name)
                data = torch.load(file_path)
                labels = data['labels']  # Assuming labels are stored under the key 'labels'
                all_labels.extend(labels)

        # Convert to numpy array for processing
        all_labels = np.array(all_labels)

        # Count normal and anomaly samples
        unique_labels, counts = np.unique(all_labels, return_counts=True)
        total_samples = len(all_labels)

        # Calculate percentages
        percentages = (counts / total_samples) * 100

        # Log the distribution
        logger.info(f"Total number of samples: {total_samples}")
        for label, count, percentage in zip(unique_labels, counts, percentages):
            label_type = "Normal" if label == 0 else "Anomaly"
            logger.info(f"{label_type}: {count} samples ({percentage:.2f}%)")

        # Create visualization
        plt.figure(figsize=(10, 6))
        plt.bar(['Normal', 'Anomaly'], counts)
        plt.title('Distribution of Normal vs Anomaly Samples')
        plt.ylabel('Number of Samples')
        plt.grid(True, alpha=0.3)

        # Add value labels on top of each bar
        for i, count in enumerate(counts):
            plt.text(i, count, f'{count}\n({percentages[i]:.1f}%)', 
                    ha='center', va='bottom')

        plt.tight_layout()
        save_path = 'experiments/figures/normal_anomaly_distribution_pt.png'
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        logger.info(f"Distribution plot saved to {save_path}")
        plt.close()

    except Exception as e:
        logger.error(f"Error analyzing directory: {str(e)}")

def main():
    # file_path = "Data/processed/lsmt/sliding_window/val/contact/batch_0.npz"
    # analyze_normal_anomaly_distribution(file_path)

    directory_path = "Data/processed/lsmt/dataset_1200s/train/contact"
    analyze_normal_anomaly_distribution_pt(directory_path)

if __name__ == "__main__":
    main()