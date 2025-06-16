import numpy as np
import matplotlib.pyplot as plt
from src.utils.logger import logger
import os
import torch
import pandas as pd

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
            if file_name.endswith('.npz'):
                file_path = os.path.join(directory_path, file_name)
                data = np.load(file_path)
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

def analyze_parquet_labels(file_path, label_column='anomaly_label'):
    """
    Analyze the distribution of normal and anomaly samples in a parquet file.
    
    Args:
        file_path (str): Path to the parquet file
        label_column (str): Name of the column containing anomaly labels (default: 'anomaly_label')
    
    Returns:
        dict: Dictionary containing distribution statistics
    """
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return None
    
    try:
        # Read the parquet file
        logger.info(f"Reading parquet file: {file_path}")
        df = pd.read_parquet(file_path)
        
        # Check if the label column exists
        if label_column not in df.columns:
            logger.error(f"Label column '{label_column}' not found in parquet file")
            return None
        
        # Count normal and anomaly samples
        value_counts = df[label_column].value_counts()
        total_samples = len(df)
        
        # Calculate percentages
        percentages = (value_counts / total_samples) * 100
        
        # Log the distribution
        logger.info(f"Total number of samples: {total_samples}")
        
        # Create results dictionary
        results = {
            'total_samples': total_samples,
            'counts': {},
            'percentages': {}
        }
        
        # Process each unique value (could be boolean True/False or 0/1)
        for label, count in value_counts.items():
            percentage = percentages[label]
            
            # Convert boolean labels to string for better readability
            label_str = str(label)
            label_type = "Anomaly" if label in [True, 1, "1", "True"] else "Normal"
            
            logger.info(f"{label_type} ({label_str}): {count} samples ({percentage:.2f}%)")
            
            results['counts'][label_str] = int(count)
            results['percentages'][label_str] = float(percentage)
        
        # Create visualization
        plt.figure(figsize=(10, 6))
        
        # Extract labels and counts for plotting
        labels_for_plot = [str(label) for label in value_counts.index]
        counts_for_plot = value_counts.values
        
        # Map labels to more readable form for plot
        label_mapping = {}
        for label in value_counts.index:
            if label in [True, 1, "1", "True"]:
                label_mapping[str(label)] = "Anomaly"
            else:
                label_mapping[str(label)] = "Normal"
        
        plot_labels = [label_mapping.get(str(label), str(label)) for label in value_counts.index]
        
        # Create the bar plot
        bars = plt.bar(plot_labels, counts_for_plot)
        plt.title('Distribution of Normal vs Anomaly Samples')
        plt.ylabel('Number of Samples')
        plt.grid(True, alpha=0.3)
        
        # Add value labels on top of each bar
        for i, (count, percentage) in enumerate(zip(counts_for_plot, percentages)):
            plt.text(i, count, f'{count}\n({percentage:.1f}%)', 
                    ha='center', va='bottom')
        
        plt.tight_layout()
        
        # Create directory for saving the figure
        file_name = os.path.basename(file_path).replace('.parquet', '')
        save_path = f'experiments/figures/normal_anomaly_distribution_{file_name}.png'
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        plt.savefig(save_path, dpi=300)
        logger.info(f"Distribution plot saved to {save_path}")
        plt.close()
        
        return results
        
    except Exception as e:
        logger.error(f"Error analyzing parquet file: {str(e)}")
        return None

def main():
    '''file_path = "Data/processed/transform/slidingWindow_noOverlap_0.7_800s/projection_pos_encoding_float16/val/contact/batch_0.pt"
    analyze_normal_anomaly_distribution(file_path)'''

    

    directory_path = "Data/processed/transform/slidingWindow_noOverlap_600_600_50_0.95_th0.3/val/contact"
    analyze_normal_anomaly_distribution_pt(directory_path)
    
    '''# Analyze parquet file with normal/anomaly labels
    parquet_path = "Data/processed/transform/slidingWindow_noOverlap_600_600_50_0.95_th0.3/train/contact"
    analyze_parquet_labels(parquet_path, label_column='anomaly_label')'''

if __name__ == "__main__":
    main()