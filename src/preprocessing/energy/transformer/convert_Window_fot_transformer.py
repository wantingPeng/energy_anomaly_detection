"""
Convert sliding window data to Transformer-ready format.

This script processes sliding window data by:
1. Loading data from .npz files
2. Applying linear projection to each time step
3. Adding positional encoding
4. Creating PyTorch datasets for Transformer training
5. Saving processed data to disk
"""

import os
import gc
import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, Dataset
import math
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import glob
from joblib import Parallel, delayed
import shutil

from src.utils.logger import logger
from src.utils.memory_left import log_memory


class PositionalEncoding(nn.Module):
    """
    Positional encoding for Transformer models.
    
    Implementation of the sinusoidal position encoding described in 
    "Attention Is All You Need" paper.
    """
    
    def __init__(self, d_model, max_seq_length=600):
        """
        Initialize positional encoding.
        
        Args:
            d_model: Dimensionality of the model
            max_seq_length: Maximum sequence length
            dropout: Dropout rate
        """
        super(PositionalEncoding, self).__init__()
        
        # Create positional encoding matrix
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # Apply sine to even indices and cosine to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add batch dimension
        pe = pe.unsqueeze(0)
        
        # Register as buffer (not a parameter)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """
        Add positional encoding to input tensor.
        
        Args:
            x: Input tensor of shape [batch_size, seq_length, d_model]
            
        Returns:
            Tensor with positional encoding added
        """
        x = x + self.pe[:, :x.size(1), :]
        return x

def process_batch_file(batch_path, output_dir, component, data_type, linear_projection, positional_encoding):
    logger.info(f"Processing {batch_path}")
    log_memory(f"Before loading {os.path.basename(batch_path)}")
    
    # Load .npz
    data = np.load(batch_path, allow_pickle=True)
    windows = data['windows']
    labels = data['labels']
    logger.info(f"Windows shape: {windows.shape}")
    logger.info(f"Labels shape: {labels.shape}")
    log_memory(f"After loading {os.path.basename(batch_path)}")

    # Convert to tensor
    windows_tensor = torch.FloatTensor(windows)
    labels_tensor = torch.LongTensor(labels)
   
    # Apply linear projection and positional encoding
    with torch.no_grad():
        projected = linear_projection(windows_tensor)
        transformer_ready_windows = positional_encoding(projected)
    
    log_memory(f"After transformation {os.path.basename(batch_path)}")

    # Save
    output_path = os.path.join(output_dir, data_type, component, 
                               os.path.basename(batch_path).replace('.npz', '.pt'))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save({
        'windows': transformer_ready_windows,
        'labels': labels_tensor,
    }, output_path)

    logger.info(f"Saved to {output_path}")
    del data, windows, labels, windows_tensor, labels_tensor, projected, transformer_ready_windows
    gc.collect()
    log_memory(f"After cleanup {os.path.basename(batch_path)}")


def process_component(input_dir, output_dir, component, data_type, linear_projection, positional_encoding):
    """
    Process all batch files for a specific component and data type.
    
    Args:
        input_dir: Input directory containing batch files
        output_dir: Output directory path
        component: Component type (e.g., 'contact', 'pcb', 'ring')
        data_type: Data type ('train', 'val', or 'test')
        linear_projection: Linear projection layer (shared)
        positional_encoding: Positional encoding layer (shared)
    """
    logger.info(f"Processing {component} {data_type} data")
    log_memory(f"Starting {component} {data_type}")
 
    component_dir = os.path.join(input_dir, data_type, component)
    
    # Check if component directory exists
    if not os.path.exists(component_dir):
        logger.warning(f"Component directory {component_dir} does not exist. Skipping.")
        return
    
    # Find all batch files
    batch_files = glob.glob(os.path.join(component_dir, "*.npz"))
    
    if not batch_files:
        logger.warning(f"No batch files found for {component} {data_type}. Skipping.")
        return
    
    logger.info(f"Found {len(batch_files)} batch files for {component} {data_type}")
    
    # Process each batch file
    for batch_file in tqdm(batch_files, desc=f"Processing {component} {data_type} batches"):
        process_batch_file(batch_file, output_dir, component, data_type, 
                           linear_projection, positional_encoding)
        
        # Force garbage collection
        gc.collect()
        log_memory(f"After batch {os.path.basename(batch_file)}")


def main():
    """
    Main function to process all data types and components.
    """
    start_time = datetime.now()
    logger.info(f"Starting sliding window to transformer conversion at {start_time}")
    log_memory("Starting conversion")
    
    # Get paths from config
    input_dir = "Data/processed/lsmt_timeFeatures/sliding_window_1200s"
    output_dir = "Data/processed/transform/slidingWindows_transform_1200s"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Explicitly define components and data types
    components = ['contact']
    data_types = ['train_down_25%', 'val', 'test']
    
    logger.info(f"Processing data types: {data_types}")
    logger.info(f"Processing components: {components}")
    

        # 动态读取 input_features
    sample_file = glob.glob(os.path.join(input_dir, data_types[0], components[0], "*.npz"))[0]
    logger.error(f'not found sample file {sample_file}')   
    sample_data = np.load(sample_file)
    input_features = sample_data['windows'].shape[2]
    logger.info(f"Input features: {input_features}")
    # Initialize shared layers
    d_model = 256
    max_seq_length = 1200
    
    # Create linear projection layer
    linear_projection = nn.Linear(input_features, d_model)
    nn.init.xavier_uniform_(linear_projection.weight)
    nn.init.zeros_(linear_projection.bias)
    
    # Create positional encoding layer
    positional_encoding = PositionalEncoding(d_model, max_seq_length)
    
    # Save the projection weights for reproducibility
    projection_path = os.path.join(output_dir, "projection_weights.pt")
    torch.save({
        'linear_projection': linear_projection.state_dict(),
        'd_model': d_model,
        'max_seq_length': max_seq_length
    }, projection_path)
    logger.info(f"Saved projection weights to {projection_path}")
    del sample_data
    gc.collect()
    log_memory("After projection weights saved")
    
    # Process each component and data type
    for data_type in data_types:
        for component in components:
            process_component(input_dir, output_dir, component, data_type, 
                              linear_projection, positional_encoding)
            log_memory(f"Completed {component} {data_type}")
    
    end_time = datetime.now()
    duration = end_time - start_time
    logger.info(f"Finished conversion at {end_time}. Duration: {duration}")
    log_memory("Conversion complete")


if __name__ == "__main__":
    main()