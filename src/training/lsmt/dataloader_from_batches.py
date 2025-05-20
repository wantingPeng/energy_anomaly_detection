import os
import pickle
import torch
from torch.utils.data import DataLoader, TensorDataset
from src.utils.logger import logger

def get_component_dataloaders(
    component_names=None,
    data_dir="Data/processed/lsmt/dataset",
    batch_size=32,
    shuffle=True,
    num_workers=4,
    pin_memory=True
):
    """
    Load data from batch .pt files and return a generator of DataLoaders, one for each component.
    
    Args:
        component_names (list): List of component names to load. If None, load all available components.
        data_dir (str): Directory containing the processed data
        batch_size (int): Batch size for DataLoader
        shuffle (bool): Whether to shuffle the data
        num_workers (int): Number of workers for DataLoader
        pin_memory (bool): Whether to pin memory in DataLoader
        
    Yields:
        tuple: (component_name, DataLoader) for each component
    """
    if component_names is None:
        # Get all component directories
        component_names = [d for d in os.listdir(data_dir) 
                          if os.path.isdir(os.path.join(data_dir, d))]
        logger.info(f"Found components: {component_names}")
    
    for component_name in component_names:
        component_dir = os.path.join(data_dir, component_name)
        metadata_path = os.path.join(component_dir, "metadata.pkl")
        
        try:
            # Load metadata
            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)
                
            file_paths = metadata.get('file_paths', [])
            print(f"File paths: {file_paths}")
            if not file_paths:
                logger.warning(f"No file paths found in metadata for component {component_name}")
                continue
                
            logger.info(f"Loading {len(file_paths)} batch files for component {component_name}")
            
            # Create a dataset from the batch files
            all_windows = []
            all_labels = []
            
            for batch_file in file_paths:
                batch_path = os.path.join(component_dir, batch_file)
                
                if not os.path.exists(batch_path):
                    logger.warning(f"Batch file not found: {batch_path}")
                    continue
                    
                # Load batch data
                batch_data = torch.load(batch_path)
                
                if 'windows' not in batch_data or 'labels' not in batch_data:
                    logger.warning(f"Batch file {batch_path} missing required keys")
                    continue
                    
                all_windows.append(batch_data['windows'])
                all_labels.append(batch_data['labels'])
            
            if not all_windows:
                logger.warning(f"No valid data found for component {component_name}")
                continue
                
            # Concatenate all windows and labels
            windows = torch.cat(all_windows, dim=0)
            labels = torch.cat(all_labels, dim=0)
            
            logger.info(f"Created dataset for {component_name} with {len(windows)} samples")
            
            # Create TensorDataset and DataLoader
            dataset = TensorDataset(windows, labels)
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                pin_memory=pin_memory
            )
            
            yield component_name, dataloader
            
        except Exception as e:
            logger.error(f"Error loading data for component {component_name}: {str(e)}")
            continue
