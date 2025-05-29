import os
import pickle
import torch
from torch.utils.data import DataLoader, TensorDataset
from src.utils.logger import logger
import glob

def get_component_dataloaders(
    component_names=None,
    data_dir=None,
    batch_size=128,
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
        logger.info(f"Found components: {component_names}")
    
    for component_name in component_names:
        component_dir = os.path.join(data_dir, component_name)
        # Check if component directory exists
        if not os.path.exists(component_dir):
            logger.warning(f"Component directory not found: {component_dir}")
            continue
            
        # metadata_path = os.path.join(component_dir, "*.pkl")
        # print('metadata_path', metadata_path)
        # try:
        #     # Load metadata
        #     if not os.path.exists(metadata_path):
        #logger.warning(f"Metadata file not found: {metadata_path}")
        
        # Try to find batch files directly
        batch_files = [f for f in os.listdir(component_dir) if f.endswith('.pt') and f.startswith('batch_')]
        if batch_files:
            logger.info(f"Found {len(batch_files)} batch files without metadata for component {component_name}")
            file_paths = batch_files
        else:
            logger.warning(f"No batch files found for component {component_name}")
            continue
            # else:
            #     # Load metadata from file
            #     with open(metadata_path, 'rb') as f:
            #         metadata = pickle.load(f)
            #         #print('metadata', metadata)
            #     pt_file_paths = [item[0] for item in metadata]
            #     if not pt_file_paths:
            #         logger.warning(f"No file paths found in metadata for component {component_name}")
            #         continue
        
        logger.info(f"Loading {len(file_paths)} batch files for component {component_name}")
        
        # Create a dataset from the batch files
        all_windows = []
        all_labels = []
        
        for batch_path in file_paths:
            batch_path = os.path.join(component_dir, batch_path)
            
            if not os.path.exists(batch_path):
                logger.warning(f"Batch file not found: {batch_path}")
                continue
                
            # Load batch data
            try:
                batch_data = torch.load(batch_path)
                
                if 'windows' not in batch_data or 'labels' not in batch_data:
                    logger.warning(f"Batch file {batch_path} missing required keys")
                    continue
                    
                windows = batch_data['windows']
                labels = batch_data['labels']
            
                    
                # Ensure labels are appropriate for binary classification (0 or 1)
                if labels.dtype != torch.long:
                    labels = labels.long()
                    
                all_windows.append(windows)
                all_labels.append(labels)
                
            except Exception as e:
                logger.error(f"Error loading batch file {batch_path}: {str(e)}")
                continue
        
        if not all_windows:
            logger.warning(f"No valid data found for component {component_name}")
            continue
            
        # Concatenate all windows and labels
        try:
            windows = torch.cat(all_windows, dim=0)
            labels = torch.cat(all_labels, dim=0)
            
            logger.info(f"Created dataset for {component_name} with {len(windows)} samples, "
                        f"window shape: {windows.shape}, label shape: {labels.shape}")
            
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
            logger.error(f"Error creating dataset for component {component_name}: {str(e)}")
            continue
        
    # except Exception as e:
    #     logger.error(f"Error loading data for component {component_name}: {str(e)}")
    #     continue
