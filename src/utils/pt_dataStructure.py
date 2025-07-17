import os
import torch
from src.utils.logger import logger
from datetime import datetime
import numpy as np
import torch.serialization

def load_display_checkpoint(checkpoint_path):
    """
    Load and display the contents of a PyTorch checkpoint file.
    
    Args:
        checkpoint_path (str): Path to the checkpoint file
        
    Returns:
        dict: The loaded checkpoint data
    """
    try:
        # Check if file exists
        if not os.path.isfile(checkpoint_path):
            logger.error(f"Checkpoint file not found: {checkpoint_path}")
            return None
        
        # Load checkpoint - handle PyTorch 2.6+ security restrictions
        logger.info(f"Loading checkpoint from: {checkpoint_path}")
        
        try:
            # First try: Add numpy scalar to safe globals list (more secure approach)
            torch.serialization.add_safe_globals(['numpy._core.multiarray.scalar'])
            checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        except Exception as e:
            logger.warning(f"First loading attempt failed: {str(e)}")
            # Second try: Use weights_only=False (less secure but compatible with older checkpoints)
            logger.info("Attempting to load with weights_only=False")
            checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'), weights_only=False)
        
        # Display checkpoint keys
        logger.info(f"Checkpoint contains the following keys: {list(checkpoint.keys())}")
        
        # Display basic information
        if 'epoch' in checkpoint:
            logger.info(f"Epoch: {checkpoint['epoch']}")
        if 'train_loss' in checkpoint:
            logger.info(f"Train Loss: {checkpoint['train_loss']}")
        if 'val_loss' in checkpoint:
            logger.info(f"Validation Loss: {checkpoint['val_loss']}")
        
        # Display metrics if available
        if 'metrics' in checkpoint:
            logger.info(f"Metrics: {checkpoint['metrics']}")
        
        # Display model architecture info
        if 'model_state_dict' in checkpoint:
            model_dict = checkpoint['model_state_dict']
            logger.info(f"Model has {len(model_dict)} layers/parameters")
            for key, value in model_dict.items():
                if hasattr(value, 'shape'):
                    logger.info(f"  {key}: {value.shape}")
                else:
                    logger.info(f"  {key}: {type(value)}")
        
        # Display optimizer info
        if 'optimizer_state_dict' in checkpoint:
            logger.info("Optimizer state dict is present")
        
        # Display config if available
        if 'config' in checkpoint:
            logger.info(f"Configuration: {checkpoint['config']}")
        
        return checkpoint
    
    except Exception as e:
        logger.error(f"Error loading checkpoint: {str(e)}")
        return None

def main():
    """
    Main function to demonstrate checkpoint loading functionality
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"load_checkpoint_{timestamp}.log"
    
    # Example usage
    checkpoint_path = "experiments/row_energyData_subsample_Transform/bestModel/op0.3_ad0.48/transformer_20250706_171102_op0.3_ad0.48/best_adj_f1/checkpoint_epoch_0.pt"
    checkpoint = load_display_checkpoint(checkpoint_path)
    
    if checkpoint:
        logger.info("Checkpoint loaded successfully")
    else:
        logger.error("Failed to load checkpoint")
    
if __name__ == "__main__":
    main()
