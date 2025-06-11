import torch
import os
import sys
from pathlib import Path
from src.utils.logger import logger
import argparse
from pprint import pformat
import numpy as np


def load_model_info(model_path):
    """
    Load and display information about a PyTorch model.
    
    Args:
        model_path (str): Path to the PyTorch model file (.pt or .pth)
    """
    logger.info(f"Loading model from: {model_path}")
    
    if not os.path.exists(model_path):
        logger.error(f"Model file not found: {model_path}")
        return
    
    try:
        # Add numpy to safe globals
        torch.serialization.add_safe_globals(['numpy._core.multiarray.scalar'])
        
        # Load the model with weights_only=False to handle the error
        model_data = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
        logger.info(f"Model loaded successfully")
        
        # Display model information
        logger.info("Model content structure:")
        
        if isinstance(model_data, dict):
            # Most saved models are dictionaries with keys like 'model_state_dict', 'optimizer_state_dict', etc.
            logger.info(f"Model is a dictionary with keys: {list(model_data.keys())}")
            
            for key in model_data.keys():
                if key == 'model_state_dict' or key == 'state_dict':
                    logger.info(f"\nModel architecture parameters:")
                    param_shapes = {k: v.shape for k, v in model_data[key].items()}
                    logger.info(f"Parameters: \n{pformat(param_shapes)}")
                elif key == 'model_config' or key == 'config':
                    logger.info(f"\nModel configuration:")
                    logger.info(f"{pformat(model_data[key])}")
                elif key == 'optimizer_state_dict':
                    logger.info(f"\nOptimizer information available")
                elif key == 'epoch' or key == 'best_epoch':
                    logger.info(f"\nEpoch information: {model_data[key]}")
                elif key == 'best_metric' or key == 'metrics':
                    logger.info(f"\nMetric information: {model_data[key]}")
                else:
                    if not isinstance(model_data[key], torch.Tensor):
                        logger.info(f"\n{key}: {model_data[key]}")
                    else:
                        logger.info(f"\n{key}: Tensor of shape {model_data[key].shape}")
        elif isinstance(model_data, torch.nn.Module):
            logger.info("Model is a direct PyTorch module instance")
            logger.info(f"Model structure: \n{model_data}")
        else:
            logger.info(f"Model type: {type(model_data)}")
            logger.info(f"Model content: {model_data}")
            
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")


def main():
    model_path = "experiments/lstm_late_fusion/lstm_late_fusion_20250604_121132_0.25_time/best_model_f1.pt"
    load_model_info(model_path)


if __name__ == "__main__":
    main() 