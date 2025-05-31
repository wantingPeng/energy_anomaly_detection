import argparse
import torch
import yaml
import os
from pathlib import Path

from src.training.lsmt.lstm_model import LSTMModel
from src.training.lsmt.dataloader_from_batches import get_component_dataloaders
from src.training.lsmt.focal_loss import FocalLoss
from src.training.lsmt.gradient_checker import gradient_diagnosis
from src.utils.logger import logger

def load_config(config_path):
    """
    Load configuration from YAML file
    
    Args:
        config_path (str): Path to configuration file
        
    Returns:
        dict: Configuration dictionary
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded configuration from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Error loading configuration from {config_path}: {str(e)}")
        raise

def load_model(model_path, config):
    """
    Load a pre-trained LSTM model
    
    Args:
        model_path (str): Path to the model checkpoint
        config (dict): Configuration dictionary
        
    Returns:
        LSTMModel: Loaded model
    """
    # Create model from config
    model_config = config.get('model', {})
    model = LSTMModel(config=model_config)
    
    # Load model weights
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # If checkpoint contains full model state_dict
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Loaded model state from checkpoint at epoch {checkpoint.get('epoch', 'unknown')}")
    else:
        # Direct model state dict
        model.load_state_dict(checkpoint)
        logger.info("Loaded model state directly from checkpoint")
    
    return model

def main():
    """Main function to diagnose gradient issues in a pre-trained model"""
    parser = argparse.ArgumentParser(description="Diagnose gradient issues in a pre-trained LSTM model")
    parser.add_argument("--config", type=str, default="configs/lstm_training.yaml",
                        help="Path to YAML configuration file")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the pre-trained model checkpoint")
    parser.add_argument("--component", type=str, default="contact",
                        help="Component to use for gradient diagnosis")
    parser.add_argument("--num_batches", type=int, default=10,
                        help="Number of batches to use for diagnosis")
    parser.add_argument("--output_dir", type=str, default="experiments/logs/gradient_diagnosis",
                        help="Directory to save diagnosis results")
    parser.add_argument("--clip_values", type=str, default="none,0.5,1.0,5.0",
                        help="Comma-separated list of gradient clipping values to test (use 'none' for no clipping)")
    
    args = parser.parse_args()
    
    # Process clip values
    clip_values = []
    for val in args.clip_values.split(','):
        if val.lower() == 'none':
            clip_values.append(None)
        else:
            try:
                clip_values.append(float(val))
            except ValueError:
                logger.warning(f"Ignoring invalid clip value: {val}")
    
    # Load configuration
    config = load_config(args.config)
    
    # Load model
    model = load_model(args.model_path, config)
    
    # Device configuration
    device_config = config.get('device', {})
    use_gpu = device_config.get('use_gpu', False) and torch.cuda.is_available()
    device = torch.device('cuda' if use_gpu else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Move model to device
    model = model.to(device)
    
    # Set up loss function
    loss_config = config.get('training', {}).get('loss', {})
    alpha = loss_config.get('focal_alpha', 0.25)
    gamma = loss_config.get('focal_gamma', 2.0)
    criterion = FocalLoss(alpha=alpha, gamma=gamma)
    
    # Set up optimizer
    train_config = config.get('training', {})
    learning_rate = train_config.get('learning_rate', 0.001)
    weight_decay = train_config.get('weight_decay', 0.0001)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Get dataloader for specified component
    data_dir = config.get('data', {}).get('data_dir', 'Data/processed/lsmt/dataset/train')
    batch_size = train_config.get('batch_size', 64)
    
    logger.info(f"Loading data for component: {args.component}")
    
    dataloader = None
    for component_name, loader in get_component_dataloaders(
        component_names=[args.component],
        data_dir=data_dir,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=use_gpu
    ):
        dataloader = loader
        break
    
    if dataloader is None:
        logger.error(f"Could not load data for component: {args.component}")
        return
    
    logger.info(f"Loaded dataloader with {len(dataloader.dataset)} samples")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run gradient diagnosis
    logger.info(f"Starting gradient diagnosis with clip values: {clip_values}")
    
    results = gradient_diagnosis(
        model=model,
        dataloader=dataloader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        plot_dir=args.output_dir,
        clip_values=clip_values
    )
    
    # Print diagnosis summary
    logger.info("Gradient Diagnosis Summary:")
    logger.info(f"Issues detected: {results['issues_detected']}")
    
    if results['issues_detected']:
        logger.info("Recommendations:")
        for rec in results['recommendations']:
            issue = rec['issue'].replace('_', ' ')
            logger.info(f"For {issue}:")
            for suggestion in rec['suggestions']:
                logger.info(f"- {suggestion}")
    else:
        logger.info("No gradient issues were detected in the model.")
    
    if 'clip_test_results' in results and results['clip_test_results']:
        logger.info("Gradient Clipping Test Results:")
        for clip_value, result in results['clip_test_results'].items():
            status = "IMPROVED" if result['improved'] else "NO IMPROVEMENT"
            logger.info(f"Clip value {clip_value}: {status}")
    
    logger.info(f"Detailed diagnosis report saved to {args.output_dir}")

if __name__ == "__main__":
    main() 