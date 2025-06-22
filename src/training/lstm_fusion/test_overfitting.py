"""
Test script for LSTM model with Late Fusion to verify its ability to overfit to a single sample.

This script uses a single data sample to test if the model can achieve near-perfect accuracy,
which is a good way to verify the model's capacity and implementation correctness.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import argparse

from src.utils.logger import logger
from src.training.lsmt.lsmt_fusion.lstm_late_fusion_model import LSTMLateFusionModel
from src.training.lsmt.lsmt_fusion.lstm_late_fusion_dataset import create_single_sample_data_loader


def load_config(config_path=None):
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    if config_path is None:
        config_path = Path("configs/lstm_late_fusion.yaml")
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    logger.info(f"Loaded configuration from {config_path}")
    return config


def train_epoch(model, data_loader, criterion, optimizer, device):
    """
    Train the model for one epoch on a single sample.
    
    Args:
        model: LSTM Late Fusion model
        data_loader: Training data loader with a single sample
        criterion: Loss function
        optimizer: Optimizer
        device: Device to use for training
        
    Returns:
        Training loss
    """
    model.train()
    total_loss = 0
    
    for windows, stat_features, labels in data_loader:
        # Move data to device
        windows = windows.to(device)
        stat_features = stat_features.to(device)
        labels = labels.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(windows, stat_features)
        
        # Calculate loss
        loss = criterion(outputs, labels)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss


def evaluate(model, data_loader, criterion, device):
    """
    Evaluate the model on a single sample.
    
    Args:
        model: LSTM Late Fusion model
        data_loader: Data loader with a single sample
        criterion: Loss function
        device: Device to use for evaluation
        
    Returns:
        Tuple of (loss, accuracy)
    """
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for windows, stat_features, labels in data_loader:
            # Move data to device
            windows = windows.to(device)
            stat_features = stat_features.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(windows, stat_features)
            
            # Calculate loss
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = correct / total
    return total_loss, accuracy


def plot_results(train_losses, accuracies, save_dir):
    """
    Plot training losses and accuracies.
    
    Args:
        train_losses: List of training losses
        accuracies: List of accuracies
        save_dir: Directory to save the plots
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Plot losses
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss on Single Sample')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'overfitting_loss.png'))
    plt.close()
    
    # Plot accuracies
    plt.figure(figsize=(10, 5))
    plt.plot(accuracies, label='Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy on Single Sample')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'overfitting_accuracy.png'))
    plt.close()


def main(args):
    """
    Main function to test the LSTM Late Fusion model's ability to overfit to a single sample.
    
    Args:
        args: Command line arguments
    """
    # Load configuration
    config = load_config(args.config)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create output directory
    output_dir = os.path.join(config['paths']['output_dir'], "overfitting_test")
    os.makedirs(output_dir, exist_ok=True)
    
    # Create data loaders with a single sample
    data_loaders = create_single_sample_data_loader(
        lstm_data_dir=config['paths']['lstm_data_dir'],
        stat_features_dir=config['paths']['stat_features_dir'],
        batch_size=1,
        num_workers=0,
        component=config['data']['component'],
        sample_idx=args.sample_idx
    )
    
    # Create model
    model = LSTMLateFusionModel(config=config['model'])
    model.to(device)
    
    # Define loss function
    criterion = nn.CrossEntropyLoss()
    
    # Define optimizer with higher learning rate for faster convergence
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=config['training']['weight_decay']
    )
    
    # Initialize tracking variables
    train_losses = []
    accuracies = []
    
    # Training loop
    for epoch in range(1, args.num_epochs + 1):
        logger.info(f"Epoch {epoch}/{args.num_epochs}")
        
        # Train
        train_loss = train_epoch(model, data_loaders['train'], criterion, optimizer, device)
        train_losses.append(train_loss)
        
        # Evaluate
        _, accuracy = evaluate(model, data_loaders['train'], criterion, device)
        accuracies.append(accuracy)
        
        logger.info(f"Epoch {epoch}: Loss = {train_loss:.6f}, Accuracy = {accuracy:.6f}")
        
        # Early stopping if we've achieved perfect accuracy
        if accuracy == 1.0 and train_loss < 0.01:
            logger.info(f"Perfect accuracy achieved at epoch {epoch}. Early stopping.")
            break
    
    # Plot results
    plot_results(train_losses, accuracies, output_dir)
    
    # Final evaluation
    final_loss, final_accuracy = evaluate(model, data_loaders['train'], criterion, device)
    logger.info(f"Final results - Loss: {final_loss:.6f}, Accuracy: {final_accuracy:.6f}")
    
    # Save model
    torch.save(model.state_dict(), os.path.join(output_dir, "overfitting_model.pt"))
    logger.info(f"Saved model to {os.path.join(output_dir, 'overfitting_model.pt')}")
    
    # Conclusion
    if final_accuracy == 1.0:
        logger.info("SUCCESS: Model successfully overfit to the single sample!")
    else:
        logger.warning(f"WARNING: Model failed to overfit completely. Final accuracy: {final_accuracy:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test LSTM Late Fusion model's ability to overfit")
    parser.add_argument("--config", type=str, default="configs/lstm_late_fusion.yaml",
                      help="Path to configuration file")
    parser.add_argument("--num_epochs", type=int, default=100,
                      help="Number of epochs to train")
    parser.add_argument("--learning_rate", type=float, default=0.01,
                      help="Learning rate (higher than normal for faster overfitting)")
    parser.add_argument("--sample_idx", type=int, default=0,
                      help="Index of the sample to use for overfitting test")
    args = parser.parse_args()
    
    main(args) 