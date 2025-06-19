"""
Test script for LSTM model with Late Fusion to verify its ability to overfit to a small dataset.

This script uses a small subset of the data to test if the model can achieve near-perfect accuracy,
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
import random
from sklearn.metrics import precision_score, recall_score, f1_score

from src.utils.logger import logger
from src.training.lsmt_fusion.lstm_late_fusion_model import LSTMLateFusionModel
from src.training.lsmt_fusion.lstm_late_fusion_dataset import LSTMLateFusionDataset
from torch.utils.data import DataLoader, Subset


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


def create_small_data_loaders(
    lstm_data_dir: str,
    stat_features_dir: str,
    batch_size: int = 8,
    num_workers: int = 0,
    component: str = 'contact',
    num_samples: int = 10,
    seed: int = 42
):
    """
    Create data loaders with a small subset of samples for overfitting tests.
    
    Args:
        lstm_data_dir: Directory containing LSTM sliding window data
        stat_features_dir: Directory containing statistical features
        batch_size: Batch size for data loaders
        num_workers: Number of worker processes for data loading
        component: Component type ('contact', 'pcb', or 'ring')
        num_samples: Number of samples to include in the small dataset
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary of data loaders with a small subset of samples
    """
    data_loaders = {}
    
    # Set random seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Create the full dataset
    full_dataset = LSTMLateFusionDataset(
        lstm_data_dir=lstm_data_dir,
        stat_features_dir=stat_features_dir,
        data_type='train',
        component=component
    )
    
    # Get total number of samples
    total_samples = len(full_dataset)
    logger.info(f"Total samples in full dataset: {total_samples}")
    
    # Make sure we don't request more samples than available
    num_samples = min(num_samples, total_samples)
    
    # Randomly select indices for the small dataset
    all_indices = list(range(total_samples))
    selected_indices = random.sample(all_indices, num_samples)
    
    # Create subset datasets
    small_dataset = Subset(full_dataset, selected_indices)
    
    # Create data loaders
    data_loaders['train'] = DataLoader(
        small_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # Use the same subset for validation and testing
    data_loaders['val'] = DataLoader(
        small_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    data_loaders['test'] = DataLoader(
        small_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    logger.info(f"Created data loaders with {num_samples} samples for overfitting test")
    
    # Check class distribution
    labels = []
    for _, _, label in small_dataset:
        labels.append(label.item())
    
    unique_labels, counts = np.unique(labels, return_counts=True)
    for label, count in zip(unique_labels, counts):
        logger.info(f"Class {label}: {count} samples ({count/len(labels)*100:.2f}%)")
    
    return data_loaders


def train_epoch(model, data_loader, criterion, optimizer, device):
    """
    Train the model for one epoch on a small dataset.
    
    Args:
        model: LSTM Late Fusion model
        data_loader: Training data loader with a small dataset
        criterion: Loss function
        optimizer: Optimizer
        device: Device to use for training
        
    Returns:
        Average training loss
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
    
    avg_loss = total_loss / len(data_loader)
    return avg_loss


def evaluate(model, data_loader, criterion, device):
    """
    Evaluate the model on a small dataset.
    
    Args:
        model: LSTM Late Fusion model
        data_loader: Data loader with a small dataset
        criterion: Loss function
        device: Device to use for evaluation
        
    Returns:
        Tuple of (average loss, accuracy, precision, recall, f1)
    """
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_predicted = []
    all_labels = []
    
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
            
            # Store predictions and labels for precision, recall, and F1
            all_predicted.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(data_loader)
    accuracy = correct / total
    
    # Calculate precision, recall, and F1 score
    # For binary classification
    all_predicted = np.array(all_predicted)
    all_labels = np.array(all_labels)
    
    # For multi-class, we'll use macro averaging
    precision = precision_score(all_labels, all_predicted, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_predicted, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_predicted, average='macro', zero_division=0)
    
    return avg_loss, accuracy, precision, recall, f1


def plot_results(train_losses, train_accuracies, val_losses, val_precisions, val_recalls, val_f1s, save_dir):
    """
    Plot training and validation losses and accuracies.
    
    Args:
        train_losses: List of training losses
        train_accuracies: List of training accuracies
        val_losses: List of validation losses
        val_precisions: List of validation precisions
        val_recalls: List of validation recalls
        val_f1s: List of validation f1 scores
        save_dir: Directory to save the plots
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Plot losses
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss on Small Dataset')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'small_dataset_loss.png'))
    plt.close()
    
    # Plot accuracies
    plt.figure(figsize=(10, 5))
    plt.plot(train_accuracies, label='Training Accuracy')
    plt.plot(val_losses, label='Validation Loss')
    plt.plot(val_precisions, label='Validation Precision')
    plt.plot(val_recalls, label='Validation Recall')
    plt.plot(val_f1s, label='Validation F1')
    plt.xlabel('Epoch')
    plt.ylabel('Metrics')
    plt.title('Training and Validation Metrics on Small Dataset')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'small_dataset_metrics.png'))
    plt.close()


def main(args):
    """
    Main function to test the LSTM Late Fusion model's ability to overfit to a small dataset.
    
    Args:
        args: Command line arguments
    """
    # Load configuration
    config = load_config(args.config)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create output directory
    output_dir = os.path.join(config['paths']['output_dir'], f"small_dataset_overfitting_test_{args.num_samples}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Create data loaders with a small dataset
    data_loaders = create_small_data_loaders(
        lstm_data_dir=config['paths']['lstm_data_dir'],
        stat_features_dir=config['paths']['stat_features_dir'],
        batch_size=args.batch_size,
        num_workers=0,
        component=config['data']['component'],
        num_samples=args.num_samples,
        seed=args.seed
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
    train_accuracies = []
    train_precisions = []
    train_recalls = []
    train_f1s = []
    val_losses = []
    val_precisions = []
    val_recalls = []
    val_f1s = []
    best_val_loss = float('inf')
    
    # Training loop
    for epoch in range(1, args.num_epochs + 1):
        logger.info(f"Epoch {epoch}/{args.num_epochs}")
        
        # Train
        train_loss = train_epoch(model, data_loaders['train'], criterion, optimizer, device)
        train_losses.append(train_loss)
        
        # Evaluate on training set
        _, train_accuracy, train_precision, train_recall, train_f1 = evaluate(model, data_loaders['train'], criterion, device)
        train_accuracies.append(train_accuracy)
        train_precisions.append(train_precision)
        train_recalls.append(train_recall)
        train_f1s.append(train_f1)
        
        # Evaluate on validation set
        val_loss, val_accuracy, val_precision, val_recall, val_f1 = evaluate(model, data_loaders['val'], criterion, device)
        val_losses.append(val_loss)
        val_precisions.append(val_precision)
        val_recalls.append(val_recall)
        val_f1s.append(val_f1)
        
        logger.info(f"Epoch {epoch}: Train Loss = {train_loss:.6f}, Train Accuracy = {train_accuracy:.6f}, "
                   f"Val Loss = {val_loss:.6f}, Val Accuracy = {val_accuracy:.6f}, "
                   f"Val Precision = {val_precision:.6f}, Val Recall = {val_recall:.6f}, Val F1 = {val_f1:.6f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(output_dir, "best_model.pt"))
            logger.info(f"Saved best model at epoch {epoch}")
        
        # Early stopping if we've achieved near-perfect accuracy
        if train_accuracy > 0.99 and val_accuracy > 0.99 and val_f1 > 0.99:
            logger.info(f"Near-perfect accuracy achieved at epoch {epoch}. Early stopping.")
            break
    
    # Plot results
    plot_results(train_losses, train_accuracies, val_losses, val_precisions, val_recalls, val_f1s, output_dir)
    
    # Final evaluation
    model.load_state_dict(torch.load(os.path.join(output_dir, "best_model.pt")))
    final_train_loss, final_train_accuracy, final_train_precision, final_train_recall, final_train_f1 = evaluate(model, data_loaders['train'], criterion, device)
    final_val_loss, final_val_accuracy, final_val_precision, final_val_recall, final_val_f1 = evaluate(model, data_loaders['test'], criterion, device)
    
    logger.info(f"Final results - Train Loss: {final_train_loss:.6f}, Train Accuracy: {final_train_accuracy:.6f}, "
                f"Train Precision: {final_train_precision:.6f}, Train Recall: {final_train_recall:.6f}, Train F1: {final_train_f1:.6f}")
    logger.info(f"Final results - Test Loss: {final_val_loss:.6f}, Test Accuracy: {final_val_accuracy:.6f}, "
                f"Test Precision: {final_val_precision:.6f}, Test Recall: {final_val_recall:.6f}, Test F1: {final_val_f1:.6f}")
    
    # Conclusion
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test LSTM Late Fusion model's ability to overfit to a small dataset")
    parser.add_argument("--config", type=str, default="configs/lstm_late_fusion.yaml",
                      help="Path to configuration file")
    parser.add_argument("--num_epochs", type=int, default=200,
                      help="Number of epochs to train")
    parser.add_argument("--learning_rate", type=float, default=0.005,
                      help="Learning rate (higher than normal for faster overfitting)")
    parser.add_argument("--num_samples", type=int, default=10,
                      help="Number of samples to include in the small dataset")
    parser.add_argument("--batch_size", type=int, default=4,
                      help="Batch size for training")
    parser.add_argument("--seed", type=int, default=42,
                      help="Random seed for reproducibility")
    args = parser.parse_args()
    
    main(args) 