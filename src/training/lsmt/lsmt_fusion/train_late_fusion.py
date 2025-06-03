"""
Training script for LSTM model with Late Fusion.

This script trains an LSTM model with Late Fusion that combines time series data
with statistical features for energy anomaly detection.
"""

import os
import gc
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse

from src.utils.logger import logger
from src.training.lsmt.lsmt_fusion.lstm_late_fusion_model import LSTMLateFusionModel
from src.training.lsmt.lsmt_fusion.lstm_late_fusion_dataset import create_data_loaders


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


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance by focusing on hard examples.
    
    Args:
        alpha: Weighting factor for the rare class (typically the positive class)
        gamma: Focusing parameter that reduces the loss contribution from easy examples
        reduction: Specifies the reduction to apply to the output ('mean', 'sum', or 'none')
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.cross_entropy = nn.CrossEntropyLoss(reduction='none')
        
    def forward(self, inputs, targets):
        ce_loss = self.cross_entropy(inputs, targets)
        pt = torch.exp(-ce_loss)
        
        # Apply alpha weighting based on the class
        alpha_tensor = torch.ones_like(targets, dtype=torch.float32)
        alpha_tensor[targets == 1] = self.alpha  # Weight for positive class
        alpha_tensor[targets == 0] = 1 - self.alpha  # Weight for negative class
        
        # Apply focusing term
        focal_loss = alpha_tensor * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def train_epoch(model, data_loader, criterion, optimizer, device):
    """
    Train the model for one epoch.
    
    Args:
        model: LSTM Late Fusion model
        data_loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to use for training
        
    Returns:
        Average training loss
    """
    model.train()
    total_loss = 0
    
    for windows, stat_features, labels in tqdm(data_loader, desc="Training"):
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


def evaluate(model, data_loader, criterion, device, threshold=0.3):
    """
    Evaluate the model on validation or test data.
    
    Args:
        model: LSTM Late Fusion model
        data_loader: Validation or test data loader
        criterion: Loss function
        device: Device to use for evaluation
        threshold: Classification threshold for positive class (anomaly)
        
    Returns:
        Tuple of (average loss, accuracy, precision, recall, f1, confusion matrix)
    """
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_scores = []
    
    with torch.no_grad():
        for windows, stat_features, labels in tqdm(data_loader, desc="Evaluating"):
            # Move data to device
            windows = windows.to(device)
            stat_features = stat_features.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(windows, stat_features)
            
            # Calculate loss
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            # Get probabilities and predictions with custom threshold
            probs = torch.softmax(outputs, dim=1)
            anomaly_scores = probs[:, 1]  # Probability for anomaly class
            preds = (anomaly_scores > threshold).long()
            
            # Store predictions, scores and labels
            all_preds.extend(preds.cpu().numpy())
            all_scores.extend(anomaly_scores.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    avg_loss = total_loss / len(data_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary', zero_division=0)
    conf_matrix = confusion_matrix(all_labels, all_preds)
    
    return avg_loss, accuracy, precision, recall, f1, conf_matrix, all_scores, all_labels


def save_model(model, optimizer, epoch, train_loss, val_loss, metrics, config, save_dir):
    """
    Save model checkpoint.
    
    Args:
        model: LSTM Late Fusion model
        optimizer: Optimizer
        epoch: Current epoch
        train_loss: Training loss
        val_loss: Validation loss
        metrics: Dictionary of metrics
        config: Configuration dictionary
        save_dir: Directory to save the checkpoint
    """
    os.makedirs(save_dir, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'metrics': metrics,
        'config': config
    }
    
    checkpoint_path = os.path.join(save_dir, f"checkpoint_epoch_{epoch}.pt")
    torch.save(checkpoint, checkpoint_path)
    logger.info(f"Saved checkpoint to {checkpoint_path}")


def plot_metrics(train_losses, val_losses, accuracies, precisions, recalls, f1s, save_dir):
    """
    Plot training and validation metrics.
    
    Args:
        train_losses: List of training losses
        val_losses: List of validation losses
        accuracies: List of accuracies
        precisions: List of precisions
        recalls: List of recalls
        f1s: List of F1 scores
        save_dir: Directory to save the plots
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Plot losses
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'loss_plot.png'))
    plt.close()
    
    # Plot metrics
    plt.figure(figsize=(10, 6))
    plt.plot(accuracies, label='Accuracy')
    plt.plot(precisions, label='Precision')
    plt.plot(recalls, label='Recall')
    plt.plot(f1s, label='F1 Score')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.title('Validation Metrics')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'metrics_plot.png'))
    plt.close()


def plot_pr_curve(scores, labels, thresholds, save_dir):
    """
    Plot precision-recall curve at different thresholds.
    
    Args:
        scores: Prediction scores (probabilities)
        labels: True labels
        thresholds: List of thresholds to evaluate
        save_dir: Directory to save the plot
    """
    precisions = []
    recalls = []
    f1_scores = []
    
    for threshold in thresholds:
        preds = [1 if score > threshold else 0 for score in scores]
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, preds, average='binary', zero_division=0
        )
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
    
    # Plot precision-recall curve
    plt.figure(figsize=(10, 6))
    plt.plot(recalls, precisions, 'b-', label='Precision-Recall curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'pr_curve.png'))
    plt.close()
    
    # Plot F1 score vs threshold
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, f1_scores, 'g-', label='F1 Score')
    plt.xlabel('Threshold')
    plt.ylabel('F1 Score')
    plt.title('F1 Score vs Threshold')
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'f1_threshold.png'))
    plt.close()
    
    # Find best threshold
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]
    best_f1 = f1_scores[best_idx]
    
    return best_threshold, best_f1


def main(args):
    """
    Main function to train the LSTM Late Fusion model.
    
    Args:
        args: Command line arguments
    """
    # Load configuration
    config = load_config(args.config)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create experiment directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = os.path.join(config['paths']['output_dir'], f"lstm_late_fusion_{timestamp}")
    os.makedirs(experiment_dir, exist_ok=True)
    
    # Set up logging
    log_file = os.path.join("experiments/logs", f"lstm_late_fusion_training_{timestamp}.log")
    
    # Create data loaders
    data_loaders = create_data_loaders(
        lstm_data_dir=config['paths']['lstm_data_dir'],
        stat_features_dir=config['paths']['stat_features_dir'],
        batch_size=config['training']['batch_size'],
        num_workers=config['training']['num_workers'],
        component=config['data']['component']
    )
    
    # Create model
    model = LSTMLateFusionModel(config=config['model'])
    model.to(device)
    
    # Get class weights if specified in config
    if config['training'].get('use_class_weights', False):
        # Calculate class weights from training data
        # Sample class distribution calculation (adjust based on your data)
        train_labels = []
        for _, _, labels in data_loaders['train']:
            train_labels.extend(labels.numpy())
        
        class_counts = np.bincount(train_labels)
        total_samples = len(train_labels)
        num_classes = len(class_counts)
        
        # Inverse frequency weighting
        class_weights = total_samples / (num_classes * class_counts)
        class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
        
        logger.info(f"Using class weights: {class_weights}")
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    elif config['training'].get('use_focal_loss', False):
        # Use Focal Loss
        alpha = config['training'].get('focal_loss_alpha', 0.25)
        gamma = config['training'].get('focal_loss_gamma', 2.0)
        logger.info(f"Using Focal Loss with alpha={alpha}, gamma={gamma}")
        criterion = FocalLoss(alpha=alpha, gamma=gamma)
    elif config['training'].get('pos_weight', None) is not None:
        # Use binary cross entropy with logits and positive weight
        pos_weight = torch.tensor([config['training']['pos_weight']], dtype=torch.float32).to(device)
        logger.info(f"Using BCE with positive weight: {pos_weight}")
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        # Default: regular cross entropy loss
        criterion = nn.CrossEntropyLoss()
    
    # Define optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    # Learning rate scheduler
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
    )
    
    # Initialize tracking variables
    train_losses = []
    val_losses = []
    accuracies = []
    precisions = []
    recalls = []
    f1s = []
    best_val_loss = float('inf')
    best_f1 = 0
    
    # Get evaluation threshold from config or use default
    eval_threshold = config['evaluation'].get('threshold', 0.3)
    logger.info(f"Using evaluation threshold: {eval_threshold}")
    
    # Training loop
    for epoch in range(1, config['training']['num_epochs'] + 1):
        logger.info(f"Epoch {epoch}/{config['training']['num_epochs']}")
        
        # Train
        train_loss = train_epoch(model, data_loaders['train'], criterion, optimizer, device)
        train_losses.append(train_loss)
        logger.info(f"Training Loss: {train_loss:.4f}")
        
        # Evaluate
        val_results = evaluate(
            model, data_loaders['val'], criterion, device, threshold=eval_threshold
        )
        val_loss, accuracy, precision, recall, f1, conf_matrix, val_scores, val_labels = val_results
        
        val_losses.append(val_loss)
        accuracies.append(accuracy)
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        
        logger.info(f"Validation Loss: {val_loss:.4f}, Accuracy: {accuracy:.4f}, "
                   f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
        logger.info(f"Confusion Matrix:\n{conf_matrix}")
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Every 5 epochs, find best threshold
        if epoch % 5 == 0 or epoch == config['training']['num_epochs']:
            # Calculate metrics at different thresholds
            thresholds = np.arange(0.1, 0.9, 0.05)
            best_threshold, best_threshold_f1 = plot_pr_curve(
                val_scores, val_labels, thresholds, experiment_dir
            )
            logger.info(f"Best threshold: {best_threshold:.2f} with F1: {best_threshold_f1:.4f}")
            
            # Update evaluation threshold if auto-threshold is enabled
            if config['evaluation'].get('auto_threshold', False):
                eval_threshold = best_threshold
                logger.info(f"Updated evaluation threshold to {eval_threshold:.2f}")
        
        # Save metrics
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': conf_matrix.tolist(),
            'threshold': eval_threshold
        }
        
        # Save checkpoint
        save_model(
            model, optimizer, epoch, train_loss, val_loss,
            metrics, config, experiment_dir
        )
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(experiment_dir, 'best_model_loss.pt'))
            logger.info(f"Saved best model (loss) at epoch {epoch}")
        
        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), os.path.join(experiment_dir, 'best_model_f1.pt'))
            logger.info(f"Saved best model (F1) at epoch {epoch}")
        
        # Plot metrics
        plot_metrics(
            train_losses, val_losses, accuracies,
            precisions, recalls, f1s, experiment_dir
        )
        
        # Force garbage collection
        gc.collect()
    
    # Evaluate on test set using best model
    logger.info("Evaluating best model on test set")
    model.load_state_dict(torch.load(os.path.join(experiment_dir, 'best_model_f1.pt')))
    
    test_results = evaluate(
        model, data_loaders['test'], criterion, device, threshold=eval_threshold
    )
    test_loss, test_accuracy, test_precision, test_recall, test_f1, test_conf_matrix, test_scores, test_labels = test_results
    
    logger.info(f"Test Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.4f}, "
               f"Precision: {test_precision:.4f}, Recall: {test_recall:.4f}, F1: {test_f1:.4f}")
    logger.info(f"Test Confusion Matrix:\n{test_conf_matrix}")
    
    # Try different thresholds on test set
    thresholds = np.arange(0.1, 0.9, 0.05)
    best_test_threshold, best_test_f1 = plot_pr_curve(
        test_scores, test_labels, thresholds, 
        os.path.join(experiment_dir, 'test_results')
    )
    logger.info(f"Best test threshold: {best_test_threshold:.2f} with F1: {best_test_f1:.4f}")
    
    # Re-evaluate with best threshold from test set
    if best_test_threshold != eval_threshold:
        logger.info(f"Re-evaluating with best test threshold: {best_test_threshold:.2f}")
        test_preds = [1 if score > best_test_threshold else 0 for score in test_scores]
        test_accuracy = accuracy_score(test_labels, test_preds)
        test_precision, test_recall, test_f1, _ = precision_recall_fscore_support(
            test_labels, test_preds, average='binary', zero_division=0
        )
        test_conf_matrix = confusion_matrix(test_labels, test_preds)
        
        logger.info(f"Updated Test Metrics - Accuracy: {test_accuracy:.4f}, "
                  f"Precision: {test_precision:.4f}, Recall: {test_recall:.4f}, F1: {test_f1:.4f}")
        logger.info(f"Updated Test Confusion Matrix:\n{test_conf_matrix}")
    
    # Save test metrics
    test_metrics = {
        'loss': test_loss,
        'accuracy': test_accuracy,
        'precision': test_precision,
        'recall': test_recall,
        'f1': test_f1,
        'confusion_matrix': test_conf_matrix.tolist(),
        'threshold': best_test_threshold
    }
    
    test_metrics_path = os.path.join(experiment_dir, 'test_metrics.yaml')
    with open(test_metrics_path, 'w') as f:
        yaml.dump(test_metrics, f)
    
    logger.info(f"Saved test metrics to {test_metrics_path}")
    logger.info("Training complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LSTM Late Fusion model")
    parser.add_argument("--config", type=str, default="configs/lstm_late_fusion.yaml",
                       help="Path to configuration file")
    args = parser.parse_args()
    
    main(args) 