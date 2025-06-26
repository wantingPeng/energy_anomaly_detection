"""
Training script for LSTM Sequence model.

This script trains an LSTM model that processes time series data 
for energy anomaly detection, making predictions for each time point in the window.
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
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix, f1_score
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse

from src.utils.logger import logger
from src.training.lstm.focal_loss import FocalLoss
from src.training.row_energyData_subsample_LSTMLateFusion.lstm_late_fusion_model import LSTMSequenceModel
from src.training.row_energyData_subsample_LSTMLateFusion.lstm_late_fusion_dataset import create_data_loaders

def load_config(config_path=None):
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    if config_path is None:
        config_path = Path("configs/lstm_sequence.yaml")
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    logger.info(f"Loaded configuration from {config_path}")
    return config


class EarlyStopping:
    """
    Early stopping to prevent overfitting.
    
    Args:
        patience: Number of epochs to wait after last improvement
        min_delta: Minimum change to qualify as an improvement
        mode: 'min' for loss, 'max' for metrics like F1
    """
    def __init__(self, patience=10, min_delta=0.0001, mode='min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, current_score):
        if self.best_score is None:
            self.best_score = current_score
            return False
        
        if self.mode == 'min':
            # For metrics like loss where lower is better
            if current_score < self.best_score - self.min_delta:
                self.best_score = current_score
                self.counter = 0
            else:
                self.counter += 1
        else:
            # For metrics like F1 where higher is better
            if current_score > self.best_score + self.min_delta:
                self.best_score = current_score
                self.counter = 0
            else:
                self.counter += 1
        
        if self.counter >= self.patience:
            self.early_stop = True
            
        return self.early_stop


def train(model, data_loader, criterion, optimizer, device, threshold=0.3, epoch=None):
    """
    Train the model for one epoch.
    
    Args:
        model: LSTM Sequence model
        data_loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to use for training
        threshold: Classification threshold for positive class (anomaly)
        epoch: Current epoch number
        
    Returns:
        Tuple of (average loss, accuracy, precision, recall, f1, confusion matrix)
    """
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    for windows, label_sequences in tqdm(data_loader, desc="Training"):
        # Move data to device
        windows = windows.to(device)  # [batch_size, seq_len, input_size]
        label_sequences = label_sequences.to(device)  # [batch_size, seq_len]
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(windows)  # [batch_size, seq_len, output_size]
        
        # Reshape for loss calculation
        batch_size, seq_len, output_size = outputs.shape
        outputs_flat = outputs.reshape(-1, output_size)  # [batch_size*seq_len, output_size]
        labels_flat = label_sequences.reshape(-1)  # [batch_size*seq_len]
        
        # Calculate loss
        loss = criterion(outputs_flat, labels_flat)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # Get probabilities and predictions with custom threshold
        probs = torch.softmax(outputs_flat, dim=1)
        anomaly_scores = probs[:, 1]  # Probability for anomaly class
        preds_flat = (anomaly_scores > threshold).long()
        
        # Store predictions and labels for metric calculation
        all_preds.extend(preds_flat.cpu().detach().numpy())
        all_labels.extend(labels_flat.cpu().detach().numpy())
    
    # Calculate metrics
    avg_loss = total_loss / len(data_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary', zero_division=0)
    conf_matrix = confusion_matrix(all_labels, all_preds)
    
    return avg_loss, accuracy, precision, recall, f1, conf_matrix


def evaluate(model, data_loader, criterion, device, epoch=None, find_optimal_threshold=True):
    """
    Evaluate the model on validation or test data.
    
    Args:
        model: LSTM Sequence model
        data_loader: Validation or test data loader
        criterion: Loss function
        device: Device to use for evaluation
        epoch: Current epoch number
        find_optimal_threshold: Whether to find the optimal threshold based on F1 score
        
    Returns:
        Tuple of (average loss, accuracy, precision, recall, f1, confusion matrix, optimal_threshold)
    """
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_scores = []
    
    with torch.no_grad():
        for windows, label_sequences in tqdm(data_loader, desc="Evaluating"):
            # Move data to device
            windows = windows.to(device)  # [batch_size, seq_len, input_size]
            label_sequences = label_sequences.to(device)  # [batch_size, seq_len]
            
            # Forward pass
            outputs = model(windows)  # [batch_size, seq_len, output_size]
            
            # Reshape for loss calculation
            batch_size, seq_len, output_size = outputs.shape
            outputs_flat = outputs.reshape(-1, output_size)  # [batch_size*seq_len, output_size]
            labels_flat = label_sequences.reshape(-1)  # [batch_size*seq_len]
            
            # Calculate loss
            loss = criterion(outputs_flat, labels_flat)
            total_loss += loss.item()
            
            # Get probabilities
            probs = torch.softmax(outputs_flat, dim=1)
            anomaly_scores = probs[:, 1]  # Probability for anomaly class
            
            # Store scores and labels
            all_scores.extend(anomaly_scores.cpu().numpy())
            all_labels.extend(labels_flat.cpu().numpy())
    
    # Convert to numpy arrays for easier processing
    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)
    
    # Find optimal threshold based on F1 score if requested
    threshold = 0.5  # Default threshold
    if find_optimal_threshold:
        thresholds = np.linspace(0.01, 0.99, 99)  # Test 99 threshold values
        f1_scores = []
        
        for thresh in thresholds:
            temp_preds = (all_scores > thresh).astype(int)
            f1 = f1_score(all_labels, temp_preds, zero_division=0)
            f1_scores.append(f1)
        
        # Find the threshold that maximizes F1 score
        best_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds[best_idx]
        
        # Use the optimal threshold
        threshold = optimal_threshold
        
    # Calculate final predictions using the threshold
    all_preds = (all_scores > threshold).astype(int)
    
    # Calculate metrics
    avg_loss = total_loss / len(data_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary', zero_division=0)
    conf_matrix = confusion_matrix(all_labels, all_preds)
    
    return avg_loss, accuracy, precision, recall, f1, conf_matrix, threshold


def load_model(model, optimizer, checkpoint_path, device):
    """
    Load model checkpoint for continuing training or evaluation.
    
    Args:
        model: LSTM Sequence model instance
        optimizer: Optimizer instance (can be None if just for evaluation)
        checkpoint_path: Path to the checkpoint file
        device: Device to load the model to
        
    Returns:
        Loaded model
    """
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    # Load model state
    model.load_state_dict(checkpoint)
    
    logger.info(f"Loaded model successfully")
    
    return model

def main(args):
    """
    Main function to train the LSTM Sequence model.
    
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
    experiment_name = f"lstm_sequence_{timestamp}"
    if args.experiment_name:
        experiment_name = args.experiment_name
    
    experiment_dir = os.path.join(config['paths']['output_dir'], 'model_save', experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)
    
    # Set up logging
    log_file = os.path.join("experiments/logs", f"lstm_sequence_training_{timestamp}.log")
    
    # Set up TensorBoard writer
    tensorboard_dir = os.path.join("experiments/lstm_sequence/tensorboard", experiment_name)
    os.makedirs(tensorboard_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=tensorboard_dir)
    logger.info(f"TensorBoard logs will be saved to {tensorboard_dir}")
    
    # Create data loaders
    data_loaders = create_data_loaders(
        lstm_data_dir=config['paths']['lstm_data_dir'],
        batch_size=config['training']['batch_size'],
        num_workers=config['training']['num_workers'],
        component=config['data']['component']
    )
    
    # Create model
    model = LSTMSequenceModel(config=config['model'])
    model.to(device)
    
    # Define optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    # Initialize tracking variables
    start_epoch = 1
    train_losses = []
    val_losses = []
    accuracies = []
    precisions = []
    recalls = []
    f1s = []
    best_val_loss = float('inf')
    best_f1 = 0
    
    # Load existing model if specified
    if args.load_model:
        model = load_model(model, None, args.load_model, device)

    # Define loss function based on config
    if config['training'].get('use_focal_loss', True):
        # Use Focal Loss
        alpha = config['training'].get('focal_loss_alpha', 0.25)
        gamma = config['training'].get('focal_loss_gamma', 2.0)
        logger.info(f"Using Focal Loss with alpha={alpha}, gamma={gamma}")
        criterion = FocalLoss(alpha=alpha, gamma=gamma)
    else:
        criterion = nn.CrossEntropyLoss()
 
    logger.info(f"Using criterion: {criterion}")
    
    # Learning rate scheduler
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
    )
    
    # Get evaluation threshold from config or use default
    eval_threshold = config['evaluation'].get('threshold', 0.3)
    logger.info(f"Using evaluation threshold: {eval_threshold}")
    
    # Log model architecture and hyperparameters to TensorBoard
    writer.add_text('Model/Architecture', str(model), 0)
    writer.add_text('Hyperparameters/Learning_Rate', str(config['training']['learning_rate']), 0)
    writer.add_text('Hyperparameters/Batch_Size', str(config['training']['batch_size']), 0)
    writer.add_text('Hyperparameters/Weight_Decay', str(config['training']['weight_decay']), 0)
    
    # Training loop
    for epoch in range(start_epoch, config['training']['num_epochs'] + 1):
        logger.info(f"Epoch {epoch}/{config['training']['num_epochs']}")
        
        # Train
        train_results = train(model, data_loaders['train'], criterion, optimizer, device, threshold=eval_threshold, epoch=epoch)
        train_loss, train_accuracy, train_precision, train_recall, train_f1, train_conf_matrix = train_results
        train_losses.append(train_loss)
        
        # Log training metrics
        logger.info(f"Training Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.4f}, "
                   f"Precision: {train_precision:.4f}, Recall: {train_recall:.4f}, F1: {train_f1:.4f}")
        
        # Log training metrics to TensorBoard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Metrics/train_accuracy', train_accuracy, epoch)
        writer.add_scalar('Metrics/train_precision', train_precision, epoch)
        writer.add_scalar('Metrics/train_recall', train_recall, epoch)
        writer.add_scalar('Metrics/train_f1', train_f1, epoch)
        
        # Evaluate
        val_results = evaluate(
            model, data_loaders['val'], criterion, device, epoch=epoch
        )
        val_loss, accuracy, precision, recall, f1, conf_matrix, threshold = val_results
        
        val_losses.append(val_loss)
        accuracies.append(accuracy)
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        
        logger.info(f"Validation Loss: {val_loss:.4f}, Accuracy: {accuracy:.4f}, "
                   f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
        logger.info(f"Confusion Matrix:\n{conf_matrix}")
        logger.info(f"Current best Threshold: {threshold}")
        
        # Log validation metrics to TensorBoard
        writer.add_scalar('Loss/validation', val_loss, epoch)
        writer.add_scalar('Metrics/val_accuracy', accuracy, epoch)
        writer.add_scalar('Metrics/val_precision', precision, epoch)
        writer.add_scalar('Metrics/val_recall', recall, epoch)
        writer.add_scalar('Metrics/val_f1', f1, epoch)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Log learning rate to TensorBoard
        current_lr = optimizer.param_groups[0]['lr']
        logger.info(f"Current learning rate: {current_lr}")
        writer.add_scalar('Learning_Rate', current_lr, epoch)
        
        # Save metrics
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': conf_matrix.tolist(),
            'threshold': threshold,
            'train_accuracy': train_accuracy,
            'train_precision': train_precision,
            'train_recall': train_recall,
            'train_f1': train_f1,
            'train_confusion_matrix': train_conf_matrix.tolist()
        }
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = os.path.join(experiment_dir, 'best_model_loss.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'metrics': metrics,
                'config': config
            }, best_model_path)
            logger.info(f"Saved best model (loss) at epoch {epoch}")
        
        if f1 > best_f1:
            best_f1 = f1
            best_model_path = os.path.join(experiment_dir, 'best_model_f1.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'metrics': metrics,
                'config': config
            }, best_model_path)
            logger.info(f"Saved best model (F1) at epoch {epoch}")
        
        # Force garbage collection
        gc.collect()
    
    # Close TensorBoard writer
    writer.close()
    
    # Evaluate on test set using best model
    if args.evaluate_test:
        logger.info("Evaluating best model on test set")
        
        # Load best model by F1 score
        best_model_path = os.path.join(experiment_dir, 'best_model_f1.pt')
        model = load_model(model, None, best_model_path, device)
        
        test_results = evaluate(
            model, data_loaders['test'], criterion, device, threshold=threshold
        )
        test_loss, test_accuracy, test_precision, test_recall, test_f1, test_conf_matrix, _ = test_results
        
        logger.info(f"Test Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.4f}, "
                   f"Precision: {test_precision:.4f}, Recall: {test_recall:.4f}, F1: {test_f1:.4f}")
        logger.info(f"Test Confusion Matrix:\n{test_conf_matrix}")
        
        # Save test metrics
        test_metrics = {
            'loss': test_loss,
            'accuracy': test_accuracy,
            'precision': test_precision,
            'recall': test_recall,
            'f1': test_f1,
            'confusion_matrix': test_conf_matrix.tolist(),
            'threshold': threshold
        }
        
        test_metrics_path = os.path.join(experiment_dir, 'test_metrics.yaml')
        with open(test_metrics_path, 'w') as f:
            yaml.dump(test_metrics, f)
        
        logger.info(f"Saved test metrics to {test_metrics_path}")
    
    logger.info("Training complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LSTM Sequence model")
    parser.add_argument("--config", type=str, default="configs/lstm_sequence.yaml",
                       help="Path to configuration file")
    parser.add_argument("--load_model", type=str, default=None,
                       help="Path to pretrained model checkpoint to continue training")
    parser.add_argument("--experiment_name", type=str, default=None,
                       help="Custom experiment name for output directory")
    parser.add_argument("--evaluate_test", action="store_true",
                       help="Evaluate the best model on test set after training")
    args = parser.parse_args()
    
    main(args) 