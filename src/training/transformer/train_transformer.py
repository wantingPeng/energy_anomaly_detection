"""
Training script for Transformer model for energy anomaly detection.

This script trains a Transformer model using PyTorch's nn.TransformerEncoder
for energy anomaly detection.
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
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import argparse
import time

from src.utils.logger import logger
from src.training.transformer.transformer_model import TransformerModel
from src.training.transformer.transformer_dataset import create_data_loaders

# Define focal loss for imbalanced dataset
class FocalLoss(nn.Module):
    """
    Focal Loss for imbalanced datasets.
    
    Args:
        alpha: Weight for the rare class
        gamma: Focusing parameter
        reduction: Reduction method ('mean', 'sum', or 'none')
    """
    def __init__(self, alpha=0.25, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        BCE_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        
        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss

def load_config(config_path=None):
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    if config_path is None:
        config_path = Path("configs/transformer.yaml")
    
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

def save_model(model, optimizer, epoch, train_loss, val_loss, metrics, config, save_dir):
    """
    Save model checkpoint.
    
    Args:
        model: Transformer model
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

def load_model(model, optimizer, checkpoint_path, device):
    """
    Load model checkpoint for continuing training or evaluation.
    
    Args:
        model: Transformer model instance
        optimizer: Optimizer instance (can be None if just for evaluation)
        checkpoint_path: Path to the checkpoint file
        device: Device to load the model to
        
    Returns:
        Tuple of (model, optimizer, start_epoch, checkpoint_data)
    """
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load optimizer state if provided
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Get starting epoch (checkpoint's epoch + 1)
    start_epoch = checkpoint.get('epoch', 0) + 1
    
    logger.info(f"Loaded model from epoch {checkpoint.get('epoch', 0)}")
    
    return model, optimizer, start_epoch, checkpoint

def train_epoch(model, data_loader, optimizer, criterion, device, scheduler=None):
    """
    Train the model for one epoch.
    
    Args:
        model: Transformer model
        data_loader: DataLoader for training data
        optimizer: Optimizer
        criterion: Loss function
        device: Device to use for training
        scheduler: Learning rate scheduler
        
    Returns:
        Average training loss for the epoch
    """
    model.train()
    total_loss = 0
    all_labels = []
    all_preds = []
    
    # Progress bar
    pbar = tqdm(data_loader, desc="Training")
    
    for batch_idx, (data, labels) in enumerate(pbar):
        # Move data to device
        data, labels = data.to(device), labels.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(data)
        
        # Calculate loss
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Update weights
        optimizer.step()
        
        # Update learning rate if scheduler is provided
        if scheduler is not None and not isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step()
        
        # Calculate metrics
        _, predicted = torch.max(outputs.data, 1)
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(predicted.cpu().numpy())
        
        # Update total loss
        total_loss += loss.item()
        
        # Update progress bar
        pbar.set_postfix({'loss': loss.item()})
    
    # Calculate average loss
    avg_loss = total_loss / len(data_loader)
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='binary', zero_division=0
    )
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }
    
    return avg_loss, metrics

def evaluate(model, data_loader, criterion, device):
    """
    Evaluate the model on validation or test data.
    
    Args:
        model: Transformer model
        data_loader: DataLoader for validation/test data
        criterion: Loss function
        device: Device to use for evaluation
        
    Returns:
        Tuple of (average loss, metrics)
    """
    model.eval()
    total_loss = 0
    all_labels = []
    all_preds = []
    all_scores = []
    
    with torch.no_grad():
        for data, labels in tqdm(data_loader, desc="Evaluating"):
            # Move data to device
            data, labels = data.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(data)
            
            # Calculate loss
            loss = criterion(outputs, labels)
            
            # Calculate metrics
            scores = torch.softmax(outputs, dim=1)[:, 1]
            _, predicted = torch.max(outputs.data, 1)
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_scores.extend(scores.cpu().numpy())
            
            # Update total loss
            total_loss += loss.item()
    
    # Calculate average loss
    avg_loss = total_loss / len(data_loader)
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='binary', zero_division=0
    )
    
    # Calculate confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm,
        'predictions': all_preds,
        'labels': all_labels,
        'scores': all_scores
    }
    
    return avg_loss, metrics


def main(args):
    """
    Main function to train the Transformer model.
    
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
    experiment_name = f"transformer_{timestamp}"
    if args.experiment_name:
        experiment_name = args.experiment_name
    
    experiment_dir = os.path.join(config['paths']['output_dir'], experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)
    
    # Set up TensorBoard writer
    tensorboard_dir = os.path.join("experiments/transformer/tensorboard", experiment_name)
    os.makedirs(tensorboard_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=tensorboard_dir)
    logger.info(f"TensorBoard logs will be saved to {tensorboard_dir}")
    
    # Create data loaders
    data_loaders = create_data_loaders(
        data_dir=config['paths']['data_dir'],
        batch_size=config['training']['batch_size'],
        num_workers=config['training']['num_workers'],
        component=config['data']['component']
    )
    
    # Get a sample batch to determine input dimension
    sample_batch, _ = next(iter(data_loaders['train']))
    input_dim = sample_batch.shape[2]  # [batch_size, seq_len, input_dim]
    seq_len = sample_batch.shape[1]
    
    logger.info(f"Input dimension: {input_dim}")
    logger.info(f"Sequence length: {seq_len}")
    
    # Create model
    model = TransformerModel(
        input_dim=input_dim,
        d_model=config['model']['d_model'],
        nhead=config['model']['nhead'],
        num_layers=config['model']['num_layers'],
        dim_feedforward=config['model']['dim_feedforward'],
        dropout=config['model']['dropout'],
        num_classes=config['model']['num_classes'],
        activation=config['model']['activation']
    )
    
    # Log model architecture and parameter count
    logger.info(f"Model architecture:\n{model}")
    
    # Move model to device
    model = model.to(device)
    
    # Define loss function (with class weighting if needed)
    if config['training']['use_focal_loss']:
        criterion = FocalLoss(
            alpha=config['training']['focal_loss_alpha'],
            gamma=config['training']['focal_loss_gamma']
        )
        logger.info(f"Using Focal Loss with alpha={config['training']['focal_loss_alpha']}, gamma={config['training']['focal_loss_gamma']}")
    else:
        # Calculate class weights if needed
        if config['training']['use_class_weights']:
            # Get class distribution from training data
            all_labels = []
            for _, labels in data_loaders['train'].dataset:
                all_labels.append(labels)
            all_labels = torch.stack(all_labels)
            
            # Calculate class weights
            class_counts = torch.bincount(all_labels)
            class_weights = 1.0 / class_counts.float()
            class_weights = class_weights / class_weights.sum()
            
            logger.info(f"Class weights: {class_weights}")
            criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
        else:
            criterion = nn.CrossEntropyLoss()
    
    # Define optimizer
    if config['training']['optimizer'] == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )
    elif config['training']['optimizer'] == 'adamw':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )
    else:
        optimizer = optim.SGD(
            model.parameters(),
            lr=config['training']['learning_rate'],
            momentum=config['training']['momentum'],
            weight_decay=config['training']['weight_decay']
        )
    
    # Define learning rate scheduler
    if config['training']['lr_scheduler'] == 'reduce_on_plateau':
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=config['training']['lr_reduce_factor'],
            patience=config['training']['lr_reduce_patience'],
        )
    elif config['training']['lr_scheduler'] == 'cosine_annealing':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config['training']['num_epochs'],
            eta_min=config['training']['min_lr']
        )
    else:
        scheduler = None
    
    # Resume training from checkpoint if provided
    start_epoch = 0
    if args.load_model:
        model, optimizer, start_epoch, _ = load_model(
            model, optimizer, args.load_model, device
        )
    
    # Set up early stopping
    early_stopping = EarlyStopping(
        patience=config['training']['early_stopping_patience'],
        min_delta=config['training']['early_stopping_min_delta'],
        mode='max' if config['training']['early_stopping_metric'] == 'f1' else 'min'
    )
    
    # Initialize best metrics
    best_val_loss = float('inf')
    best_val_f1 = 0.0
    
    # Training loop
    for epoch in range(start_epoch, config['training']['num_epochs']):
        logger.info(f"Epoch {epoch+1}/{config['training']['num_epochs']}")
        
        # Record epoch start time
        epoch_start_time = time.time()
        
        # Train for one epoch
        train_loss, train_metrics = train_epoch(
            model, data_loaders['train'], optimizer, criterion, device
        )
        
        # Evaluate on validation set
        val_loss, val_metrics = evaluate(
            model, data_loaders['val'], criterion, device
        )
        
        # Update learning rate scheduler if using ReduceLROnPlateau
        if isinstance(scheduler, ReduceLROnPlateau):
            if config['training']['early_stopping_metric'] == 'f1':
                scheduler.step(val_metrics['f1'])
            else:
                scheduler.step(val_loss)
        elif scheduler is not None and not isinstance(scheduler, optim.lr_scheduler.CosineAnnealingLR):
            scheduler.step()
        
        # Calculate epoch duration
        epoch_duration = time.time() - epoch_start_time
        
        # Log metrics
        logger.info(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        logger.info(f"Train F1: {train_metrics['f1']:.4f}, Val F1: {val_metrics['f1']:.4f}")
        logger.info(f"Val Precision: {val_metrics['precision']:.4f}, Val Recall: {val_metrics['recall']:.4f}")
        logger.info(f"Epoch completed in {epoch_duration:.2f} seconds")
        
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('train/f1', train_metrics['f1'], epoch)
        writer.add_scalar('val/f1', val_metrics['f1'], epoch)
        writer.add_scalar('train/precision', train_metrics['precision'], epoch)
        writer.add_scalar('val/precision', val_metrics['precision'], epoch)
        writer.add_scalar('train/recall', train_metrics['recall'], epoch)
        writer.add_scalar('val/recall', val_metrics['recall'], epoch)
        
        # Save current model checkpoint
        save_model(
            model, optimizer, epoch, train_loss, val_loss, val_metrics,
            config, os.path.join(experiment_dir, "checkpoints")
        )
        
        # Save best model based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_model(
                model, optimizer, epoch, train_loss, val_loss, val_metrics,
                config, os.path.join(experiment_dir, "best_loss")
            )
            logger.info(f"New best validation loss: {best_val_loss:.4f}")
        
        # Save best model based on F1 score
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            save_model(
                model, optimizer, epoch, train_loss, val_loss, val_metrics,
                config, os.path.join(experiment_dir, "best_f1")
            )
            logger.info(f"New best validation F1: {best_val_f1:.4f}")
        
        # Check for early stopping
        if config['training']['early_stopping_metric'] == 'f1':
            if early_stopping(val_metrics['f1']):
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break
        else:
            if early_stopping(val_loss):
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break
    
    # Evaluate on test set using best model (based on F1)
    logger.info("Evaluating best model (based on F1) on test set")
    best_model_path = os.path.join(experiment_dir, "best_f1", f"checkpoint_epoch_{epoch}.pt")
    if os.path.exists(best_model_path):
        model, _, _, _ = load_model(model, None, best_model_path, device)
        test_loss, test_metrics = evaluate(model, data_loaders['test'], criterion, device)
        
        logger.info(f"Test Loss: {test_loss:.4f}")
        logger.info(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
        logger.info(f"Test Precision: {test_metrics['precision']:.4f}")
        logger.info(f"Test Recall: {test_metrics['recall']:.4f}")
        logger.info(f"Test F1: {test_metrics['f1']:.4f}")
        
        # Log test metrics to TensorBoard
        writer.add_scalar('test/loss', test_loss, 0)
    
    # Close TensorBoard writer
    writer.close()
    
    logger.info("Training completed successfully")

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train Transformer model for energy anomaly detection")
    parser.add_argument("--config", type=str, default=None, help="Path to configuration file")
    parser.add_argument("--experiment_name", type=str, default=None, help="Name of the experiment")
    parser.add_argument("--test", action="store_true", help="Test the model")
    parser.add_argument("--load_model", type=str, default=None, help="Path to checkpoint to load model from")
    args = parser.parse_args()
    
    # Run main function
    main(args)