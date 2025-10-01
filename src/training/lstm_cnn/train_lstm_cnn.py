"""
Training script for LSTM-CNN model for energy anomaly detection.

This script trains a hybrid LSTM-CNN model using PyTorch for energy anomaly 
detection with per-timestep predictions.
"""

import os
import gc
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from sklearn.metrics import (
    precision_recall_fscore_support, 
    accuracy_score, 
    confusion_matrix, 
    precision_recall_curve, 
    f1_score, 
    auc, 
    average_precision_score
)
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import argparse
import time
import sys

# Add project root to path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from src.utils.logger import logger
from src.training.lstm_cnn.lstm_cnn_model import LSTMCNN, SimpleLSTMCNN
from src.preprocessing.row_energyData_subsample_Transform.dataloader import create_data_loaders


def point_adjustment(gt, pred):
    """
    Point adjustment strategy for anomaly detection evaluation.
    
    This function adjusts predictions based on the principle that if any point
    in a true anomaly segment is detected, the entire segment should be considered
    as detected. This makes evaluation more practical for real-world applications.
    
    Args:
        gt: Ground truth labels (numpy array)
        pred: Predicted labels (numpy array)
    
    Returns:
        Tuple of (adjusted_gt, adjusted_pred)
    """
    gt = gt.copy()
    pred = pred.copy()
    anomaly_state = False
    
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            # Adjust backward: mark entire anomaly segment as detected
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            # Adjust forward: mark entire anomaly segment as detected
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
    
    return gt, pred


def evaluate_with_adjustment(flat_preds, flat_labels):
    """
    Evaluate predictions with point adjustment.
    
    Args:
        flat_preds: Flattened predictions
        flat_labels: Flattened ground truth labels
    
    Returns:
        Dictionary of adjusted metrics
    """
    flat_labels, adjusted_preds = point_adjustment(flat_labels, flat_preds)

    original_anomaly_count = np.sum(flat_preds == 1)
    adjusted_anomaly_count = np.sum(adjusted_preds == 1)
    logger.info(f"Original anomaly count: {original_anomaly_count}, "
                f"Adjusted: {adjusted_anomaly_count}, "
                f"Diff: {adjusted_anomaly_count - original_anomaly_count}")
  
    # Calculate metrics with adjusted predictions
    adj_accuracy = accuracy_score(flat_labels, adjusted_preds)
    adj_precision, adj_recall, adj_f1, _ = precision_recall_fscore_support(
        flat_labels, adjusted_preds, average='binary', zero_division=0
    )

    adjusted_preds_count = np.sum(adjusted_preds == 1)
    
    return {
        'adj_accuracy': adj_accuracy,
        'adj_precision': adj_precision,
        'adj_recall': adj_recall,
        'adj_f1': adj_f1,
        'adjusted_preds_count': adjusted_preds_count
    }


class FocalLoss(nn.Module):
    """
    Focal Loss for imbalanced datasets.
    
    Args:
        alpha: Weight for the rare class
        gamma: Focusing parameter
        reduction: Reduction method ('mean', 'sum', or 'none')
    """
    def __init__(self, alpha=0.75, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        """
        Calculate focal loss.
        
        Args:
            inputs: Predictions tensor of shape [batch_size, seq_len, num_classes]
            targets: Target tensor of shape [batch_size, seq_len]
            
        Returns:
            Focal loss value
        """
        # Reshape inputs and targets
        if inputs.dim() > 2:
            batch_size, seq_len, num_classes = inputs.size()
            inputs = inputs.reshape(-1, num_classes)
            targets = targets.reshape(-1)
        
        BCE_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        
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
        config_path = Path("configs/lstm_cnn_config.yaml")
    
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
    def __init__(self, patience=10, min_delta=0.0001, mode='max'):
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
        model: LSTM-CNN model
        optimizer: Optimizer
        epoch: Current epoch
        train_loss: Training loss
        val_loss: Validation loss
        metrics: Dictionary of metrics
        config: Configuration dictionary
        save_dir: Directory to save the checkpoint
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Save config as a separate YAML file
    config_dir = os.path.join(save_dir, "config")
    os.makedirs(config_dir, exist_ok=True)
    config_path = os.path.join(config_dir, "config.yaml")
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    logger.info(f"Saved config to {config_path}")
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'metrics': metrics
    }
    
    checkpoint_path = os.path.join(save_dir, "best_model.pt")
    torch.save(checkpoint, checkpoint_path)
    logger.info(f"Saved checkpoint to {checkpoint_path}")


def load_model(model, optimizer, checkpoint_path, device):
    """
    Load model checkpoint.
    
    Args:
        model: Model instance
        optimizer: Optimizer instance
        checkpoint_path: Path to checkpoint file
        device: Device to load model to
        
    Returns:
        Tuple of (model, optimizer, start_epoch, checkpoint)
    """
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    start_epoch = checkpoint.get('epoch', 0) + 1
    
    logger.info(f"Loaded model from epoch {checkpoint.get('epoch', 0)}")
    
    return model, optimizer, start_epoch, checkpoint


def train_epoch(model, data_loader, optimizer, criterion, device, scheduler=None):
    """
    Train the model for one epoch.

    Args:
        model: LSTM-CNN model
        data_loader: DataLoader for training data
        optimizer: Optimizer
        criterion: Loss function
        device: Device to use for training
        scheduler: Learning rate scheduler

    Returns:
        Average training loss and metrics dict
    """
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []

    pbar = tqdm(data_loader, desc="Training")

    for batch_idx, (data, labels) in enumerate(pbar):
        data, labels = data.to(device), labels.to(device)
        data = data.float()

        optimizer.zero_grad()
        
        # Forward pass
        logits = model(data)
        loss = criterion(logits, labels)
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if scheduler is not None and not isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step()

        preds = torch.argmax(logits, dim=2)  # [batch_size, seq_len]
        
        all_preds.append(preds.detach().cpu())
        all_labels.append(labels.cpu())

        total_loss += loss.item()
        pbar.set_postfix({'loss': loss.item()})

    all_preds = torch.cat(all_preds, dim=0).numpy().flatten()
    all_labels = torch.cat(all_labels, dim=0).numpy().flatten()
    
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

    avg_loss = total_loss / len(data_loader)
    return avg_loss, metrics


def evaluate(model, data_loader, criterion, device, config, print_samples=True):
    """
    Evaluate the model on validation or test data.
    
    Args:
        model: LSTM-CNN model
        data_loader: DataLoader for validation or test data
        criterion: Loss function
        device: Device to use for evaluation
        config: Configuration dictionary
        print_samples: Whether to print sample predictions
    
    Returns:
        Average loss and metrics dictionary
    """
    model.eval()
    total_loss = 0
    all_logits = []
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch_idx, (data, labels) in enumerate(tqdm(data_loader, desc="Evaluating")):
            data, labels = data.to(device), labels.to(device)
            data = data.float()

            # Forward pass
            logits = model(data)
            loss = criterion(logits, labels)
            total_loss += loss.item()

            probs = torch.softmax(logits, dim=2)
            preds = torch.argmax(logits, dim=2)

            all_logits.append(logits.cpu())
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

    avg_loss = total_loss / len(data_loader)

    all_logits = torch.cat(all_logits, dim=0)
    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()
    
    flat_preds = all_preds.flatten()
    flat_labels = all_labels.flatten()
    
    all_probs = torch.softmax(all_logits, dim=2)[:, :, 1].numpy()
    flat_probs = all_probs.flatten()
    
    # Original point-wise evaluation
    auprc = average_precision_score(flat_labels, flat_probs)
    
    precision_curve, recall_curve, thresholds = precision_recall_curve(flat_labels, flat_probs)
    f1_scores = 2 * precision_curve * recall_curve / (precision_curve + recall_curve + 1e-10)
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
    
    optimal_preds = (flat_probs >= optimal_threshold).astype(int)
    optimal_accuracy = accuracy_score(flat_labels, optimal_preds)
    optimal_precision, optimal_recall, optimal_f1, _ = precision_recall_fscore_support(
        flat_labels, optimal_preds, average='binary', zero_division=0
    )
    
    # Point adjustment evaluation
    logger.info(f"\n===== Before Point Adjustment =====")
    logger.info(f"Using optimal_preds for point adjustment (threshold: {optimal_threshold:.6f})")
    
    adj_metrics = evaluate_with_adjustment(optimal_preds, flat_labels)
    
    logger.info(f"\n===== Original Point-wise Evaluation =====")
    logger.info(f"Optimal threshold: {optimal_threshold:.6f}")
    logger.info(f"Optimal F1: {optimal_f1:.4f}, Precision: {optimal_precision:.4f}, Recall: {optimal_recall:.4f}")
    logger.info(f"AUPRC: {auprc:.4f}")
    
    logger.info(f"\n===== Point Adjustment Evaluation =====")
    logger.info(f"Adjusted F1: {adj_metrics['adj_f1']:.4f}, "
                f"Precision: {adj_metrics['adj_precision']:.4f}, "
                f"Recall: {adj_metrics['adj_recall']:.4f}")

    metrics = {
        'auprc': auprc,
        'optimal_threshold': optimal_threshold,
        'optimal_accuracy': optimal_accuracy,
        'optimal_precision': optimal_precision,
        'optimal_recall': optimal_recall,
        'optimal_f1': optimal_f1,
        'adj_accuracy': adj_metrics['adj_accuracy'],
        'adj_precision': adj_metrics['adj_precision'],
        'adj_recall': adj_metrics['adj_recall'],
        'adj_f1': adj_metrics['adj_f1'],
        'adjusted_preds_count': adj_metrics['adjusted_preds_count']
    }

    return avg_loss, metrics


def main(args):
    """
    Main function to train the LSTM-CNN model.
    
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
    experiment_name = f"lstm_cnn_{timestamp}"
    if args.experiment_name:
        experiment_name = args.experiment_name
    
    experiment_dir = os.path.join(config['paths']['output_dir'], 'model_save', experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)
    
    # Set up TensorBoard writer
    tensorboard_dir = os.path.join(config['paths']['output_dir'], 'tensorboard', experiment_name)
    os.makedirs(tensorboard_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=tensorboard_dir)
    logger.info(f"TensorBoard logs will be saved to {tensorboard_dir}")
    
    # Create data loaders
    logger.info(f"Loading data from: {config['paths']['data_dir']}")
    data_loaders = create_data_loaders(
        data_dir=config['paths']['data_dir'],
        batch_size=config['training']['batch_size'],
        num_workers=config['training']['num_workers'],
        window_size=config['data']['window_size'],
        step_size=config['data']['step_size'],
        train_ratio=config['data']['train_ratio'],
        val_ratio=config['data']['val_ratio'],
        test_ratio=config['data']['test_ratio'],
        exclude_columns=config['data']['exclude_columns'],
        target_anomaly_ratio=config['data'].get('target_anomaly_ratio', None)
    )
    
    # Get a sample batch to determine input dimension
    sample_batch, _ = next(iter(data_loaders['train']))
    input_dim = sample_batch.shape[2]
    seq_len = sample_batch.shape[1]
    
    logger.info(f"Input dimension: {input_dim}")
    logger.info(f"Sequence length: {seq_len}")
    
    # Create model based on model variant
    model_variant = config['model'].get('variant', 'standard')
    
    if model_variant == 'simple':
        logger.info("Creating SimpleLSTMCNN model...")
        model = SimpleLSTMCNN(
            input_dim=input_dim,
            cnn_channels=config['model']['simple']['cnn_channels'],
            lstm_hidden_dim=config['model']['simple']['lstm_hidden_dim'],
            dropout=config['model']['dropout'],
            num_classes=config['model']['num_classes']
        )
    else:
        logger.info("Creating LSTMCNN model...")
        model = LSTMCNN(
            input_dim=input_dim,
            cnn_channels=config['model']['cnn_channels'],
            cnn_kernel_sizes=config['model']['cnn_kernel_sizes'],
            lstm_hidden_dim=config['model']['lstm_hidden_dim'],
            lstm_num_layers=config['model']['lstm_num_layers'],
            dropout=config['model']['dropout'],
            num_classes=config['model']['num_classes'],
            use_bidirectional=config['model']['use_bidirectional']
        )
    
    # Log model parameters
    param_counts = model.get_num_params()
    logger.info(f"Model parameters: {param_counts}")
    logger.info(f"Total parameters: {param_counts['total']:,}")
    logger.info(f"Trainable parameters: {param_counts['trainable']:,}")
    
    # Move model to device
    model = model.to(device)
    
    # Define loss function
    if config['training']['use_focal_loss']:
        criterion = FocalLoss(
            alpha=config['training']['focal_loss_alpha'],
            gamma=config['training']['focal_loss_gamma']
        )
        logger.info(f"Using Focal Loss with alpha={config['training']['focal_loss_alpha']}, "
                   f"gamma={config['training']['focal_loss_gamma']}")
    else:
        criterion = nn.CrossEntropyLoss()
        logger.info("Using Cross Entropy Loss")
    
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
    
    logger.info(f"Using {config['training']['optimizer'].upper()} optimizer with "
               f"lr={config['training']['learning_rate']}")
    
    # Define learning rate scheduler
    scheduler = None
    if config['training']['lr_scheduler'] == 'reduce_on_plateau':
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=config['training']['lr_reduce_factor'],
            patience=config['training']['lr_reduce_patience'],
            min_lr=config['training']['min_lr']
        )
        logger.info("Using ReduceLROnPlateau scheduler")
    elif config['training']['lr_scheduler'] == 'cosine_annealing':
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=config['training']['num_epochs'],
            eta_min=config['training']['min_lr']
        )
        logger.info("Using CosineAnnealingLR scheduler")
    
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
        mode='max'
    )
    
    # Initialize best metrics
    best_val_loss = float('inf')
    best_val_f1 = 0.0
    best_val_auprc = 0.0
    best_val_adj_f1 = 0.0
    
    # Create directories for different best model types
    best_loss_dir = os.path.join(experiment_dir, "best_loss")
    best_f1_dir = os.path.join(experiment_dir, "best_f1") 
    best_auprc_dir = os.path.join(experiment_dir, "best_auprc")
    best_adj_f1_dir = os.path.join(experiment_dir, "best_adj_f1")
    
    # Training loop
    logger.info(f"Starting training for {config['training']['num_epochs']} epochs")
    for epoch in range(start_epoch, config['training']['num_epochs']):
        logger.info(f"\n{'='*50}")
        logger.info(f"Epoch {epoch+1}/{config['training']['num_epochs']}")
        logger.info(f"{'='*50}")
        
        epoch_start_time = time.time()
        
        # Train for one epoch
        train_loss, train_metrics = train_epoch(
            model, data_loaders['train'], optimizer, criterion, device, scheduler
        )
        
        # Evaluate on validation set
        val_loss, val_metrics = evaluate(
            model, data_loaders['val'], criterion, device, config,
            print_samples=(epoch == 0)
        )
        
        # Update learning rate scheduler
        if isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(val_metrics['optimal_f1'])
        elif scheduler is not None:
            scheduler.step()
        
        epoch_duration = time.time() - epoch_start_time
        current_lr = optimizer.param_groups[0]['lr']

        # Log metrics
        logger.info(f"\nEpoch {epoch+1} Summary:")
        logger.info(f"Time: {epoch_duration:.2f}s | LR: {current_lr:.6f}")
        logger.info(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        logger.info(f"Train F1: {train_metrics['f1']:.4f}")
        logger.info(f"Val Optimal F1: {val_metrics['optimal_f1']:.4f} | "
                   f"Val Adjusted F1: {val_metrics['adj_f1']:.4f}")
        logger.info(f"Val AUPRC: {val_metrics['auprc']:.4f}")
        
        # TensorBoard logging
        writer.add_scalar('lr', current_lr, epoch)
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/f1', train_metrics['f1'], epoch)
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/auprc', val_metrics['auprc'], epoch)
        writer.add_scalar('val/optimal_f1', val_metrics['optimal_f1'], epoch)
        writer.add_scalar('val/optimal_precision', val_metrics['optimal_precision'], epoch)
        writer.add_scalar('val/optimal_recall', val_metrics['optimal_recall'], epoch)
        writer.add_scalar('val/adj_f1', val_metrics['adj_f1'], epoch)
        writer.add_scalar('val/adj_precision', val_metrics['adj_precision'], epoch)
        writer.add_scalar('val/adj_recall', val_metrics['adj_recall'], epoch)
        
        # Save best models
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_model(model, optimizer, epoch, train_loss, val_loss, val_metrics, config, best_loss_dir)
            logger.info(f"✓ New best validation loss: {best_val_loss:.4f}")
        
        if val_metrics['optimal_f1'] > best_val_f1:
            best_val_f1 = val_metrics['optimal_f1']
            save_model(model, optimizer, epoch, train_loss, val_loss, val_metrics, config, best_f1_dir)
            logger.info(f"✓ New best validation F1: {best_val_f1:.4f}")
        
        if val_metrics['auprc'] > best_val_auprc:
            best_val_auprc = val_metrics['auprc']
            save_model(model, optimizer, epoch, train_loss, val_loss, val_metrics, config, best_auprc_dir)
            logger.info(f"✓ New best validation AUPRC: {best_val_auprc:.4f}")
        
        if val_metrics['adj_f1'] > best_val_adj_f1:
            best_val_adj_f1 = val_metrics['adj_f1']
            save_model(model, optimizer, epoch, train_loss, val_loss, val_metrics, config, best_adj_f1_dir)
            logger.info(f"✓ New best validation adjusted F1: {best_val_adj_f1:.4f}")
    
        # Early stopping check
        if early_stopping(val_metrics['optimal_f1']):
            logger.info(f"\nEarly stopping triggered after {epoch+1} epochs")
            logger.info(f"Best validation F1: {best_val_f1:.4f}")
            break
    
    # Evaluate on test set
    logger.info(f"\n{'='*50}")
    logger.info("Final Test Set Evaluation")
    logger.info(f"{'='*50}")
    
    # Test with best F1 model
    logger.info("\nEvaluating best F1 model on test set...")
    best_f1_model_path = os.path.join(best_f1_dir, "best_model.pt")
    if os.path.exists(best_f1_model_path):
        model, _, _, _ = load_model(model, None, best_f1_model_path, device)
        test_loss, test_metrics = evaluate(model, data_loaders['test'], criterion, device, config)
        
        logger.info(f"\nTest Results (Best F1 Model):")
        logger.info(f"Test Loss: {test_loss:.4f}")
        logger.info(f"Test AUPRC: {test_metrics['auprc']:.4f}")
        logger.info(f"Test Optimal F1: {test_metrics['optimal_f1']:.4f}")
        logger.info(f"Test Optimal Precision: {test_metrics['optimal_precision']:.4f}")
        logger.info(f"Test Optimal Recall: {test_metrics['optimal_recall']:.4f}")
        logger.info(f"Test Adjusted F1: {test_metrics['adj_f1']:.4f}")
        logger.info(f"Test Adjusted Precision: {test_metrics['adj_precision']:.4f}")
        logger.info(f"Test Adjusted Recall: {test_metrics['adj_recall']:.4f}")
        
        # Log to TensorBoard
        writer.add_scalar('test/loss', test_loss, 0)
        writer.add_scalar('test/auprc', test_metrics['auprc'], 0)
        writer.add_scalar('test/optimal_f1', test_metrics['optimal_f1'], 0)
        writer.add_scalar('test/adj_f1', test_metrics['adj_f1'], 0)
    
    writer.close()
    logger.info("\n✓ Training completed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LSTM-CNN model for energy anomaly detection")
    parser.add_argument("--config", type=str, default="configs/lstm_cnn_config.yaml",
                       help="Path to configuration file")
    parser.add_argument("--experiment_name", type=str, default=None,
                       help="Name of the experiment")
    parser.add_argument("--load_model", type=str, default=None,
                       help="Path to checkpoint to load model from")
    args = parser.parse_args()
    
    main(args)

