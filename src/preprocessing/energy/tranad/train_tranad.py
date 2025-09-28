"""
Training script for TranAD model.

This script trains a TranAD model for anomaly detection in energy time series data.
It implements the two-phase training approach with adversarial training.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml
import argparse
import time
from datetime import datetime
from tqdm import tqdm
from pathlib import Path
import json
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, average_precision_score

# Import local modules
from tranad_model import TranAD, create_tranad_config
from tranad_dataloader import create_tranad_data_loaders
from utils.logger import logger


def set_seed(seed):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_config(config, save_path):
    """Save configuration to YAML file."""
    with open(save_path, 'w') as f:
        yaml.dump(config, f)


def get_optimizer(model, config):
    """Get optimizer based on configuration."""
    optimizer_name = config.get('optimizer', 'adam').lower()
    lr = config.get('learning_rate', 0.001)
    weight_decay = config.get('weight_decay', 0.0)
    
    if optimizer_name == 'adam':
        return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'adamw':
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'sgd':
        momentum = config.get('momentum', 0.9)
        return optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")


def get_lr_scheduler(optimizer, config):
    """Get learning rate scheduler based on configuration."""
    scheduler_name = config.get('lr_scheduler', 'none').lower()
    
    if scheduler_name == 'none':
        return None
    elif scheduler_name == 'steplr':
        step_size = config.get('lr_step_size', 30)
        gamma = config.get('lr_gamma', 0.1)
        return optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif scheduler_name == 'cosine':
        T_max = config.get('lr_T_max', 100)
        eta_min = config.get('lr_eta_min', 0)
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)
    elif scheduler_name == 'plateau':
        patience = config.get('lr_patience', 10)
        factor = config.get('lr_factor', 0.1)
        return optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=factor, patience=patience)
    else:
        raise ValueError(f"Unsupported scheduler: {scheduler_name}")


def train_epoch(model, dataloader, optimizer, device, phase):
    """
    Train the model for one epoch.
    
    Args:
        model: TranAD model
        dataloader: Training data loader
        optimizer: Optimizer
        device: Device to use (cuda or cpu)
        phase: Training phase (1 or 2)
        
    Returns:
        epoch_loss: Average loss for the epoch
        loss_components: Dictionary of loss components
    """
    model.train()
    total_loss = 0.0
    loss_components_sum = {}
    
    for batch_idx, (data, labels) in enumerate(tqdm(dataloader, desc=f"Training (Phase {phase})")):
        # Move data to device
        data = data.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        loss, loss_components = model.compute_loss(data, phase=phase)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Update statistics
        total_loss += loss.item()
        
        # Accumulate loss components
        for name, value in loss_components.items():
            if name not in loss_components_sum:
                loss_components_sum[name] = 0.0
            loss_components_sum[name] += value.item()
    
    # Calculate average loss
    epoch_loss = total_loss / len(dataloader)
    
    # Calculate average loss components
    avg_loss_components = {name: value / len(dataloader) for name, value in loss_components_sum.items()}
    
    return epoch_loss, avg_loss_components


def validate(model, dataloader, device, phase):
    """
    Validate the model.
    
    Args:
        model: TranAD model
        dataloader: Validation data loader
        device: Device to use (cuda or cpu)
        phase: Validation phase (1 or 2)
        
    Returns:
        val_loss: Average validation loss
        loss_components: Dictionary of loss components
        anomaly_scores: Anomaly scores
        true_labels: True labels
    """
    model.eval()
    total_loss = 0.0
    loss_components_sum = {}
    all_anomaly_scores = []
    all_labels = []
    
    with torch.no_grad():
        for batch_idx, (data, labels) in enumerate(tqdm(dataloader, desc=f"Validating (Phase {phase})")):
            # Move data to device
            data = data.to(device)
            
            # Forward pass
            loss, loss_components = model.compute_loss(data, phase=phase)
            
            # Get anomaly scores
            _, anomaly_scores, _ = model(data, phase=phase)
            
            # Update statistics
            total_loss += loss.item()
            
            # Accumulate loss components
            for name, value in loss_components.items():
                if name not in loss_components_sum:
                    loss_components_sum[name] = 0.0
                loss_components_sum[name] += value.item()
            
            # Collect anomaly scores and labels
            all_anomaly_scores.append(anomaly_scores.cpu().numpy())
            all_labels.append(labels.numpy())
    
    # Calculate average loss
    val_loss = total_loss / len(dataloader)
    
    # Calculate average loss components
    avg_loss_components = {name: value / len(dataloader) for name, value in loss_components_sum.items()}
    
    # Concatenate anomaly scores and labels
    all_anomaly_scores = np.concatenate(all_anomaly_scores)
    all_labels = np.concatenate(all_labels)
    
    return val_loss, avg_loss_components, all_anomaly_scores, all_labels


def evaluate_anomaly_detection(anomaly_scores, true_labels, threshold_percentile=95):
    """
    Evaluate anomaly detection performance.
    
    Args:
        anomaly_scores: Anomaly scores [batch_size, seq_len]
        true_labels: True labels [batch_size, seq_len]
        threshold_percentile: Percentile for threshold calculation
        
    Returns:
        metrics: Dictionary of evaluation metrics
    """
    # Flatten arrays
    anomaly_scores_flat = anomaly_scores.flatten()
    true_labels_flat = true_labels.flatten()
    
    # Calculate threshold
    threshold = np.percentile(anomaly_scores_flat, threshold_percentile)
    
    # Detect anomalies
    predicted_anomalies = (anomaly_scores_flat > threshold).astype(int)
    
    # Calculate metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels_flat, predicted_anomalies, average='binary'
    )
    
    # Calculate AUC-ROC and AUC-PR
    auroc = roc_auc_score(true_labels_flat, anomaly_scores_flat)
    auprc = average_precision_score(true_labels_flat, anomaly_scores_flat)
    
    # Calculate true positives, false positives, true negatives, false negatives
    tp = np.sum((predicted_anomalies == 1) & (true_labels_flat == 1))
    fp = np.sum((predicted_anomalies == 1) & (true_labels_flat == 0))
    tn = np.sum((predicted_anomalies == 0) & (true_labels_flat == 0))
    fn = np.sum((predicted_anomalies == 0) & (true_labels_flat == 1))
    
    # Calculate additional metrics
    accuracy = (tp + tn) / (tp + fp + tn + fn)
    
    metrics = {
        'threshold': threshold,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auroc': auroc,
        'auprc': auprc,
        'tp': int(tp),
        'fp': int(fp),
        'tn': int(tn),
        'fn': int(fn)
    }
    
    return metrics


def save_checkpoint(model, optimizer, epoch, loss, metrics, save_dir, filename):
    """
    Save model checkpoint.
    
    Args:
        model: TranAD model
        optimizer: Optimizer
        epoch: Current epoch
        loss: Validation loss
        metrics: Evaluation metrics
        save_dir: Directory to save checkpoint
        filename: Checkpoint filename
    """
    os.makedirs(save_dir, exist_ok=True)
    checkpoint_path = os.path.join(save_dir, filename)
    
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'metrics': metrics
    }, checkpoint_path)
    
    logger.info(f"Checkpoint saved to {checkpoint_path}")


def plot_training_curves(train_losses, val_losses, save_path):
    """
    Plot training and validation loss curves.
    
    Args:
        train_losses: List of training losses
        val_losses: List of validation losses
        save_path: Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()
    
    logger.info(f"Training curves saved to {save_path}")


def plot_anomaly_scores(anomaly_scores, true_labels, threshold, save_path):
    """
    Plot anomaly scores and true labels.
    
    Args:
        anomaly_scores: Anomaly scores [batch_size, seq_len]
        true_labels: True labels [batch_size, seq_len]
        threshold: Anomaly threshold
        save_path: Path to save the plot
    """
    # Flatten arrays
    anomaly_scores_flat = anomaly_scores.flatten()
    true_labels_flat = true_labels.flatten()
    
    # Create time axis
    time = np.arange(len(anomaly_scores_flat))
    
    plt.figure(figsize=(15, 8))
    
    # Plot anomaly scores
    plt.plot(time, anomaly_scores_flat, label='Anomaly Score', color='blue', alpha=0.7)
    
    # Plot threshold
    plt.axhline(y=threshold, color='r', linestyle='--', label=f'Threshold ({threshold:.3f})')
    
    # Plot true anomalies
    anomaly_indices = np.where(true_labels_flat == 1)[0]
    plt.scatter(anomaly_indices, anomaly_scores_flat[anomaly_indices], color='red', label='True Anomalies', marker='x')
    
    plt.xlabel('Time')
    plt.ylabel('Anomaly Score')
    plt.title('Anomaly Scores and True Anomalies')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()
    
    logger.info(f"Anomaly scores plot saved to {save_path}")


def train_tranad(config):
    """
    Train TranAD model with two-phase training.
    
    Args:
        config: Configuration dictionary
    """
    # Extract configuration
    data_dir = config['data']['data_dir']
    batch_size = config['data']['batch_size']
    window_size = config['data']['window_size']
    step_size = config['data']['step_size']
    num_workers = config['data']['num_workers']
    
    model_size = config['model']['model_size']
    input_dim = config['model']['input_dim']
    
    num_epochs_phase1 = config['training']['num_epochs_phase1']
    num_epochs_phase2 = config['training']['num_epochs_phase2']
    early_stopping_patience = config['training']['early_stopping_patience']
    
    experiment_name = config['experiment']['name']
    output_dir = config['experiment']['output_dir']
    
    # Set random seed
    set_seed(config['experiment']['seed'])
    
    # Create output directory
    experiment_dir = os.path.join(output_dir, experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)
    
    # Save configuration
    config_save_path = os.path.join(experiment_dir, 'config.yaml')
    save_config(config, config_save_path)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create data loaders
    data_loaders = create_tranad_data_loaders(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        window_size=window_size,
        step_size=step_size,
        scaler_save_path=os.path.join(experiment_dir, 'scaler.pkl')
    )
    
    # Create model
    model_config = create_tranad_config(model_size)
    model = TranAD(input_dim=input_dim, **model_config)
    model = model.to(device)
    
    logger.info(f"Created TranAD model with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Create optimizer
    optimizer = get_optimizer(model, config['training'])
    
    # Create learning rate scheduler
    scheduler = get_lr_scheduler(optimizer, config['training'])
    
    # Training loop - Phase 1
    logger.info("=== Starting Phase 1 Training ===")
    train_losses_phase1 = []
    val_losses_phase1 = []
    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0
    
    for epoch in range(1, num_epochs_phase1 + 1):
        epoch_start_time = time.time()
        
        # Train
        train_loss, train_loss_components = train_epoch(model, data_loaders['train'], optimizer, device, phase=1)
        
        # Validate
        val_loss, val_loss_components, val_anomaly_scores, val_labels = validate(model, data_loaders['val'], device, phase=1)
        
        # Update learning rate
        if scheduler is not None:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()
        
        # Record losses
        train_losses_phase1.append(train_loss)
        val_losses_phase1.append(val_loss)
        
        # Calculate metrics
        val_metrics = evaluate_anomaly_detection(val_anomaly_scores, val_labels)
        
        # Log progress
        epoch_time = time.time() - epoch_start_time
        logger.info(f"Phase 1 - Epoch {epoch}/{num_epochs_phase1} | "
                   f"Train Loss: {train_loss:.6f} | "
                   f"Val Loss: {val_loss:.6f} | "
                   f"F1: {val_metrics['f1']:.4f} | "
                   f"AUROC: {val_metrics['auroc']:.4f} | "
                   f"Time: {epoch_time:.2f}s")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            
            # Save checkpoint
            save_checkpoint(
                model, optimizer, epoch, val_loss, val_metrics,
                experiment_dir, 'best_model_phase1.pth'
            )
            
            # Plot anomaly scores
            plot_anomaly_scores(
                val_anomaly_scores, val_labels, val_metrics['threshold'],
                os.path.join(experiment_dir, f'anomaly_scores_phase1_epoch{epoch}.png')
            )
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= early_stopping_patience:
            logger.info(f"Early stopping triggered after {epoch} epochs")
            break
    
    # Plot training curves for Phase 1
    plot_training_curves(
        train_losses_phase1, val_losses_phase1,
        os.path.join(experiment_dir, 'training_curves_phase1.png')
    )
    
    logger.info(f"Phase 1 completed. Best validation loss: {best_val_loss:.6f} at epoch {best_epoch}")
    
    # Load best model from Phase 1
    checkpoint_path = os.path.join(experiment_dir, 'best_model_phase1.pth')
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Training loop - Phase 2
    logger.info("=== Starting Phase 2 Training ===")
    train_losses_phase2 = []
    val_losses_phase2 = []
    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0
    
    for epoch in range(1, num_epochs_phase2 + 1):
        epoch_start_time = time.time()
        
        # Train
        train_loss, train_loss_components = train_epoch(model, data_loaders['train'], optimizer, device, phase=2)
        
        # Validate
        val_loss, val_loss_components, val_anomaly_scores, val_labels = validate(model, data_loaders['val'], device, phase=2)
        
        # Update learning rate
        if scheduler is not None:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()
        
        # Record losses
        train_losses_phase2.append(train_loss)
        val_losses_phase2.append(val_loss)
        
        # Calculate metrics
        val_metrics = evaluate_anomaly_detection(val_anomaly_scores, val_labels)
        
        # Log progress
        epoch_time = time.time() - epoch_start_time
        logger.info(f"Phase 2 - Epoch {epoch}/{num_epochs_phase2} | "
                   f"Train Loss: {train_loss:.6f} | "
                   f"Val Loss: {val_loss:.6f} | "
                   f"F1: {val_metrics['f1']:.4f} | "
                   f"AUROC: {val_metrics['auroc']:.4f} | "
                   f"Time: {epoch_time:.2f}s")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            
            # Save checkpoint
            save_checkpoint(
                model, optimizer, epoch, val_loss, val_metrics,
                experiment_dir, 'best_model_phase2.pth'
            )
            
            # Plot anomaly scores
            plot_anomaly_scores(
                val_anomaly_scores, val_labels, val_metrics['threshold'],
                os.path.join(experiment_dir, f'anomaly_scores_phase2_epoch{epoch}.png')
            )
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= early_stopping_patience:
            logger.info(f"Early stopping triggered after {epoch} epochs")
            break
    
    # Plot training curves for Phase 2
    plot_training_curves(
        train_losses_phase2, val_losses_phase2,
        os.path.join(experiment_dir, 'training_curves_phase2.png')
    )
    
    logger.info(f"Phase 2 completed. Best validation loss: {best_val_loss:.6f} at epoch {best_epoch}")
    
    # Load best model from Phase 2
    checkpoint_path = os.path.join(experiment_dir, 'best_model_phase2.pth')
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Evaluate on test set
    logger.info("=== Evaluating on Test Set ===")
    test_loss, test_loss_components, test_anomaly_scores, test_labels = validate(model, data_loaders['test'], device, phase=2)
    test_metrics = evaluate_anomaly_detection(test_anomaly_scores, test_labels)
    
    logger.info(f"Test Loss: {test_loss:.6f}")
    logger.info(f"Test Metrics:")
    for name, value in test_metrics.items():
        logger.info(f"  {name}: {value}")
    
    # Plot test anomaly scores
    plot_anomaly_scores(
        test_anomaly_scores, test_labels, test_metrics['threshold'],
        os.path.join(experiment_dir, 'test_anomaly_scores.png')
    )
    
    # Save test metrics
    with open(os.path.join(experiment_dir, 'test_metrics.json'), 'w') as f:
        json.dump(test_metrics, f, indent=4)
    
    logger.info(f"Training completed. Results saved to {experiment_dir}")


def create_default_config():
    """Create default configuration for TranAD training."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    config = {
        'experiment': {
            'name': f'tranad_{timestamp}',
            'output_dir': 'experiments/tranad',
            'seed': 42
        },
        'data': {
            'data_dir': 'Data/downsampleData_scratch_1minut/energy_data_labeled.parquet',
            'batch_size': 64,
            'window_size': 60,
            'step_size': 1,
            'num_workers': 4
        },
        'model': {
            'model_size': 'base',  # 'small', 'base', or 'large'
            'input_dim': 10  # Update based on your data
        },
        'training': {
            'optimizer': 'adam',
            'learning_rate': 0.001,
            'weight_decay': 1e-5,
            'num_epochs_phase1': 50,
            'num_epochs_phase2': 30,
            'early_stopping_patience': 10,
            'lr_scheduler': 'plateau',
            'lr_patience': 5,
            'lr_factor': 0.5
        }
    }
    
    return config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train TranAD model")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--data_dir", type=str, help="Path to data directory")
    parser.add_argument("--output_dir", type=str, help="Path to output directory")
    parser.add_argument("--experiment_name", type=str, help="Experiment name")
    parser.add_argument("--batch_size", type=int, help="Batch size")
    parser.add_argument("--window_size", type=int, help="Window size")
    parser.add_argument("--input_dim", type=int, help="Input dimension")
    
    args = parser.parse_args()
    
    # Load or create configuration
    if args.config:
        config = load_config(args.config)
        logger.info(f"Loaded configuration from {args.config}")
    else:
        config = create_default_config()
        logger.info("Using default configuration")
    
    # Override configuration with command-line arguments
    if args.data_dir:
        config['data']['data_dir'] = args.data_dir
    if args.output_dir:
        config['experiment']['output_dir'] = args.output_dir
    if args.experiment_name:
        config['experiment']['name'] = args.experiment_name
    if args.batch_size:
        config['data']['batch_size'] = args.batch_size
    if args.window_size:
        config['data']['window_size'] = args.window_size
    if args.input_dim:
        config['model']['input_dim'] = args.input_dim
    
    # Train model
    train_tranad(config)

