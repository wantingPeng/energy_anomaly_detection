"""
Training script for Transformer model with soft labels for energy anomaly detection.

This script trains a Transformer model using PyTorch's nn.TransformerEncoder
for energy anomaly detection with soft labels (anomaly scores between 0 and 1).
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
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import argparse
import time
import matplotlib.pyplot as plt

from src.utils.logger import logger
from src.training.transform_soft_label.transformer_model_regression import TransformerModelSoftLabel
from src.training.transformer.transfomer_dataset_no_pro_pos import create_data_loaders


def load_config(config_path=None):
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:    
        Configuration dictionary
    """
    if config_path is None:
        config_path = Path("configs/transform_soft_label.yaml")
    
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
        mode: 'min' for metrics like MSE, 'max' for metrics like R2
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
            # For metrics like MSE where lower is better
            if current_score < self.best_score - self.min_delta:
                self.best_score = current_score
                self.counter = 0
            else:
                self.counter += 1
        else:
            # For metrics like R2 where higher is better
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
        model: Transformer model with soft labels
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
        'config': config,
        'input_dim': model.input_dim,
        'd_model': model.d_model
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
        model: Transformer model with soft labels
        data_loader: DataLoader for training data
        optimizer: Optimizer
        criterion: Loss function
        device: Device to use for training
        scheduler: Learning rate scheduler

    Returns:
        Average training loss for the epoch and MSE/MAE metrics
    """
    model.train()
    total_loss = 0
    all_targets = []
    all_outputs = []

    pbar = tqdm(data_loader, desc="Training")

    for batch_idx, (data, targets) in enumerate(pbar):
        data = data.float().to(device)
        targets = targets.float().to(device)  # Ensure targets are float for soft labels

        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if scheduler is not None and not isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step()

        all_outputs.extend(outputs.detach().cpu().numpy())
        all_targets.extend(targets.cpu().numpy())

        total_loss += loss.item()
        pbar.set_postfix({'loss': loss.item()})

    avg_loss = total_loss / len(data_loader)

    # Calculate regression metrics
    all_outputs = np.array(all_outputs)
    all_targets = np.array(all_targets)
    print(f"[Train Prediction Range] min: {all_outputs.min():.4f}, max: {all_outputs.max():.4f}, mean: {all_outputs.mean():.4f}, std: {all_outputs.std():.4f}")
    print(f"[Train Target Range] min: {all_targets.min():.4f}, max: {all_targets.max():.4f}, mean: {all_targets.mean():.4f}, std: {all_targets.std():.4f}")
    mse = mean_squared_error(all_targets, all_outputs)
    mae = mean_absolute_error(all_targets, all_outputs)

    try:
        r2 = r2_score(all_targets, all_outputs)
    except:
        r2 = 0.0

    metrics = {
        'mse': mse,
        'mae': mae,
        'r2': r2
    }

    return avg_loss, metrics


def evaluate(model, data_loader, criterion, device, config, print_samples=True):
    """
    Evaluate the model on validation or test data.
    
    Args:
        model: Transformer model with soft labels
        data_loader: DataLoader for validation/test data
        criterion: Loss function
        device: Device to use for evaluation
        config: Configuration dictionary
        print_samples: Whether to print sample outputs
    
    Returns:
        Average loss and metrics dictionary
    """
    model.eval()
    total_loss = 0
    all_targets = []
    all_outputs = []
    sample_outputs = []

    with torch.no_grad():
        for batch_idx, (data, targets) in enumerate(tqdm(data_loader, desc="Evaluating")):
            data = data.float().to(device)
            targets = targets.float().to(device)  # Ensure targets are float for soft labels

            # Forward pass
            outputs = model(data)
            loss = criterion(outputs, targets)
            total_loss += loss.item()

            all_targets.extend(targets.cpu().numpy())
            all_outputs.extend(outputs.cpu().numpy())

            # Sample output logging
            if print_samples and batch_idx == 0:
                sample_size = min(5, len(data))
                for i in range(sample_size):
                    sample_outputs.append({
                        'prediction': outputs[i].item(),
                        'actual': targets[i].item()
                    })

    avg_loss = total_loss / len(data_loader)

    if print_samples and sample_outputs:
        logger.info("\n===== Sample Outputs =====")
        for i, sample in enumerate(sample_outputs):
            logger.info(f"Sample {i+1}:")
            logger.info(f"  Prediction: {sample['prediction']:.6f}")
            logger.info(f"  Actual: {sample['actual']:.6f}")
            logger.info(f"  Difference: {abs(sample['prediction'] - sample['actual']):.6f}")
            logger.info("------------------------")

    all_outputs = np.array(all_outputs)
    all_targets = np.array(all_targets)
    print(f"[Val Prediction Range] min: {all_outputs.min():.4f}, max: {all_outputs.max():.4f}, mean: {all_outputs.mean():.4f}, std: {all_outputs.std():.4f}")
    print(f"[Val Target Range] min: {all_targets.min():.4f}, max: {all_targets.max():.4f}, mean: {all_targets.mean():.4f}, std: {all_targets.std():.4f}")
    # Calculate regression metrics
    mse = mean_squared_error(all_targets, all_outputs)
    mae = mean_absolute_error(all_targets, all_outputs)
    
    try:
        r2 = r2_score(all_targets, all_outputs)
    except:
        r2 = 0.0

    metrics = {
        'mse': mse,
        'mae': mae,
        'r2': r2,
        'predictions': all_outputs,
        'targets': all_targets
    }

    return avg_loss, metrics


def main(args):
    """
    Main function to train the Transformer model with soft labels.
    
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
    experiment_name = f"transformer_soft_{timestamp}"
    if args.experiment_name:
        experiment_name = args.experiment_name
    
    experiment_dir = os.path.join(config['paths']['output_dir'], 'model_save', experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)
    
    # Set up TensorBoard writer
    tensorboard_dir = os.path.join("experiments/transformer_soft/tensorboard", experiment_name)
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
    model = TransformerModelSoftLabel(
        input_dim=input_dim,
        d_model=config['model']['d_model'],
        nhead=config['model']['nhead'],
        num_layers=config['model']['num_layers'],
        dim_feedforward=config['model']['dim_feedforward'],
        dropout=config['model']['dropout'],
        activation=config['model']['activation']
    )
    
    # Log model architecture and parameter count
    logger.info(f"Model architecture:\n{model}")
    
    # Move model to device
    model = model.to(device)
    
    # Define loss function
    criterion = nn.MSELoss() if config['training']['loss'] == 'mse' else nn.BCELoss()
    logger.info(f"Using loss function: {criterion}")
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
        mode='min' if config['training']['early_stopping_metric'] in ['loss', 'mse', 'mae'] else 'max'
    )
    
    # Initialize best metrics
    best_val_loss = float('inf')
    best_val_mse = float('inf')
    best_val_r2 = float('-inf')
    
    # Training loop
    for epoch in range(start_epoch, config['training']['num_epochs']):
        logger.info(f"Epoch {epoch+1}/{config['training']['num_epochs']}")
        
        # Record epoch start time
        epoch_start_time = time.time()
        
        # Train for one epoch
        train_loss, train_metrics = train_epoch(
            model, data_loaders['train'], optimizer, criterion, device,
            scheduler=scheduler
        )
        
        # Evaluate on validation set
        val_loss, val_metrics = evaluate(
            model, data_loaders['val_200'], criterion, device, config,
            print_samples=(epoch == 0)  # Print samples only for first epoch
        )
        
        # Update learning rate scheduler if using ReduceLROnPlateau
        if isinstance(scheduler, ReduceLROnPlateau):
            if config['training']['early_stopping_metric'] == 'mse':
                scheduler.step(val_metrics['mse'])
            elif config['training']['early_stopping_metric'] == 'r2':
                # Negative since scheduler is in 'min' mode
                scheduler.step(-val_metrics['r2'])
            else:
                scheduler.step(val_loss)
        elif scheduler is not None and not isinstance(scheduler, optim.lr_scheduler.CosineAnnealingLR):
            scheduler.step()
        
        # Calculate epoch duration
        epoch_duration = time.time() - epoch_start_time
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']

        # Log learning rate
        logger.info(f"Current learning rate: {current_lr:.6f}")
        writer.add_scalar('lr', current_lr, epoch)

        # Log metrics
        logger.info(f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        logger.info(f"Train MSE: {train_metrics['mse']:.6f}, Val MSE: {val_metrics['mse']:.6f}")
        logger.info(f"Train MAE: {train_metrics['mae']:.6f}, Val MAE: {val_metrics['mae']:.6f}")
        logger.info(f"Train R²: {train_metrics['r2']:.6f}, Val R²: {val_metrics['r2']:.6f}")
        logger.info(f"Epoch completed in {epoch_duration:.2f} seconds")
        
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('train/mse', train_metrics['mse'], epoch)
        writer.add_scalar('val/mse', val_metrics['mse'], epoch)
        writer.add_scalar('train/mae', train_metrics['mae'], epoch)
        writer.add_scalar('val/mae', val_metrics['mae'], epoch)
        writer.add_scalar('train/r2', train_metrics['r2'], epoch)
        writer.add_scalar('val/r2', val_metrics['r2'], epoch)
        
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
            logger.info(f"New best validation loss: {best_val_loss:.6f}")
        
        # Save best model based on MSE
        if val_metrics['mse'] < best_val_mse:
            best_val_mse = val_metrics['mse']
            save_model(
                model, optimizer, epoch, train_loss, val_loss, val_metrics,
                config, os.path.join(experiment_dir, "best_mse")
            )
            logger.info(f"New best validation MSE: {best_val_mse:.6f}")
        
        # Save best model based on R²
        if val_metrics['r2'] > best_val_r2:
            best_val_r2 = val_metrics['r2']
            save_model(
                model, optimizer, epoch, train_loss, val_loss, val_metrics,
                config, os.path.join(experiment_dir, "best_r2")
            )
            logger.info(f"New best validation R²: {best_val_r2:.6f}")
        
        # Check for early stopping
        if config['training']['early_stopping_metric'] == 'mse':
            if early_stopping(val_metrics['mse']):
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break
        elif config['training']['early_stopping_metric'] == 'r2':
            # Negative since scheduler is in 'min' mode but we want to maximize R²
            if early_stopping(-val_metrics['r2']):
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break
        else:
            if early_stopping(val_loss):
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break
    
    # Evaluate on test set using best model (based on MSE)
    logger.info("Evaluating best model (based on MSE) on test set")
    best_model_path = os.path.join(experiment_dir, "best_mse", f"checkpoint_epoch_{epoch}.pt")
    if os.path.exists(best_model_path):
        model, _, _, _ = load_model(model, None, best_model_path, device)
        test_loss, test_metrics = evaluate(model, data_loaders['val'], criterion, device, config)
        
        logger.info(f"Test Loss: {test_loss:.6f}")
        logger.info(f"Test MSE: {test_metrics['mse']:.6f}")
        logger.info(f"Test MAE: {test_metrics['mae']:.6f}")
        logger.info(f"Test R²: {test_metrics['r2']:.6f}")
        
        # Log test metrics to TensorBoard
        writer.add_scalar('test/loss', test_loss, 0)
        writer.add_scalar('test/mse', test_metrics['mse'], 0)
        writer.add_scalar('test/mae', test_metrics['mae'], 0)
        writer.add_scalar('test/r2', test_metrics['r2'], 0)
    
    # Evaluate on test set using best model (based on R²)
    logger.info("Evaluating best model (based on R²) on test set")
    best_r2_model_path = os.path.join(experiment_dir, "best_r2", f"checkpoint_epoch_{epoch}.pt")
    if os.path.exists(best_r2_model_path):
        model, _, _, _ = load_model(model, None, best_r2_model_path, device)
        test_loss, test_metrics = evaluate(model, data_loaders['val'], criterion, device, config)
        
        logger.info(f"Test Loss (R² model): {test_loss:.6f}")
        logger.info(f"Test MSE (R² model): {test_metrics['mse']:.6f}")
        logger.info(f"Test MAE (R² model): {test_metrics['mae']:.6f}")
        logger.info(f"Test R² (R² model): {test_metrics['r2']:.6f}")
        
        # Log test metrics to TensorBoard for R² model
        writer.add_scalar('test_r2_model/loss', test_loss, 0)
        writer.add_scalar('test_r2_model/mse', test_metrics['mse'], 0)
        writer.add_scalar('test_r2_model/mae', test_metrics['mae'], 0)
        writer.add_scalar('test_r2_model/r2', test_metrics['r2'], 0)
    
    # Close TensorBoard writer
    writer.close()
    
    logger.info("Training completed successfully")


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train Transformer model with soft labels for energy anomaly detection")
    parser.add_argument("--config", type=str, default=None, help="Path to configuration file")
    parser.add_argument("--experiment_name", type=str, default=None, help="Name of the experiment")
    parser.add_argument("--test", action="store_true", help="Test the model")
    parser.add_argument("--load_model", type=str, default=None, help="Path to checkpoint to load model from")
    args = parser.parse_args()
    
    # Run main function
    main(args) 