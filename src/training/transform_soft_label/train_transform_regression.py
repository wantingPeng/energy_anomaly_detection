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
from sklearn.metrics import mean_absolute_error, median_absolute_error, precision_recall_curve, average_precision_score
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import argparse
import time
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance
import numpy as np

from src.utils.logger import logger
from src.training.transform_soft_label.transformer_model_regression import TransformerModelSoftLabel
from src.training.transformer.transfomer_dataset_no_pro_pos import create_data_loaders


def calculate_threshold_metrics(targets, predictions):
    """
    Calculate classification metrics based on thresholds and AUPRC.
    
    Args:
        targets: Ground truth values (numpy array)
        predictions: Predicted values (numpy array)
        
    Returns:
        Dictionary with threshold-based metrics and AUPRC values
    """
    # Define thresholds for classification
    thresholds = [0.0, 0.2, 0.4, 0.6, 1.0]
    class_names = ["No fault", "Medium risk", "High risk", "Fault"]
    
    # Classification based on thresholds
    target_classes = np.zeros_like(targets, dtype=int)
    pred_classes = np.zeros_like(predictions, dtype=int)
    
    for i in range(len(thresholds) - 1):
        target_classes[(targets >= thresholds[i]) & (targets < thresholds[i + 1])] = i
        pred_classes[(predictions >= thresholds[i]) & (predictions < thresholds[i + 1])] = i
    
    # Count samples in each class
    class_counts = {}
    for i in range(len(class_names)):
        target_count = np.sum(target_classes == i)
        pred_count = np.sum(pred_classes == i)
        class_counts[class_names[i]] = {
            "target_count": int(target_count),
            "pred_count": int(pred_count),
            "percentage_target": float(target_count) / len(targets) * 100 if len(targets) > 0 else 0,
            "percentage_pred": float(pred_count) / len(predictions) * 100 if len(predictions) > 0 else 0
        }
    
    # Calculate confusion metrics for each class
    confusion_metrics = {}
    accuracy = np.mean(target_classes == pred_classes)
    
    # Calculate AUPRC for binary classification of each threshold
    auprc_metrics = {}
    for i in range(len(thresholds) - 1):
        # Binary classification for this threshold range
        # 1 if value is in or above this range, 0 otherwise
        for j, threshold in enumerate(thresholds[:-1]):
            binary_targets = (targets >= threshold).astype(int)
            binary_preds = predictions  # Keep the continuous predictions for PR curve
            
            try:
                ap_score = average_precision_score(binary_targets, binary_preds)
                precision, recall, _ = precision_recall_curve(binary_targets, binary_preds)
                auprc_metrics[f">= {threshold:.1f}"] = {
                    "average_precision": float(ap_score),
                    "positive_samples": int(np.sum(binary_targets)),
                    "positive_rate": float(np.mean(binary_targets)) * 100
                }
            except Exception as e:
                auprc_metrics[f">= {threshold:.1f}"] = {
                    "error": str(e),
                    "positive_samples": int(np.sum(binary_targets)),
                    "positive_rate": float(np.mean(binary_targets)) * 100
                }
    
    return {
        "class_distribution": class_counts,
        "accuracy": float(accuracy),
        "auprc": auprc_metrics
    }


class TweedieLoss(nn.Module):
    """
    Tweedie loss for regression with zero-inflated or highly skewed data.
    
    Args:
        variance_power (float): Variance power parameter p where variance ~ mean^p
                               p=1 is Poisson
                               p=2 is Gamma
                               p=3 is Inverse Gaussian
                               1<p<2 is compound Poisson-Gamma (good for zero-inflated data)
    """
    def __init__(self, variance_power=1.5):
        super(TweedieLoss, self).__init__()
        self.p = variance_power
        
    def forward(self, pred, target):
        # Avoid numerical instability with small values
        eps = 1e-5
        pred = torch.clamp(pred, min=eps,max=10.0)
        
        if self.p == 2.0:  # Gamma case
            return torch.mean(pred / target + torch.log(target))
        else:
            # General Tweedie case
            term1 = torch.pow(target, 2.0 - self.p) / ((1.0 - self.p) * (2.0 - self.p))
            term2 = torch.pow(pred, 2.0 - self.p) / ((1.0 - self.p) * (2.0 - self.p))
            term3 = target * torch.pow(pred, 1.0 - self.p) / (1.0 - self.p)
            return torch.mean(term1 - term3 + term2)


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
        # Ensure non-negative predictions
        outputs = torch.clamp(outputs,min=1e-5, max=5.0)
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
    mae = mean_absolute_error(all_targets, all_outputs)
    median_ae = median_absolute_error(all_targets, all_outputs)
    
    # Calculate Wasserstein distance (Earth Mover's Distance)
    try:
        w_distance = wasserstein_distance(all_targets.flatten(), all_outputs.flatten())
    except:
        w_distance = float('inf')

    metrics = {
        'mae': mae,
        'median_ae': median_ae,
        'wasserstein': w_distance
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
            # Ensure non-negative predictions
            outputs = torch.clamp(outputs, min=1e-5, max=5.0)
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
    mae = mean_absolute_error(all_targets, all_outputs)
    median_ae = median_absolute_error(all_targets, all_outputs)
    
    # Calculate Wasserstein distance (Earth Mover's Distance)
    try:
        w_distance = wasserstein_distance(all_targets.flatten(), all_outputs.flatten())
    except:
        w_distance = float('inf')

    # Add threshold-based classification and AUPRC metrics
    threshold_metrics = calculate_threshold_metrics(all_targets, all_outputs)
    
    metrics = {
        'mae': mae,
        'median_ae': median_ae,
        'wasserstein': w_distance,
        'predictions': all_outputs,
        'targets': all_targets,
        'threshold_metrics': threshold_metrics
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
    if config['training']['loss'] == 'mse':
        criterion = nn.MSELoss()
    elif config['training']['loss'] == 'bce':
        criterion = nn.BCELoss()
    elif config['training']['loss'] == 'tweedie':
        variance_power = config['training'].get('tweedie_variance_power', 1.5)
        criterion = TweedieLoss(variance_power=variance_power)
        logger.info(f"Using Tweedie Loss with variance power: {variance_power}")
    else:
        criterion = nn.MSELoss()  # Default to MSE if not specified
    logger.info(f"Using loss function: {config['training']['loss']}")
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
        mode='min'  # All our metrics (loss, mae, median_ae, wasserstein) are "lower is better"
    )
    
    # Initialize best metrics
    best_val_loss = float('inf')
    best_val_wasserstein = float('inf')
    best_val_mae = float('inf')
    best_val_median_ae = float('inf')
    
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
            if config['training']['early_stopping_metric'] == 'wasserstein':
                scheduler.step(val_metrics['wasserstein'])
            elif config['training']['early_stopping_metric'] == 'mae':
                scheduler.step(val_metrics['mae'])
            elif config['training']['early_stopping_metric'] == 'median_ae':
                scheduler.step(val_metrics['median_ae'])
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
        logger.info(f"Train MAE: {train_metrics['mae']:.6f}, Val MAE: {val_metrics['mae']:.6f}")
        logger.info(f"Train Median AE: {train_metrics['median_ae']:.6f}, Val Median AE: {val_metrics['median_ae']:.6f}")
        logger.info(f"Train Wasserstein: {train_metrics['wasserstein']:.6f}, Val Wasserstein: {val_metrics['wasserstein']:.6f}")
        
        # Log threshold classification metrics
        logger.info("=== Validation Threshold Classification Metrics ===")
        # Log class distribution
        logger.info("Class Distribution (Ground Truth):")
        for class_name, stats in val_metrics['threshold_metrics']['class_distribution'].items():
            logger.info(f"  {class_name}: {stats['target_count']} samples ({stats['percentage_target']:.2f}%)")
            
        logger.info("Class Distribution (Predictions):")
        for class_name, stats in val_metrics['threshold_metrics']['class_distribution'].items():
            logger.info(f"  {class_name}: {stats['pred_count']} samples ({stats['percentage_pred']:.2f}%)")
        
        # Log classification accuracy
        logger.info(f"Classification Accuracy: {val_metrics['threshold_metrics']['accuracy']:.4f}")
        
        # Log AUPRC metrics
        logger.info("AUPRC Metrics:")
        for threshold, auprc_data in val_metrics['threshold_metrics']['auprc'].items():
            if "error" in auprc_data:
                logger.info(f"  {threshold}: Error - {auprc_data['error']}")
            else:
                logger.info(f"  {threshold}: AUPRC = {auprc_data['average_precision']:.4f} (Positive rate: {auprc_data['positive_rate']:.2f}%)")
        
        logger.info(f"Epoch completed in {epoch_duration:.2f} seconds")
        
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('train/mae', train_metrics['mae'], epoch)
        writer.add_scalar('val/mae', val_metrics['mae'], epoch)
        writer.add_scalar('train/median_ae', train_metrics['median_ae'], epoch)
        writer.add_scalar('val/median_ae', val_metrics['median_ae'], epoch)
        writer.add_scalar('train/wasserstein', train_metrics['wasserstein'], epoch)
        writer.add_scalar('val/wasserstein', val_metrics['wasserstein'], epoch)
        
        # Add AUPRC metrics to TensorBoard
        for threshold, auprc_data in val_metrics['threshold_metrics']['auprc'].items():
            if "average_precision" in auprc_data:
                writer.add_scalar(f'val/auprc/{threshold}', auprc_data['average_precision'], epoch)
        
        # Add classification accuracy to TensorBoard
        writer.add_scalar('val/classification_accuracy', val_metrics['threshold_metrics']['accuracy'], epoch)
        
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
            
        # Save best model based on Wasserstein distance
        if val_metrics['wasserstein'] < best_val_wasserstein:
            best_val_wasserstein = val_metrics['wasserstein']
            save_model(
                model, optimizer, epoch, train_loss, val_loss, val_metrics,
                config, os.path.join(experiment_dir, "best_wasserstein")
            )
            logger.info(f"New best validation Wasserstein distance: {best_val_wasserstein:.6f}")
            
        # Save best model based on MAE
        if val_metrics['mae'] < best_val_mae:
            best_val_mae = val_metrics['mae']
            save_model(
                model, optimizer, epoch, train_loss, val_loss, val_metrics,
                config, os.path.join(experiment_dir, "best_mae")
            )
            logger.info(f"New best validation MAE: {best_val_mae:.6f}")
            
        # Save best model based on Median AE
        if val_metrics['median_ae'] < best_val_median_ae:
            best_val_median_ae = val_metrics['median_ae']
            save_model(
                model, optimizer, epoch, train_loss, val_loss, val_metrics,
                config, os.path.join(experiment_dir, "best_median_ae")
            )
            logger.info(f"New best validation Median AE: {best_val_median_ae:.6f}")
        
        # Check for early stopping
        if config['training']['early_stopping_metric'] == 'wasserstein':
            if early_stopping(val_metrics['wasserstein']):
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break
        elif config['training']['early_stopping_metric'] == 'mae':
            if early_stopping(val_metrics['mae']):
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break
        elif config['training']['early_stopping_metric'] == 'median_ae':
            if early_stopping(val_metrics['median_ae']):
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break
        else:
            if early_stopping(val_loss):
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break
    
    # Evaluate on test set using best model (based on Wasserstein)
    logger.info("Evaluating best model (based on Wasserstein) on test set")
    best_wasserstein_model_path = os.path.join(experiment_dir, "best_wasserstein", f"checkpoint_epoch_{epoch}.pt")
    if os.path.exists(best_wasserstein_model_path):
        model, _, _, _ = load_model(model, None, best_wasserstein_model_path, device)
        test_loss, test_metrics = evaluate(model, data_loaders['val'], criterion, device, config)
        
        logger.info(f"Test Loss (Wasserstein model): {test_loss:.6f}")
        logger.info(f"Test MAE (Wasserstein model): {test_metrics['mae']:.6f}")
        logger.info(f"Test Median AE (Wasserstein model): {test_metrics['median_ae']:.6f}")
        logger.info(f"Test Wasserstein (Wasserstein model): {test_metrics['wasserstein']:.6f}")
        
        # Log detailed test threshold metrics for Wasserstein model
        logger.info("=== Test Threshold Classification Metrics (Wasserstein model) ===")
        logger.info("Class Distribution (Ground Truth):")
        for class_name, stats in test_metrics['threshold_metrics']['class_distribution'].items():
            logger.info(f"  {class_name}: {stats['target_count']} samples ({stats['percentage_target']:.2f}%)")
            
        logger.info("Class Distribution (Predictions):")
        for class_name, stats in test_metrics['threshold_metrics']['class_distribution'].items():
            logger.info(f"  {class_name}: {stats['pred_count']} samples ({stats['percentage_pred']:.2f}%)")
        
        logger.info(f"Classification Accuracy: {test_metrics['threshold_metrics']['accuracy']:.4f}")
        
        logger.info("AUPRC Metrics:")
        for threshold, auprc_data in test_metrics['threshold_metrics']['auprc'].items():
            if "error" in auprc_data:
                logger.info(f"  {threshold}: Error - {auprc_data['error']}")
            else:
                logger.info(f"  {threshold}: AUPRC = {auprc_data['average_precision']:.4f} (Positive rate: {auprc_data['positive_rate']:.2f}%)")
        
        # Log test metrics to TensorBoard
        writer.add_scalar('test_wasserstein_model/loss', test_loss, 0)
        writer.add_scalar('test_wasserstein_model/mae', test_metrics['mae'], 0)
        writer.add_scalar('test_wasserstein_model/median_ae', test_metrics['median_ae'], 0)
        writer.add_scalar('test_wasserstein_model/wasserstein', test_metrics['wasserstein'], 0)
        writer.add_scalar('test_wasserstein_model/classification_accuracy', test_metrics['threshold_metrics']['accuracy'], 0)
        
        # Add AUPRC metrics
        for threshold, auprc_data in test_metrics['threshold_metrics']['auprc'].items():
            if "average_precision" in auprc_data:
                writer.add_scalar(f'test_wasserstein_model/auprc/{threshold}', auprc_data['average_precision'], 0)
    
    # Evaluate on test set using best model (based on MAE)
    logger.info("Evaluating best model (based on MAE) on test set")
    best_mae_model_path = os.path.join(experiment_dir, "best_mae", f"checkpoint_epoch_{epoch}.pt")
    if os.path.exists(best_mae_model_path):
        model, _, _, _ = load_model(model, None, best_mae_model_path, device)
        test_loss, test_metrics = evaluate(model, data_loaders['val'], criterion, device, config)
        
        logger.info(f"Test Loss (MAE model): {test_loss:.6f}")
        logger.info(f"Test MAE (MAE model): {test_metrics['mae']:.6f}")
        logger.info(f"Test Median AE (MAE model): {test_metrics['median_ae']:.6f}")
        logger.info(f"Test Wasserstein (MAE model): {test_metrics['wasserstein']:.6f}")
        
        # Log detailed test threshold metrics for MAE model
        logger.info("=== Test Threshold Classification Metrics (MAE model) ===")
        logger.info("Class Distribution (Ground Truth):")
        for class_name, stats in test_metrics['threshold_metrics']['class_distribution'].items():
            logger.info(f"  {class_name}: {stats['target_count']} samples ({stats['percentage_target']:.2f}%)")
            
        logger.info("Class Distribution (Predictions):")
        for class_name, stats in test_metrics['threshold_metrics']['class_distribution'].items():
            logger.info(f"  {class_name}: {stats['pred_count']} samples ({stats['percentage_pred']:.2f}%)")
        
        logger.info(f"Classification Accuracy: {test_metrics['threshold_metrics']['accuracy']:.4f}")
        
        logger.info("AUPRC Metrics:")
        for threshold, auprc_data in test_metrics['threshold_metrics']['auprc'].items():
            if "error" in auprc_data:
                logger.info(f"  {threshold}: Error - {auprc_data['error']}")
            else:
                logger.info(f"  {threshold}: AUPRC = {auprc_data['average_precision']:.4f} (Positive rate: {auprc_data['positive_rate']:.2f}%)")
        
        # Log test metrics to TensorBoard
        writer.add_scalar('test_mae_model/loss', test_loss, 0)
        writer.add_scalar('test_mae_model/mae', test_metrics['mae'], 0)
        writer.add_scalar('test_mae_model/median_ae', test_metrics['median_ae'], 0)
        writer.add_scalar('test_mae_model/wasserstein', test_metrics['wasserstein'], 0)
        writer.add_scalar('test_mae_model/classification_accuracy', test_metrics['threshold_metrics']['accuracy'], 0)
        
        # Add AUPRC metrics
        for threshold, auprc_data in test_metrics['threshold_metrics']['auprc'].items():
            if "average_precision" in auprc_data:
                writer.add_scalar(f'test_mae_model/auprc/{threshold}', auprc_data['average_precision'], 0)
    
    # Evaluate on test set using best model (based on Median AE)
    logger.info("Evaluating best model (based on Median AE) on test set")
    best_median_ae_model_path = os.path.join(experiment_dir, "best_median_ae", f"checkpoint_epoch_{epoch}.pt")
    if os.path.exists(best_median_ae_model_path):
        model, _, _, _ = load_model(model, None, best_median_ae_model_path, device)
        test_loss, test_metrics = evaluate(model, data_loaders['val'], criterion, device, config)
        
        logger.info(f"Test Loss (Median AE model): {test_loss:.6f}")
        logger.info(f"Test MAE (Median AE model): {test_metrics['mae']:.6f}")
        logger.info(f"Test Median AE (Median AE model): {test_metrics['median_ae']:.6f}")
        logger.info(f"Test Wasserstein (Median AE model): {test_metrics['wasserstein']:.6f}")
        
        # Log detailed test threshold metrics for Median AE model
        logger.info("=== Test Threshold Classification Metrics (Median AE model) ===")
        logger.info("Class Distribution (Ground Truth):")
        for class_name, stats in test_metrics['threshold_metrics']['class_distribution'].items():
            logger.info(f"  {class_name}: {stats['target_count']} samples ({stats['percentage_target']:.2f}%)")
            
        logger.info("Class Distribution (Predictions):")
        for class_name, stats in test_metrics['threshold_metrics']['class_distribution'].items():
            logger.info(f"  {class_name}: {stats['pred_count']} samples ({stats['percentage_pred']:.2f}%)")
        
        logger.info(f"Classification Accuracy: {test_metrics['threshold_metrics']['accuracy']:.4f}")
        
        logger.info("AUPRC Metrics:")
        for threshold, auprc_data in test_metrics['threshold_metrics']['auprc'].items():
            if "error" in auprc_data:
                logger.info(f"  {threshold}: Error - {auprc_data['error']}")
            else:
                logger.info(f"  {threshold}: AUPRC = {auprc_data['average_precision']:.4f} (Positive rate: {auprc_data['positive_rate']:.2f}%)")
        
        # Log test metrics to TensorBoard
        writer.add_scalar('test_median_ae_model/loss', test_loss, 0)
        writer.add_scalar('test_median_ae_model/mae', test_metrics['mae'], 0)
        writer.add_scalar('test_median_ae_model/median_ae', test_metrics['median_ae'], 0)
        writer.add_scalar('test_median_ae_model/wasserstein', test_metrics['wasserstein'], 0)
        writer.add_scalar('test_median_ae_model/classification_accuracy', test_metrics['threshold_metrics']['accuracy'], 0)
        
        # Add AUPRC metrics
        for threshold, auprc_data in test_metrics['threshold_metrics']['auprc'].items():
            if "average_precision" in auprc_data:
                writer.add_scalar(f'test_median_ae_model/auprc/{threshold}', auprc_data['average_precision'], 0)
    
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