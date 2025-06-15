"""
Training script for Transformer model with binary classification for energy anomaly detection.

This script trains a Transformer model using PyTorch's nn.TransformerEncoder
for energy anomaly detection with binary classification (normal:0 / abnormal:1) 
using a threshold of 0.5 on the model outputs.
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
from sklearn.metrics import mean_absolute_error, median_absolute_error, precision_recall_curve, average_precision_score, precision_score, recall_score, f1_score
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


def calculate_binary_metrics(targets, predictions, threshold=0.5):
    """
    Calculate binary classification metrics using a threshold of 0.5.
    
    Args:
        targets: Ground truth values (numpy array)
        predictions: Predicted values (numpy array)
        threshold: Threshold for binary classification (default: 0.5)
        
    Returns:
        Dictionary with binary classification metrics
    """
    # Convert to binary labels using threshold
    binary_targets = (targets >= threshold).astype(int)
    binary_preds = (predictions >= threshold).astype(int)
    
    # Calculate class distribution
    class_names = ["Normal (0)", "Abnormal (1)"]
    class_counts = {}
    for i in range(len(class_names)):
        target_count = np.sum(binary_targets == i)
        pred_count = np.sum(binary_preds == i)
        class_counts[class_names[i]] = {
            "target_count": int(target_count),
            "pred_count": int(pred_count),
            "percentage_target": float(target_count) / len(targets) * 100 if len(targets) > 0 else 0,
            "percentage_pred": float(pred_count) / len(predictions) * 100 if len(predictions) > 0 else 0
        }
    
    # Calculate confusion metrics
    accuracy = np.mean(binary_targets == binary_preds)
    
    # Calculate precision, recall, f1
    try:
        prec = precision_score(binary_targets, binary_preds)
        rec = recall_score(binary_targets, binary_preds)
        f1 = f1_score(binary_targets, binary_preds)
    except Exception as e:
        prec, rec, f1 = 0.0, 0.0, 0.0
        logger.warning(f"Error calculating precision/recall/f1: {str(e)}")
    
    # Calculate AUPRC
    try:
        ap_score = average_precision_score(binary_targets, predictions)
        precision, recall, _ = precision_recall_curve(binary_targets, predictions)
        auprc = {
            "average_precision": float(ap_score),
            "positive_samples": int(np.sum(binary_targets)),
            "positive_rate": float(np.mean(binary_targets)) * 100
        }
    except Exception as e:
        auprc = {
            "error": str(e),
            "positive_samples": int(np.sum(binary_targets)),
            "positive_rate": float(np.mean(binary_targets)) * 100
        }
    
    return {
        "class_distribution": class_counts,
        "accuracy": float(accuracy),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "auprc": auprc
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

class QuantileLoss(nn.Module):
    """
    Quantile Loss for quantile regression.
    
    This loss is useful for predicting specific quantiles of the target distribution,
    which is helpful for uncertainty estimation or asymmetric error penalties.
    
    Args:
        quantile (float): Quantile to predict, between 0 and 1
                         0.5 corresponds to median regression (symmetric loss)
                         Lower values penalize overestimation more
                         Higher values penalize underestimation more
    """
    def __init__(self, quantile=0.5):
        super(QuantileLoss, self).__init__()
        self.quantile = quantile
        
    def forward(self, pred, target):
        # Calculate quantile loss
        errors = target - pred
        losses = torch.max(self.quantile * errors, (self.quantile - 1) * errors)
        return torch.mean(losses)


class DynamicWeightedMSELoss(nn.Module):
    """
    Dynamically Weighted MSE Loss that calculates weights inversely proportional to the frequency
    of target values in the batch. This puts more emphasis on rare target values.
    
    Args:
        num_bins (int): Number of bins to divide the target range [0,1] into
        beta (float): Smoothing factor for weight calculation, higher values give more weight to rare values
        min_weight (float): Minimum weight to apply to any sample
    """
    def __init__(self, num_bins=20, beta=0.9, min_weight=1.0):
        super(DynamicWeightedMSELoss, self).__init__()
        self.num_bins = num_bins
        self.beta = beta
        self.min_weight = min_weight
        
    def forward(self, pred, target):
        # Calculate squared error
        squared_error = (pred - target) ** 2
        
        # Calculate histogram of target values
        bin_edges = torch.linspace(0, 1, self.num_bins + 1, device=target.device)
        bin_width = 1.0 / self.num_bins
        
        # Assign each target value to a bin
        target_bins = torch.floor(target / bin_width).long()
        target_bins = torch.clamp(target_bins, 0, self.num_bins - 1)
        
        # Count samples in each bin
        bin_counts = torch.zeros(self.num_bins, device=target.device)
        for i in range(self.num_bins):
            bin_counts[i] = torch.sum(target_bins == i).float()
            
        # Avoid division by zero
        bin_counts = torch.clamp(bin_counts, min=1.0)
        
        # Calculate weights inversely proportional to bin counts
        bin_weights = torch.pow(bin_counts, -self.beta)
        
        # Normalize weights
        if torch.sum(bin_weights) > 0:
            bin_weights = bin_weights / torch.sum(bin_weights) * self.num_bins
        
        # Ensure minimum weight
        bin_weights = torch.clamp(bin_weights, min=self.min_weight)
        
        # Assign weights to each sample based on its bin
        sample_weights = torch.zeros_like(target)
        for i in range(len(target)):
            bin_idx = target_bins[i]
            sample_weights[i] = bin_weights[bin_idx]
        
        # Apply weights to squared errors
        weighted_squared_error = sample_weights * squared_error
        
        # Return mean of weighted squared errors
        return torch.mean(weighted_squared_error)


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
        Average training loss for the epoch and metrics
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
        #outputs = torch.clamp(outputs,min=1e-5, max=5.0)
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
    
    # Calculate binary classification metrics
    binary_metrics = calculate_binary_metrics(all_targets, all_outputs)

    metrics = {
        'mae': mae,
        'median_ae': median_ae,
        'wasserstein': w_distance,
        'precision': binary_metrics['precision'],
        'recall': binary_metrics['recall'],
        'f1': binary_metrics['f1'],
        'accuracy': binary_metrics['accuracy']
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
            #outputs = torch.clamp(outputs, min=1e-5, max=5.0)
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
                        'actual': targets[i].item(),
                        'binary_pred': 1 if outputs[i].item() >= 0.5 else 0,
                        'binary_actual': 1 if targets[i].item() >= 0.5 else 0
                    })

    avg_loss = total_loss / len(data_loader)

    if print_samples and sample_outputs:
        logger.info("\n===== Sample Outputs =====")
        for i, sample in enumerate(sample_outputs):
            logger.info(f"Sample {i+1}:")
            logger.info(f"  Prediction: {sample['prediction']:.6f} (Binary: {sample['binary_pred']})")
            logger.info(f"  Actual: {sample['actual']:.6f} (Binary: {sample['binary_actual']})")
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

    # Calculate binary classification metrics
    binary_metrics = calculate_binary_metrics(all_targets, all_outputs)
    
    metrics = {
        'mae': mae,
        'median_ae': median_ae,
        'wasserstein': w_distance,
        'predictions': all_outputs,
        'targets': all_targets,
        'binary_metrics': binary_metrics,
        'precision': binary_metrics['precision'],
        'recall': binary_metrics['recall'],
        'f1': binary_metrics['f1'],
        'accuracy': binary_metrics['accuracy']
    }

    return avg_loss, metrics


def main(args):
    """
    Main function to train the Transformer model with binary classification.
    
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
    experiment_name = f"transformer_binary_{timestamp}"
    if args.experiment_name:
        experiment_name = args.experiment_name
    
    experiment_dir = os.path.join(config['paths']['output_dir'], 'model_save', experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)
    
    # Set up TensorBoard writer
    tensorboard_dir = os.path.join("experiments/transformer_binary/tensorboard", experiment_name)
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
    elif config['training']['loss'] == 'dynamic_weighted_mse':
        num_bins = config['training'].get('dynamic_weighted_mse_num_bins', 20)
        beta = config['training'].get('dynamic_weighted_mse_beta', 0.9)
        min_weight = config['training'].get('dynamic_weighted_mse_min_weight', 1.0)
        criterion = DynamicWeightedMSELoss(num_bins=num_bins, beta=beta, min_weight=min_weight)
        logger.info(f"Using Dynamic Weighted MSE Loss with num_bins: {num_bins}, beta: {beta}, min_weight: {min_weight}")
    elif config['training']['loss'] == 'quantile':
        quantile = config['training'].get('quantile_value', 0.5)
        criterion = QuantileLoss(quantile=quantile)
        logger.info(f"Using Quantile Loss with quantile: {quantile}")
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
        mode='min'  # Use 'min' for loss and 'max' for f1
    )
    
    # Initialize best metrics
    best_val_loss = float('inf')
    best_val_f1 = 0.0
    best_val_accuracy = 0.0
    
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
            if config['training']['early_stopping_metric'] == 'f1':
                scheduler.step(-val_metrics['f1'])  # Negate because ReduceLROnPlateau uses 'min' mode
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
        logger.info(f"Train F1: {train_metrics['f1']:.6f}, Val F1: {val_metrics['f1']:.6f}")
        logger.info(f"Train Precision: {train_metrics['precision']:.6f}, Val Precision: {val_metrics['precision']:.6f}")
        logger.info(f"Train Recall: {train_metrics['recall']:.6f}, Val Recall: {val_metrics['recall']:.6f}")
        logger.info(f"Train Accuracy: {train_metrics['accuracy']:.6f}, Val Accuracy: {val_metrics['accuracy']:.6f}")
        
        # Log binary classification metrics
        logger.info("=== Validation Binary Classification Metrics ===")
        # Log class distribution
        logger.info("Class Distribution (Ground Truth):")
        for class_name, stats in val_metrics['binary_metrics']['class_distribution'].items():
            logger.info(f"  {class_name}: {stats['target_count']} samples ({stats['percentage_target']:.2f}%)")
            
        logger.info("Class Distribution (Predictions):")
        for class_name, stats in val_metrics['binary_metrics']['class_distribution'].items():
            logger.info(f"  {class_name}: {stats['pred_count']} samples ({stats['percentage_pred']:.2f}%)")
        
        # Log AUPRC metrics
        auprc_data = val_metrics['binary_metrics']['auprc']
        if "error" in auprc_data:
            logger.info(f"  AUPRC Error: {auprc_data['error']}")
        else:
            logger.info(f"  AUPRC: {auprc_data['average_precision']:.4f} (Positive rate: {auprc_data['positive_rate']:.2f}%)")
        
        logger.info(f"Epoch completed in {epoch_duration:.2f} seconds")
        
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('train/mae', train_metrics['mae'], epoch)
        writer.add_scalar('val/mae', val_metrics['mae'], epoch)
        writer.add_scalar('train/f1', train_metrics['f1'], epoch)
        writer.add_scalar('val/f1', val_metrics['f1'], epoch)
        writer.add_scalar('train/precision', train_metrics['precision'], epoch)
        writer.add_scalar('val/precision', val_metrics['precision'], epoch)
        writer.add_scalar('train/recall', train_metrics['recall'], epoch)
        writer.add_scalar('val/recall', val_metrics['recall'], epoch)
        writer.add_scalar('train/accuracy', train_metrics['accuracy'], epoch)
        writer.add_scalar('val/accuracy', val_metrics['accuracy'], epoch)
        
        # Add AUPRC metrics to TensorBoard
        if "average_precision" in val_metrics['binary_metrics']['auprc']:
            writer.add_scalar('val/auprc', val_metrics['binary_metrics']['auprc']['average_precision'], epoch)
        
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
            
        # Save best model based on F1 score
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            save_model(
                model, optimizer, epoch, train_loss, val_loss, val_metrics,
                config, os.path.join(experiment_dir, "best_f1")
            )
            logger.info(f"New best validation F1: {best_val_f1:.6f}")
            
        # Save best model based on accuracy
        if val_metrics['accuracy'] > best_val_accuracy:
            best_val_accuracy = val_metrics['accuracy']
            save_model(
                model, optimizer, epoch, train_loss, val_loss, val_metrics,
                config, os.path.join(experiment_dir, "best_accuracy")
            )
            logger.info(f"New best validation accuracy: {best_val_accuracy:.6f}")
        
        # Check for early stopping
        if config['training']['early_stopping_metric'] == 'f1':
            if early_stopping(-val_metrics['f1']):  # Negate because EarlyStopping uses 'min' mode
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break
        else:
            if early_stopping(val_loss):
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break
    
    # Evaluate on test set using best model (based on F1)
    logger.info("Evaluating best model (based on F1) on test set")
    best_f1_model_path = os.path.join(experiment_dir, "best_f1", f"checkpoint_epoch_{epoch}.pt")
    if os.path.exists(best_f1_model_path):
        model, _, _, _ = load_model(model, None, best_f1_model_path, device)
        test_loss, test_metrics = evaluate(model, data_loaders['val'], criterion, device, config)
        
        logger.info(f"Test Loss (F1 model): {test_loss:.6f}")
        logger.info(f"Test F1 (F1 model): {test_metrics['f1']:.6f}")
        logger.info(f"Test Precision (F1 model): {test_metrics['precision']:.6f}")
        logger.info(f"Test Recall (F1 model): {test_metrics['recall']:.6f}")
        logger.info(f"Test Accuracy (F1 model): {test_metrics['accuracy']:.6f}")
        
        # Log detailed test binary metrics for F1 model
        logger.info("=== Test Binary Classification Metrics (F1 model) ===")
        logger.info("Class Distribution (Ground Truth):")
        for class_name, stats in test_metrics['binary_metrics']['class_distribution'].items():
            logger.info(f"  {class_name}: {stats['target_count']} samples ({stats['percentage_target']:.2f}%)")
            
        logger.info("Class Distribution (Predictions):")
        for class_name, stats in test_metrics['binary_metrics']['class_distribution'].items():
            logger.info(f"  {class_name}: {stats['pred_count']} samples ({stats['percentage_pred']:.2f}%)")
        
        # Log AUPRC metrics
        auprc_data = test_metrics['binary_metrics']['auprc']
        if "error" in auprc_data:
            logger.info(f"  AUPRC Error: {auprc_data['error']}")
        else:
            logger.info(f"  AUPRC: {auprc_data['average_precision']:.4f} (Positive rate: {auprc_data['positive_rate']:.2f}%)")
        
        # Log test metrics to TensorBoard
        writer.add_scalar('test_f1_model/loss', test_loss, 0)
        writer.add_scalar('test_f1_model/f1', test_metrics['f1'], 0)
        writer.add_scalar('test_f1_model/precision', test_metrics['precision'], 0)
        writer.add_scalar('test_f1_model/recall', test_metrics['recall'], 0)
        writer.add_scalar('test_f1_model/accuracy', test_metrics['accuracy'], 0)
        
        if "average_precision" in test_metrics['binary_metrics']['auprc']:
            writer.add_scalar('test_f1_model/auprc', test_metrics['binary_metrics']['auprc']['average_precision'], 0)
    
    # Close TensorBoard writer
    writer.close()
    
    logger.info("Training completed successfully")


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train Transformer model with binary classification for energy anomaly detection")
    parser.add_argument("--config", type=str, default=None, help="Path to configuration file")
    parser.add_argument("--experiment_name", type=str, default=None, help="Name of the experiment")
    parser.add_argument("--test", action="store_true", help="Test the model")
    parser.add_argument("--load_model", type=str, default=None, help="Path to checkpoint to load model from")
    args = parser.parse_args()
    
    # Run main function
    main(args) 