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
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse

from src.utils.logger import logger
from src.training.lsmt.focal_loss import FocalLoss
from src.training.lsmt_fusion.BiGRU_late_fusion_model import BiGRULateFusionModel
#from src.training.lsmt.lsmt_fusion.lstm_late_fusion_model import LSTMLateFusionModel
from src.training.lsmt_fusion.lstm_late_fusion_dataset import create_data_loaders
#from src.training.lsmt.lsmt_fusion.attention_weight import visualize_attention_weights
from src.training.lsmt_fusion.train import train
from src.training.lsmt_fusion.evaluate import evaluate
#from src.training.lsmt.lsmt_fusion.watch_weight import visualize_lstm_gradients

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



'''def save_model(model, optimizer, epoch, train_loss, val_loss, metrics, config, save_dir):
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
    logger.info(f"Saved checkpoint to {checkpoint_path}")'''


def load_model(model, optimizer, checkpoint_path, device):
    """
    Load model checkpoint for continuing training or evaluation.
    
    Args:
        model: LSTM Late Fusion model instance
        optimizer: Optimizer instance (can be None if just for evaluation)
        checkpoint_path: Path to the checkpoint file
        device: Device to load the model to
        
    Returns:
        Tuple of (model, optimizer, start_epoch, checkpoint_data)
    """
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device,weights_only=False)
    # Load model state
    model.load_state_dict(checkpoint)
    #model.load_state_dict(checkpoint['model_state_dict'])
    # Load optimizer state if provided
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Get starting epoch (checkpoint's epoch + 1)
    start_epoch = checkpoint.get('epoch', 0) + 1
    
    logger.info(f"Loaded model from epoch {checkpoint.get('epoch', 0)}")
    
    #return model, optimizer, start_epoch, checkpoint
    logger.error(f"return model")
    return model

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
    experiment_name = f"lstm_late_fusion_{timestamp}"
    if args.experiment_name:
        experiment_name = args.experiment_name
    
    experiment_dir = os.path.join(config['paths']['output_dir'],'model_save', experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)
    
    # Set up logging
    log_file = os.path.join("experiments/logs", f"lstm_late_fusion_training_{timestamp}.log")
    
    # Set up TensorBoard writer
    tensorboard_dir = os.path.join("experiments/lstm_late_fusion/tensorboard", experiment_name)
    os.makedirs(tensorboard_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=tensorboard_dir)
    logger.info(f"TensorBoard logs will be saved to {tensorboard_dir}")
    
    # Create data loaders
    data_loaders = create_data_loaders(
        lstm_data_dir=config['paths']['lstm_data_dir'],
        stat_features_dir=config['paths']['stat_features_dir'],
        batch_size=config['training']['batch_size'],
        num_workers=config['training']['num_workers'],
        component=config['data']['component']
    )
    
    # Create model
    model = BiGRULateFusionModel(config=config['model'])
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
        '''      model, optimizer, start_epoch, checkpoint = load_model(
            model, optimizer, args.load_model, device
        )
        
        # Get best metrics from checkpoint if available
        if 'val_loss' in checkpoint:
            best_val_loss = checkpoint['val_loss']
            logger.info(f"Loaded best validation loss: {best_val_loss:.4f}")
        
        if 'metrics' in checkpoint and 'f1' in checkpoint['metrics']:
            best_f1 = checkpoint['metrics']['f1']
            logger.info(f"Loaded best F1 score: {best_f1:.4f}")
         '''

                  
        model= load_model(
        model, None, args.load_model, device
    )

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
        
    if config['training'].get('use_class_weights', False):
        logger.info(f"Using class weights: {class_weights}")
        criterion = nn.CrossEntropyLoss(weight=class_weights)

    elif config['training'].get('use_focal_loss', True):
        # Use Focal Loss
        alpha = config['training'].get('focal_loss_alpha', 0.25)
        gamma = config['training'].get('focal_loss_gamma', 2.0)
        logger.info(f"Using Focal Loss with alpha={alpha}, gamma={gamma}")
        criterion = FocalLoss(alpha=alpha, gamma=gamma)
 
    logger.info(f"Using criterion: {criterion}")
    # Learning rate scheduler
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
    )
    '''
    # Initialize Early Stopping
    early_stopping_metric = config['training'].get('early_stopping_metric', 'loss')
    early_stopping_patience = config['training'].get('early_stopping_patience', 10)
    early_stopping_min_delta = config['training'].get('early_stopping_min_delta', 0.0001)
    
    # Set mode based on metric (min for loss, max for F1)
    early_stopping_mode = 'min' if early_stopping_metric == 'loss' else 'max'
    
    logger.info(f"Using Early Stopping with patience={early_stopping_patience}, "
               f"metric={early_stopping_metric}, mode={early_stopping_mode}")
    
    early_stopping = EarlyStopping(
        patience=early_stopping_patience,
        min_delta=early_stopping_min_delta,
        mode=early_stopping_mode
    )
    '''
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
        logger.info(f"current best Threshold: {threshold}")
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
            'threshold': eval_threshold,
            'train_accuracy': train_accuracy,
            'train_precision': train_precision,
            'train_recall': train_recall,
            'train_f1': train_f1,
            'train_confusion_matrix': train_conf_matrix.tolist()
        }
        
        '''# Save checkpoint
        save_model(
            model, optimizer, epoch, train_loss, val_loss,
            metrics, config, experiment_dir
        )'''
        
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
        '''
        # Check early stopping
        early_stopping_value = val_loss if early_stopping_metric == 'loss' else f1
       if early_stopping(early_stopping_value):
            logger.info(f"Early stopping triggered after {epoch} epochs")
            logger.info(f"Best {early_stopping_metric}: {early_stopping.best_score:.4f}")
            break'''
        
        # Force garbage collection
        gc.collect()
        
        # Visualize attention weights
        #visualize_attention_weights(val_attn_weights, experiment_dir, epoch)
    
    # Close TensorBoard writer
    writer.close()
    
    # Evaluate on test set using best model
    if args.evaluate_test:
        logger.info("Evaluating best model on test set")
        
        # Load best model by F1 score
        best_model_path = os.path.join(experiment_dir, 'best_model_f1.pt')
        model, _, _, _ = load_model(model, None, best_model_path, device)
        
        test_results = evaluate(
            model, data_loaders['test'], criterion, device, threshold=eval_threshold
        )
        test_loss, test_accuracy, test_precision, test_recall, test_f1, test_conf_matrix = test_results
        
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
           # 'threshold': best_test_threshold
        }
        
        test_metrics_path = os.path.join(experiment_dir, 'test_metrics.yaml')
        with open(test_metrics_path, 'w') as f:
            yaml.dump(test_metrics, f)
        
        logger.info(f"Saved test metrics to {test_metrics_path}")
        
        # Visualize attention weights
        #visualize_attention_weights(test_attn_weights, experiment_dir, 0, prefix='test')
    
    logger.info("Training complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LSTM Late Fusion model")
    parser.add_argument("--config", type=str, default="configs/lstm_late_fusion.yaml",
                       help="Path to configuration file")
    parser.add_argument("--load_model", type=str, default=None,
                       help="Path to pretrained model checkpoint to continue training")
    parser.add_argument("--experiment_name", type=str, default=None,
                       help="Custom experiment name for output directory")
    parser.add_argument("--evaluate_test", action="store_true",
                       help="Evaluate the best model on test set after training")
    args = parser.parse_args()
    
    main(args) 