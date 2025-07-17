"""
Training script for FEDformer model for energy anomaly detection.

This script trains a FEDformer model using Fourier Enhanced Decomposition
for energy anomaly detection with enhanced frequency domain features.
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
from sklearn.metrics import (precision_recall_fscore_support, accuracy_score, 
                           confusion_matrix, precision_recall_curve, f1_score, 
                           auc, average_precision_score, roc_auc_score)
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import argparse
import time
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.append('src')

from utils.logger import logger
from models.fedformer_anomaly_detection import FEDformerAnomalyDetector, create_fedformer_anomaly_model
from training.fedformer.dataset_fedformer import create_fedformer_data_loaders, get_feature_info

class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance in anomaly detection
    """
    def __init__(self, alpha=1, gamma=2, logits=False, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            ce_loss = F.cross_entropy(inputs, targets, reduce=False)
        else:
            ce_loss = nn.CrossEntropyLoss(reduce=False)(inputs, targets)
            
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss

        if self.reduce:
            return torch.mean(focal_loss)
        else:
            return focal_loss

class FEDformerLoss(nn.Module):
    """
    Combined loss for FEDformer anomaly detection
    """
    def __init__(self, 
                 classification_weight=1.0, 
                 reconstruction_weight=0.5,
                 anomaly_weight=0.3,
                 use_focal=True,
                 focal_alpha=1.0,
                 focal_gamma=2.0):
        super().__init__()
        self.classification_weight = classification_weight
        self.reconstruction_weight = reconstruction_weight
        self.anomaly_weight = anomaly_weight
        self.use_focal = use_focal
        
        if use_focal:
            self.classification_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma, logits=True)
        else:
            self.classification_loss = nn.CrossEntropyLoss()
        
        self.reconstruction_loss = nn.MSELoss()
        self.anomaly_loss = nn.BCELoss()
    
    def forward(self, outputs, targets, seq_x=None):
        """
        Calculate combined loss
        
        Args:
            outputs: Dict with 'logits', 'anomaly_scores', 'encoded_features'
            targets: Anomaly labels [B, L]
            seq_x: Original input sequences [B, L, F] for reconstruction
        """
        logits = outputs['logits']  # [B, L, num_classes]
        anomaly_scores = outputs['anomaly_scores']  # [B, L]
        
        # Classification loss
        B, L, _ = logits.shape
        logits_flat = logits.view(-1, logits.size(-1))  # [B*L, num_classes]
        targets_flat = targets.view(-1)  # [B*L]
        
        cls_loss = self.classification_loss(logits_flat, targets_flat)
        
        # Anomaly score loss (treat as regression to binary labels)
        anomaly_targets = targets.float()  # [B, L]
        ano_loss = self.anomaly_loss(anomaly_scores, anomaly_targets)
        
        # Reconstruction loss (optional, for encoder regularization)
        recon_loss = 0.0
        if seq_x is not None and 'encoded_features' in outputs:
            # Simple reconstruction from encoded features to input
            encoded = outputs['encoded_features']  # [B, L, d_model]
            # This would require a reconstruction head in the model
            # For now, use a simple L2 regularization on encoded features
            recon_loss = torch.mean(encoded ** 2) * 0.001
        
        # Combine losses
        total_loss = (self.classification_weight * cls_loss + 
                     self.anomaly_weight * ano_loss + 
                     self.reconstruction_weight * recon_loss)
        
        return {
            'total_loss': total_loss,
            'classification_loss': cls_loss,
            'anomaly_loss': ano_loss,
            'reconstruction_loss': recon_loss
        }

def point_adjustment(gt, pred):
    """
    Point adjustment strategy for anomaly detection evaluation.
    Same as in the original transformer training script.
    """
    gt = gt.copy()
    pred = pred.copy()
    anomaly_state = False
    
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            # Adjust backward
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            # Adjust forward
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
    """Evaluate with point adjustment"""
    flat_labels_adj, adjusted_preds = point_adjustment(flat_labels, flat_preds)
    
    # Calculate metrics
    adj_accuracy = accuracy_score(flat_labels_adj, adjusted_preds)
    adj_precision, adj_recall, adj_f1, _ = precision_recall_fscore_support(
        flat_labels_adj, adjusted_preds, average='binary', zero_division=0
    )
    
    adjusted_preds_count = np.sum(adjusted_preds == 1)
    
    return {
        'adj_accuracy': adj_accuracy,
        'adj_precision': adj_precision,
        'adj_recall': adj_recall,
        'adj_f1': adj_f1,
        'adjusted_preds_count': adjusted_preds_count
    }

def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    total_cls_loss = 0.0
    total_ano_loss = 0.0
    total_recon_loss = 0.0
    
    all_predictions = []
    all_labels = []
    all_anomaly_scores = []
    
    progress_bar = tqdm(dataloader, desc=f'Training Epoch {epoch}')
    
    for batch_idx, batch in enumerate(progress_bar):
        seq_x, seq_y, seq_x_mark, seq_y_mark, labels = batch
        seq_x = seq_x.to(device).float()
        seq_y = seq_y.to(device).float()
        seq_x_mark = seq_x_mark.to(device).float()
        seq_y_mark = seq_y_mark.to(device).float()
        labels = labels.to(device).long()
        
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(seq_x)
        
        # Calculate loss
        loss_dict = criterion(outputs, labels, seq_x)
        total_loss_batch = loss_dict['total_loss']
        
        # Backward pass
        total_loss_batch.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Accumulate losses
        total_loss += total_loss_batch.item()
        total_cls_loss += loss_dict['classification_loss'].item()
        total_ano_loss += loss_dict['anomaly_loss'].item()
        total_recon_loss += loss_dict['reconstruction_loss']
        
        # Get predictions for metrics
        logits = outputs['logits']  # [B, L, num_classes]
        anomaly_scores = outputs['anomaly_scores']  # [B, L]
        
        predictions = torch.argmax(logits, dim=-1)  # [B, L]
        
        all_predictions.extend(predictions.cpu().numpy().flatten())
        all_labels.extend(labels.cpu().numpy().flatten())
        all_anomaly_scores.extend(anomaly_scores.detach().cpu().numpy().flatten())
        
        # Update progress bar
        progress_bar.set_postfix({
            'Loss': f'{total_loss_batch.item():.4f}',
            'CLS': f'{loss_dict["classification_loss"].item():.4f}',
            'ANO': f'{loss_dict["anomaly_loss"].item():.4f}'
        })
    
    # Calculate epoch metrics
    avg_loss = total_loss / len(dataloader)
    avg_cls_loss = total_cls_loss / len(dataloader)
    avg_ano_loss = total_ano_loss / len(dataloader)
    avg_recon_loss = total_recon_loss / len(dataloader)
    
    # Calculate basic metrics
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_anomaly_scores = np.array(all_anomaly_scores)
    
    accuracy = accuracy_score(all_labels, all_predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_predictions, average='binary', zero_division=0
    )
    
    # ROC AUC with anomaly scores
    try:
        roc_auc = roc_auc_score(all_labels, all_anomaly_scores)
        pr_auc = average_precision_score(all_labels, all_anomaly_scores)
    except:
        roc_auc = 0.0
        pr_auc = 0.0
    
    return {
        'avg_loss': avg_loss,
        'avg_cls_loss': avg_cls_loss,
        'avg_ano_loss': avg_ano_loss,
        'avg_recon_loss': avg_recon_loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'predictions': all_predictions,
        'labels': all_labels,
        'anomaly_scores': all_anomaly_scores
    }

def validate_epoch(model, dataloader, criterion, device, epoch):
    """Validate for one epoch"""
    model.eval()
    total_loss = 0.0
    total_cls_loss = 0.0
    total_ano_loss = 0.0
    
    all_predictions = []
    all_labels = []
    all_anomaly_scores = []
    
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc=f'Validation Epoch {epoch}')
        
        for batch in progress_bar:
            seq_x, seq_y, seq_x_mark, seq_y_mark, labels = batch
            seq_x = seq_x.to(device).float()
            seq_y = seq_y.to(device).float()
            seq_x_mark = seq_x_mark.to(device).float()
            seq_y_mark = seq_y_mark.to(device).float()
            labels = labels.to(device).long()
            
            # Forward pass
            outputs = model(seq_x)
            
            # Calculate loss
            loss_dict = criterion(outputs, labels, seq_x)
            
            total_loss += loss_dict['total_loss'].item()
            total_cls_loss += loss_dict['classification_loss'].item()
            total_ano_loss += loss_dict['anomaly_loss'].item()
            
            # Get predictions
            logits = outputs['logits']
            anomaly_scores = outputs['anomaly_scores']
            predictions = torch.argmax(logits, dim=-1)
            
            all_predictions.extend(predictions.cpu().numpy().flatten())
            all_labels.extend(labels.cpu().numpy().flatten())
            all_anomaly_scores.extend(anomaly_scores.cpu().numpy().flatten())
    
    # Calculate metrics
    avg_loss = total_loss / len(dataloader)
    avg_cls_loss = total_cls_loss / len(dataloader)
    avg_ano_loss = total_ano_loss / len(dataloader)
    
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_anomaly_scores = np.array(all_anomaly_scores)
    
    accuracy = accuracy_score(all_labels, all_predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_predictions, average='binary', zero_division=0
    )
    
    # Calculate adjusted metrics
    adjusted_metrics = evaluate_with_adjustment(all_predictions, all_labels)
    
    # ROC AUC
    try:
        roc_auc = roc_auc_score(all_labels, all_anomaly_scores)
        pr_auc = average_precision_score(all_labels, all_anomaly_scores)
    except:
        roc_auc = 0.0
        pr_auc = 0.0
    
    return {
        'avg_loss': avg_loss,
        'avg_cls_loss': avg_cls_loss,
        'avg_ano_loss': avg_ano_loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'predictions': all_predictions,
        'labels': all_labels,
        'anomaly_scores': all_anomaly_scores,
        **adjusted_metrics
    }

def save_checkpoint(model, optimizer, scheduler, epoch, best_f1, checkpoint_dir):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'best_f1': best_f1,
    }
    
    checkpoint_path = os.path.join(checkpoint_dir, f'fedformer_epoch_{epoch}.pth')
    torch.save(checkpoint, checkpoint_path)
    logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    return checkpoint_path

def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None):
    """Load model checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    return checkpoint['epoch'], checkpoint.get('best_f1', 0.0)

def main():
    parser = argparse.ArgumentParser(description='Train FEDformer for Energy Anomaly Detection')
    parser.add_argument('--config', type=str, default='configs/fedformer_minute_level_config.yaml',
                       help='Path to config file')
    parser.add_argument('--data_dir', type=str, 
                       default='Data/row_energyData_subsample_Transform/labeled',
                       help='Path to data directory')
    parser.add_argument('--component', type=str, default='contact',
                       help='Component type (contact, pcb, ring)')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (cuda, cpu, or auto)')
    
    args = parser.parse_args()
    
    # Setup device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    logger.info(f"Using device: {device}")
    
    # Load or create config
    if os.path.exists(args.config):
        with open(args.config, 'r') as f:
            yaml_config = yaml.safe_load(f)
        logger.info(f"Loaded config from {args.config}")
        
        # Merge model and training configs, apply mode-specific overrides
        config = {**yaml_config['model'], **yaml_config['training']}
        
        # Apply mode-specific configurations
        config_mode = yaml_config.get('config_mode', 'standard')
        if config_mode == 'memory' and 'memory_mode' in yaml_config:
            memory_overrides = yaml_config['memory_mode']
            config.update(memory_overrides.get('model', {}))
            config.update(memory_overrides.get('training', {}))
            if 'data' in memory_overrides:
                config.update(memory_overrides['data'])
        elif config_mode == 'weekly' and 'weekly_mode' in yaml_config:
            weekly_overrides = yaml_config['weekly_mode']
            config.update(weekly_overrides.get('model', {}))
            config.update(weekly_overrides.get('training', {}))
            if 'data' in weekly_overrides:
                config.update(weekly_overrides['data'])
        
        # Add data config
        config.update(yaml_config.get('data', {}))
        
    else:
        logger.error(f"Config file not found: {args.config}")
        logger.error("请确保配置文件存在，或使用默认路径：configs/fedformer_minute_level_config.yaml")
        raise FileNotFoundError(f"Configuration file not found: {args.config}")
    
    # Get feature info from actual data
    logger.info("Getting feature information from data...")
    try:
        feature_info = get_feature_info(args.data_dir, config.get('component', args.component))
        config['enc_in'] = feature_info['num_features']
        config['dec_in'] = feature_info['num_features'] 
        config['c_out'] = feature_info['num_features']
        logger.info(f"Found {feature_info['num_features']} features in data")
        logger.info(f"Anomaly ratio: {feature_info['anomaly_ratio']:.3f}")
    except Exception as e:
        logger.warning(f"Could not get feature info from data: {e}")
        logger.warning("Using default enc_in=33 (update this based on your actual data)")
        config['enc_in'] = 33
        config['dec_in'] = 33
        config['c_out'] = 33
    
    # Create data loaders - pass all parameters from config
    logger.info("Creating data loaders...")
    data_loaders = create_fedformer_data_loaders(
        data_dir=args.data_dir,
        batch_size=config['batch_size'],
        component=config.get('component', args.component),
        seq_len=config['seq_len'],
        label_len=config['label_len'],
        pred_len=config['pred_len'],
        freq=config.get('freq', 't'),
        num_workers=config.get('num_workers', 4)
    )
    
    if 'train' not in data_loaders:
        raise ValueError("Training data loader not created!")
    
    # Create model
    logger.info("Creating FEDformer model...")
    model = create_fedformer_anomaly_model(config).to(device)
    
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create criterion
    criterion = FEDformerLoss(
        classification_weight=1.0,
        reconstruction_weight=0.1,
        anomaly_weight=0.5,
        use_focal=True
    )
    
    # Create optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=1e-4
    )
    
    # Create scheduler
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=config['epochs'],
        eta_min=config['learning_rate'] * 0.01
    )
    
    # Setup directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = f"experiments/fedformer_{args.component}_{timestamp}"
    checkpoint_dir = os.path.join(experiment_dir, "checkpoints")
    log_dir = os.path.join(experiment_dir, "logs")
    
    for dir_path in [experiment_dir, checkpoint_dir, log_dir]:
        os.makedirs(dir_path, exist_ok=True)
    
    # Setup tensorboard
    writer = SummaryWriter(log_dir)
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_f1 = 0.0
    
    if args.resume and os.path.exists(args.resume):
        logger.info(f"Resuming from checkpoint: {args.resume}")
        start_epoch, best_f1 = load_checkpoint(args.resume, model, optimizer, scheduler)
        start_epoch += 1
    
    # Training history
    history = {
        'train_loss': [], 'val_loss': [],
        'train_f1': [], 'val_f1': [],
        'train_precision': [], 'val_precision': [],
        'train_recall': [], 'val_recall': [],
        'train_roc_auc': [], 'val_roc_auc': [],
        'lr': [], 'val_adj_f1': []
    }
    
    # Training loop
    logger.info("Starting training...")
    patience_counter = 0
    
    for epoch in range(start_epoch, config['epochs']):
        epoch_start_time = time.time()
        
        # Train
        train_metrics = train_epoch(model, data_loaders['train'], criterion, optimizer, device, epoch)
        
        # Validate
        val_metrics = None
        if 'val' in data_loaders:
            val_metrics = validate_epoch(model, data_loaders['val'], criterion, device, epoch)
        
        # Update scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        # Log metrics
        epoch_time = time.time() - epoch_start_time
        
        logger.info(f"Epoch {epoch}/{config['epochs']} ({epoch_time:.2f}s)")
        logger.info(f"  Train - Loss: {train_metrics['avg_loss']:.4f}, "
                   f"F1: {train_metrics['f1']:.4f}, "
                   f"ROC-AUC: {train_metrics['roc_auc']:.4f}")
        
        if val_metrics:
            logger.info(f"  Val   - Loss: {val_metrics['avg_loss']:.4f}, "
                       f"F1: {val_metrics['f1']:.4f}, "
                       f"Adj-F1: {val_metrics['adj_f1']:.4f}, "
                       f"ROC-AUC: {val_metrics['roc_auc']:.4f}")
        
        # Update history
        history['train_loss'].append(train_metrics['avg_loss'])
        history['train_f1'].append(train_metrics['f1'])
        history['train_precision'].append(train_metrics['precision'])
        history['train_recall'].append(train_metrics['recall'])
        history['train_roc_auc'].append(train_metrics['roc_auc'])
        history['lr'].append(current_lr)
        
        if val_metrics:
            history['val_loss'].append(val_metrics['avg_loss'])
            history['val_f1'].append(val_metrics['f1'])
            history['val_precision'].append(val_metrics['precision'])
            history['val_recall'].append(val_metrics['recall'])
            history['val_roc_auc'].append(val_metrics['roc_auc'])
            history['val_adj_f1'].append(val_metrics['adj_f1'])
        
        # Tensorboard logging
        writer.add_scalar('Loss/Train', train_metrics['avg_loss'], epoch)
        writer.add_scalar('F1/Train', train_metrics['f1'], epoch)
        writer.add_scalar('Learning_Rate', current_lr, epoch)
        
        if val_metrics:
            writer.add_scalar('Loss/Val', val_metrics['avg_loss'], epoch)
            writer.add_scalar('F1/Val', val_metrics['f1'], epoch)
            writer.add_scalar('F1_Adjusted/Val', val_metrics['adj_f1'], epoch)
        
        # Save checkpoint and check early stopping
        current_f1 = val_metrics['adj_f1'] if val_metrics else train_metrics['f1']
        
        if current_f1 > best_f1:
            best_f1 = current_f1
            patience_counter = 0
            
            # Save best model
            checkpoint_path = save_checkpoint(
                model, optimizer, scheduler, epoch, best_f1, checkpoint_dir
            )
            
            # Also save as best model
            best_model_path = os.path.join(checkpoint_dir, 'best_model.pth')
            torch.save(model.state_dict(), best_model_path)
            
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= config['patience']:
            logger.info(f"Early stopping triggered after {patience_counter} epochs without improvement")
            break
        
        # Save training history plot
        if (epoch + 1) % 10 == 0:
            plot_path = os.path.join(experiment_dir, 'training_history.png')
        
        # Cleanup
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    
    # Final evaluation on test set
    if 'test' in data_loaders:
        logger.info("Evaluating on test set...")
        
        # Load best model
        best_model_path = os.path.join(checkpoint_dir, 'best_model.pth')
        if os.path.exists(best_model_path):
            model.load_state_dict(torch.load(best_model_path, map_location=device))
        
        test_metrics = validate_epoch(model, data_loaders['test'], criterion, device, 'Test')
        
        logger.info("Test Results:")
        logger.info(f"  Loss: {test_metrics['avg_loss']:.4f}")
        logger.info(f"  Accuracy: {test_metrics['accuracy']:.4f}")
        logger.info(f"  Precision: {test_metrics['precision']:.4f}")
        logger.info(f"  Recall: {test_metrics['recall']:.4f}")
        logger.info(f"  F1: {test_metrics['f1']:.4f}")
        logger.info(f"  Adjusted F1: {test_metrics['adj_f1']:.4f}")
        logger.info(f"  ROC AUC: {test_metrics['roc_auc']:.4f}")
        logger.info(f"  PR AUC: {test_metrics['pr_auc']:.4f}")
        
        # Save test results
        test_results = {
            'config': config,
            'test_metrics': test_metrics,
            'best_epoch': epoch - patience_counter,
            'total_epochs': epoch + 1
        }
        
        results_path = os.path.join(experiment_dir, 'test_results.yaml')
        with open(results_path, 'w') as f:
            yaml.dump(test_results, f, default_flow_style=False)
    
    # Final training history plot
    plot_path = os.path.join(experiment_dir, 'final_training_history.png')
    
    writer.close()
    logger.info(f"Training completed! Results saved in {experiment_dir}")

if __name__ == "__main__":
    main() 