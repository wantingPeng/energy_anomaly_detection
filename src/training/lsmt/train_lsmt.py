import os
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
import time
import json
from datetime import datetime
from tqdm import tqdm
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import f1_score, precision_score, recall_score

from src.training.lsmt.lstm_model import LSTMModel
from src.training.lsmt.dataloader_from_batches import get_component_dataloaders
from src.utils.logger import logger
from src.training.lsmt.val_lsmt import validate
from src.training.lsmt.focal_loss import FocalLoss

def load_config(config_path):
    """
    Load configuration from YAML file
    
    Args:
        config_path (str): Path to configuration file
        
    Returns:
        dict: Configuration dictionary
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded configuration from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Error loading configuration from {config_path}: {str(e)}")
        raise

def train_model(model, dataloader, criterion, optimizer, device):
    """
    Train the model for one epoch
    
    Args:
        model: PyTorch model
        dataloader: PyTorch DataLoader
        criterion: Loss function
        optimizer: PyTorch optimizer
        device: Device to train on
        
    Returns:
        tuple: (epoch_loss, epoch_accuracy, epoch_precision, epoch_recall, epoch_f1)
    """
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    all_predictions = []
    all_targets = []
    
    # Use tqdm for progress bar
    pbar = tqdm(dataloader, desc="Training")
    
    for batch_idx, (data, targets) in enumerate(pbar):
        # Move data to device
        data, targets = data.to(device), targets.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(data)
        
        # Calculate loss
        loss = criterion(outputs, targets.long())
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Calculate accuracy for binary classification
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()
        
        # Store predictions and targets for metric calculation
        all_predictions.extend(predicted.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())
        
        # Update total loss
        total_loss += loss.item()
        
        # Update progress bar
        avg_loss = total_loss / (batch_idx + 1)
        accuracy = 100. * correct / total
        pbar.set_postfix({'loss': f'{avg_loss:.4f}', 'accuracy': f'{accuracy:.2f}%'})
    
    # Calculate metrics
    epoch_loss = total_loss / len(dataloader)
    epoch_accuracy = 100. * correct / total
    epoch_precision = precision_score(all_targets, all_predictions, average='weighted')
    epoch_recall = recall_score(all_targets, all_predictions, average='weighted')
    epoch_f1 = f1_score(all_targets, all_predictions, average='weighted')
    
    return epoch_loss, epoch_accuracy, epoch_precision, epoch_recall, epoch_f1

def train_epoch(config):
    """
    Train the LSTM model using the provided configuration
    
    Args:
        config (dict): Configuration dictionary
        components (list, optional): List of component names to train on. If None, use all available components
    """
    # Device configuration
    device_config = config.get('device', {})
    use_gpu = device_config.get('use_gpu', False) and torch.cuda.is_available()
    precision = device_config.get('precision', 'full')
    device = torch.device('cuda' if use_gpu else 'cpu')
    logger.info(f"Using device: {device}, precision: {precision}")
    
    # Model configuration
    model_config = config.get('model', {})
    model = LSTMModel(config=model_config)
    model.to(device)
    logger.info(f"Model created with input_size={model_config.get('input_size')}, hidden_size={model_config.get('hidden_size')}")
    
    # Training configuration
    train_config = config.get('training', {})
    batch_size = train_config.get('batch_size', 128)
    epochs = train_config.get('epochs', 100)
    learning_rate = train_config.get('learning_rate', 0.001)
    weight_decay = train_config.get('weight_decay', 0.0001)
    
    # Loss function configuration
    loss_config = train_config.get('loss', {})
    alpha = loss_config.get('focal_alpha', 0.25)
    gamma = loss_config.get('focal_gamma', 2.0)
    
    # Initialize FocalLoss
    criterion = FocalLoss(alpha=alpha, gamma=gamma)
    logger.info(f"Using FocalLoss with alpha={alpha}, gamma={gamma} for training")
    
    # Class weights for imbalanced datasets
    class_weights = train_config.get('class_weights', None)
    if class_weights:
        class_weights = torch.tensor(class_weights).to(device)
    
    # Optimizer
    optimizer_type = train_config.get('optimizer', 'adam')
    if optimizer_type.lower() == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        logger.info(f"Using Adam optimizer with lr={learning_rate}, weight_decay={weight_decay}")
    elif optimizer_type.lower() == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        logger.info(f"Using SGD optimizer with lr={learning_rate}, weight_decay={weight_decay}")
    else:
        logger.error(f"Unsupported optimizer: {optimizer_type}, defaulting to Adam")
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Learning rate scheduler
    lr_scheduler_config = train_config.get('lr_scheduler', {})
    use_scheduler = lr_scheduler_config.get('use', True)
    scheduler = None
    
    if use_scheduler:
        scheduler_type = lr_scheduler_config.get('type', 'reduce_on_plateau')
        if scheduler_type == 'reduce_on_plateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=lr_scheduler_config.get('factor', 0.5),
                patience=lr_scheduler_config.get('patience', 5),
                min_lr=lr_scheduler_config.get('min_lr', 0.00001)
            )
            logger.info(f"Using ReduceLROnPlateau scheduler with patience={lr_scheduler_config.get('patience', 5)}")
    
    # Data configuration
    data_config = config.get('data', {})
    train_data_dir = data_config.get('data_dir', 'Data/processed/lsmt/dataset/train')
    
    # Components to train on
    #component_names = ["contact", "pcb", "ring"]
    component_names = ["contact"]

    # Logging configuration
    logging_config = config.get('logging', {})
    log_interval = logging_config.get('log_interval', 10)
    save_checkpoint = logging_config.get('save_checkpoint', True)
    checkpoint_dir = logging_config.get('checkpoint_dir', 'src/training/lsmt/checkpoints')
    
    # Create checkpoint directory if it doesn't exist
    os.makedirs(checkpoint_dir, exist_ok=True)
    logger.info(f"Checkpoint directory: {checkpoint_dir}")
    
    # Set up TensorBoard
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tensorboard_dir = os.path.join('src/training/lsmt/tensorboard', f'run_{timestamp}')
    os.makedirs(tensorboard_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=tensorboard_dir)
    logger.info(f"TensorBoard logs will be saved to {tensorboard_dir}")
    
    # Set up mixed precision training if specified
    scaler = None
    if precision == 'mixed' and use_gpu and torch.cuda.is_available():
        scaler = torch.amp.GradScaler('cuda')
        logger.info("Using mixed precision training")
    
    # Initialize dictionary to store training history
    history = {
        "overall": {
            "train_losses": [],
            "val_losses": [],
            "train_accuracies": [],
            "train_precisions": [],
            "train_recalls": [],
            "train_f1_scores": [],
            "val_f1_scores": [],
            "learning_rates": []
        }
    }
    
    for component_name in component_names:
        history[component_name] = {
            "train_losses": [],
            "val_losses": [],
            "train_accuracies": [],
            "train_precisions": [],
            "train_recalls": [],
            "train_f1_scores": [],
            "val_f1_scores": []
        }
    
    # Training loop
    logger.info(f"Starting training for {epochs} epochs")
    best_loss = float('inf')
    early_stopping_counter = 0
    early_stopping_patience = train_config.get('early_stopping', 10)
    
    # Global step counter for TensorBoard
    global_step = 0
    
    for epoch in range(epochs):
        logger.info(f"Epoch {epoch+1}/{epochs}")
        epoch_start_time = time.time()
        epoch_train_loss = 0
        num_components = len(component_names)
        
        # Iterate over each component for training
        for component_name, train_dataloader in get_component_dataloaders(
            component_names=component_names,
            data_dir=train_data_dir,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=use_gpu
        ):
            logger.info(f"Training on component: {component_name} with {len(train_dataloader.dataset)} samples")
            
            # Train for one epoch
            '''
            if precision == 'mixed' and scaler:
                # Implement mixed precision training
                logger.info("Using mixed precision training")
                model.train()
                total_loss = 0
                correct = 0
                total = 0
                
                pbar = tqdm(train_dataloader, desc=f"Training on {component_name}")
                
                for batch_idx, (data, targets) in enumerate(pbar):
                    data, targets = data.to(device), targets.to(device)
                    optimizer.zero_grad()
                    
                    with torch.amp.autocast('cuda'):
                        outputs = model(data)
                        loss = criterion(outputs, targets.long())
                    
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    
                    _, predicted = torch.max(outputs.data, 1)
                    total += targets.size(0)
                    correct += (predicted == targets).sum().item()
                    
                    total_loss += loss.item()
                    avg_loss = total_loss / (batch_idx + 1)
                    accuracy = 100. * correct / total
                    
                    # Log to TensorBoard per batch
                    writer.add_scalar(f'Loss/train_batch/{component_name}', loss.item(), global_step)
                    global_step += 1
                    
                    if batch_idx % log_interval == 0:
                        pbar.set_postfix({'loss': f'{avg_loss:.4f}', 'accuracy': f'{accuracy:.2f}%'})
                
                train_loss, train_accuracy, train_precision, train_recall, train_f1 = train_model(
                    model=model,
                    dataloader=train_dataloader,
                    criterion=criterion,
                    optimizer=optimizer,
                    device=device
                )
                
                # Log to TensorBoard
                writer.add_scalar(f'Loss/train/{component_name}', train_loss, epoch)
                writer.add_scalar(f'Accuracy/train/{component_name}', train_accuracy, epoch)
                writer.add_scalar(f'Precision/train/{component_name}', train_precision, epoch)
                writer.add_scalar(f'Recall/train/{component_name}', train_recall, epoch)
                writer.add_scalar(f'F1/train/{component_name}', train_f1, epoch)
                
            else:
            '''
                # Standard precision training
            logger.info("Using standard precision training")
            train_loss, train_accuracy, train_precision, train_recall, train_f1 = train_model(
                model=model,
                dataloader=train_dataloader,
                criterion=criterion,
                optimizer=optimizer,
                device=device
            )
            
            # Log to TensorBoard
            writer.add_scalar(f'Loss/train/{component_name}', train_loss, epoch)
            writer.add_scalar(f'Accuracy/train/{component_name}', train_accuracy, epoch)
            writer.add_scalar(f'Precision/train/{component_name}', train_precision, epoch)
            writer.add_scalar(f'Recall/train/{component_name}', train_recall, epoch)
            writer.add_scalar(f'F1/train/{component_name}', train_f1, epoch)
            
            
            logger.info(f"Component {component_name} - Training Loss: {train_loss:.4f}, "
                        f"Accuracy: {train_accuracy:.2f}%, "
                        f"Precision: {train_precision:.4f}, "
                        f"Recall: {train_recall:.4f}, "
                        f"F1-score: {train_f1:.4f}")
            epoch_train_loss += train_loss
        
        # Calculate average training loss across all components
        avg_train_loss = epoch_train_loss / num_components
        
        # Run full validation on all components at the end of each epoch
        logger.info(f"Running full validation for epoch {epoch+1}")
        
        # Check if best model exists and load it for validation
        best_model_path = os.path.join(checkpoint_dir, "lstm_best_model.pt")
        
        # if epoch > 0 and os.path.exists(best_model_path):
        #     logger.info(f"Loading best model from {best_model_path} for validation")
        #     component_metrics, overall_metrics = validate(config, model_path=best_model_path, model=None)
        #     logger.info(f"Validation performed using the best model")
        # else:
        #     # First epoch or no best model saved yet, use current model
        #     logger.info(f"Using current model for validation (no best model saved yet)")
        component_metrics, overall_metrics = validate(config, model_path=None, model=model)
        
        # Store validation metrics in history
        if overall_metrics:
            avg_val_loss = overall_metrics['loss']
            avg_val_f1 = overall_metrics['f1_score']
            
            # Store component-specific metrics and log to TensorBoard
            for component_name, metrics in component_metrics.items():
                if component_name in history:

                    
                    # Log component metrics to TensorBoard
                    writer.add_scalar(f'Loss/val/{component_name}', metrics['loss'], epoch)
                    writer.add_scalar(f'F1/val/{component_name}', metrics['f1_score'], epoch)
                    writer.add_scalar(f'Accuracy/val/{component_name}', metrics['accuracy'], epoch)
                    writer.add_scalar(f'Precision/val/{component_name}', metrics['precision'], epoch)
                    writer.add_scalar(f'Recall/val/{component_name}', metrics['recall'], epoch)
            
            # Log to TensorBoard
            writer.add_scalar('Loss/val', avg_val_loss, epoch)
            writer.add_scalar('F1/val', avg_val_f1, epoch)
            writer.add_scalar('Accuracy/val', overall_metrics['accuracy'], epoch)
            writer.add_scalar('Precision/val', overall_metrics['precision'], epoch)
            writer.add_scalar('Recall/val', overall_metrics['recall'], epoch)
        else:
            # Fallback if validation fails
            logger.warning("Validation failed to return metrics, using training loss as fallback")
            avg_val_loss = avg_train_loss
            avg_val_f1 = 0.0
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        

        # Log overall metrics to TensorBoard
        writer.add_scalar('Loss/train', avg_train_loss, epoch)
        writer.add_scalar('LR', current_lr, epoch)
        
        logger.info(f"Average training loss across all components: {avg_train_loss:.4f}")
        logger.info(f"Average validation loss across all components: {avg_val_loss:.4f}")
        logger.info(f"Average validation F1 score across all components: {avg_val_f1:.4f}")
        logger.info(f"Current learning rate: {current_lr}")
        
        # Update learning rate scheduler
        if scheduler is not None:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(avg_val_loss)
                current_lr = optimizer.param_groups[0]['lr']
                logger.info(f"Current learning rate updated to: {current_lr}")
        
        # Calculate time taken for epoch
        epoch_time = time.time() - epoch_start_time
        logger.info(f"Epoch completed in {epoch_time:.2f} seconds")
        
        # Save checkpoint if specified
        if save_checkpoint:
            # Save model checkpoint
            checkpoint_path = os.path.join(checkpoint_dir, f"lstm_epoch_{epoch+1}.pt")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_train_loss,
                'val_loss': avg_val_loss,
            }, checkpoint_path)
            logger.info(f"Checkpoint saved to {checkpoint_path}")
            
            # Save best model
            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                best_model_path = os.path.join(checkpoint_dir, "lstm_best_model.pt")
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': avg_train_loss,
                    'val_loss': best_loss,
                }, best_model_path)
                logger.info(f"Best model saved to {best_model_path} with validation loss {best_loss:.4f}")
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1
                logger.info(f"Early stopping counter: {early_stopping_counter}/{early_stopping_patience}")
        
        # Early stopping
        if early_stopping_counter >= early_stopping_patience:
            logger.info(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    # Close TensorBoard writer
    writer.close()
    
    logger.info("Training completed")
    
    # Return final model and history
    return model, history

def main():
    """Main function to run the training script"""
    # Load configuration
    config_path = "configs/lstm_training.yaml"
    config = load_config(config_path)
    
    # Train model
    model, history = train_epoch(config)
    
    # Save final model if not already saved in training loop
    if not config.get('logging', {}).get('save_checkpoint', True):
        # Create timestamp for model name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_save_path = f"experiments/models/lstm_model_{timestamp}.pt"
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        
        # Save model
        torch.save(model.state_dict(), model_save_path)
        logger.info(f"Final model saved to {model_save_path}")

if __name__ == "__main__":
    main()
