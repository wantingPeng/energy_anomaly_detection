import os
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
import time
from datetime import datetime
from tqdm import tqdm
from pathlib import Path

from src.training.lsmt.lstm_model import LSTMModel
from src.training.lsmt.dataloader_from_batches import get_component_dataloaders
from src.utils.logger import logger

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

def train_epoch(model, dataloader, criterion, optimizer, device):
    """
    Train the model for one epoch
    
    Args:
        model: PyTorch model
        dataloader: PyTorch DataLoader
        criterion: Loss function
        optimizer: PyTorch optimizer
        device: Device to train on
        
    Returns:
        float: Average loss for the epoch
    """
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
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
        
        # Update total loss
        total_loss += loss.item()
        
        # Update progress bar
        avg_loss = total_loss / (batch_idx + 1)
        accuracy = 100. * correct / total
        pbar.set_postfix({'loss': f'{avg_loss:.4f}', 'accuracy': f'{accuracy:.2f}%'})
    
    # Calculate average loss and accuracy for the epoch
    epoch_loss = total_loss / len(dataloader)
    epoch_accuracy = 100. * correct / total
    
    return epoch_loss, epoch_accuracy

def train_model(config, components=None):
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
    
    # Loss function
    loss_function = train_config.get('loss_function', 'bce_with_logits')
    class_weights = train_config.get('class_weights', None)
    
    if class_weights:
        class_weights = torch.tensor(class_weights).to(device)
    
    if loss_function == 'bce_with_logits':
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        logger.info("Using CrossEntropyLoss for binary classification")
    elif loss_function == 'mse':
        criterion = nn.MSELoss()
        logger.info("Using MSELoss")
    else:
        logger.error(f"Unsupported loss function: {loss_function}, defaulting to CrossEntropyLoss")
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    
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
    use_scheduler = lr_scheduler_config.get('use', False)
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
    data_dir = data_config.get('data_dir', 'Data/processed/lsmt/dataset/train')
    
    # Components to train on
    component_names = ["contact", "pcb", "ring"]
    logger.info(f"Training on components: {component_names}")
    
    # Logging configuration
    logging_config = config.get('logging', {})
    log_interval = logging_config.get('log_interval', 10)
    save_checkpoint = logging_config.get('save_checkpoint', True)
    checkpoint_dir = logging_config.get('checkpoint_dir', 'experiments/checkpoints/lstm')
    
    # Create checkpoint directory if it doesn't exist
    os.makedirs(checkpoint_dir, exist_ok=True)
    logger.info(f"Checkpoint directory: {checkpoint_dir}")
    
    # Set up mixed precision training if specified
    scaler = None
    if precision == 'mixed' and use_gpu and torch.cuda.is_available():
        scaler = torch.cuda.amp.GradScaler()
        logger.info("Using mixed precision training")
    
    # Training loop
    logger.info(f"Starting training for {epochs} epochs")
    best_loss = float('inf')
    early_stopping_counter = 0
    early_stopping_patience = train_config.get('early_stopping', 10)
    
    for epoch in range(epochs):
        logger.info(f"Epoch {epoch+1}/{epochs}")
        epoch_start_time = time.time()
        epoch_total_loss = 0
        num_components = len(component_names)
        
        # Iterate over each component
        for component_name, dataloader in get_component_dataloaders(
            component_names=component_names,
            data_dir=data_dir,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=use_gpu
        ):
            logger.info(f"Training on component: {component_name} with {len(dataloader.dataset)} samples")
            
            # Train for one epoch
            if precision == 'mixed' and scaler:
                # Implement mixed precision training
                logger.info("Using mixed precision training")
                model.train()
                total_loss = 0
                correct = 0
                total = 0
                
                pbar = tqdm(dataloader, desc=f"Training on {component_name}")
                
                for batch_idx, (data, targets) in enumerate(pbar):
                    data, targets = data.to(device), targets.to(device)
                    optimizer.zero_grad()
                    
                    with torch.amp.autocast():
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
                    
                    if batch_idx % log_interval == 0:
                        pbar.set_postfix({'loss': f'{avg_loss:.4f}', 'accuracy': f'{accuracy:.2f}%'})
                
                epoch_loss = total_loss / len(dataloader)
                epoch_accuracy = 100. * correct / total
            else:
                # Standard precision training
                logger.info("Using standard precision training")
                epoch_loss, epoch_accuracy = train_epoch(
                    model=model,
                    dataloader=dataloader,
                    criterion=criterion,
                    optimizer=optimizer,
                    device=device
                )
            
            logger.info(f"Component {component_name} - Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%")
            epoch_total_loss += epoch_loss
        
        # Calculate average loss across all components
        avg_epoch_loss = epoch_total_loss / num_components
        logger.info(f"Average loss across all components: {avg_epoch_loss:.4f}")
        # Update learning rate scheduler
        if scheduler is not None:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(avg_epoch_loss)
                current_lr = optimizer.param_groups[0]['lr']
                logger.info(f"Current learning rate: {current_lr}")
        
        # Calculate time taken for epoch
        epoch_time = time.time() - epoch_start_time
        logger.info(f"Epoch {epoch+1} completed in {epoch_time:.2f} seconds, average loss: {avg_epoch_loss:.4f}")
        
        # Save checkpoint if specified
        if save_checkpoint:
            # Save model checkpoint
            checkpoint_path = os.path.join(checkpoint_dir, f"lstm_epoch_{epoch+1}.pt")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_epoch_loss,
            }, checkpoint_path)
            logger.info(f"Checkpoint saved to {checkpoint_path}")
            
            # Save best model
            if avg_epoch_loss < best_loss:
                best_loss = avg_epoch_loss
                best_model_path = os.path.join(checkpoint_dir, "lstm_best_model.pt")
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': best_loss,
                }, best_model_path)
                logger.info(f"Best model saved to {best_model_path} with loss {best_loss:.4f}")
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1
                logger.info(f"Early stopping counter: {early_stopping_counter}/{early_stopping_patience}")
        
        # Early stopping
        if early_stopping_counter >= early_stopping_patience:
            logger.info(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    logger.info("Training completed")
    
    # Return final model
    return model

def main():
    """Main function to run the training script"""
    # Load configuration
    config_path = "configs/lstm_training.yaml"
    config = load_config(config_path)
    
    # Train model
    model = train_model(config)
    
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
