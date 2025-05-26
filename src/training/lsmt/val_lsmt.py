import os
import torch
import yaml
import numpy as np
from tqdm import tqdm
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

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

def load_model(model_path, config):
    """
    Load a trained model from checkpoint
    
    Args:
        model_path (str): Path to the model checkpoint
        config (dict): Model configuration
        
    Returns:
        LSTMModel: Loaded model
    """
    try:
        # Create model with config
        model = LSTMModel(config=config.get('model', {}))
        
        # Load model state
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        
        # Check if the checkpoint contains full model or just state dict
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"Loaded model from checkpoint {model_path}, trained for {checkpoint.get('epoch', 'unknown')} epochs")
        else:
            model.load_state_dict(checkpoint)
            logger.info(f"Loaded model state from {model_path}")
            
        return model
    except Exception as e:
        logger.error(f"Error loading model from {model_path}: {str(e)}")
        raise

def validate_model(model, dataloader, criterion, device):
    """
    Validate the model on validation data
    
    Args:
        model: PyTorch model
        dataloader: PyTorch DataLoader
        criterion: Loss function
        device: Device to validate on
        
    Returns:
        tuple: (validation_loss, accuracy, all_predictions, all_targets)
    """
    model.eval()
    total_loss = 0
    all_predictions = []
    all_targets = []
    
    # Use tqdm for progress bar
    pbar = tqdm(dataloader, desc="Validating")
    
    with torch.no_grad():
        for data, targets in pbar:
            # Move data to device
            data, targets = data.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(data)
            
            # Using CrossEntropyLoss with multi-class outputs
            loss = criterion(outputs, targets.long())
            #_, predictions = torch.max(outputs.data, 1)
            probs = torch.softmax(outputs, dim=1)
            predictions = (probs[:, 1] > 0.8).long()
            
            # Update total loss
            total_loss += loss.item()
            
            # Store predictions and targets for metrics calculation
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            
            # Update progress bar
            pbar.set_postfix({'val_loss': f'{total_loss / (pbar.n + 1):.4f}'})
    
    # Calculate validation loss
    val_loss = total_loss / len(dataloader)
    
    # Calculate accuracy
    accuracy = accuracy_score(all_targets, all_predictions)
    
    return val_loss, accuracy, np.array(all_predictions), np.array(all_targets)

def calculate_metrics(predictions, targets):
    """
    Calculate evaluation metrics
    
    Args:
        predictions: Model predictions
        targets: True labels
        
    Returns:
        dict: Dictionary of evaluation metrics
    """
    accuracy = accuracy_score(targets, predictions)
    precision = precision_score(targets, predictions, average='binary', zero_division=0)
    recall = recall_score(targets, predictions, average='binary', zero_division=0)
    f1 = f1_score(targets, predictions, average='binary', zero_division=0)
    conf_matrix = confusion_matrix(targets, predictions)
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': conf_matrix
    }
    
    return metrics

def validate(config, model_path=None, model=None):
    """
    Validate the LSTM model using the provided configuration
    
    Args:
        config (dict): Configuration dictionary
        model_path (str, optional): Path to the model checkpoint, if None use the best model from checkpoint dir
        model (LSTMModel, optional): Already loaded model instance. If provided, model_path is ignored.
        
    Returns:
        tuple: (component_metrics, overall_metrics)
    """
    # Device configuration
    device_config = config.get('device', {})
    use_gpu = device_config.get('use_gpu', False) and torch.cuda.is_available()
    device = torch.device('cuda' if use_gpu else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load model if not provided
    if model is None:
        if model_path is None:
            checkpoint_dir = config.get('logging', {}).get('checkpoint_dir', 'src/training/lsmt/checkpoints')
            model_path = os.path.join(checkpoint_dir, "lstm_best_model.pt")
            if not os.path.exists(model_path):
                logger.error(f"Best model checkpoint not found at {model_path}")
                # Look for any model checkpoint
                checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pt')]
                if checkpoints:
                    model_path = os.path.join(checkpoint_dir, checkpoints[0])
                    logger.info(f"Using alternative checkpoint: {model_path}")
                else:
                    logger.error(f"No model checkpoints found in {checkpoint_dir}")
                    return {}, None
        
        # Load model
        model = load_model(model_path, config)
        logger.info(f"Model loaded from checkpoint for validation")
    else:
        logger.info(f"Using provided model instance for validation")
        
    # Ensure model is on the correct device
    model.to(device)
    
    # Training configuration for class weights
    train_config = config.get('training', {})
    class_weights = train_config.get('class_weights', None)
    
    if class_weights:
        class_weights = torch.tensor(class_weights).to(device)
    
    # Always use CrossEntropyLoss for validation
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    logger.info("Using CrossEntropyLoss for validation")
    
    # Data configuration
    data_config = config.get('data', {})
    val_data_dir = data_config.get('val_data_dir', 'Data/processed/lsmt/dataset/val')
    
    # Components to validate on
    component_names = ["contact", "pcb", "ring"]
    logger.info(f"Validating on components: {component_names}")
    
    # Batch size for validation
    batch_size = train_config.get('batch_size', 128)
    
    # Validation loop
    logger.info(f"Starting validation on {val_data_dir}")
    
    component_metrics = {}
    all_predictions = []
    all_targets = []
    overall_metrics = {}
    # Iterate over each component
    for component_name, dataloader in get_component_dataloaders(
        component_names=component_names,
        data_dir=val_data_dir,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=use_gpu
    ):
        logger.info(f"Validating on component: {component_name} with {len(dataloader.dataset)} samples")
        
        # Validate
        val_loss, accuracy, predictions, targets = validate_model(
            model=model,
            dataloader=dataloader,
            criterion=criterion,
            device=device
        )
        
        # Calculate detailed metrics
        metrics = calculate_metrics(predictions, targets)
        
        # Store component metrics
        component_metrics[component_name] = {
            'loss': val_loss,
            'accuracy': metrics['accuracy'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1_score': metrics['f1_score'],
            'confusion_matrix': metrics['confusion_matrix']
        }
        
        # Append to overall predictions and targets
        all_predictions.extend(predictions)
        all_targets.extend(targets)
        '''
        # Log component results
        logger.info(f"Component: {component_name}")
        logger.info(f"  Loss: {val_loss:.4f}")
        logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"  Precision: {metrics['precision']:.4f}")
        logger.info(f"  Recall: {metrics['recall']:.4f}")
        logger.info(f"  F1 Score: {metrics['f1_score']:.4f}")
        logger.info(f"  Confusion Matrix:\n{metrics['confusion_matrix']}")
        '''
    # Calculate overall metrics
    if all_predictions:
        overall_metrics = calculate_metrics(np.array(all_predictions), np.array(all_targets))
        
        # Add loss to overall metrics
        overall_val_loss = sum(m['loss'] for m in component_metrics.values()) / len(component_metrics)
        overall_metrics['loss'] = overall_val_loss
        
        # Log overall results
        logger.info("Overall Results:")
        logger.info(f"  Loss: {overall_val_loss:.4f}")
        logger.info(f"  Accuracy: {overall_metrics['accuracy']:.4f}")
        logger.info(f"  Precision: {overall_metrics['precision']:.4f}")
        logger.info(f"  Recall: {overall_metrics['recall']:.4f}")
        logger.info(f"  F1 Score: {overall_metrics['f1_score']:.4f}")
        logger.info(f"  Confusion Matrix:\n{overall_metrics['confusion_matrix']}")
        
        return component_metrics, overall_metrics
    
    return component_metrics, overall_metrics

def main():
    """Main function to run the validation script"""
    # Load configuration
    config_path = "configs/lstm_training.yaml"
    config = load_config(config_path)
    
    # Validate model
    component_metrics, overall_metrics = validate(config)
    
    logger.info("Validation completed")

if __name__ == "__main__":
    main()
