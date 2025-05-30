import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import time
from datetime import datetime
from pathlib import Path

from src.training.lsmt.lstm_model import LSTMModel
from src.training.lsmt.focal_loss import FocalLoss
from src.utils.logger import logger

def setup_logger():
    """
    Set up logger for the test script
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path("experiments/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"test_overfit_single_sample_{timestamp}.log"
    logger.info(f"Logs will be saved to {log_file}")

def load_single_sample(data_path):
    """
    Load a single sample from the dataset for overfitting test
    
    Args:
        data_path (str): Path to the dataset file
        
    Returns:
        tuple: (data, label) - Single sample data and label
    """
    logger.info(f"Loading data from {data_path}")
    
    try:
        batch_data = torch.load(data_path)
        
        if 'windows' not in batch_data or 'labels' not in batch_data:
            logger.error(f"Batch file {data_path} missing required keys")
            raise KeyError("Required keys 'windows' or 'labels' not found in data")
            
        windows = batch_data['windows']
        labels = batch_data['labels']
        
        # Ensure we have at least one sample
        if len(windows) == 0:
            logger.error(f"No samples found in {data_path}")
            raise ValueError("No samples found in dataset")
            
        logger.info(f"Loaded data with {len(windows)} samples")
        logger.info(f"Data shape: {windows.shape}, Labels shape: {labels.shape}")
        
        # Take the first sample
        data_sample = windows[0].unsqueeze(0)  # Add batch dimension
        label_sample = labels[0].unsqueeze(0)  # Add batch dimension
        
        logger.info(f"Selected sample - Data shape: {data_sample.shape}, Label: {label_sample.item()}")
        
        return data_sample, label_sample
        
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise

def test_overfit_single_sample(
    data_path="Data/processed/lsmt/dataset/train/contact/batch_0.pt",
    learning_rate=1e-3,
    epochs=200,
    device=None,
    model_config=None
):
    """
    Test if the LSTM model can overfit a single sample
    
    Args:
        data_path (str): Path to the dataset file
        learning_rate (float): Learning rate for training
        epochs (int): Number of epochs to train
        device (torch.device): Device to train on
        model_config (dict): Model configuration
    """
    setup_logger()
    
    # Set device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load a single sample
    data_sample, label_sample = load_single_sample(data_path)
    data_sample = data_sample.to(device)
    label_sample = label_sample.to(device)
    
    # Create model
    if model_config is None:
        model_config = {
            'input_size': data_sample.shape[2],  # Feature dimension
            'hidden_size': 128,
            'num_layers': 2,
            'dropout': 0.2,
            'output_size': 2  # Binary classification
        }
    
    model = LSTMModel(config=model_config)
    model.to(device)
    logger.info(f"Model created with config: {model_config}")
    
    # Define loss function (using FocalLoss as per project)
    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    logger.info("Using FocalLoss with alpha=0.25, gamma=2.0")
    
    # Define optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    logger.info(f"Using Adam optimizer with learning_rate={learning_rate}")
    
    # Training loop
    logger.info(f"Starting training for {epochs} epochs...")
    
    history = {
        'loss': [],
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1_score': []
    }
    
    for epoch in tqdm(range(epochs), desc="Training"):
        # Set model to training mode
        model.train()
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(data_sample)
        
        # Compute loss
        loss = criterion(outputs, label_sample.long())
        
        # Backward pass
        loss.backward()
        
        # Update weights
        optimizer.step()
        
        # Get predictions
        _, predicted = torch.max(outputs.data, 1)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        
        # Compute metrics
        accuracy = (predicted == label_sample).float().mean().item()
        
        # Move tensors to CPU for sklearn metrics
        y_true = label_sample.cpu().numpy()
        y_pred = predicted.cpu().numpy()
        
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # Store metrics
        history['loss'].append(loss.item())
        history['accuracy'].append(accuracy)
        history['precision'].append(precision)
        history['recall'].append(recall)
        history['f1_score'].append(f1)
        
        # Log metrics every 20 epochs
        if (epoch + 1) % 20 == 0:
            logger.info(f"Epoch {epoch+1}/{epochs}")
            logger.info(f"Loss: {loss.item():.4f}")
            logger.info(f"Accuracy: {accuracy:.4f}")
            logger.info(f"Precision: {precision:.4f}")
            logger.info(f"Recall: {recall:.4f}")
            logger.info(f"F1 Score: {f1:.4f}")
            logger.info(f"Probabilities: {probabilities.detach().cpu().numpy()}")
    
    logger.info("Training complete")
    
    # Final evaluation
    model.eval()
    with torch.no_grad():
        outputs = model(data_sample)
        _, predicted = torch.max(outputs.data, 1)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        
        logger.info("Final evaluation:")
        logger.info(f"True label: {label_sample.item()}")
        logger.info(f"Predicted label: {predicted.item()}")
        logger.info(f"Probabilities: {probabilities.cpu().numpy()}")
        
        # Calculate final metrics
        accuracy = (predicted == label_sample).float().mean().item()
        y_true = label_sample.cpu().numpy()
        y_pred = predicted.cpu().numpy()
        
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        logger.info(f"Final Accuracy: {accuracy:.4f}")
        logger.info(f"Final Precision: {precision:.4f}")
        logger.info(f"Final Recall: {recall:.4f}")
        logger.info(f"Final F1 Score: {f1:.4f}")
    
    return model, history

if __name__ == "__main__":
    # Path to data
    data_path = "Data/processed/lsmt/dataset/train/contact/batch_0.pt"
    
    # Check if data exists
    if not os.path.exists(data_path):
        logger.error(f"Data file not found: {data_path}")
        exit(1)
    
    # Training parameters
    learning_rate = 1e-3
    epochs = 200
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Run test
    model, history = test_overfit_single_sample(
        data_path=data_path,
        learning_rate=learning_rate,
        epochs=epochs,
        device=device
    )
    
    logger.info("Overfitting test complete") 