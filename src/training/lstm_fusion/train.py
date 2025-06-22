import torch
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import numpy as np
#from src.training.lsmt.lsmt_fusion.watch_weight import visualize_lstm_gradients


def train(model, data_loader, criterion, optimizer, device, threshold=0.3, epoch=None):
    """
    Train the model for one epoch.
    
    Args:
        model: LSTM Late Fusion model
        data_loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to use for training
        threshold: Classification threshold for positive class (anomaly)
        epoch: Current epoch number for gradient visualization
        
    Returns:
        Tuple of (average loss, accuracy, precision, recall, f1, confusion matrix)
    """
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    for windows, stat_features, labels in tqdm(data_loader, desc="Training"):
        # Move data to device
        windows = windows.to(device)
        stat_features = stat_features.to(device)
        labels = labels.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        #print(f"[Batch Input mean: {windows.mean().item():.6f}, std: {windows.std().item():.6f}")
        # Forward pass
        #outputs, attn_weights = model(windows, stat_features)
        outputs= model(windows, stat_features)

        # Calculate loss
        loss = criterion(outputs, labels)
        '''entropy = - (attn_weights * torch.log(attn_weights + 1e-8)).sum(dim=1).mean()
        loss += 0.01 * entropy '''
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # Get probabilities and predictions with custom threshold
        probs = torch.softmax(outputs, dim=1)
        anomaly_scores = probs[:, 1]  # Probability for anomaly class
        preds = (anomaly_scores > threshold).long()
        
        # Store predictions and labels for metric calculation
        all_preds.extend(preds.cpu().detach().numpy())
        all_labels.extend(labels.cpu().detach().numpy())
    
    # Calculate metrics
    avg_loss = total_loss / len(data_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary', zero_division=0)
    conf_matrix = confusion_matrix(all_labels, all_preds)
    
    '''# Visualize LSTM gradients if epoch is provid
    if epoch is not None:
        visualize_lstm_gradients(model, epoch, prefix='train')
    '''
    return avg_loss, accuracy, precision, recall, f1, conf_matrix
