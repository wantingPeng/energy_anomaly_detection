import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, f1_score
#from src.training.lsmt.lsmt_fusion.watch_weight import visualize_lstm_gradients

def evaluate(model, data_loader, criterion, device, epoch=None, find_optimal_threshold=True):
    """
    Evaluate the model on validation or test data.
    
    Args:
        model: LSTM Late Fusion model
        data_loader: Validation or test data loader
        criterion: Loss function
        device: Device to use for evaluation
        threshold: Default classification threshold for positive class (anomaly)
        epoch: Current epoch number for gradient visualization
        find_optimal_threshold: Whether to find the optimal threshold based on F1 score
        
    Returns:
        Tuple of (average loss, accuracy, precision, recall, f1, confusion matrix, optimal_threshold)
    """
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_scores = []
    all_attn_weights = []  # Store attention weights for visualization
    
    with torch.no_grad():
        for windows, stat_features, labels in tqdm(data_loader, desc="Evaluating"):
            # Move data to device
            windows = windows.to(device)
            stat_features = stat_features.to(device)
            labels = labels.to(device)
            
            # Forward pass
            #outputs, attn_weights = model(windows, stat_features)
            outputs = model(windows, stat_features)

            # Store attention weights for visualization
            #all_attn_weights.append(attn_weights.detach().cpu())
            
            # Calculate loss
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            # Get probabilities
            probs = torch.softmax(outputs, dim=1)
            anomaly_scores = probs[:, 1]  # Probability for anomaly class
            
            # Store scores and labels
            all_scores.extend(anomaly_scores.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Convert to numpy arrays for easier processing
    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)
    
    # Find optimal threshold based on F1 score if requested
    if find_optimal_threshold:
        thresholds = np.linspace(0.01, 0.99, 99)  # Test 99 threshold values
        f1_scores = []
        
        for thresh in thresholds:
            temp_preds = (all_scores > thresh).astype(int)
            f1 = f1_score(all_labels, temp_preds, zero_division=0)
            f1_scores.append(f1)
        
        # Find the threshold that maximizes F1 score
        best_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds[best_idx]
        
        # Use the optimal threshold
        threshold = optimal_threshold
        
    # Calculate final predictions using the threshold
    all_preds = (all_scores > threshold).astype(int)
    
    # Calculate metrics
    avg_loss = total_loss / len(data_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary', zero_division=0)
    conf_matrix = confusion_matrix(all_labels, all_preds)
    
    # # Visualize LSTM gradients if epoch is provided
    # if epoch is not None:
    #     visualize_lstm_gradients(model, epoch, prefix='val')
    
    return avg_loss, accuracy, precision, recall, f1, conf_matrix, threshold

