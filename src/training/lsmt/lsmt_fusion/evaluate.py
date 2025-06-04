import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

def evaluate(model, data_loader, criterion, device, threshold=0.3):
    """
    Evaluate the model on validation or test data.
    
    Args:
        model: LSTM Late Fusion model
        data_loader: Validation or test data loader
        criterion: Loss function
        device: Device to use for evaluation
        threshold: Classification threshold for positive class (anomaly)
        
    Returns:
        Tuple of (average loss, accuracy, precision, recall, f1, confusion matrix)
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
            outputs, attn_weights = model(windows, stat_features)
            
            # Store attention weights for visualization
            all_attn_weights.append(attn_weights.detach().cpu())
            
            # Calculate loss
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            # Get probabilities and predictions with custom threshold
            probs = torch.softmax(outputs, dim=1)
            anomaly_scores = probs[:, 1]  # Probability for anomaly class
            preds = (anomaly_scores > threshold).long()
            
            # Store predictions, scores and labels
            all_preds.extend(preds.cpu().numpy())
            all_scores.extend(anomaly_scores.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    avg_loss = total_loss / len(data_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary', zero_division=0)
    conf_matrix = confusion_matrix(all_labels, all_preds)
    
    return avg_loss, accuracy, precision, recall, f1, conf_matrix, all_scores, all_labels, all_attn_weights

