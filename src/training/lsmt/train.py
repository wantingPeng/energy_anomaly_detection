import torch
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score


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
