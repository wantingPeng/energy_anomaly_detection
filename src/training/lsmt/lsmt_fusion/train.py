import torch
from tqdm import tqdm


def train_epoch(model, data_loader, criterion, optimizer, device):
    """
    Train the model for one epoch.
    
    Args:
        model: LSTM Late Fusion model
        data_loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to use for training
        
    Returns:
        Average training loss
    """
    model.train()
    total_loss = 0
    
    for windows, stat_features, labels in tqdm(data_loader, desc="Training"):
        # Move data to device
        windows = windows.to(device)
        stat_features = stat_features.to(device)
        labels = labels.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs,attn_weights = model(windows, stat_features)
        # Calculate loss
        loss = criterion(outputs, labels)
        entropy = - (attn_weights * torch.log(attn_weights + 1e-8)).sum(dim=1).mean()
        loss += 0.01 * entropy 
        # Backward pass and optimize
        loss.backward()
        '''     
           print(f"Loss: {loss.item():.6f}")
        print("Logits:", outputs[:3].detach().cpu().numpy())  # 防止打印太多
        print("Logits max/min:", outputs.max().item(), outputs.min().item())'''
        #print('attention_layer.weight.grad.abs().mean()', model.attention_layer.weight.grad.abs().mean())

        optimizer.step()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(data_loader)
    print('len(data_loader)', len(data_loader))
    return avg_loss
