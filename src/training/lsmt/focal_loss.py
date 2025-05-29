import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utils.logger import logger


class FocalLoss(nn.Module):
    """
    Focal Loss for binary classification tasks.
    
    This implementation follows the paper "Focal Loss for Dense Object Detection"
    with support for class imbalance through alpha parameter and easy/hard example
    weighting through gamma parameter.
    
    Args:
        alpha (float): Weight for the rare class (anomaly). Default: 0.25
        gamma (float): Focusing parameter for hard examples. Default: 2.0
        reduction (str): Specifies the reduction to apply to the output. Default: 'mean'
    """
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        logger.info(f"Initialized FocalLoss with alpha={alpha}, gamma={gamma}")
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calculate the focal loss.
        
        Args:
            inputs (torch.Tensor): Model predictions of shape (N, C) where C = 2 for binary classification
            targets (torch.Tensor): Ground truth labels of shape (N,)
            
        Returns:
            torch.Tensor: Computed focal loss
        """
        # Convert targets to one-hot encoding
        targets = F.one_hot(targets, num_classes=2).float()
        
        # Apply softmax to get probabilities
        probs = F.softmax(inputs, dim=1)
        
        # Calculate focal loss
        ce_loss = F.cross_entropy(inputs, targets.argmax(dim=1), reduction='none')
        pt = torch.sum(targets * probs, dim=1)  # Get the probability of the true class
        
        # Calculate the focal term
        focal_term = (1 - pt) ** self.gamma
        
        # Apply alpha weighting
        alpha_weight = self.alpha * targets[:, 1] + (1 - self.alpha) * targets[:, 0]
        
        # Combine all terms
        loss = alpha_weight * focal_term * ce_loss
        
        # Apply reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss 