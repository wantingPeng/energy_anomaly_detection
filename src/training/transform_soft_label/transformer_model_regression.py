import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional
from src.utils.logger import logger


class AttentionPooling(nn.Module):
    """
    Attention pooling layer to create a weighted sum of sequence elements.
    """
    def __init__(self, d_model):
        super(AttentionPooling, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
        
    def forward(self, x):
        """
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            
        Returns:
            Pooled representation [batch_size, d_model]
        """
        # x shape: [batch_size, seq_len, d_model]
        batch_size, seq_len, _ = x.size()
        
        # Calculate attention scores
        attn_scores = self.attention(x)  # [batch_size, seq_len, 1]
        
        # Apply softmax to get attention weights
        attn_weights = F.softmax(attn_scores, dim=1)  # [batch_size, seq_len, 1]
        
        # Weighted sum to get the final representation
        weighted_sum = torch.bmm(attn_weights.transpose(1, 2), x)  # [batch_size, 1, d_model]
        pooled_output = weighted_sum.squeeze(1)  # [batch_size, d_model]
        
        return pooled_output


class TransformerModelSoftLabel(nn.Module):
    """
    Transformer model for anomaly detection with soft labels.
    
    This model uses a transformer encoder to process time series data
    and predict soft anomaly scores between 0 and 1.
    """
    def __init__(
        self,
        input_dim: int,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 3,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        activation: str = 'gelu'
    ):
        """
        Initialize the transformer model for soft labels.
        
        Args:
            input_dim: Dimension of input features
            d_model: Dimension of the model
            nhead: Number of heads in multi-head attention
            num_layers: Number of transformer layers
            dim_feedforward: Dimension of the feedforward network
            dropout: Dropout probability
            activation: Activation function for transformer layers
        """
        super(TransformerModelSoftLabel, self).__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        
        # Linear projection and positional encoding are assumed to be already applied in data processing
        
        # Transformer encoder
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers,
            num_layers=num_layers
        )
        
        # Attention pooling layer
        self.attention_pooling = AttentionPooling(d_model)
        
        # Output layer for soft label prediction (single output node)
        self.regressor = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        '''nn.Linear(d_model, 64),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(64, 1),
        #nn.Sigmoid()  # Ensure output is between 0 and 1'''
        logger.info(f"Initialized TransformerModelSoftLabel with input_dim={input_dim}, d_model={d_model}")
        
    def forward(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of the transformer model.
        
        Args:
            src: Input tensor [batch_size, seq_len, input_dim]
            src_mask: Optional mask for src sequence
            
        Returns:
            Output tensor with soft label scores between 0 and 1
        """
        # Apply transformer encoder (using batch_first=True)
        encoder_output = self.transformer_encoder(src, mask=src_mask)
        
        # Apply attention pooling
        pooled_output = self.attention_pooling(encoder_output)
        
        # Apply regressor to get soft label prediction
        output = self.regressor(pooled_output)
        
        return output.squeeze(-1)   # Return [batch_size] instead of [batch_size, 1] 