import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import yaml
import os
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
        
        return pooled_output, attn_weights

class TransformerLateFusionModel(nn.Module):
    """
    Transformer model with Late Fusion for anomaly detection.
    
    This model uses a transformer encoder to process time series data
    and combines it with statistical features through late fusion
    for improved anomaly detection.
    """
    def __init__(
        self,
        input_dim: int,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 3,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        num_classes: int = 2,
        activation: str = 'gelu',
        stat_features_size: int = 47,
        config=None,
    ):
        """
        Initialize the transformer model with late fusion.
        
        Args:
            input_dim: Dimension of input features
            d_model: Dimension of the model
            nhead: Number of heads in multi-head attention
            num_layers: Number of transformer layers
            dim_feedforward: Dimension of the feedforward network
            dropout: Dropout probability
            num_classes: Number of output classes (2 for binary classification)
            activation: Activation function for transformer layers
            stat_features_size: Number of statistical features
            config: Configuration dictionary (optional)
            config_path: Path to YAML configuration file (optional)
        """
        super(TransformerLateFusionModel, self).__init__()
        
        
        self.input_dim = input_dim
        self.d_model = d_model
        self.stat_features_size = stat_features_size
        
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
        
        # Transformer branch fully connected layers
        self.transformer_fc = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Statistical features branch
        self.stat_fc = nn.Sequential(
            nn.Linear(stat_features_size, stat_features_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(stat_features_size // 2, stat_features_size // 4),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        # Fusion layer
        fusion_input_size = (d_model // 2) + (stat_features_size // 4)
        self.fusion_fc = nn.Sequential(
            nn.Linear(fusion_input_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )
        
        self.output_activation = nn.Identity()
        
        logger.info(f"Initialized Transformer Late Fusion model with input_dim={input_dim}, "
                   f"d_model={d_model}, nhead={nhead}, num_layers={num_layers}, "
                   f"stat_features_size={stat_features_size}, num_classes={num_classes}")
        
        # Initialize weights
        self._init_weights()
        
    def forward(self, src: torch.Tensor, stat_features: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of the transformer late fusion model.
        
        Args:
            src: Input tensor [batch_size, seq_len, input_dim]
            stat_features: Statistical features tensor [batch_size, stat_features_size]
            src_mask: Optional mask for src sequence
            
        Returns:
            Output tensor with class probabilities
        """
        # Check input shapes
        batch_size, seq_len, features = src.size()
        if features != self.input_dim and features != self.d_model:
            logger.warning(f"Expected input dimension {self.input_dim} or {self.d_model}, got {features}. Attempting to proceed.")
            
        if stat_features.size(1) != self.stat_features_size:
            logger.warning(f"Expected stat features size {self.stat_features_size}, got {stat_features.size(1)}. Attempting to proceed.")
        
        # Apply transformer encoder
        encoder_output = self.transformer_encoder(src, mask=src_mask)
        
        # Apply attention pooling
        pooled_output, attn_weights = self.attention_pooling(encoder_output)
        
        # Transformer branch features
        transformer_features = self.transformer_fc(pooled_output)
        
        # Statistical features branch forward pass
        stat_features_out = self.stat_fc(stat_features)
        
        # Concatenate features for fusion
        fused_features = torch.cat([transformer_features, stat_features_out], dim=1)
        
        # Pass through fusion layers
        out = self.fusion_fc(fused_features)
        
        # Apply output activation
        out = self.output_activation(out)
        
        return out, attn_weights
    
    def _init_weights(self):
        """
        Initialize the weights for transformer and linear layers
        """
        for name, param in self.named_parameters():
            if 'weight' in name:
                if 'norm' not in name:  # Skip LayerNorm weights
                    nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
  