import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

class PositionalEncoding(nn.Module):
    """
    Positional encoding for Transformer models.
    
    Implementation of the sinusoidal position encoding described in 
    "Attention Is All You Need" paper.
    """
    
    def __init__(self, d_model, max_seq_length=1200):
        """
        Initialize positional encoding.
        
        Args:
            d_model: Dimensionality of the model
            max_seq_length: Maximum sequence length
        """
        super(PositionalEncoding, self).__init__()
        
        # Create positional encoding matrix
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # Apply sine to even indices and cosine to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add batch dimension
        pe = pe.unsqueeze(0)
        
        # Register as buffer (not a parameter)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """
        Add positional encoding to input tensor.
        
        Args:
            x: Input tensor of shape [batch_size, seq_length, d_model]
            
        Returns:
            Tensor with positional encoding added
        """
        x = x + self.pe[:, :x.size(1), :]
        return x



class TransformerModel(nn.Module):
    """
    Transformer model for anomaly detection with both per-timestep predictions
    and sequence-level predictions using attention pooling.
    
    This model uses a transformer encoder to process time series data
    and predicts anomalies for each timestep in the sequence and
    provides a sequence-level classification using attention pooling.
    """
    def __init__(
        self,
        input_dim: int,
        d_model: int = 64,
        nhead: int = 8,
        num_layers: int = 2,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        num_classes: int = 2,
        activation: str = 'gelu'
    ):
        """
        Initialize the transformer model.
        
        Args:
            input_dim: Dimension of input features
            d_model: Dimension of the model
            nhead: Number of heads in multi-head attention
            num_layers: Number of transformer layers
            dim_feedforward: Dimension of the feedforward network
            dropout: Dropout probability
            num_classes: Number of output classes (2 for binary classification)
            activation: Activation function for transformer layers
        """
        super(TransformerModel, self).__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        
        # Input projection layer to convert from input_dim to d_model
        self.input_projection = nn.Linear(input_dim, d_model)
        # Initialize weights properly
        nn.init.xavier_uniform_(self.input_projection.weight)
        nn.init.zeros_(self.input_projection.bias)
        
        # Positional encoding
        self.pe = PositionalEncoding(d_model)
        
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
        
        # Output layer for per-timestep classification
        self.timestep_classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )
        
    
        
    def forward(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of the transformer model.
        
        Args:
            src: Input tensor [batch_size, seq_len, input_dim]
            src_mask: Optional mask for src sequence
            
        Returns:
            A tuple containing:
            - timestep_logits: Class logits for each timestep [batch_size, seq_len, num_classes]
        """
        # Project input from input_dim to d_model
        src = self.input_projection(src)
        
        # Apply positional encoding
        src = self.pe(src)
        encoder_output = self.transformer_encoder(src)

        timestep_logits = self.timestep_classifier(encoder_output)
        
        
        return timestep_logits

