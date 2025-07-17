"""
FEDformer model implementation for energy anomaly detection.

FEDformer (Frequency Enhanced Decomposition former) uses Fourier-based frequency domain 
decomposition to better capture seasonal and trend patterns in time series data.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from torch.nn.utils import weight_norm
from typing import Optional, Tuple

class FourierBlock(nn.Module):
    """
    Fourier Block for enhanced frequency domain processing
    """
    def __init__(self, d_model: int, seq_len: int, modes: int = 32, mode_select: str = 'random'):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.modes = modes
        self.mode_select = mode_select
        
        # Fourier coefficients (complex weights)
        self.weights = nn.Parameter(torch.randn(modes, d_model, d_model, dtype=torch.cfloat))
        self.init_weights()
    
    def init_weights(self):
        """Initialize Fourier weights"""
        with torch.no_grad():
            self.weights.real = torch.randn_like(self.weights.real) * 0.02
            self.weights.imag = torch.randn_like(self.weights.imag) * 0.02
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, d_model]
        Returns:
            Fourier enhanced features
        """
        B, L, E = x.shape
        
        # Apply FFT
        x_ft = torch.fft.rfft(x, dim=1)  # [B, L//2+1, E]
        
        # Initialize output
        out_ft = torch.zeros(B, L//2+1, E, dtype=torch.cfloat, device=x.device)
        
        # Apply Fourier transform with learned weights
        if self.mode_select == 'random':
            # Random mode selection
            if self.modes < L//2+1:
                indices = torch.randperm(L//2+1)[:self.modes]
            else:
                indices = torch.arange(L//2+1)
        else:
            # Low frequency mode selection
            indices = torch.arange(min(self.modes, L//2+1))
        
        for i, mode_idx in enumerate(indices):
            if i < self.modes and mode_idx < L//2+1:
                # Apply learned Fourier weights
                out_ft[:, mode_idx, :] = torch.einsum('be,eio->bio', 
                                                    x_ft[:, mode_idx, :], 
                                                    self.weights[i])
        
        # Inverse FFT
        x_out = torch.fft.irfft(out_ft, n=L, dim=1)
        
        return x_out

class FourierCrossAttention(nn.Module):
    """
    Fourier Cross Attention for seasonal-trend decomposition
    """
    def __init__(self, d_model: int, n_heads: int = 8, modes: int = 32, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.modes = modes
        
        # Query, Key, Value projections
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        
        # Fourier transform layer
        self.fourier_block = FourierBlock(d_model, seq_len=1024, modes=modes)
        
        # Output projection
        self.out_projection = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor,
                attn_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Fourier Cross Attention forward pass
        """
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads
        
        # Project to Q, K, V
        queries = self.W_q(queries).view(B, L, H, self.d_k)
        keys = self.W_k(keys).view(B, S, H, self.d_k)
        values = self.W_v(values).view(B, S, H, self.d_k)
        
        # Apply Fourier enhancement to queries
        queries_fourier = self.fourier_block(queries.contiguous().view(B, L, -1))
        queries_fourier = queries_fourier.view(B, L, H, self.d_k)
        
        # Compute attention scores
        scores = torch.einsum('blhd,bshd->bhls', queries_fourier, keys)
        
        if attn_mask is not None:
            scores.masked_fill_(attn_mask, -1e9)
        
        # Apply softmax
        A = torch.softmax(scores / math.sqrt(self.d_k), dim=-1)
        A = self.dropout(A)
        
        # Apply attention to values
        V = torch.einsum('bhls,bshd->blhd', A, values)
        V = V.contiguous().view(B, L, -1)
        
        # Output projection
        output = self.out_projection(V)
        
        return output, A

class DecompositionLayer(nn.Module):
    """
    Seasonal-Trend decomposition layer
    """
    def __init__(self, kernel_size: int = 25):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg_pool = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=kernel_size//2)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decompose input into seasonal and trend components
        
        Args:
            x: [batch_size, seq_len, features]
        Returns:
            seasonal, trend components
        """
        # Moving average for trend
        x_permuted = x.permute(0, 2, 1)  # [B, features, seq_len]
        trend = self.avg_pool(x_permuted).permute(0, 2, 1)  # [B, seq_len, features]
        
        # Seasonal = original - trend
        seasonal = x - trend
        
        return seasonal, trend

class FEDformerEncoderLayer(nn.Module):
    """
    FEDformer Encoder Layer with Fourier Enhanced Attention
    """
    def __init__(self, d_model: int, n_heads: int = 8, d_ff: int = 2048, 
                 modes: int = 32, dropout: float = 0.1, activation: str = 'gelu'):
        super().__init__()
        
        # Seasonal-Trend Decomposition
        self.decomp = DecompositionLayer()
        
        # Fourier Cross Attention for seasonal component
        self.seasonal_attention = FourierCrossAttention(d_model, n_heads, modes, dropout)
        
        # Standard attention for trend component  
        self.trend_attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        
        # Feed Forward Networks
        self.seasonal_ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU() if activation == 'gelu' else nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        
        self.trend_ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU() if activation == 'gelu' else nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        
        # Layer Normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        FEDformer Encoder Layer forward pass
        """
        # Seasonal-Trend Decomposition
        seasonal, trend = self.decomp(x)
        
        # Seasonal branch with Fourier attention
        seasonal_attn, _ = self.seasonal_attention(seasonal, seasonal, seasonal, attn_mask)
        seasonal = self.norm1(seasonal + self.dropout(seasonal_attn))
        seasonal_ffn_out = self.seasonal_ffn(seasonal)
        seasonal = self.norm2(seasonal + self.dropout(seasonal_ffn_out))
        
        # Trend branch with standard attention
        trend_attn, _ = self.trend_attention(trend, trend, trend, attn_mask)
        trend = self.norm3(trend + self.dropout(trend_attn))
        trend_ffn_out = self.trend_ffn(trend)
        trend = self.norm4(trend + self.dropout(trend_ffn_out))
        
        # Combine seasonal and trend
        output = seasonal + trend
        
        return output

class FEDformerEncoder(nn.Module):
    """
    FEDformer Encoder with multiple layers
    """
    def __init__(self, encoder_layer: FEDformerEncoderLayer, num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList([encoder_layer for _ in range(num_layers)])
        self.num_layers = num_layers
    
    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        output = x
        for layer in self.layers:
            output = layer(output, attn_mask)
        return output

class PositionalEncoding(nn.Module):
    """
    Positional Encoding for FEDformer
    """
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:x.size(1), :].transpose(0, 1)

class FEDformerModel(nn.Module):
    """
    FEDformer Model for Energy Anomaly Detection
    """
    def __init__(
        self,
        input_size: int,
        d_model: int = 512,
        n_heads: int = 8,
        num_encoder_layers: int = 6,
        d_ff: int = 2048,
        modes: int = 32,
        seq_len: int = 60,
        dropout: float = 0.1,
        num_classes: int = 2,
        activation: str = 'gelu'
    ):
        super().__init__()
        
        self.d_model = d_model
        self.seq_len = seq_len
        
        # Input embedding
        self.input_embedding = nn.Linear(input_size, d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model)
        
        # FEDformer Encoder
        encoder_layer = FEDformerEncoderLayer(
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            modes=modes,
            dropout=dropout,
            activation=activation
        )
        self.encoder = FEDformerEncoder(encoder_layer, num_encoder_layers)
        
        # Classification heads
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )
        
        # Anomaly detection head (regression for anomaly scores)
        self.anomaly_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )
        
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights"""
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor, return_attention: bool = False) -> dict:
        """
        Forward pass for FEDformer
        
        Args:
            x: Input tensor [batch_size, seq_len, input_size]
            return_attention: Whether to return attention weights
            
        Returns:
            Dictionary containing predictions and anomaly scores
        """
        # Input embedding
        x = self.input_embedding(x)  # [B, L, d_model]
        
        # Add positional encoding
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        # FEDformer encoding
        encoded = self.encoder(x)  # [B, L, d_model]
        
        # Classification for each timestep
        logits = self.classifier(encoded)  # [B, L, num_classes]
        
        # Anomaly scores for each timestep
        anomaly_scores = self.anomaly_head(encoded)  # [B, L, 1]
        
        return {
            'logits': logits,
            'anomaly_scores': anomaly_scores.squeeze(-1),  # [B, L]
            'encoded_features': encoded
        }
    
    def get_attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract attention weights for interpretability
        """
        # This would require modifications to store attention weights
        # during forward pass - placeholder for now
        pass 