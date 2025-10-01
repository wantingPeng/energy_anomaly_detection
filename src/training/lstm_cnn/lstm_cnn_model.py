"""
LSTM-CNN Hybrid Model for Energy Anomaly Detection.

This module implements a hybrid architecture that combines:
- CNN layers for local feature extraction and pattern recognition
- LSTM layers for temporal sequence modeling
- Per-timestep anomaly classification

The model processes time series data with sliding windows and outputs
anomaly predictions for each timestep in the sequence.
"""

import torch
import torch.nn as nn
from typing import Optional


class LSTMCNN(nn.Module):
    """
    LSTM-CNN Hybrid Model for time series anomaly detection.
    
    Architecture:
        1. CNN branch: Multi-scale 1D convolutions for local pattern extraction
        2. LSTM branch: Bidirectional LSTM for temporal dependencies
        3. Feature fusion: Concatenate CNN and LSTM features
        4. Classification: Per-timestep binary classification
    
    Args:
        input_dim: Number of input features
        cnn_channels: List of channel numbers for CNN layers
        cnn_kernel_sizes: List of kernel sizes for CNN layers
        lstm_hidden_dim: Hidden dimension for LSTM layers
        lstm_num_layers: Number of LSTM layers
        dropout: Dropout probability
        num_classes: Number of output classes (2 for binary classification)
        use_bidirectional: Whether to use bidirectional LSTM
    """
    
    def __init__(
        self,
        input_dim: int,
        cnn_channels: list = [64, 128, 128],
        cnn_kernel_sizes: list = [3, 3, 3],
        lstm_hidden_dim: int = 128,
        lstm_num_layers: int = 2,
        dropout: float = 0.3,
        num_classes: int = 2,
        use_bidirectional: bool = True
    ):
        super(LSTMCNN, self).__init__()
        
        self.input_dim = input_dim
        self.cnn_channels = cnn_channels
        self.lstm_hidden_dim = lstm_hidden_dim
        self.lstm_num_layers = lstm_num_layers
        self.num_classes = num_classes
        self.use_bidirectional = use_bidirectional
        
        # ===== CNN Branch for Local Feature Extraction =====
        cnn_layers = []
        in_channels = input_dim
        
        for i, (out_channels, kernel_size) in enumerate(zip(cnn_channels, cnn_kernel_sizes)):
            # 1D Convolution
            cnn_layers.append(
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,  # Same padding
                    bias=False
                )
            )
            # Batch Normalization
            cnn_layers.append(nn.BatchNorm1d(out_channels))
            # Activation
            cnn_layers.append(nn.ReLU(inplace=True))
            # Dropout
            if i < len(cnn_channels) - 1:  # No dropout after last CNN layer
                cnn_layers.append(nn.Dropout(dropout))
            
            in_channels = out_channels
        
        self.cnn_branch = nn.Sequential(*cnn_layers)
        self.cnn_output_dim = cnn_channels[-1]
        
        # ===== LSTM Branch for Temporal Modeling =====
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout=dropout if lstm_num_layers > 1 else 0,
            bidirectional=use_bidirectional
        )
        
        # Calculate LSTM output dimension
        lstm_output_dim = lstm_hidden_dim * 2 if use_bidirectional else lstm_hidden_dim
        
        # ===== Feature Fusion =====
        # Combine CNN and LSTM features
        fusion_dim = self.cnn_output_dim + lstm_output_dim
        
        # ===== Classification Head =====
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LSTM):
                for name, param in module.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.xavier_uniform_(param.data)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param.data)
                    elif 'bias' in name:
                        nn.init.constant_(param.data, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the LSTM-CNN model.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, input_dim]
        
        Returns:
            Output logits of shape [batch_size, seq_len, num_classes]
        """
        batch_size, seq_len, input_dim = x.size()
        
        # ===== CNN Branch =====
        # Transpose for Conv1d: [batch_size, seq_len, input_dim] -> [batch_size, input_dim, seq_len]
        x_cnn = x.transpose(1, 2)
        
        # Apply CNN layers
        cnn_features = self.cnn_branch(x_cnn)  # [batch_size, cnn_channels[-1], seq_len]
        
        # Transpose back: [batch_size, cnn_channels[-1], seq_len] -> [batch_size, seq_len, cnn_channels[-1]]
        cnn_features = cnn_features.transpose(1, 2)
        
        # ===== LSTM Branch =====
        # x is already in the right shape: [batch_size, seq_len, input_dim]
        lstm_output, _ = self.lstm(x)  # [batch_size, seq_len, lstm_output_dim]
        
        # ===== Feature Fusion =====
        # Concatenate CNN and LSTM features along the feature dimension
        fused_features = torch.cat([cnn_features, lstm_output], dim=2)  # [batch_size, seq_len, fusion_dim]
        
        # ===== Classification =====
        # Apply classifier to each timestep
        logits = self.classifier(fused_features)  # [batch_size, seq_len, num_classes]
        
        return logits
    
    def get_num_params(self) -> dict:
        """
        Get the number of parameters in the model.
        
        Returns:
            Dictionary with parameter counts
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        cnn_params = sum(p.numel() for p in self.cnn_branch.parameters())
        lstm_params = sum(p.numel() for p in self.lstm.parameters())
        classifier_params = sum(p.numel() for p in self.classifier.parameters())
        
        return {
            'total': total_params,
            'trainable': trainable_params,
            'cnn_branch': cnn_params,
            'lstm_branch': lstm_params,
            'classifier': classifier_params
        }


class SimpleLSTMCNN(nn.Module):
    """
    Simplified LSTM-CNN Model for faster training and inference.
    
    This is a lightweight version with fewer parameters, suitable for
    smaller datasets or when computational resources are limited.
    
    Args:
        input_dim: Number of input features
        cnn_channels: Number of channels for the CNN layer
        lstm_hidden_dim: Hidden dimension for LSTM
        dropout: Dropout probability
        num_classes: Number of output classes
    """
    
    def __init__(
        self,
        input_dim: int,
        cnn_channels: int = 64,
        lstm_hidden_dim: int = 64,
        dropout: float = 0.2,
        num_classes: int = 2
    ):
        super(SimpleLSTMCNN, self).__init__()
        
        self.input_dim = input_dim
        self.cnn_channels = cnn_channels
        self.lstm_hidden_dim = lstm_hidden_dim
        self.num_classes = num_classes
        
        # Single CNN layer
        self.cnn = nn.Sequential(
            nn.Conv1d(input_dim, cnn_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        
        # Single bidirectional LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        
        # Fusion and classification
        fusion_dim = cnn_channels + lstm_hidden_dim * 2
        
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, (nn.Conv1d, nn.Linear)):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, input_dim]
        
        Returns:
            Output logits of shape [batch_size, seq_len, num_classes]
        """
        # CNN branch
        x_cnn = x.transpose(1, 2)
        cnn_features = self.cnn(x_cnn).transpose(1, 2)
        
        # LSTM branch
        lstm_output, _ = self.lstm(x)
        
        # Fusion and classification
        fused = torch.cat([cnn_features, lstm_output], dim=2)
        logits = self.classifier(fused)
        
        return logits
    
    def get_num_params(self) -> dict:
        """Get parameter counts."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total': total_params,
            'trainable': trainable_params
        }


if __name__ == "__main__":
    # Test the model
    batch_size = 32
    seq_len = 60
    input_dim = 10
    
    # Create sample data
    x = torch.randn(batch_size, seq_len, input_dim)
    
    # Test LSTMCNN
    print("Testing LSTMCNN model...")
    model = LSTMCNN(input_dim=input_dim)
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Parameters: {model.get_num_params()}")
    
    # Test SimpleLSTMCNN
    print("\nTesting SimpleLSTMCNN model...")
    simple_model = SimpleLSTMCNN(input_dim=input_dim)
    simple_output = simple_model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {simple_output.shape}")
    print(f"Parameters: {simple_model.get_num_params()}")

