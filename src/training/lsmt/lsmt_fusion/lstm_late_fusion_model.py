import torch
import torch.nn as nn
import yaml
import os
from src.utils.logger import logger

class LSTMLateFusionModel(nn.Module):
    """
    LSTM model with Late Fusion for energy anomaly detection.
    
    This model processes time series data using LSTM layers and combines it with
    statistical features through late fusion for improved classification.
    """
    
    def __init__(self, config=None, config_path=None):
        """
        Initialize the LSTM Late Fusion model with given configuration.
        
        Args:
            config (dict, optional): Configuration dictionary with model parameters
            config_path (str, optional): Path to YAML configuration file
        """
        super(LSTMLateFusionModel, self).__init__()
        
        # Load config from file if provided
        if config is None and config_path is not None:
            config = self._load_config(config_path)
        elif config is None:
             raise ValueError("No configuration provided and default config not found")
        
        # Extract model parameters from config
        self.input_size = config.get('input_size', 31)  # LSTM input features
        self.hidden_size = config.get('hidden_size', 128)
        self.num_layers = config.get('num_layers', 2)
        self.dropout = config.get('dropout', 0.2)
        self.output_size = config.get('output_size', 2)
        
        # Statistical features parameters
        self.stat_features_size = config.get('stat_features_size', 47)  # Number of statistical features
        
        # LSTM branch
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout if self.num_layers > 1 else 0,
        )
        
        '''# Attention layer for LSTM outputs
        self.attention_layer = nn.Linear(self.hidden_size, 1)
        nn.init.xavier_uniform_(self.attention_layer.weight, gain=2.0)
        nn.init.zeros_(self.attention_layer.bias)'''

        # LSTM branch fully connected layers
        self.lstm_fc = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        )
        
        # Statistical features branch
        self.stat_fc = nn.Sequential(
            nn.Linear(self.stat_features_size, self.stat_features_size // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.stat_features_size // 2, self.stat_features_size // 4),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        )
        
        # Fusion layer
        fusion_input_size = (self.hidden_size // 2) + (self.stat_features_size // 4)
        self.fusion_fc = nn.Sequential(
            nn.Linear(fusion_input_size, fusion_input_size // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(fusion_input_size // 2, self.output_size)
        )
        
        self.output_activation = nn.Identity()
            
        logger.info(f"Initialized LSTM Late Fusion model with LSTM input_size={self.input_size}, "
                   f"hidden_size={self.hidden_size}, num_layers={self.num_layers}, "
                   f"stat_features_size={self.stat_features_size}, output_size={self.output_size}")
        
        # Initialize weights
        self._init_weights()
            
    def forward(self, x, stat_features):
        """
        Forward pass through the LSTM Late Fusion model.
        
        Args:
            x: LSTM input tensor with shape (batch_size, sequence_length, input_size)
            stat_features: Statistical features tensor with shape (batch_size, stat_features_size)
            
        Returns:
            Output tensor and attention weights
        """
        # Check input shapes
        batch_size, seq_len, features = x.size()
        if features != self.input_size:
            logger.warning(f"Expected LSTM input size {self.input_size}, got {features}. Attempting to reshape.")
            
        if stat_features.size(1) != self.stat_features_size:
            logger.warning(f"Expected stat features size {self.stat_features_size}, got {stat_features.size(1)}. Attempting to reshape.")

        # LSTM branch forward pass
        lstm_out, _ = self.lstm(x)
        #logger.info(f"LSTM output STD per time step: {lstm_out.std(dim=1)}")

        
        lstm_out = lstm_out[:, -1, :]  # Use the output from the last time step
        lstm_features = self.lstm_fc(lstm_out)
        
        '''
        # Apply attention mechanism
        attn_scores = self.attention_layer(lstm_out)             # (B, T, 1)
        attn_weights = torch.softmax(attn_scores, dim=1)         # (B, T, 1)

        context_vector = torch.sum(attn_weights * lstm_out, dim=1)  # (B, H)
        lstm_features = self.lstm_fc(context_vector)
        '''
        ''' scores = self.attention_layer(lstm_out)           # (B, T, 1)
        weights = torch.sigmoid(scores)                   # [0, 1]
        weights = weights / weights.sum(dim=1, keepdim=True)
        context = torch.sum(weights * lstm_out, dim=1)
        lstm_features = self.lstm_fc(context)'''
        
        # Statistical features branch forward pass
        stat_out = self.stat_fc(stat_features)
        
        # Concatenate features for fusion
        fused_features = torch.cat([lstm_features, stat_out], dim=1)
        
        # Pass through fusion layers
        out = self.fusion_fc(fused_features)
        
        # Apply output activation
        out = self.output_activation(out)
        
        #return out, attn_weights
        #return out,weights
        return out

    def _init_weights(self):
        """
        Initialize the weights for LSTM and linear layers
        """
        for name, param in self.named_parameters():
            if 'lstm' in name:
                if 'weight_ih' in name:
                    nn.init.xavier_uniform_(param)
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(param)
                elif 'bias' in name:
                    nn.init.zeros_(param)
            elif 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def _load_config(self, config_path):
        """
        Load model configuration from YAML file.
        
        Args:
            config_path (str): Path to configuration file
            
        Returns:
            dict: Configuration dictionary
        """
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                
            # Extract model configuration
            model_config = config.get('model', {})
            logger.info(f"Loaded model configuration from {config_path}")
            return model_config
        except Exception as e:
            logger.error(f"Error loading configuration from {config_path}: {str(e)}")
            raise 