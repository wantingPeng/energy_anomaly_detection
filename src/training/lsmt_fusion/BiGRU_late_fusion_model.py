import torch
import torch.nn as nn
import yaml
import os
from src.utils.logger import logger

class BiGRULateFusionModel(nn.Module):
    """
    Bidirectional GRU model with Late Fusion for energy anomaly detection.
    
    This model processes time series data using Bidirectional GRU layers with residual connections
    and layer normalization, and combines it with statistical features through 
    late fusion for improved classification.
    """
    
    def __init__(self, config=None, config_path=None):
        """
        Initialize the BiGRU Late Fusion model with given configuration.
        
        Args:
            config (dict, optional): Configuration dictionary with model parameters
            config_path (str, optional): Path to YAML configuration file
        """
        super(BiGRULateFusionModel, self).__init__()
        
        # Load config from file if provided
        if config is None and config_path is not None:
            config = self._load_config(config_path)
        elif config is None:
             raise ValueError("No configuration provided and default config not found")
        
        # Extract model parameters from config
        self.input_size = config.get('input_size', 31)  # GRU input features
        self.hidden_size = config.get('hidden_size', 128)
        self.num_layers = config.get('num_layers', 2)
        self.dropout = config.get('dropout', 0.2)
        self.output_size = config.get('output_size', 2)
        
        # Statistical features parameters
        self.stat_features_size = config.get('stat_features_size', 47)  # Number of statistical features
        
        # BiGRU layers
        self.gru_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        
        # First BiGRU layer
        self.gru_layers.append(
            nn.GRU(
                input_size=self.input_size,
                hidden_size=self.hidden_size // 2,  # Half size for bidirectional
                num_layers=1,
                batch_first=True,
                bidirectional=True,
            )
        )
        self.layer_norms.append(nn.LayerNorm(self.hidden_size))
        
        # Additional BiGRU layers with residual connections
        for i in range(1, self.num_layers):
            self.gru_layers.append(
                nn.GRU(
                    input_size=self.hidden_size,
                    hidden_size=self.hidden_size // 2,  # Half size for bidirectional
                    num_layers=1,
                    batch_first=True,
                    bidirectional=True,
                )
            )
            self.layer_norms.append(nn.LayerNorm(self.hidden_size))
            
        # Dropout layer for BiGRU outputs
        self.dropout_layer = nn.Dropout(self.dropout)
        
        # Attention layer for BiGRU outputs
        self.attention_layer = nn.Linear(self.hidden_size, 1)
        nn.init.xavier_uniform_(self.attention_layer.weight, gain=2.0)
        nn.init.zeros_(self.attention_layer.bias)

        # BiGRU branch fully connected layers
        self.gru_fc = nn.Sequential(
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
            
        logger.info(f"Initialized BiGRU Late Fusion model with BiGRU input_size={self.input_size}, "
                   f"hidden_size={self.hidden_size}, num_layers={self.num_layers}, "
                   f"stat_features_size={self.stat_features_size}, output_size={self.output_size}")
        
        # Initialize weights
        self._init_weights()
            
    def forward(self, x, stat_features):
        """
        Forward pass through the BiGRU Late Fusion model.
        
        Args:
            x: BiGRU input tensor with shape (batch_size, sequence_length, input_size)
            stat_features: Statistical features tensor with shape (batch_size, stat_features_size)
            
        Returns:
            Output tensor and attention weights
        """
        # Check input shapes
        batch_size, seq_len, features = x.size()
        if features != self.input_size:
            logger.warning(f"Expected BiGRU input size {self.input_size}, got {features}. Attempting to reshape.")
            
        if stat_features.size(1) != self.stat_features_size:
            logger.warning(f"Expected stat features size {self.stat_features_size}, got {stat_features.size(1)}. Attempting to reshape.")

        # BiGRU layers with residual connections and layer normalization
        residual = None
        hidden_states = x
        
        for i, (gru, layer_norm) in enumerate(zip(self.gru_layers, self.layer_norms)):
            gru_out, _ = gru(hidden_states)
            
            # Apply residual connection if not first layer
            if i > 0 and residual is not None:
                gru_out = gru_out + residual
                
            # Apply layer normalization
            gru_out = layer_norm(gru_out)
            
            # Apply dropout
            gru_out = self.dropout_layer(gru_out)
            
            # Save for next residual connection
            residual = gru_out
            hidden_states = gru_out
        
        '''# Apply attention mechanism
        scores = self.attention_layer(hidden_states)      # (B, T, 1)
        weights = torch.sigmoid(scores)                   # [0, 1]
        weights = weights / weights.sum(dim=1, keepdim=True)
        context = torch.sum(weights * hidden_states, dim=1)
        gru_features = self.gru_fc(context)'''
        
        gru_out = gru_out[:, -1, :]  # Use the output from the last time step
        gru_features = self.gru_fc(gru_out)

        # Statistical features branch forward pass
        stat_out = self.stat_fc(stat_features)
        
        # Concatenate features for fusion
        fused_features = torch.cat([gru_features, stat_out], dim=1)
        
        # Pass through fusion layers
        out = self.fusion_fc(fused_features)
        
        # Apply output activation
        out = self.output_activation(out)
        
        return out
        #return out, weights

    def _init_weights(self):
        """
        Initialize the weights for GRU and linear layers
        """
        for name, param in self.named_parameters():
            if 'gru' in name:
                if 'weight_ih' in name:
                    nn.init.xavier_uniform_(param)
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(param)
                elif 'bias' in name:
                    nn.init.zeros_(param)
            elif 'weight' in name and len(param.shape) >= 2:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name or (len(param.shape) < 2 and 'weight' in name):
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