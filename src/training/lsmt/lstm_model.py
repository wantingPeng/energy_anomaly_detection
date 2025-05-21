import torch
import torch.nn as nn
import yaml
import os
from src.utils.logger import logger

class LSTMModel(nn.Module):
    """
    LSTM model for energy anomaly detection.
    
    This model processes time series data using LSTM layers followed by
    fully connected layers for classification or regression tasks.
    """
    
    def __init__(self, config=None, config_path=None):
        """
        Initialize the LSTM model with given configuration.
        
        Args:
            config (dict, optional): Configuration dictionary with model parameters
            config_path (str, optional): Path to YAML configuration file
        """
        super(LSTMModel, self).__init__()
        
        # Load config from file if provided
        if config is None and config_path is not None:
            config = self._load_config(config_path)
        elif config is None:
             raise ValueError("No configuration provided and default config not found")
        
        # Extract model parameters from config
        self.input_size = config.get('input_size', 31)  # Updated default to match dataset dimensions
        self.hidden_size = config.get('hidden_size', 128)
        self.num_layers = config.get('num_layers', 2)
        self.dropout = config.get('dropout', 0.2)
        self.bidirectional = config.get('bidirectional', False)
        self.output_size = config.get('output_size', 2)
        
        # Calculate the output dimension from LSTM (accounting for bidirectional)
        lstm_output_dim = self.hidden_size * 2 if self.bidirectional else self.hidden_size
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout if self.num_layers > 1 else 0,
            bidirectional=self.bidirectional
        )
        
        # Fully connected layers for output
        fc_layers = []
        
        # First FC layer
        fc_layers.append(nn.Linear(lstm_output_dim, lstm_output_dim // 2))
        fc_layers.append(nn.ReLU())
        fc_layers.append(nn.Dropout(self.dropout))
        
        # Optional middle layer for larger hidden dimensions
        if lstm_output_dim > 128:
            middle_dim = lstm_output_dim // 4
            fc_layers.append(nn.Linear(lstm_output_dim // 2, middle_dim))
            fc_layers.append(nn.ReLU())
            fc_layers.append(nn.Dropout(self.dropout))
            fc_layers.append(nn.Linear(middle_dim, self.output_size))
        else:
            fc_layers.append(nn.Linear(lstm_output_dim // 2, self.output_size))
        
        self.fc = nn.Sequential(*fc_layers)
        
        # Add sigmoid if binary classification
        if config.get('task_type', 'binary_classification') == 'binary_classification' and self.output_size == 1:
            self.output_activation = nn.Sigmoid()
        else:
            # For CrossEntropyLoss, don't use sigmoid activation
            self.output_activation = nn.Identity()
            
        logger.info(f"Initialized LSTM model with input_size={self.input_size}, "
                   f"hidden_size={self.hidden_size}, num_layers={self.num_layers}, "
                   f"bidirectional={self.bidirectional}, output_size={self.output_size}")
        
        # Initialize weights
        self._init_weights()
            
    def forward(self, x):
        """
        Forward pass through the LSTM model.
        
        Args:
            x: Input tensor with shape (batch_size, sequence_length, input_size)
            
        Returns:
            Output tensor
        """
        # Check input shape
        batch_size, seq_len, features = x.size()
        if features != self.input_size:
            logger.warning(f"Expected input size {self.input_size}, got {features}. Attempting to reshape.")
            if features == 1:
                # If single feature, assume it's a univariate series and the features are combined
                x = x.reshape(batch_size, seq_len // self.input_size, self.input_size)
        
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)
        
        # Use the output from the last time step
        out = lstm_out[:, -1, :]
        
        # Pass through fully connected layers
        out = self.fc(out)
        
        # Apply output activation
        out = self.output_activation(out)
        
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
            elif 'fc' in name:
                if 'weight' in name:
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