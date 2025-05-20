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
        self.input_size = config.get('input_size', 1)
        self.hidden_size = config.get('hidden_size', 64)
        self.num_layers = config.get('num_layers', 2)
        self.dropout = config.get('dropout', 0.2)
        self.bidirectional = config.get('bidirectional', False)
        self.output_size = config.get('output_size', 1)
        
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
        self.fc = nn.Sequential(
            nn.Linear(lstm_output_dim, lstm_output_dim // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(lstm_output_dim // 2, self.output_size)
        )
        
        # Add sigmoid if binary classification
        if config.get('task_type', 'binary_classification') == 'binary_classification':
            self.output_activation = nn.Sigmoid()
        else:
            self.output_activation = nn.Identity()
            
        logger.info(f"Initialized LSTM model with input_size={self.input_size}, "
                   f"hidden_size={self.hidden_size}, num_layers={self.num_layers}, "
                   f"bidirectional={self.bidirectional}, output_size={self.output_size}")
            
    def forward(self, x):
        """
        Forward pass through the LSTM model.
        
        Args:
            x: Input tensor with shape (batch_size, sequence_length, input_size)
            
        Returns:
            Output tensor
        """
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)
        
        # Use the output from the last time step
        out = lstm_out[:, -1, :]
        
        # Pass through fully connected layers
        out = self.fc(out)
        
        # Apply output activation
        out = self.output_activation(out)
        
        return out
    
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