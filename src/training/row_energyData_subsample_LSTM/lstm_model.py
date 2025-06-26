import torch
import torch.nn as nn
import yaml
import os
from src.utils.logger import logger

class LSTMSequenceModel(nn.Module):
    """
    LSTM model for energy anomaly detection.
    
    This model processes time series data using LSTM layers and makes predictions
    for each time point in the window.
    """
    
    def __init__(self, config=None, config_path=None):
        """
        Initialize the LSTM Sequence model with given configuration.
        
        Args:
            config (dict, optional): Configuration dictionary with model parameters
            config_path (str, optional): Path to YAML configuration file
        """
        super(LSTMSequenceModel, self).__init__()
        
        # Load config from file if provided
        if config is None and config_path is not None:
            config = self._load_config(config_path)
        elif config is None:
             raise ValueError("No configuration provided and default config not found")
        
        # Extract model parameters from config
        self.input_size = config.get('input_size', 27)  # LSTM input features
        self.hidden_size = config.get('hidden_size', 128)
        self.num_layers = config.get('num_layers', 2)
        self.dropout = config.get('dropout', 0.2)
        self.output_size = config.get('output_size', 2)
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout if self.num_layers > 1 else 0,
            bidirectional=True  # Using bidirectional LSTM for better context
        )
        
        # LSTM output projection
        self.lstm_projection = nn.Linear(self.hidden_size * 2, self.hidden_size)  # *2 for bidirectional
        
        # Output layers
        self.output_layers = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size // 2, self.output_size)
        )
        
        self.output_activation = nn.Identity()
            
        logger.info(f"Initialized LSTM Sequence model with input_size={self.input_size}, "
                   f"hidden_size={self.hidden_size}, num_layers={self.num_layers}, "
                   f"bidirectional=True, output_size={self.output_size}")
        
        # Initialize weights
        self._init_weights()
            
    def forward(self, x):
        """
        Forward pass through the LSTM Sequence model.
        
        Args:
            x: LSTM input tensor with shape (batch_size, sequence_length, input_size)
            
        Returns:
            Output tensor with shape (batch_size, sequence_length, output_size)
        """
        # Check input shapes
        batch_size, seq_len, features = x.size()
        if features != self.input_size:
            logger.warning(f"Expected LSTM input size {self.input_size}, got {features}. Attempting to reshape.")

        # LSTM forward pass
        lstm_out, _ = self.lstm(x)  # shape: (batch_size, seq_len, hidden_size*2)
        
        # Project bidirectional output to original hidden size
        lstm_out = self.lstm_projection(lstm_out)  # shape: (batch_size, seq_len, hidden_size)
        
        # Apply output layers to each time step
        # Reshape for efficient processing
        lstm_out_reshaped = lstm_out.reshape(-1, self.hidden_size)  # shape: (batch_size*seq_len, hidden_size)
        outputs = self.output_layers(lstm_out_reshaped)  # shape: (batch_size*seq_len, output_size)
        
        # Apply output activation
        outputs = self.output_activation(outputs)
        
        # Reshape back to sequence format
        outputs = outputs.reshape(batch_size, seq_len, self.output_size)  # shape: (batch_size, seq_len, output_size)
        
        return outputs

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