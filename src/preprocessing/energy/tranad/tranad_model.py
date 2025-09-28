import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict, List, Any
from utils.logger import logger


class PositionalEncoding(nn.Module):
    """
    Positional encoding for Transformer models.
    """
    def __init__(self, d_model, max_seq_length=1200):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class FocusScoreAttention(nn.Module):
    """
    Focus Score Attention mechanism for TranAD.
    This is a key component that adjusts attention weights based on reconstruction errors.
    """
    def __init__(self, d_model):
        super(FocusScoreAttention, self).__init__()
        
        # Attention mechanism
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        
        # Focus score adjustment
        self.focus_projection = nn.Linear(1, d_model)
        
        # Output projection
        self.output_projection = nn.Linear(d_model, d_model)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(d_model)
        
        # Scaling factor
        self.scale = math.sqrt(d_model)
        
    def forward(self, x, focus_scores=None):
        """
        Apply focus score attention.
        
        Args:
            x: Input features [batch_size, seq_len, d_model]
            focus_scores: Focus scores [batch_size, seq_len, 1]
            
        Returns:
            attended_features: Features with focus-adjusted attention
            attention_weights: Attention weights
        """
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        
        # Compute attention scores
        attention_scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        
        # Apply focus score adjustment if provided
        if focus_scores is not None:
            # Project focus scores to feature dimension
            focus_features = self.focus_projection(focus_scores)
            
            # Adjust attention with focus scores
            # Higher focus scores amplify attention on anomalous regions
            attention_scores = attention_scores + torch.matmul(focus_features, focus_features.transpose(-2, -1))
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        # Apply attention to values
        attended_features = torch.matmul(attention_weights, v)
        
        # Output projection and residual connection
        output = self.layer_norm(x + self.output_projection(attended_features))
        
        return output, attention_weights


class TranADEncoder(nn.Module):
    """
    TranAD Encoder with self-attention and focus score mechanism.
    """
    def __init__(self, input_dim, d_model=64, nhead=8, num_layers=3, 
                 dim_feedforward=256, dropout=0.1, activation='gelu'):
        super(TranADEncoder, self).__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)
        nn.init.xavier_uniform_(self.input_projection.weight)
        nn.init.zeros_(self.input_projection.bias)
        
        # Positional encoding
        self.pe = PositionalEncoding(d_model)
        
        # Transformer encoder layers
        encoder_layers = []
        for _ in range(num_layers):
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation=activation,
                batch_first=True
            )
            encoder_layers.append(encoder_layer)
        self.encoder_layers = nn.ModuleList(encoder_layers)
        
        # Focus score attention
        self.focus_attention = FocusScoreAttention(d_model)
        
    def forward(self, x, focus_scores=None):
        """
        Forward pass of the encoder.
        
        Args:
            x: Input tensor [batch_size, seq_len, input_dim]
            focus_scores: Focus scores for attention adjustment
            
        Returns:
            encoded_features: Encoded features [batch_size, seq_len, d_model]
            attention_weights: Attention weights from focus attention
        """
        # Project input to d_model
        x_proj = self.input_projection(x)
        
        # Add positional encoding
        x_pe = self.pe(x_proj)
        
        # Apply transformer encoder layers
        encoded = x_pe
        for layer in self.encoder_layers:
            encoded = layer(encoded)
        
        # Apply focus score attention if focus scores are provided
        if focus_scores is not None:
            encoded, attention_weights = self.focus_attention(encoded, focus_scores)
        else:
            attention_weights = None
            
        return encoded, attention_weights


class TranADDecoder(nn.Module):
    """
    TranAD Decoder for reconstruction.
    """
    def __init__(self, d_model, input_dim, num_layers=3, dropout=0.1):
        super(TranADDecoder, self).__init__()
        
        self.d_model = d_model
        self.input_dim = input_dim
        
        # Decoder layers
        layers = []
        for i in range(num_layers):
            if i == 0:
                layers.append(nn.Linear(d_model, d_model))
            else:
                layers.append(nn.Linear(d_model, d_model))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        
        self.decoder_layers = nn.Sequential(*layers)
        
        # Output projection
        self.output_projection = nn.Linear(d_model, input_dim)
        nn.init.xavier_uniform_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)
        
    def forward(self, encoded_features):
        """
        Forward pass of the decoder.
        
        Args:
            encoded_features: Encoded features [batch_size, seq_len, d_model]
            
        Returns:
            reconstructed: Reconstructed input [batch_size, seq_len, input_dim]
        """
        # Apply decoder layers
        x = self.decoder_layers(encoded_features)
        
        # Project to input dimension
        reconstructed = self.output_projection(x)
        
        return reconstructed


class AdversarialDecoder(nn.Module):
    """
    Adversarial Decoder for dual-decoder training.
    This decoder is trained to maximize the difference from the main decoder.
    """
    def __init__(self, d_model, input_dim, num_layers=3, dropout=0.1):
        super(AdversarialDecoder, self).__init__()
        
        self.d_model = d_model
        self.input_dim = input_dim
        
        # Adversarial decoder layers
        layers = []
        for i in range(num_layers):
            if i == 0:
                layers.append(nn.Linear(d_model, d_model))
            else:
                layers.append(nn.Linear(d_model, d_model))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        
        self.decoder_layers = nn.Sequential(*layers)
        
        # Output projection
        self.output_projection = nn.Linear(d_model, input_dim)
        nn.init.xavier_uniform_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)
        
    def forward(self, encoded_features):
        """
        Forward pass of the adversarial decoder.
        
        Args:
            encoded_features: Encoded features [batch_size, seq_len, d_model]
            
        Returns:
            adversarial_reconstructed: Adversarial reconstruction [batch_size, seq_len, input_dim]
        """
        # Apply adversarial decoder layers
        x = self.decoder_layers(encoded_features)
        
        # Project to input dimension
        adversarial_reconstructed = self.output_projection(x)
        
        return adversarial_reconstructed


class TranAD(nn.Module):
    """
    Complete TranAD model with encoder-decoder architecture, focus score mechanism,
    and adversarial training for anomaly detection in time series data.
    
    This model implements the architecture from the paper:
    "TranAD: Deep Transformer Networks for Anomaly Detection in Multivariate Time Series Data"
    """
    def __init__(
        self, 
        input_dim, 
        d_model=64, 
        nhead=8, 
        num_encoder_layers=3,
        num_decoder_layers=3, 
        dim_feedforward=256, 
        dropout=0.1,
        activation='gelu', 
        use_adversarial=True
    ):
        super(TranAD, self).__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        self.use_adversarial = use_adversarial
        
        logger.info(f"Initializing TranAD model with input_dim={input_dim}, d_model={d_model}")
        
        # Encoder
        self.encoder = TranADEncoder(
            input_dim=input_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_encoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation
        )
        
        # Main decoder
        self.decoder = TranADDecoder(
            d_model=d_model,
            input_dim=input_dim,
            num_layers=num_decoder_layers,
            dropout=dropout
        )
        
        # Adversarial decoder (if enabled)
        if use_adversarial:
            self.adversarial_decoder = AdversarialDecoder(
                d_model=d_model,
                input_dim=input_dim,
                num_layers=num_decoder_layers,
                dropout=dropout
            )
        
        logger.info("TranAD model initialized successfully")
        
    def forward(self, x, phase=1, return_all=False):
        """
        Forward pass of the TranAD model.
        
        Args:
            x: Input tensor [batch_size, seq_len, input_dim]
            phase: Training phase (1 or 2)
                   Phase 1: Initial reconstruction
                   Phase 2: Focus-adjusted reconstruction
            return_all: Whether to return all components
            
        Returns:
            If return_all is False:
                - reconstructed: Main reconstruction [batch_size, seq_len, input_dim]
                - anomaly_scores: Anomaly scores [batch_size, seq_len]
            If return_all is True:
                Dictionary containing:
                - 'reconstructed': Main reconstruction
                - 'anomaly_scores': Anomaly scores
                - 'adversarial_reconstructed': Adversarial reconstruction (if enabled)
                - 'attention_weights': Attention weights (if phase 2)
                - 'encoded_features': Encoded features
        """
        if phase == 1:
            # Phase 1: Initial reconstruction without focus scores
            encoded_features, _ = self.encoder(x)
            reconstructed = self.decoder(encoded_features)
            
            # Calculate reconstruction error
            reconstruction_error = F.mse_loss(reconstructed, x, reduction='none')
            anomaly_scores = torch.mean(reconstruction_error, dim=-1)  # [batch_size, seq_len]
            
            # Calculate focus scores (normalized reconstruction error)
            focus_scores = torch.sigmoid(reconstruction_error.mean(dim=-1, keepdim=True))
            
            if return_all:
                result = {
                    'reconstructed': reconstructed,
                    'anomaly_scores': anomaly_scores,
                    'focus_scores': focus_scores,
                    'encoded_features': encoded_features
                }
                
                if self.use_adversarial:
                    adversarial_reconstructed = self.adversarial_decoder(encoded_features)
                    result['adversarial_reconstructed'] = adversarial_reconstructed
                
                return result
            
            return reconstructed, anomaly_scores, focus_scores
            
        elif phase == 2:
            # Phase 1 forward pass to get focus scores
            with torch.no_grad():
                _, _, focus_scores = self.forward(x, phase=1)
            
            # Phase 2: Focus-adjusted reconstruction
            encoded_features, attention_weights = self.encoder(x, focus_scores)
            reconstructed = self.decoder(encoded_features)
            
            # Calculate reconstruction error
            reconstruction_error = F.mse_loss(reconstructed, x, reduction='none')
            anomaly_scores = torch.mean(reconstruction_error, dim=-1)  # [batch_size, seq_len]
            
            if return_all:
                result = {
                    'reconstructed': reconstructed,
                    'anomaly_scores': anomaly_scores,
                    'attention_weights': attention_weights,
                    'focus_scores': focus_scores,
                    'encoded_features': encoded_features
                }
                
                if self.use_adversarial:
                    adversarial_reconstructed = self.adversarial_decoder(encoded_features)
                    result['adversarial_reconstructed'] = adversarial_reconstructed
                
                return result
            
            return reconstructed, anomaly_scores, attention_weights
        
        else:
            raise ValueError(f"Invalid phase {phase}. Must be 1 or 2.")
    
    def compute_loss(self, x, phase=1):
        """
        Compute loss for the TranAD model.
        
        Args:
            x: Input tensor [batch_size, seq_len, input_dim]
            phase: Training phase (1 or 2)
            
        Returns:
            loss: Total loss
            loss_components: Dictionary of loss components
        """
        if phase == 1:
            # Phase 1: Basic reconstruction loss
            outputs = self.forward(x, phase=1, return_all=True)
            reconstructed = outputs['reconstructed']
            
            # Reconstruction loss (MSE)
            recon_loss = F.mse_loss(reconstructed, x)
            
            loss_components = {'recon_loss': recon_loss}
            total_loss = recon_loss
            
            # Adversarial loss if enabled
            if self.use_adversarial and 'adversarial_reconstructed' in outputs:
                adv_reconstructed = outputs['adversarial_reconstructed']
                
                # Main decoder tries to minimize reconstruction error
                # Adversarial decoder tries to maximize difference from main decoder
                adv_loss = -F.mse_loss(adv_reconstructed, reconstructed)
                
                loss_components['adv_loss'] = adv_loss
                total_loss = total_loss + 0.1 * adv_loss  # Weight for adversarial loss
            
            return total_loss, loss_components
            
        elif phase == 2:
            # Phase 2: Focus-adjusted reconstruction loss
            outputs = self.forward(x, phase=2, return_all=True)
            reconstructed = outputs['reconstructed']
            
            # Reconstruction loss (MSE)
            recon_loss = F.mse_loss(reconstructed, x)
            
            loss_components = {'recon_loss': recon_loss}
            total_loss = recon_loss
            
            # Adversarial loss if enabled
            if self.use_adversarial and 'adversarial_reconstructed' in outputs:
                adv_reconstructed = outputs['adversarial_reconstructed']
                
                # Main decoder tries to minimize reconstruction error
                # Adversarial decoder tries to maximize difference from main decoder
                adv_loss = -F.mse_loss(adv_reconstructed, reconstructed)
                
                loss_components['adv_loss'] = adv_loss
                total_loss = total_loss + 0.1 * adv_loss  # Weight for adversarial loss
            
            return total_loss, loss_components
        
        else:
            raise ValueError(f"Invalid phase {phase}. Must be 1 or 2.")
    
    def detect_anomalies(self, x, threshold=None, percentile=95):
        """
        Detect anomalies using reconstruction error.
        
        Args:
            x: Input tensor [batch_size, seq_len, input_dim]
            threshold: Anomaly threshold (if None, use percentile)
            percentile: Percentile for threshold calculation
            
        Returns:
            anomaly_scores: Anomaly scores [batch_size, seq_len]
            anomalies: Binary anomaly indicators [batch_size, seq_len]
            threshold: Threshold used for detection
        """
        # Phase 2 forward pass (with focus adjustment)
        _, anomaly_scores, _ = self.forward(x, phase=2)
        
        # Calculate threshold if not provided
        if threshold is None:
            threshold = torch.quantile(anomaly_scores, percentile/100)
        
        # Detect anomalies
        anomalies = (anomaly_scores > threshold).float()
        
        return anomaly_scores, anomalies, threshold


def create_tranad_config(model_size='base'):
    """
    Create TranAD model configuration for different model sizes.
    
    Args:
        model_size: Size of the model ('small', 'base', 'large')
        
    Returns:
        Model configuration dictionary
    """
    configs = {
        'small': {
            'd_model': 32,
            'nhead': 4,
            'num_encoder_layers': 2,
            'num_decoder_layers': 2,
            'dim_feedforward': 128,
            'dropout': 0.1,
            'use_adversarial': True
        },
        'base': {
            'd_model': 64,
            'nhead': 8,
            'num_encoder_layers': 3,
            'num_decoder_layers': 3,
            'dim_feedforward': 256,
            'dropout': 0.1,
            'use_adversarial': True
        },
        'large': {
            'd_model': 128,
            'nhead': 16,
            'num_encoder_layers': 4,
            'num_decoder_layers': 4,
            'dim_feedforward': 512,
            'dropout': 0.15,
            'use_adversarial': True
        }
    }
    
    return configs.get(model_size, configs['base'])


if __name__ == "__main__":
    # Test the TranAD model
    import torch
    
    # Model parameters
    batch_size = 32
    seq_len = 100
    input_dim = 10
    
    # Create model
    config = create_tranad_config('base')
    model = TranAD(input_dim=input_dim, **config)
    
    # Test input
    test_input = torch.randn(batch_size, seq_len, input_dim)
    
    # Forward pass
    with torch.no_grad():
        # Phase 1
        reconstructed, anomaly_scores, focus_scores = model(test_input, phase=1)
        print(f"Phase 1 - Reconstructed shape: {reconstructed.shape}")
        print(f"Phase 1 - Anomaly scores shape: {anomaly_scores.shape}")
        print(f"Phase 1 - Focus scores shape: {focus_scores.shape}")
        
        # Phase 2
        reconstructed, anomaly_scores, attention_weights = model(test_input, phase=2)
        print(f"Phase 2 - Reconstructed shape: {reconstructed.shape}")
        print(f"Phase 2 - Anomaly scores shape: {anomaly_scores.shape}")
        
        # Test anomaly detection
        anomaly_scores, anomalies, threshold = model.detect_anomalies(test_input)
        print(f"Anomaly detection - Scores shape: {anomaly_scores.shape}")
        print(f"Anomaly detection - Anomalies shape: {anomalies.shape}")
        print(f"Anomaly detection - Threshold: {threshold.item()}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
