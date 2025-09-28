import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict, List
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


class FocusScoreModule(nn.Module):
    """
    Focus Score Module for self-regulation based on reconstruction errors.
    This is a key component of TranAD that adjusts attention weights based on reconstruction errors.
    """
    def __init__(self, d_model, focus_threshold=0.1):
        super(FocusScoreModule, self).__init__()
        self.d_model = d_model
        self.focus_threshold = focus_threshold
        
        # Learnable parameters for focus score calculation
        self.focus_weight = nn.Parameter(torch.ones(1))
        self.focus_bias = nn.Parameter(torch.zeros(1))
        
        # Attention adjustment network
        self.attention_adjuster = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, d_model),
            nn.Sigmoid()
        )
        
    def forward(self, x, reconstruction_error):
        """
        Calculate focus scores and adjust attention weights.
        
        Args:
            x: Input features [batch_size, seq_len, d_model]
            reconstruction_error: Reconstruction error [batch_size, seq_len, input_dim]
            
        Returns:
            adjusted_features: Features with adjusted attention weights
            focus_scores: Focus scores for each timestep
        """
        batch_size, seq_len, _ = x.size()
        
        # Calculate focus scores based on reconstruction error
        # Focus score = sigmoid(focus_weight * error + focus_bias)
        error_magnitude = torch.norm(reconstruction_error, dim=-1, keepdim=True)  # [batch_size, seq_len, 1]
        focus_scores = torch.sigmoid(self.focus_weight * error_magnitude + self.focus_bias)
        
        # Apply focus scores to adjust attention weights
        attention_adjustment = self.attention_adjuster(x)
        adjusted_features = x * attention_adjustment * focus_scores
        
        return adjusted_features, focus_scores


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
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Focus score module
        self.focus_score_module = FocusScoreModule(d_model)
        
    def forward(self, x, reconstruction_error=None):
        """
        Forward pass of the encoder.
        
        Args:
            x: Input tensor [batch_size, seq_len, input_dim]
            reconstruction_error: Reconstruction error for focus score calculation
            
        Returns:
            encoded_features: Encoded features [batch_size, seq_len, d_model]
            focus_scores: Focus scores [batch_size, seq_len, 1]
        """
        # Project input to d_model
        x_proj = self.input_projection(x)
        
        # Add positional encoding
        x_pe = self.pe(x_proj)
        
        # Apply transformer encoder
        encoded = self.transformer_encoder(x_pe)
        
        # Apply focus score mechanism if reconstruction error is provided
        if reconstruction_error is not None:
            encoded, focus_scores = self.focus_score_module(encoded, reconstruction_error)
        else:
            focus_scores = None
            
        return encoded, focus_scores


class TranADDecoder(nn.Module):
    """
    TranAD Decoder for reconstruction.
    """
    def __init__(self, d_model, input_dim, num_layers=3, dropout=0.1):
        super(TranADDecoder, self).__init__()
        
        self.d_model = d_model
        self.input_dim = input_dim
        
        # Decoder layers
        decoder_layers = []
        for i in range(num_layers):
            decoder_layers.append(
                nn.Sequential(
                    nn.Linear(d_model if i == 0 else d_model, d_model),
                    nn.ReLU(),
                    nn.Dropout(dropout)
                )
            )
        self.decoder_layers = nn.ModuleList(decoder_layers)
        
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
        x = encoded_features
        
        # Apply decoder layers
        for layer in self.decoder_layers:
            x = layer(x)
        
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
        
        # Adversarial decoder layers (similar structure but different weights)
        decoder_layers = []
        for i in range(num_layers):
            decoder_layers.append(
                nn.Sequential(
                    nn.Linear(d_model if i == 0 else d_model, d_model),
                    nn.ReLU(),
                    nn.Dropout(dropout)
                )
            )
        self.decoder_layers = nn.ModuleList(decoder_layers)
        
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
        x = encoded_features
        
        # Apply adversarial decoder layers
        for layer in self.decoder_layers:
            x = layer(x)
        
        # Project to input dimension
        adversarial_reconstructed = self.output_projection(x)
        
        return adversarial_reconstructed


class MAMLModule(nn.Module):
    """
    Model-Agnostic Meta-Learning (MAML) module for fast adaptation.
    """
    def __init__(self, model, inner_lr=0.01, meta_lr=0.001):
        super(MAMLModule, self).__init__()
        self.model = model
        self.inner_lr = inner_lr
        self.meta_lr = meta_lr
        
    def forward(self, support_data, query_data, num_inner_steps=5):
        """
        Perform MAML adaptation.
        
        Args:
            support_data: Support set for adaptation
            query_data: Query set for evaluation
            num_inner_steps: Number of inner loop steps
            
        Returns:
            adapted_predictions: Predictions on query set after adaptation
        """
        # Store original parameters
        original_params = {name: param.clone() for name, param in self.model.named_parameters()}
        
        # Inner loop: adapt parameters on support set
        for step in range(num_inner_steps):
            # Forward pass on support set
            support_output = self.model(support_data)
            support_loss = F.mse_loss(support_output, support_data)
            
            # Compute gradients
            grads = torch.autograd.grad(support_loss, self.model.parameters(), create_graph=True)
            
            # Update parameters
            for (name, param), grad in zip(self.model.named_parameters(), grads):
                param.data = param.data - self.inner_lr * grad
        
        # Outer loop: evaluate on query set
        query_output = self.model(query_data)
        
        # Restore original parameters
        for name, param in self.model.named_parameters():
            param.data = original_params[name]
        
        return query_output


class TranADModel(nn.Module):
    """
    Complete TranAD model with encoder-decoder architecture, focus score mechanism,
    adversarial training, and MAML for anomaly detection.
    """
    def __init__(self, input_dim, d_model=64, nhead=8, num_encoder_layers=3,
                 num_decoder_layers=3, dim_feedforward=256, dropout=0.1,
                 activation='gelu', use_adversarial=True, use_maml=True):
        super(TranADModel, self).__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        self.use_adversarial = use_adversarial
        self.use_maml = use_maml
        
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
        
        # MAML module (if enabled)
        if use_maml:
            self.maml_module = MAMLModule(self, inner_lr=0.01, meta_lr=0.001)
        
        logger.info("TranAD model initialized successfully")
        
    def forward(self, x, return_components=False, reconstruction_error=None):
        """
        Forward pass of the TranAD model.
        
        Args:
            x: Input tensor [batch_size, seq_len, input_dim]
            return_components: Whether to return individual components
            reconstruction_error: Reconstruction error for focus score calculation
            
        Returns:
            If return_components is False:
                - reconstructed: Main reconstruction [batch_size, seq_len, input_dim]
                - anomaly_scores: Anomaly scores [batch_size, seq_len]
            If return_components is True:
                - reconstructed: Main reconstruction
                - anomaly_scores: Anomaly scores
                - adversarial_reconstructed: Adversarial reconstruction (if enabled)
                - focus_scores: Focus scores (if available)
        """
        # Encode input
        encoded_features, focus_scores = self.encoder(x, reconstruction_error)
        
        # Main reconstruction
        reconstructed = self.decoder(encoded_features)
        
        # Calculate anomaly scores (reconstruction error)
        reconstruction_error = F.mse_loss(reconstructed, x, reduction='none')
        anomaly_scores = torch.mean(reconstruction_error, dim=-1)  # [batch_size, seq_len]
        
        if return_components:
            components = {
                'reconstructed': reconstructed,
                'anomaly_scores': anomaly_scores,
                'focus_scores': focus_scores
            }
            
            # Adversarial reconstruction (if enabled)
            if self.use_adversarial:
                adversarial_reconstructed = self.adversarial_decoder(encoded_features)
                components['adversarial_reconstructed'] = adversarial_reconstructed
            
            return components
        
        return reconstructed, anomaly_scores
    
    def compute_adversarial_loss(self, x, reconstructed, adversarial_reconstructed):
        """
        Compute adversarial loss for dual-decoder training.
        
        Args:
            x: Original input
            reconstructed: Main decoder reconstruction
            adversarial_reconstructed: Adversarial decoder reconstruction
            
        Returns:
            adversarial_loss: Adversarial loss
        """
        # Main decoder loss (minimize reconstruction error)
        main_loss = F.mse_loss(reconstructed, x)
        
        # Adversarial decoder loss (maximize difference from main decoder)
        adversarial_loss = -F.mse_loss(adversarial_reconstructed, reconstructed)
        
        # Total adversarial loss
        total_loss = main_loss + adversarial_loss
        
        return total_loss
    
    def adapt_to_new_data(self, support_data, query_data, num_steps=5):
        """
        Adapt model to new data using MAML.
        
        Args:
            support_data: Support set for adaptation
            query_data: Query set for evaluation
            num_steps: Number of adaptation steps
            
        Returns:
            adapted_predictions: Predictions on query set after adaptation
        """
        if not self.use_maml:
            raise ValueError("MAML is not enabled for this model")
        
        return self.maml_module(support_data, query_data, num_steps)
    
    def get_attention_weights(self, x):
        """
        Extract attention weights from the encoder.
        
        Args:
            x: Input tensor [batch_size, seq_len, input_dim]
            
        Returns:
            attention_weights: Attention weights from the encoder
        """
        with torch.no_grad():
            # Get encoded features
            encoded_features, _ = self.encoder(x)
            
            # Extract attention weights from transformer layers
            attention_weights = []
            for layer in self.encoder.transformer_encoder.layers:
                # This would require modifying the transformer to return attention weights
                # For now, return None as placeholder
                attention_weights.append(None)
            
            return attention_weights


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
            'use_adversarial': True,
            'use_maml': True
        },
        'base': {
            'd_model': 64,
            'nhead': 8,
            'num_encoder_layers': 3,
            'num_decoder_layers': 3,
            'dim_feedforward': 256,
            'dropout': 0.1,
            'use_adversarial': True,
            'use_maml': True
        },
        'large': {
            'd_model': 128,
            'nhead': 16,
            'num_encoder_layers': 4,
            'num_decoder_layers': 4,
            'dim_feedforward': 512,
            'dropout': 0.15,
            'use_adversarial': True,
            'use_maml': True
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
    model = TranADModel(input_dim=input_dim, **config)
    
    # Test input
    test_input = torch.randn(batch_size, seq_len, input_dim)
    
    # Forward pass
    with torch.no_grad():
        # Basic forward pass
        reconstructed, anomaly_scores = model(test_input)
        print(f"Reconstructed shape: {reconstructed.shape}")
        print(f"Anomaly scores shape: {anomaly_scores.shape}")
        
        # With components
        components = model(test_input, return_components=True)
        print(f"Components: {list(components.keys())}")
        
        # Test adversarial loss
        if model.use_adversarial:
            adv_loss = model.compute_adversarial_loss(
                test_input, 
                components['reconstructed'], 
                components['adversarial_reconstructed']
            )
            print(f"Adversarial loss: {adv_loss.item()}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

