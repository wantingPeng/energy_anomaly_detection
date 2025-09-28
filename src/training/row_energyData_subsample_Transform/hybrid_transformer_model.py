import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
from utils.logger import logger

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


class LocalFeatureCNN(nn.Module):
    """
    CNN branch for extracting local patterns and sharp anomalous changes.
    Uses multiple kernel sizes to capture different temporal patterns.
    """
    def __init__(self, input_dim, output_dim, dropout=0.1):
        super(LocalFeatureCNN, self).__init__()
        
        # Multi-scale 1D convolutions with different kernel sizes
        self.conv1_small = nn.Conv1d(input_dim, output_dim // 4, kernel_size=3, padding=1)
        self.conv1_medium = nn.Conv1d(input_dim, output_dim // 4, kernel_size=5, padding=2)
        self.conv1_large = nn.Conv1d(input_dim, output_dim // 4, kernel_size=7, padding=3)
        self.conv1_xlarge = nn.Conv1d(input_dim, output_dim // 4, kernel_size=11, padding=5)
        
        # Batch normalization for each scale
        self.bn1_small = nn.BatchNorm1d(output_dim // 4)
        self.bn1_medium = nn.BatchNorm1d(output_dim // 4)
        self.bn1_large = nn.BatchNorm1d(output_dim // 4)
        self.bn1_xlarge = nn.BatchNorm1d(output_dim // 4)
        
        # Second convolution layer for feature refinement
        self.conv2 = nn.Conv1d(output_dim, output_dim, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(output_dim)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize convolutional weights using He initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Forward pass of CNN branch.
        
        Args:
            x: Input tensor [batch_size, seq_len, input_dim]
            
        Returns:
            CNN features [batch_size, seq_len, output_dim]
        """
        # Transpose for conv1d: [batch_size, input_dim, seq_len]
        x = x.transpose(1, 2)
        
        # Multi-scale convolutions
        conv_small = F.relu(self.bn1_small(self.conv1_small(x)))
        conv_medium = F.relu(self.bn1_medium(self.conv1_medium(x)))
        conv_large = F.relu(self.bn1_large(self.conv1_large(x)))
        conv_xlarge = F.relu(self.bn1_xlarge(self.conv1_xlarge(x)))
        
        # Concatenate multi-scale features
        multi_scale_features = torch.cat([conv_small, conv_medium, conv_large, conv_xlarge], dim=1)
        
        # Second convolution for feature refinement
        refined_features = F.relu(self.bn2(self.conv2(multi_scale_features)))
        refined_features = self.dropout(refined_features)
        
        # Transpose back: [batch_size, seq_len, output_dim]
        return refined_features.transpose(1, 2)


class TemporalConvolutionalNetwork(nn.Module):
    """
    Temporal Convolutional Network (TCN) for capturing long-range dependencies
    with causal convolutions and dilations.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=4, kernel_size=3, dropout=0.1):
        super(TemporalConvolutionalNetwork, self).__init__()
        
        self.num_layers = num_layers
        
        # TCN layers with increasing dilation
        self.tcn_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        
        # First layer: input_dim -> hidden_dim
        self.tcn_layers.append(
            nn.Conv1d(input_dim, hidden_dim, kernel_size, padding=(kernel_size-1), dilation=1)
        )
        self.norm_layers.append(nn.BatchNorm1d(hidden_dim))
        self.dropout_layers.append(nn.Dropout(dropout))
        
        # Middle layers: hidden_dim -> hidden_dim with increasing dilation
        for i in range(1, num_layers - 1):
            dilation = 2 ** i
            padding = (kernel_size - 1) * dilation
            self.tcn_layers.append(
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding=padding, dilation=dilation)
            )
            self.norm_layers.append(nn.BatchNorm1d(hidden_dim))
            self.dropout_layers.append(nn.Dropout(dropout))
        
        # Final layer: hidden_dim -> output_dim
        dilation = 2 ** (num_layers - 1)
        padding = (kernel_size - 1) * dilation
        self.tcn_layers.append(
            nn.Conv1d(hidden_dim, output_dim, kernel_size, padding=padding, dilation=dilation)
        )
        self.norm_layers.append(nn.BatchNorm1d(output_dim))
        self.dropout_layers.append(nn.Dropout(dropout))
        
        # Residual connections
        self.residual_conv = nn.Conv1d(input_dim, output_dim, 1) if input_dim != output_dim else None
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using He initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Forward pass of TCN.
        
        Args:
            x: Input tensor [batch_size, seq_len, input_dim]
            
        Returns:
            TCN features [batch_size, seq_len, output_dim]
        """
        # Transpose for conv1d: [batch_size, input_dim, seq_len]
        x = x.transpose(1, 2)
        residual = x
        
        # Apply TCN layers
        for i, (tcn_layer, norm_layer, dropout_layer) in enumerate(
            zip(self.tcn_layers, self.norm_layers, self.dropout_layers)
        ):
            out = tcn_layer(x)
            
            # Causal masking: remove future information
            if out.size(2) > x.size(2):
                out = out[:, :, :x.size(2)]
            
            out = F.relu(norm_layer(out))
            out = dropout_layer(out)
            x = out
        
        # Add residual connection
        if self.residual_conv is not None:
            residual = self.residual_conv(residual)
        if residual.size() == x.size():
            x = x + residual
        
        # Transpose back: [batch_size, seq_len, output_dim]
        return x.transpose(1, 2)


class MultiScaleAttention(nn.Module):
    """
    Multi-scale attention mechanism to focus on different temporal scales.
    """
    def __init__(self, d_model, num_scales=3):
        super(MultiScaleAttention, self).__init__()
        
        self.num_scales = num_scales
        self.d_model = d_model
        
        # Attention layers for different scales
        self.scale_attentions = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model // 4),
                nn.ReLU(),
                nn.Linear(d_model // 4, 1)
            ) for _ in range(num_scales)
        ])
        
        # Scale fusion layer
        self.fusion_layer = nn.Linear(d_model * num_scales, d_model)
        
    def forward(self, x):
        """
        Apply multi-scale attention.
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            
        Returns:
            Attended features [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, _ = x.size()
        scale_outputs = []
        
        for i, attention in enumerate(self.scale_attentions):
            # Different pooling scales
            scale = 2 ** i + 1
            
            # Apply average pooling to create different temporal scales
            if scale > 1:
                pooled = F.avg_pool1d(
                    x.transpose(1, 2), 
                    kernel_size=scale, 
                    stride=1, 
                    padding=scale // 2
                )
                pooled = pooled.transpose(1, 2)
            else:
                pooled = x
            
            # Resize to original sequence length if needed
            if pooled.size(1) != seq_len:
                pooled = F.interpolate(
                    pooled.transpose(1, 2), 
                    size=seq_len, 
                    mode='linear', 
                    align_corners=False
                ).transpose(1, 2)
            
            # Apply attention
            attn_scores = attention(pooled)
            attn_weights = F.softmax(attn_scores, dim=1)
            attended = pooled * attn_weights
            
            scale_outputs.append(attended)
        
        # Concatenate and fuse scale outputs
        multi_scale_features = torch.cat(scale_outputs, dim=-1)
        fused_features = self.fusion_layer(multi_scale_features)
        
        return fused_features


class FeatureFusion(nn.Module):
    """
    Feature fusion module to combine CNN, TCN, and Transformer features.
    """
    def __init__(self, feature_dim, dropout=0.1):
        super(FeatureFusion, self).__init__()
        
        # Attention weights for different feature types
        self.cnn_attention = nn.Linear(feature_dim, 1)
        self.tcn_attention = nn.Linear(feature_dim, 1)
        self.transformer_attention = nn.Linear(feature_dim, 1)
        
        # Feature refinement layers
        self.feature_refiner = nn.Sequential(
            nn.Linear(feature_dim * 3, feature_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim * 2, feature_dim),
            nn.LayerNorm(feature_dim)
        )
        
    def forward(self, cnn_features, tcn_features, transformer_features):
        """
        Fuse features from different branches.
        
        Args:
            cnn_features: CNN features [batch_size, seq_len, feature_dim]
            tcn_features: TCN features [batch_size, seq_len, feature_dim]
            transformer_features: Transformer features [batch_size, seq_len, feature_dim]
            
        Returns:
            Fused features [batch_size, seq_len, feature_dim]
        """
        # Calculate attention weights for each feature type
        cnn_attn = torch.sigmoid(self.cnn_attention(cnn_features))
        tcn_attn = torch.sigmoid(self.tcn_attention(tcn_features))
        transformer_attn = torch.sigmoid(self.transformer_attention(transformer_features))
        
        # Apply attention weights
        weighted_cnn = cnn_features * cnn_attn
        weighted_tcn = tcn_features * tcn_attn
        weighted_transformer = transformer_features * transformer_attn
        
        # Concatenate features
        concatenated = torch.cat([weighted_cnn, weighted_tcn, weighted_transformer], dim=-1)
        
        # Refine fused features
        fused_features = self.feature_refiner(concatenated)
        
        return fused_features


class HybridTransformerModel(nn.Module):
    """
    Hybrid Transformer model combining CNN, TCN, and Transformer architectures
    for enhanced anomaly detection in energy time series data.
    
    This model captures:
    1. Local patterns and sharp changes (CNN)
    2. Long-range temporal dependencies (TCN)
    3. Global attention patterns (Transformer)
    """
    
    def __init__(
        self,
        input_dim: int,
        d_model: int = 128,
        nhead: int = 8,
        num_transformer_layers: int = 2,
        num_tcn_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        num_classes: int = 2,
        activation: str = 'gelu'
    ):
        """
        Initialize the hybrid transformer model.
        
        Args:
            input_dim: Dimension of input features
            d_model: Dimension of the model
            nhead: Number of heads in multi-head attention
            num_transformer_layers: Number of transformer layers
            num_tcn_layers: Number of TCN layers
            dim_feedforward: Dimension of the feedforward network
            dropout: Dropout probability
            num_classes: Number of output classes
            activation: Activation function for transformer layers
        """
        super(HybridTransformerModel, self).__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        
        logger.info(f"Initializing HybridTransformerModel with d_model={d_model}, input_dim={input_dim}")
        
        # Input projection layer
        self.input_projection = nn.Linear(input_dim, d_model)
        nn.init.xavier_uniform_(self.input_projection.weight)
        nn.init.zeros_(self.input_projection.bias)
        
        # CNN branch for local feature extraction
        self.cnn_branch = LocalFeatureCNN(input_dim, d_model, dropout)
        
        # TCN branch for temporal convolution
        self.tcn_branch = TemporalConvolutionalNetwork(
            input_dim, d_model, d_model, num_tcn_layers, dropout=dropout
        )
        
        # Positional encoding for transformer
        self.pe = PositionalEncoding(d_model)
        
        # Transformer encoder branch
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
            num_layers=num_transformer_layers
        )
        
        # Multi-scale attention
        self.multi_scale_attention = MultiScaleAttention(d_model)
        
        # Feature fusion module
        self.feature_fusion = FeatureFusion(d_model, dropout)
        
        # Final classification layers
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 4, num_classes)
        )
        
        # Auxiliary classifiers for each branch (for multi-task learning)
        self.cnn_classifier = nn.Linear(d_model, num_classes)
        self.tcn_classifier = nn.Linear(d_model, num_classes)
        self.transformer_classifier = nn.Linear(d_model, num_classes)
        
        logger.info("HybridTransformerModel initialized successfully")
        
    def forward(
        self, 
        src: torch.Tensor, 
        src_mask: Optional[torch.Tensor] = None,
        return_auxiliary: bool = False
    ) -> Tuple[torch.Tensor, ...]:
        """
        Forward pass of the hybrid model.
        
        Args:
            src: Input tensor [batch_size, seq_len, input_dim]
            src_mask: Optional mask for src sequence
            return_auxiliary: Whether to return auxiliary predictions
            
        Returns:
            If return_auxiliary is False:
                - main_logits: Main classification logits [batch_size, seq_len, num_classes]
            If return_auxiliary is True:
                - main_logits: Main classification logits [batch_size, seq_len, num_classes]
                - cnn_logits: CNN branch logits [batch_size, seq_len, num_classes]
                - tcn_logits: TCN branch logits [batch_size, seq_len, num_classes]
                - transformer_logits: Transformer branch logits [batch_size, seq_len, num_classes]
        """
        batch_size, seq_len, _ = src.size()
        
        # CNN branch: Extract local patterns and anomalous jumps
        cnn_features = self.cnn_branch(src)
        
        # TCN branch: Capture long-range temporal dependencies
        tcn_features = self.tcn_branch(src)
        
        # Transformer branch: Global attention patterns
        transformer_input = self.input_projection(src)
        transformer_input = self.pe(transformer_input)
        transformer_features = self.transformer_encoder(transformer_input, src_mask)
        
        # Apply multi-scale attention to transformer features
        transformer_features = self.multi_scale_attention(transformer_features)
        
        # Fuse features from all branches
        fused_features = self.feature_fusion(cnn_features, tcn_features, transformer_features)
        
        # Main classification
        main_logits = self.classifier(fused_features)
        
        if return_auxiliary:
            # Auxiliary predictions for multi-task learning
            cnn_logits = self.cnn_classifier(cnn_features)
            tcn_logits = self.tcn_classifier(tcn_features)
            transformer_logits = self.transformer_classifier(transformer_features)
            
            return main_logits, cnn_logits, tcn_logits, transformer_logits
        
        return main_logits
    
    def get_attention_weights(self, src: torch.Tensor) -> dict:
        """
        Extract attention weights from different components for visualization.
        
        Args:
            src: Input tensor [batch_size, seq_len, input_dim]
            
        Returns:
            Dictionary containing attention weights from different components
        """
        with torch.no_grad():
            # Get transformer attention weights
            transformer_input = self.input_projection(src)
            transformer_input = self.pe(transformer_input)
            
            # Store original forward method
            original_forward = self.transformer_encoder.layers[0].self_attn.forward
            attention_weights = []
            
            def hook_fn(module, input, output):
                if len(output) > 1:
                    attention_weights.append(output[1])
            
            # Register hook to capture attention weights
            hook = self.transformer_encoder.layers[0].self_attn.register_forward_hook(hook_fn)
            
            # Forward pass
            _ = self.transformer_encoder(transformer_input)
            
            # Remove hook
            hook.remove()
            
            return {
                'transformer_attention': attention_weights[0] if attention_weights else None
            }


# Model configuration factory
def create_hybrid_model_config(model_size='base'):
    """
    Create model configuration for different model sizes.
    
    Args:
        model_size: Size of the model ('small', 'base', 'large')
        
    Returns:
        Model configuration dictionary
    """
    configs = {
        'small': {
            'd_model': 64,
            'nhead': 4,
            'num_transformer_layers': 2,
            'num_tcn_layers': 3,
            'dim_feedforward': 256,
            'dropout': 0.1
        },
        'base': {
            'd_model': 128,
            'nhead': 8,
            'num_transformer_layers': 3,
            'num_tcn_layers': 4,
            'dim_feedforward': 512,
            'dropout': 0.1
        },
        'large': {
            'd_model': 256,
            'nhead': 16,
            'num_transformer_layers': 4,
            'num_tcn_layers': 6,
            'dim_feedforward': 1024,
            'dropout': 0.15
        }
    }
    
    return configs.get(model_size, configs['base'])


if __name__ == "__main__":
    # Test the model
    import torch
    
    # Model parameters
    batch_size = 32
    seq_len = 100
    input_dim = 10
    
    # Create model
    config = create_hybrid_model_config('base')
    model = HybridTransformerModel(input_dim=input_dim, **config)
    
    # Test input
    test_input = torch.randn(batch_size, seq_len, input_dim)
    
    # Forward pass
    with torch.no_grad():
        # Main prediction
        main_output = model(test_input)
        print(f"Main output shape: {main_output.shape}")
        
        # Auxiliary predictions
        aux_outputs = model(test_input, return_auxiliary=True)
        print(f"Auxiliary outputs shapes: {[out.shape for out in aux_outputs]}")
        
        # Attention weights
        attention_weights = model.get_attention_weights(test_input)
        print(f"Attention weights available: {list(attention_weights.keys())}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
