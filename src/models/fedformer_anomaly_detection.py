"""
Official FEDformer implementation adapted for Energy Anomaly Detection
Based on ICML 2022 paper, modified for binary classification instead of forecasting
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

# Import official FEDformer layers
from src.layers.Embed import DataEmbedding, DataEmbedding_wo_pos
from src.layers.AutoCorrelation import AutoCorrelation, AutoCorrelationLayer
from src.layers.FourierCorrelation import FourierBlock, FourierCrossAttention
from src.layers.MultiWaveletCorrelation import MultiWaveletCross, MultiWaveletTransform
from src.layers.Autoformer_EncDec import Encoder, Decoder, EncoderLayer, DecoderLayer, my_Layernorm, series_decomp, series_decomp_multi

# Import utility for dynamic object creation  
from types import SimpleNamespace

# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class FEDformerAnomalyDetector(nn.Module):
    """
    FEDformer adapted for anomaly detection with binary classification
    """
    def __init__(self, configs):
        super(FEDformerAnomalyDetector, self).__init__()
        
        # Configuration parameters
        self.version = configs.version
        self.mode_select = configs.mode_select
        self.modes = configs.modes
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len  # For this model, we'll use pred_len as context length
        self.output_attention = configs.output_attention
        self.d_model = configs.d_model
        self.num_classes = getattr(configs, 'num_classes', 2)  # Binary classification

        # Decomposition
        kernel_size = configs.moving_avg
        if isinstance(kernel_size, list):
            self.decomp = series_decomp_multi(kernel_size)
        else:
            self.decomp = series_decomp(kernel_size)

        # Embedding
        self.enc_embedding = DataEmbedding_wo_pos(
            configs.enc_in, configs.d_model, configs.embed, configs.freq, configs.dropout
        )
        
        # For anomaly detection, we'll use encoder-only architecture
        # Build attention mechanism based on version
        if configs.version == 'Wavelets':
            encoder_self_att = MultiWaveletTransform(
                ich=configs.d_model, L=configs.L, base=configs.base
            )
        else:
            encoder_self_att = FourierBlock(
                in_channels=configs.d_model,
                out_channels=configs.d_model,
                seq_len=self.seq_len,
                modes=configs.modes,
                mode_select_method=configs.mode_select
            )

        # Encoder
        enc_modes = int(min(configs.modes, configs.seq_len//2))
        print('enc_modes: {}'.format(enc_modes))

        self.encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(
                        encoder_self_att,
                        configs.d_model, 
                        configs.n_heads
                    ),
                    configs.d_model,
                    configs.d_ff,
                    moving_avg=configs.moving_avg,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=my_Layernorm(configs.d_model)
        )
        
        # Classification heads for anomaly detection
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Multi-scale feature extraction
        self.seasonal_classifier = nn.Sequential(
            nn.Linear(configs.d_model, configs.d_model // 2),
            nn.GELU(),
            nn.Dropout(configs.dropout),
            nn.Linear(configs.d_model // 2, self.num_classes)
        )
        
        self.trend_classifier = nn.Sequential(
            nn.Linear(configs.d_model, configs.d_model // 2),
            nn.GELU(),
            nn.Dropout(configs.dropout),
            nn.Linear(configs.d_model // 2, self.num_classes)
        )
        
        # Combined classifier
        self.combined_classifier = nn.Sequential(
            nn.Linear(configs.d_model * 2, configs.d_model),
            nn.GELU(),
            nn.Dropout(configs.dropout),
            nn.Linear(configs.d_model, self.num_classes)
        )
        
        # Anomaly score regressor (for threshold-based detection)
        self.anomaly_score = nn.Sequential(
            nn.Linear(configs.d_model, configs.d_model // 2),
            nn.GELU(),
            nn.Dropout(configs.dropout),
            nn.Linear(configs.d_model // 2, 1),
            nn.Sigmoid()
        )
        
        # Sequence-level classification (for sequence labeling)
        self.sequence_classifier = nn.Sequential(
            nn.Linear(configs.d_model, configs.d_model // 2),
            nn.GELU(),
            nn.Dropout(configs.dropout),
            nn.Linear(configs.d_model // 2, self.num_classes)
        )

    def forward(self, x_enc, x_mark_enc, return_attention=False):
        """
        Forward pass for anomaly detection
        
        Args:
            x_enc: [batch_size, seq_len, features] - Input sequences
            x_mark_enc: [batch_size, seq_len, time_features] - Time embeddings
            return_attention: Whether to return attention weights
            
        Returns:
            Dictionary containing classification logits and anomaly scores
        """
        # Initial decomposition
        seasonal_init, trend_init = self.decomp(x_enc)
        
        # Embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        
        # Encoder
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        # enc_out shape: [batch_size, seq_len, d_model]
        
        # Decompose the encoded features
        seasonal_features, trend_features = self.decomp(enc_out)
        
        # Global pooling for sequence-level features
        seasonal_global = self.global_pool(seasonal_features.transpose(1, 2)).squeeze(-1)  # [B, d_model]
        trend_global = self.global_pool(trend_features.transpose(1, 2)).squeeze(-1)  # [B, d_model]
        combined_global = self.global_pool(enc_out.transpose(1, 2)).squeeze(-1)  # [B, d_model]
        
        # Classification outputs
        seasonal_logits = self.seasonal_classifier(seasonal_global)  # [B, num_classes]
        trend_logits = self.trend_classifier(trend_global)  # [B, num_classes]
        
        # Combined features for final classification
        combined_features = torch.cat([seasonal_global, trend_global], dim=-1)  # [B, d_model*2]
        combined_logits = self.combined_classifier(combined_features)  # [B, num_classes]
        
        # Anomaly scores
        anomaly_scores = self.anomaly_score(combined_global).squeeze(-1)  # [B]
        
        # Point-wise classification (for each timestep)
        pointwise_logits = self.sequence_classifier(enc_out)  # [B, seq_len, num_classes]
        
        # Results dictionary
        results = {
            'seasonal_logits': seasonal_logits,
            'trend_logits': trend_logits,
            'combined_logits': combined_logits,
            'anomaly_scores': anomaly_scores,
            'pointwise_logits': pointwise_logits,
            'encoded_features': enc_out,
            'seasonal_features': seasonal_features,
            'trend_features': trend_features
        }
        
        if return_attention or self.output_attention:
            results['attention_weights'] = attns
            
        return results


def create_fedformer_anomaly_model(config_dict):
    """
    Factory function to create FEDformer anomaly detection model from config dict
    Args:
        config_dict: Configuration dictionary from YAML file
    Returns:
        model: FEDformerAnomalyDetector instance
    """
    # Convert dict to object for dot notation access
    config = SimpleNamespace(**config_dict)
    
    model = FEDformerAnomalyDetector(config)
    return model
