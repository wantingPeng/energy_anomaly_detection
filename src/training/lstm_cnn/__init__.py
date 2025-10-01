"""
LSTM-CNN hybrid model for energy anomaly detection.

This package provides LSTM-CNN models and training utilities for 
time series anomaly detection in energy data.
"""

from .lstm_cnn_model import LSTMCNN, SimpleLSTMCNN

__all__ = ['LSTMCNN', 'SimpleLSTMCNN']


