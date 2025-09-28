# TranAD: Transformer-based Anomaly Detection

This module implements the TranAD model for anomaly detection in energy time series data. TranAD is a transformer-based deep learning architecture specifically designed for multivariate time series anomaly detection.

## Key Features

- **Two-Phase Training**: The model employs a two-phase training approach where the first phase learns normal patterns and the second phase focuses on potential anomalies.
- **Focus Score Mechanism**: Uses reconstruction errors to adjust attention weights, amplifying the model's focus on potential anomalous regions.
- **Adversarial Training**: Implements dual-decoder adversarial training to enhance anomaly detection sensitivity.
- **Specialized Data Loading**: Training is performed only on normal data to learn normal patterns, while validation and testing use both normal and anomalous data.

## Files

- `tranad_model.py`: Implementation of the TranAD model architecture
- `tranad_dataloader.py`: Specialized data loader that filters anomalous data for training
- `train_tranad.py`: Training script with two-phase training and evaluation

## Usage

1. Configure the model in `configs/tranad_config.yaml`
2. Run the training script:

```bash
python src/preprocessing/energy/tranad/train_tranad.py --config configs/tranad_config.yaml
```

## Model Architecture

TranAD consists of:

1. **Encoder**: Transformer-based encoder with self-attention mechanism
2. **Focus Score Module**: Adjusts attention based on reconstruction errors
3. **Main Decoder**: Reconstructs the input sequence
4. **Adversarial Decoder**: Trained to maximize difference from the main decoder

## Training Process

1. **Phase 1**: Initial training on normal data to learn normal patterns
2. **Phase 2**: Focus-adjusted training where the model pays special attention to regions with high reconstruction errors

## References

- "TranAD: Deep Transformer Networks for Anomaly Detection in Multivariate Time Series Data" (https://arxiv.org/abs/2201.07284)

