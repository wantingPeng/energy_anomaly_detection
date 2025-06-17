#!/bin/bash

# This script launches TensorBoard to visualize the training logs for the Transformer model.

# Set the default log directory
LOG_DIR="experiments/transformer/bestVal_f1:0.21_600_600_50_0_0.5_entrop/transformer_20250617_104754"

# Check if a specific log directory is provided as an argument
if [ "$#" -gt 0 ]; then
    LOG_DIR="$1"
fi

# Launch TensorBoard

echo "Launching TensorBoard with log directory: $LOG_DIR"
tensorboard --logdir "$LOG_DIR" --host 0.0.0.0 --port 6006 