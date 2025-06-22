#!/bin/bash

# This script launches TensorBoard to visualize the training logs for the Transformer model.

# Set the default log directory
#LOG_DIR="experiments/transformer/bestVal_F10.24_600_window_statistic_focalloss/transformer_20250618_210738 copy"
LOG_DIR="experiments/lstm_late_fusion/tensorboard_inportant/lstm_late_fusion_20250619_204539_600_down20__focloss_0.4_f10.26"


# Check if a specific log directory is provided as an argument
if [ "$#" -gt 0 ]; then
    LOG_DIR="$1"
fi

# Launch TensorBoard

echo "Launching TensorBoard with log directory: $LOG_DIR"
tensorboard --logdir "$LOG_DIR" --host 0.0.0.0 --port 6006 