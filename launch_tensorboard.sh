#!/bin/bash

# Script to launch TensorBoard for transformer experiment
# Created on October 2, 2025

# Exit on error
set -e

# Path to log directory
LOG_DIR="experiments/row_energyData_subsample_Transform/tensorboard/transformer_20251001_051939"

# Check if the log directory exists
if [ ! -d "$LOG_DIR" ]; then
    echo "Error: Log directory '$LOG_DIR' does not exist."
    exit 1
fi

# Check if virtual environment exists and activate it
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
elif [ -d "env" ]; then
    echo "Activating virtual environment..."
    source env/bin/activate
else
    echo "Warning: No virtual environment found (venv or env). Using system Python."
fi

# Check if tensorboard is installed
if ! command -v tensorboard &> /dev/null; then
    echo "Error: TensorBoard is not installed. Please install it using:"
    echo "pip install tensorboard"
    exit 1
fi

# Launch TensorBoard
echo "Launching TensorBoard for directory: $LOG_DIR"
tensorboard --logdir="$LOG_DIR" --port=6006 --host=0.0.0.0

# Note: --host=0.0.0.0 allows access from other devices on the network
# Port 6006 is the default TensorBoard port
