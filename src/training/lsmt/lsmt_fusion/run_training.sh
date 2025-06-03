#!/bin/bash
# Example script for running LSTM Late Fusion training

# Make the script exit on any error
set -e

# Create directories if they don't exist
mkdir -p experiments/logs
mkdir -p experiments/tensorboard

# Function to display help message
function show_help {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  --new                Start training from scratch (default)"
    echo "  --continue PATH      Continue training from checkpoint at PATH"
    echo "  --test               Run evaluation on test set after training"
    echo "  --name NAME          Set custom experiment name"
    echo "  --config PATH        Path to config file (default: configs/lstm_late_fusion.yaml)"
    echo "  --help               Show this help message"
}

# Default values
START_NEW=true
CHECKPOINT_PATH=""
EVALUATE_TEST=false
EXPERIMENT_NAME=""
CONFIG_PATH="configs/lstm_late_fusion.yaml"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --new)
            START_NEW=true
            CHECKPOINT_PATH=""
            shift
            ;;
        --continue)
            START_NEW=false
            CHECKPOINT_PATH="$2"
            shift 2
            ;;
        --test)
            EVALUATE_TEST=true
            shift
            ;;
        --name)
            EXPERIMENT_NAME="$2"
            shift 2
            ;;
        --config)
            CONFIG_PATH="$2"
            shift 2
            ;;
        --help)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Build the command
CMD="python -m src.training.lsmt.lsmt_fusion.train_late_fusion --config ${CONFIG_PATH}"

# Add options based on arguments
if [ "$START_NEW" = false ] && [ -n "$CHECKPOINT_PATH" ]; then
    echo "Continuing training from checkpoint: $CHECKPOINT_PATH"
    CMD="${CMD} --load_model ${CHECKPOINT_PATH}"
else
    echo "Starting new training"
fi

if [ "$EVALUATE_TEST" = true ]; then
    echo "Will evaluate on test set after training"
    CMD="${CMD} --evaluate_test"
fi

if [ -n "$EXPERIMENT_NAME" ]; then
    echo "Using experiment name: $EXPERIMENT_NAME"
    CMD="${CMD} --experiment_name ${EXPERIMENT_NAME}"
fi

# Run the command
echo "Executing: $CMD"
eval $CMD

# Launch TensorBoard after training
echo "Training complete. Launching TensorBoard..."
python -m src.training.lsmt.lsmt_fusion.launch_tensorboard 