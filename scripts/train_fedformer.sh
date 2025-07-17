#!/bin/bash

# FEDformer Training Script for Energy Anomaly Detection
# Based on periodic analysis results

echo "🚀 Starting FEDformer Training for Energy Anomaly Detection"
echo "============================================================="

# Set default parameters
DATA_DIR="Data/row_energyData_subsample_Transform/labeled"
COMPONENT="contact"
CONFIG="configs/fedformer_config.yaml"
DEVICE="auto"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --data_dir)
            DATA_DIR="$2"
            shift 2
            ;;
        --component)
            COMPONENT="$2"
            shift 2
            ;;
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --resume)
            RESUME="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --data_dir PATH     Path to data directory (default: $DATA_DIR)"
            echo "  --component NAME    Component type: contact, pcb, ring (default: $COMPONENT)"
            echo "  --config PATH       Path to config file (default: $CONFIG)"
            echo "  --device DEVICE     Device: auto, cuda, cpu (default: $DEVICE)"
            echo "  --resume PATH       Resume from checkpoint"
            echo "  -h, --help          Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Check if data directory exists
if [ ! -d "$DATA_DIR" ]; then
    echo "❌ Error: Data directory '$DATA_DIR' does not exist!"
    echo "Please check the path or run the data preprocessing first."
    exit 1
fi

# Check if config file exists
if [ ! -f "$CONFIG" ]; then
    echo "⚠️  Warning: Config file '$CONFIG' does not exist!"
    echo "Will create default config based on data characteristics."
fi

# Create necessary directories
echo "📁 Creating necessary directories..."
mkdir -p experiments/fedformer
mkdir -p configs
mkdir -p logs

# Display configuration
echo "📋 Training Configuration:"
echo "   Data Directory: $DATA_DIR"
echo "   Component: $COMPONENT" 
echo "   Config File: $CONFIG"
echo "   Device: $DEVICE"
if [ ! -z "$RESUME" ]; then
    echo "   Resume From: $RESUME"
fi

# Check GPU availability
if command -v nvidia-smi &> /dev/null; then
    echo "🖥️  GPU Information:"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits | head -1
else
    echo "💻 No GPU detected, will use CPU"
fi

# Set Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Prepare training command
TRAIN_CMD="python src/training/fedformer/train_fedformer.py"
TRAIN_CMD="$TRAIN_CMD --data_dir $DATA_DIR"
TRAIN_CMD="$TRAIN_CMD --component $COMPONENT"
TRAIN_CMD="$TRAIN_CMD --config $CONFIG"
TRAIN_CMD="$TRAIN_CMD --device $DEVICE"

if [ ! -z "$RESUME" ]; then
    TRAIN_CMD="$TRAIN_CMD --resume $RESUME"
fi

echo "🎯 Starting training with command:"
echo "   $TRAIN_CMD"
echo ""

# Check if we're in the right directory
if [ ! -f "src/training/fedformer/train_fedformer.py" ]; then
    echo "❌ Error: Training script not found!"
    echo "Please run this script from the project root directory."
    exit 1
fi

# Start training
echo "🏃 Starting FEDformer training..."
echo "============================================================="

# Run with proper error handling
if ! $TRAIN_CMD; then
    echo ""
    echo "❌ Training failed!"
    echo "Please check the error messages above."
    exit 1
fi

echo ""
echo "✅ Training completed successfully!"
echo "Check the experiments/fedformer_* directory for results."

# Show latest experiment directory
LATEST_EXP=$(ls -td experiments/fedformer_* 2>/dev/null | head -1)
if [ ! -z "$LATEST_EXP" ]; then
    echo ""
    echo "📊 Results available in: $LATEST_EXP"
    echo "   - Checkpoints: $LATEST_EXP/checkpoints/"
    echo "   - Logs: $LATEST_EXP/logs/"
    echo "   - Plots: $LATEST_EXP/*.png"
    
    # Show test results if available
    if [ -f "$LATEST_EXP/test_results.yaml" ]; then
        echo ""
        echo "🎯 Test Results Summary:"
        grep -E "(f1|precision|recall|roc_auc)" "$LATEST_EXP/test_results.yaml" | head -10
    fi
fi

echo ""
echo "🔍 To visualize training progress:"
echo "   tensorboard --logdir $LATEST_EXP/logs"
echo ""
echo "🚀 Happy anomaly detecting!" 