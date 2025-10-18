#!/bin/bash
# Script to run hyperparameter tuning for XGBoost with Optuna
# 
# Usage:
#   ./scripts/run_hyperparameter_tuning.sh [OPTIONS]
#
# Examples:
#   # GPU mode with 100 trials
#   ./scripts/run_hyperparameter_tuning.sh --gpu --trials 100
#
#   # CPU parallel mode with 4 jobs
#   ./scripts/run_hyperparameter_tuning.sh --cpu --trials 100 --jobs 4
#
#   # Quick test with 10 trials
#   ./scripts/run_hyperparameter_tuning.sh --gpu --trials 10 --name quick_test

set -e  # Exit on error

# Default values
USE_GPU=true
N_TRIALS=100
N_JOBS=1
TIMEOUT=""
EXPERIMENT_NAME=""
CONFIG="configs/xgboost_tuning_config.yaml"
USE_POINT_ADJUSTMENT=true

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --gpu)
            USE_GPU=true
            N_JOBS=1
            shift
            ;;
        --cpu)
            USE_GPU=false
            shift
            ;;
        --trials)
            N_TRIALS="$2"
            shift 2
            ;;
        --jobs)
            N_JOBS="$2"
            shift 2
            ;;
        --timeout)
            TIMEOUT="$2"
            shift 2
            ;;
        --name)
            EXPERIMENT_NAME="$2"
            shift 2
            ;;
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --no_point_adjustment)
            USE_POINT_ADJUSTMENT=false
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --gpu              Use GPU acceleration (default)"
            echo "  --cpu              Use CPU mode"
            echo "  --trials N         Number of trials (default: 100)"
            echo "  --jobs N           Number of parallel jobs (default: 1 for GPU, can increase for CPU)"
            echo "  --timeout SECONDS      Timeout in seconds (optional)"
            echo "  --name NAME            Experiment name (optional)"
            echo "  --config PATH          Config file path (default: configs/xgboost_tuning_config.yaml)"
            echo "  --no_point_adjustment  Disable point adjustment evaluation (enabled by default)"
            echo "  --help, -h             Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0 --gpu --trials 100"
            echo "  $0 --cpu --trials 100 --jobs 4"
            echo "  $0 --gpu --trials 50 --timeout 3600 --name my_experiment"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Get the script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

# Change to project root
cd "$PROJECT_ROOT"

# Check if virtual environment is activated
if [[ -z "$VIRTUAL_ENV" ]] && [[ ! -d "venv" ]]; then
    echo "⚠️  Warning: Virtual environment not activated and venv directory not found"
    echo "   Consider activating your virtual environment first:"
    echo "   source venv/bin/activate  # or conda activate your_env"
    echo ""
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Build command
CMD="python src/training/xgboost/hyperparameter_tuning.py"
CMD="$CMD --config $CONFIG"
CMD="$CMD --n_trials $N_TRIALS"
CMD="$CMD --n_jobs $N_JOBS"

if [ "$USE_GPU" = true ]; then
    CMD="$CMD --use_gpu"
    echo "🚀 Running hyperparameter tuning with GPU acceleration"
    echo "   Using device: CUDA"
    echo "   Parallel jobs: $N_JOBS (GPU mode: recommend keeping n_jobs=1)"
else
    CMD="$CMD --no_gpu"
    echo "🚀 Running hyperparameter tuning with CPU"
    echo "   Using device: CPU"
    echo "   Parallel jobs: $N_JOBS"
fi

if [ "$USE_POINT_ADJUSTMENT" = false ]; then
    CMD="$CMD --no_point_adjustment"
    echo "   Point Adjustment: Disabled (using AUPRC as optimization metric)"
else
    echo "   Point Adjustment: Enabled (using Adjusted F1 as optimization metric)"
fi

if [ -n "$TIMEOUT" ]; then
    CMD="$CMD --timeout $TIMEOUT"
    echo "   Timeout: ${TIMEOUT}s"
fi

if [ -n "$EXPERIMENT_NAME" ]; then
    CMD="$CMD --experiment_name $EXPERIMENT_NAME"
    echo "   Experiment name: $EXPERIMENT_NAME"
fi

echo "   Number of trials: $N_TRIALS"
echo "   Config file: $CONFIG"
echo ""
echo "Command: $CMD"
echo ""
echo "Starting in 3 seconds... (Ctrl+C to cancel)"
sleep 3

# Run the command
$CMD

echo ""
echo "✅ Hyperparameter tuning completed!"
echo "   Results saved in: experiments/xgboost_timeseries/hyperparameter_tuning/"

