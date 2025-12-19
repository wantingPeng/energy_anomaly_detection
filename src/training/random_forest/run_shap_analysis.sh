#!/bin/bash

# Example script to run SHAP analysis on trained Random Forest model
# This script demonstrates how to use the SHAP analysis tool with different sampling strategies

# Configuration
#MODEL_PATH="experiments/random_forest/downsampleData_scratch_1minut/optuna_tuning_20251019_161338_contact/best_model"
#MODEL_PATH="experiments/random_forest/downsampleData_scratch_1minut/optuna_tuning_20251019_170714_ring/best_model"
MODEL_PATH="experiments/random_forest/downsampleData_scratch_1minut/optuna_tuning_20251129_151813_pcb/best_model"
CONFIG_PATH="configs/random_forest_tuning_config.yaml"
DATASET="test"  # Options: train, val, test
SAMPLE_SIZE=5000  # Number of samples to use (remove or set to larger value for all samples)
SAMPLING_STRATEGY="balanced"  # Options: random, stratified, balanced, anomaly_focused
RANDOM_STATE=42  # Random seed for reproducibility
TOP_N_DEPENDENCE=5  # Number of top features for dependence plots
N_WATERFALL_EXAMPLES=5  # Number of waterfall plot examples
MAX_DISPLAY=20  # Maximum number of features to display in plots
OUTPUT_DIR="experiments/random_forest/shap_analysis"

echo "========================================="
echo "Running SHAP Analysis"
echo "========================================="
echo "Sampling Strategy: $SAMPLING_STRATEGY"
echo "Sample Size: $SAMPLE_SIZE"
echo "Random State: $RANDOM_STATE"
echo "========================================="

# Run SHAP analysis
python src/training/random_forest/shap_analysis.py \
    --model_path "$MODEL_PATH" \
    --config "$CONFIG_PATH" \
    --dataset "$DATASET" \
    --sample_size "$SAMPLE_SIZE" \
    --sampling_strategy "$SAMPLING_STRATEGY" \
    --random_state "$RANDOM_STATE" \
    --top_n_dependence "$TOP_N_DEPENDENCE" \
    --n_waterfall_examples "$N_WATERFALL_EXAMPLES" \
    --max_display "$MAX_DISPLAY" \
    --output_dir "$OUTPUT_DIR"

echo ""
echo "========================================="
echo "SHAP analysis completed!"
echo "Results saved to: $OUTPUT_DIR"
echo "========================================="
