# Data paths
paths:
  input_dir: 'Data/processed/lsmt_timeFeatures/add_timeFeatures'
  output_dir: 'Data/processed/transform/slidingWindow_noOverlap_0.8_no_stats'
  anomaly_dict: 'Data/interim/Anomaly_Data/anomaly_dict.pkl'

# Sliding window parameters
sliding_window:
  window_size: 1200  # Window size in seconds
  step_size: 1200    # Step size in seconds
  anomaly_step_size: 200
  anomaly_threshold: 0.6  # Minimum overlap ratio to label as anomaly (30%)
# 1200 300     60 10      800 150     600 100       600 600 60 0.8
# Component processing configuration
components:
  #processing_order: ['contact', 'pcb', 'ring']
  processing_order: ['contact']
# # Logging configuration
# logging:
#   log_dir: 'experiments/logs'
#   log_format: '[%(asctime)s] [%(levelname)s] [%(name)s] - %(message)s'
#   date_format: '%Y-%m-%d %H:%M:%S'

# Memory management
memory:
  gc_collect_frequency: 1  # How often to force garbage collection
  log_memory_frequency: 1  # How often to log memory usage
  threshold_gb: 2.0  # Memory threshold in GB for dynamic resource management
  max_jobs: 6  # Maximum number of parallel jobs
  batch_size: 60  # Number of segments to process in one batch


  # Transformer Model Configuration for Energy Anomaly Detection

# Paths
paths:
  data_dir: "Data/processed/transform/slidingWindow_noOverlap_0.8_no_stats/projection_pos_encoding_float16"
  output_dir: "experiments/transformer"

# Data configuration
data:
  component: "contact"  # Options: contact, pcb, ring

# Model configuration
model:
  d_model: 256  # Dimension of the model
  nhead: 8  # Number of heads in multi-head attention
  num_layers: 2  # Number of transformer layers
  dim_feedforward: 512  # Dimension of the feedforward network
  dropout: 0.4  # Dropout probability
  num_classes: 2  # Number of output classes (binary classification)
  activation: "gelu"  # Activation function for transformer layers

# Training configuration
training:
  batch_size: 64
  num_workers: 2
  num_epochs: 150
  learning_rate: 0.0001
  weight_decay: 0.0001
  momentum: 0.9  # Only used for SGD optimizer
  optimizer: "adam"  # Options: adam, adamw, sgd
  
  # Learning rate scheduling
  lr_scheduler: "reduce_on_plateau"  # Options: reduce_on_plateau, cosine_annealing, none
  lr_reduce_factor: 0.5  # Factor by which the learning rate will be reduced (ReduceLROnPlateau)
  lr_reduce_patience: 5  # Number of epochs with no improvement after which learning rate will be reduced
  min_lr: 1.0e-6  # Minimum learning rate for CosineAnnealingLR
  
  # Early stopping
  early_stopping_patience: 15  # Number of epochs with no improvement after which training will be stopped
  early_stopping_min_delta: 0.0001  # Minimum change to qualify as an improvement
  early_stopping_metric: "f1"  # Metric to monitor for early stopping, options: loss, f1
  
  # Class weighting and loss function
  use_class_weights: true  # Whether to use class weights for loss calculation
  use_focal_loss: false  # Whether to use focal loss instead of cross entropy loss
  focal_loss_alpha: 0.5  # Alpha parameter for focal loss
  focal_loss_gamma: 2.0  # Gamma parameter for focal loss

# Hardware configuration
hardware:
  precision: "float32"  # Options: float32, float16 (mixed precision) 