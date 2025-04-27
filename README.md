# Task Specification: Anomaly Detection Based on Energy Data

The database is based on one year of energy data from a production cell, as well as a list of anomalies and their corresponding periods.

---
remember:
-This project is running on a remote server connected via ssh.
-dataset is pretty large, about 32 G.
-Use Dask to process big data in the data preprocessing stage.
-Use PyTorch DataLoader to efficiently send data to the GPU in the model training stage
-The cleaned data is saved as a compressed Parquet file (.parquet)
## Rough Task Overview

### Literature Research
- Investigate the current state of research on anomaly and fault detection in energy data from production systems using AI methods.
- Analyze existing approaches:
  - Statistical methods
  - Machine Learning (e.g., Random Forests, Support Vector Machines)
  - Deep Learning (e.g., LSTMs, Autoencoders, XGBoost, etc.)
- Identify specific challenges in analyzing energy data from production cells.

### Data Description and Preprocessing
- **Data Analysis**: Analyze the provided energy data (e.g., time series of power consumption, performance, voltage values) regarding structure, scope, and quality.
- **Preprocessing Workflow**:
  - **Cleaning**: Handle missing values, noise, and inconsistent data points caused by sensor errors or operational interruptions.
  - **Normalization/Scaling**: Adjust data to a consistent value range to account for differences in measurement units.
  - **Feature Engineering**: Extract relevant features indicative of fault states (e.g., consumption peaks, frequency changes, deviations from normal operation).
  - **Data Preparation**: Transform the data into a format suitable for AI models (e.g., time windows for time series analysis).
- **Correlation Analysis**:
  - Document preprocessing steps and their impact on fault state detection.

### Model Development
- Select appropriate AI approaches for fault detection:
  - **Supervised learning** using labeled fault data
  - **Unsupervised learning** for detecting unknown anomalies
  - **Hybrid approaches** combining both
- Implement at least the following models:
  - A classical machine learning model (e.g., Isolation Forest)
  - A deep learning model (e.g., Autoencoder or LSTM network)
  - XGBoost
- Train the models with the prepared energy data, considering training, validation, and test splits.

### Evaluation and Validation
- Apply metrics to evaluate model performance:
  - Precision
  - Recall
  - F1-Score
  - AUC-ROC (Area Under the ROC Curve)
- Focus on:
  - Correct detection of fault states
  - Minimization of false alarms
- Analyze model robustness against typical disturbances in production cells (e.g., short-term power fluctuations).

### Optimization and Practical Applicability
- Perform hyperparameter tuning to improve model performance.
- Investigate the computational time and scalability of models for use in real-time monitoring systems in production cells.
- Discuss possible approaches for integrating models into existing production control systems (e.g., PLC or SCADA).
