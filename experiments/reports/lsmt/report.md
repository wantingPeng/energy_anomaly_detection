# LSTM Sliding Window Preprocessing Report

## Summary
- Total number of windows: {total_windows}
- Number of anomalous windows: {anomalous_windows}
- Number of skipped segments: {skipped_segments}

## Details
- Window size: 60 seconds
- Step size: 10 seconds
- Anomaly threshold: 30%

This report provides an overview of the sliding window preprocessing for LSTM model training. The data was processed to ensure all windows are continuous, of equal length, and free of NaN values. Anomalous windows were identified based on the overlap with predefined anomaly periods. 