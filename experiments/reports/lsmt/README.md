# LSTM Sliding Window Reports

This directory contains reports and statistics generated during the LSTM sliding window preprocessing.

## Overview

The reports provide insights into the sliding window preprocessing results, including:

- Total number of windows generated
- Number of anomaly windows
- Skipped segments
- Window size statistics

## Report Structure

For each data split (train, val, test) and component type (contact, pcb, ring), a separate report is generated with the naming convention:

```
{split}.md
```

For example:
- `train.md` - Statistics for the training set
- `val.md` - Statistics for the validation set
- `test.md` - Statistics for the test set

## How to Generate Reports

Reports are automatically generated when running the sliding window preprocessing script:

```bash
python -m src.preprocessing.energy.lstm.slinding_window
```

## Example Report Content

```markdown
# Sliding Window Statistics - Contact - Train

Generated on: 2024-05-17 10:00:00

## Summary

- Total windows: 50000
- Anomaly windows: 500 (1.00%)
- Skipped segments: 5
- Min window length: 60
- Max window length: 60
``` 