# Random Forest Model Validation Results

## Performance Summary

| Rank | Accuracy | Precision | Recall | F1 Score | Threshold | Parameters |
|------|----------|-----------|---------|-----------|-----------|------------|
| 1 | 0.5857 | 0.0682 | 0.7345 | 0.1247 | 0.1001 | n_estimators=300, max_depth=30, min_samples_split=10, class_weight=balanced |

## Detailed Classification Reports

### Model 1

Parameters: {'n_estimators': 300, 'max_depth': 30, 'min_samples_split': 10, 'class_weight': 'balanced'}

```
Optimal Threshold: 0.1001
Threshold Metrics:
  Precision: 0.0682
  Recall: 0.7345
  F1 Score: 0.1247

Final Metrics:
  Accuracy:  0.5857
  Precision: 0.0682
  Recall:    0.7345
  F1 Score:  0.1247
```

