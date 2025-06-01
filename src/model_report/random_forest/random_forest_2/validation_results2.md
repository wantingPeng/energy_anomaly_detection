# Random Forest Model Validation Results

## Performance Summary

| Rank | Accuracy | Precision | Recall | F1 Score | Threshold | Parameters |
|------|----------|-----------|---------|-----------|-----------|------------|
| 1 | 0.6265 | 0.0668 | 0.6389 | 0.1209 | 0.1208 | n_estimators=300, max_depth=30, min_samples_split=10, class_weight=balanced |

## Detailed Classification Reports

### Model 1

Parameters: {'n_estimators': 300, 'max_depth': 30, 'min_samples_split': 10, 'class_weight': 'balanced'}

```
Optimal Threshold: 0.1208
Threshold Metrics:
  Precision: 0.0668
  Recall: 0.6389
  F1 Score: 0.1209

Final Metrics:
  Accuracy:  0.6265
  Precision: 0.0668
  Recall:    0.6389
  F1 Score:  0.1209
```

