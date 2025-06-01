# Random Forest Model Validation Results

## Performance Summary

| Rank | Accuracy | Precision | Recall | F1 Score | Threshold | Parameters |
|------|----------|-----------|---------|-----------|-----------|------------|
| 1 | 0.5887 | 0.3009 | 0.7490 | 0.4293 | 0.0911 | n_estimators=300, max_depth=30, min_samples_split=10, class_weight=balanced |

## Detailed Classification Reports

### Model 1

Parameters: {'n_estimators': 300, 'max_depth': 30, 'min_samples_split': 10, 'class_weight': 'balanced'}

```
Optimal Threshold: 0.0911
Threshold Metrics:
  Precision: 0.3009
  Recall: 0.7490
  F1 Score: 0.4293

Final Metrics:
  Accuracy:  0.5887
  Precision: 0.3009
  Recall:    0.7490
  F1 Score:  0.4293
```

