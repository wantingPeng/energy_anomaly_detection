# time series Report



## Time Series Analysis

### Time Interval Statistics

- Minimum interval: 0.000 seconds

- Maximum interval: 680077.000 seconds

- Mean interval: 1.060 seconds

- Standard deviation: 144.347 seconds


### Segmentation Statistics

- Total segments: 502

| 指标                  | 数值                                        | 解读 |
| ------------------- | ----------------------------------------- | -- |
| `count = 502`       | ✅ **总段数 502**，相比之前的 17664，大幅减少，碎段被有效清理    |    |
| `mean ≈ 69163 秒`    | ≈ **19 小时/段**，段内非常长，可以容纳大量滑窗样本            |    |
| `min = 1 秒`         | 极端短段，可能是孤立点，可选择丢弃                         |    |
| `25% = 149 秒`       | 有些段还是很短（不到 3 分钟），也可丢弃                     |    |
| `50% = 3020 秒`      | ✅ 中位段长度为 **50 分钟**，**完美覆盖你建议的滑窗（600 秒）**  |    |
| `75% = 76799 秒`     | **超过 21 小时**，说明很多 segment 已经接近整天级别，非常适合训练 |    |
| `max = 1,260,148 秒` | ≈ **14.6 天**！说明你有完整超长连续数据段，很棒             |    |

