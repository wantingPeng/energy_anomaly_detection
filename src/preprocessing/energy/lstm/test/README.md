# 能源数据划分与分布漂移分析

## 功能概述

本目录包含用于处理能源数据的脚本，特别是`split_and_analyze_data.py`，它实现了以下功能：

1. 读取`Data/row/Energy_Data/Contacting`目录下的所有CSV数据文件
2. 将所有数据按时间戳从早到晚排序
3. 按照75:15:15的比例划分为训练集、验证集和测试集
4. 分析验证集和测试集之间的数据分布漂移情况
5. 生成分布漂移的可视化图表和统计报告

## 使用方法

### 直接运行

```bash
python src/preprocessing/energy/lstm/test/split_and_analyze_data.py
```

### 作为模块导入

```python
from src.preprocessing.energy.lstm.test.split_and_analyze_data import split_and_analyze_contact_data

# 执行完整的数据处理与分析流程
drift_results = split_and_analyze_contact_data()
```

## 主要函数

- `load_csv_data(data_dir)`: 加载指定目录下的所有CSV数据文件
- `preprocess_timestamp(df)`: 处理时间戳字段，确保其为日期时间格式
- `sort_and_split_data(df)`: 按时间戳排序并划分数据集
- `save_split_datasets(train_df, val_df, test_df)`: 保存划分后的数据集
- `analyze_distribution_shift(val_path, test_path)`: 分析验证集和测试集之间的分布漂移
- `split_and_analyze_contact_data()`: 主函数，执行完整的处理流程

## 输出内容

1. 分割后的数据集将保存在 `Data/processed/lstm/split/contact/` 目录下：
   - `train.parquet`
   - `val.parquet`
   - `test.parquet`

2. 分布漂移分析结果将保存在 `experiments/results/` 目录下：
   - `val_test_covariate_shift_analysis.csv`

3. 可视化图表将保存在 `experiments/plots/` 目录下：
   - `raw_data_covariate_shift_top_features.png`
   - `raw_data_ks_statistics_distribution.png`
   - `raw_data_drift_statistics_correlation.png`

4. 日志文件将保存在 `experiments/logs/` 目录下，文件名格式为：
   - `split_and_analyze_data_YYYYMMDD_HHMMSS.log`

## 依赖项

- pandas
- numpy
- matplotlib
- seaborn
- scipy
- tqdm
- pyarrow (用于parquet文件处理)

## 注意事项

- 脚本假设CSV文件具有`TimeStamp`列，并且该列可以转换为日期时间格式
- 分布漂移分析使用Kolmogorov-Smirnov检验（KS检验）来评估分布差异
- 对于大型数据集，会进行采样以提高计算效率 