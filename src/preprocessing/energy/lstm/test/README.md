# 能源数据划分与分布漂移分析

## 功能概述

本目录包含用于处理能源数据的脚本：

### 1. split_and_analyze_data.py

按时间顺序划分数据并分析分布漂移，实现以下功能：

1. 读取`Data/row/Energy_Data/Contacting`目录下的所有CSV数据文件
2. 将所有数据按时间戳从早到晚排序
3. 按照75:15:15的比例划分为训练集、验证集和测试集
4. 分析验证集和测试集之间的数据分布漂移情况
5. 生成分布漂移的可视化图表和统计报告

### 2. random_split_and_analyze.py

随机划分数据并分析分布漂移，实现以下功能：

1. 读取`Data/row/Energy_Data/Contacting`目录下的所有CSV数据文件
2. 使用随机采样方式按照75:15:15的比例划分为训练集、验证集和测试集
3. 分析验证集和测试集之间的数据分布漂移情况
4. 生成分布漂移的可视化图表和统计报告
5. 对比随机划分与时间顺序划分的分布漂移差异（如果时间顺序划分结果存在）

## 使用方法

### 按时间顺序划分

```bash
python src/preprocessing/energy/lstm/test/split_and_analyze_data.py
```

### 随机划分

```bash
python src/preprocessing/energy/lstm/test/random_split_and_analyze.py
```

### 作为模块导入

```python
# 使用时间顺序划分
from src.preprocessing.energy.lstm.test.split_and_analyze_data import split_and_analyze_contact_data
drift_results = split_and_analyze_contact_data()

# 使用随机划分
from src.preprocessing.energy.lstm.test.random_split_and_analyze import random_split_and_analyze_data
drift_results = random_split_and_analyze_data()
```

## 主要函数

### 时间顺序划分脚本

- `load_csv_data(data_dir)`: 加载指定目录下的所有CSV数据文件
- `preprocess_timestamp(df)`: 处理时间戳字段，确保其为日期时间格式
- `sort_and_split_data(df)`: 按时间戳排序并划分数据集
- `save_split_datasets(train_df, val_df, test_df)`: 保存划分后的数据集
- `analyze_distribution_shift(val_path, test_path)`: 分析验证集和测试集之间的分布漂移
- `split_and_analyze_contact_data()`: 主函数，执行完整的处理流程

### 随机划分脚本

- `load_csv_data(data_dir)`: 加载指定目录下的所有CSV数据文件
- `preprocess_timestamp(df)`: 处理时间戳字段，确保其为日期时间格式
- `random_split_data(df)`: 随机划分数据集
- `save_split_datasets(train_df, val_df, test_df)`: 保存划分后的数据集
- `analyze_distribution_shift(val_path, test_path)`: 分析验证集和测试集之间的分布漂移
- `compare_with_time_split(random_split_result_path, time_split_result_path)`: 比较随机划分和时间顺序划分的分布漂移差异
- `random_split_and_analyze_data()`: 主函数，执行完整的处理流程

## 输出内容

### 时间顺序划分

1. 分割后的数据集将保存在 `Data/processed/lsmt/test_data_drift/` 目录下：
   - `train.parquet`
   - `val.parquet`
   - `test.parquet`

2. 分布漂移分析结果将保存在 `experiments/results/` 目录下：
   - `val_test_covariate_shift_analysis.csv`

### 随机划分

1. 分割后的数据集将保存在 `Data/processed/lsmt/random_split/` 目录下：
   - `train.parquet`
   - `val.parquet`
   - `test.parquet`

2. 分布漂移分析结果将保存在 `experiments/results/` 目录下：
   - `random_split_covariate_shift_analysis.csv`
   - `random_vs_time_split_comparison.csv` (如果时间顺序划分结果存在)

3. 可视化图表将保存在 `experiments/plots/` 目录下：
   - 分布漂移相关图表
   - `random_vs_time_split_ks_comparison.png` (随机划分与时间顺序划分的对比图)

4. 日志文件将保存在 `experiments/logs/` 目录下，文件名格式为：
   - `split_and_analyze_data_YYYYMMDD_HHMMSS.log`
   - `random_split_and_analyze_YYYYMMDD_HHMMSS.log`

## 依赖项

- pandas
- numpy
- matplotlib
- seaborn
- scipy
- tqdm
- pyarrow (用于parquet文件处理)
- scikit-learn (用于随机划分)

## 注意事项

- 脚本假设CSV文件具有`TimeStamp`列，并且该列可以转换为日期时间格式
- 分布漂移分析使用Kolmogorov-Smirnov检验（KS检验）来评估分布差异
- 对于大型数据集，会进行采样以提高计算效率
- 随机划分脚本使用固定的随机种子(42)，以确保结果可重现 