"""
随机采样数据集划分与分布漂移分析

该脚本用于:
1. 读取Data/row/Energy_Data/Contacting下的CSV数据
2. 随机采样的方式划分为训练集、验证集和测试集（比例为75:15:15）
3. 分析验证集与测试集之间的数据分布漂移情况
"""

import os
import sys
import pandas as pd
import numpy as np
import glob
import time
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# 添加项目根目录到Python路径
sys.path.append(str(Path(__file__).parents[4]))
# 导入日志工具
from src.utils.logger import logger

# 导入分布漂移检测相关函数
from scripts.raw_data_covariate_shift_detection import (
    preprocess_data,
    sample_data_for_analysis, 
    compare_distributions,
    plot_drifted_features,
    visualize_feature_correlation
)

def load_csv_data(data_dir: str):
    """
    加载指定目录下所有CSV数据文件
    
    Args:
        data_dir: 数据目录路径
        
    Returns:
        pd.DataFrame: 合并后的数据
    """
    logger.info(f"开始加载CSV数据从: {data_dir}")
    start_time = time.time()
    
    # 获取目录下所有CSV文件
    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
    if not csv_files:
        logger.error(f"在 {data_dir} 目录下未找到CSV文件")
        raise FileNotFoundError(f"No CSV files found in {data_dir}")
    
    logger.info(f"找到 {len(csv_files)} 个CSV文件")
    
    # 准备列表存储各个文件的数据
    df_list = []
    total_rows = 0
    
    # 读取并合并所有CSV文件
    for file_path in tqdm(csv_files, desc="Loading CSV files"):
        try:
            # 假设CSV文件编码为UTF-8，分隔符为逗号
            df = pd.read_csv(file_path, low_memory=False)
            total_rows += len(df)
            
            # 提取文件名作为来源标识（可选）
            file_name = os.path.basename(file_path)
            df['source_file'] = file_name
            
            df_list.append(df)
            logger.info(f"成功加载 {file_name}, 行数: {len(df)}")
        except Exception as e:
            logger.error(f"加载 {file_path} 时出错: {str(e)}")
    
    # 合并所有数据帧
    if not df_list:
        logger.error("没有成功加载任何数据")
        raise ValueError("No data loaded")
    
    combined_df = pd.concat(df_list, ignore_index=True)
    logger.info(f"数据加载完成, 总行数: {len(combined_df)}, 耗时: {time.time() - start_time:.2f}秒")
    
    return combined_df

def random_split_data(df: pd.DataFrame, train_ratio: float = 0.75, val_ratio: float = 0.15, 
                     random_state: int = 42):
    """
    随机划分数据集
    
    Args:
        df: 数据帧
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        random_state: 随机种子
        
    Returns:
        tuple: (train_df, val_df, test_df)
    """
    logger.info(f"开始随机划分数据集")
    
    # 计算测试集比例
    test_ratio = 1.0 - train_ratio - val_ratio
    
    # 第一次划分：训练集 vs (验证集+测试集)
    train_df, temp_df = train_test_split(
        df, 
        train_size=train_ratio, 
        random_state=random_state
    )
    
    # 第二次划分：验证集 vs 测试集
    # 计算验证集在临时数据集中的比例
    val_ratio_adjusted = val_ratio / (val_ratio + test_ratio)
    
    val_df, test_df = train_test_split(
        temp_df,
        train_size=val_ratio_adjusted,
        random_state=random_state
    )
    
    # 记录划分结果
    n = len(df)
    logger.info(f"数据划分完成:")
    logger.info(f"训练集: {len(train_df)} 行 ({len(train_df)/n*100:.2f}%)")
    logger.info(f"验证集: {len(val_df)} 行 ({len(val_df)/n*100:.2f}%)")
    logger.info(f"测试集: {len(test_df)} 行 ({len(test_df)/n*100:.2f}%)")
    
    # 如果数据集有时间戳，记录各个集合的时间范围
    if 'TimeStamp' in df.columns:
        logger.info("各数据集时间分布情况:")
        logger.info(f"训练集时间范围: {train_df['TimeStamp'].min()} 至 {train_df['TimeStamp'].max()}")
        logger.info(f"验证集时间范围: {val_df['TimeStamp'].min()} 至 {val_df['TimeStamp'].max()}")
        logger.info(f"测试集时间范围: {test_df['TimeStamp'].min()} 至 {test_df['TimeStamp'].max()}")
    
    return train_df, val_df, test_df

def save_split_datasets(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, 
                        output_dir: str = "Data/processed/lsmt/random_split"):
    """
    保存划分后的数据集
    
    Args:
        train_df: 训练集
        val_df: 验证集
        test_df: 测试集
        output_dir: 输出目录
    """
    logger.info(f"保存划分后的数据集到 {output_dir}")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存为parquet格式
    train_path = os.path.join(output_dir, "train.parquet")
    val_path = os.path.join(output_dir, "val.parquet")
    test_path = os.path.join(output_dir, "test.parquet")
    
    train_df.to_parquet(train_path, engine='pyarrow', index=False)
    val_df.to_parquet(val_path, engine='pyarrow', index=False)
    test_df.to_parquet(test_path, engine='pyarrow', index=False)
    
    logger.info(f"训练集已保存至: {train_path}")
    logger.info(f"验证集已保存至: {val_path}")
    logger.info(f"测试集已保存至: {test_path}")
    
    return train_path, val_path, test_path

def analyze_distribution_shift(val_path: str, test_path: str):
    """
    分析验证集和测试集之间的分布漂移
    
    Args:
        val_path: 验证集文件路径
        test_path: 测试集文件路径
        
    Returns:
        pd.DataFrame: 分布漂移分析结果
    """
    logger.info(f"开始分析验证集和测试集之间的分布漂移")
    
    # 加载数据
    logger.info(f"加载验证集: {val_path}")
    val_df = pd.read_parquet(val_path, engine='pyarrow')
    
    logger.info(f"加载测试集: {test_path}")
    test_df = pd.read_parquet(test_path, engine='pyarrow')
    
    # 预处理数据，提取特征
    val_features, test_features, feature_columns = preprocess_data(val_df, test_df)
    
    # 对大数据集进行采样以提高效率
    sampled_val, sampled_test = sample_data_for_analysis(val_features, test_features)
    
    # 比较分布
    logger.info("计算分布漂移统计量...")
    df_diff = compare_distributions(sampled_val, sampled_test)
    
    # 保存结果
    os.makedirs("experiments/results", exist_ok=True)
    result_path = "experiments/results/random_split_covariate_shift_analysis.csv"
    df_diff.to_csv(result_path, index=False)
    logger.info(f"分析结果保存至: {result_path}")
    

    # 结果摘要
    drifted_features = df_diff[df_diff['is_drifted']]
    
    logger.info("\n" + "="*80)
    logger.info("随机划分后的验证集与测试集分布漂移分析结果")
    logger.info("="*80)
    logger.info(f"总特征数: {len(df_diff)}")
    logger.info(f"漂移特征数: {len(drifted_features)} (p < 0.05)")
    logger.info(f"漂移比例: {len(drifted_features) / len(df_diff) * 100:.2f}%")
    logger.info(f"平均KS统计量: {df_diff['ks_statistic'].mean():.4f}")
    logger.info(f"最大KS统计量: {df_diff['ks_statistic'].max():.4f}")
    
    if len(drifted_features) > 0:
        logger.info(f"\n前10个漂移最严重的特征:")
        for idx, row in drifted_features.head(10).iterrows():
            logger.info(f"特征: {row['feature']}, KS={row['ks_statistic']:.4f}, p={row['p_value']:.2e}")
    else:
        logger.info("\n未检测到显著的分布漂移")
    
    return df_diff

def compare_with_time_split(random_split_result_path: str, time_split_result_path: str = "experiments/results/val_test_covariate_shift_analysis.csv"):
    """
    比较随机划分和时间顺序划分的分布漂移结果差异
    
    Args:
        random_split_result_path: 随机划分的分析结果路径
        time_split_result_path: 时间顺序划分的分析结果路径
        
    Returns:
        pd.DataFrame: 比较结果
    """
    logger.info("比较随机划分与时间顺序划分的分布漂移差异")
    
    # 检查时间顺序划分结果是否存在
    if not os.path.exists(time_split_result_path):
        logger.warning(f"时间顺序划分结果文件不存在: {time_split_result_path}")
        logger.warning("跳过比较分析")
        return None
    
    # 加载两种划分方法的结果
    random_results = pd.read_csv(random_split_result_path)
    time_results = pd.read_csv(time_split_result_path)
    
    # 确保两个结果有相同的特征
    common_features = set(random_results['feature']).intersection(set(time_results['feature']))
    logger.info(f"两种划分方法共有 {len(common_features)} 个共同特征")
    
    # 筛选共同特征并合并结果
    random_results = random_results[random_results['feature'].isin(common_features)]
    time_results = time_results[time_results['feature'].isin(common_features)]
    
    # 合并结果
    comparison = pd.merge(
        random_results[['feature', 'ks_statistic', 'p_value', 'is_drifted']],
        time_results[['feature', 'ks_statistic', 'p_value', 'is_drifted']],
        on='feature',
        suffixes=('_random', '_time')
    )
    
    # 计算差异
    comparison['ks_diff'] = comparison['ks_statistic_random'] - comparison['ks_statistic_time']
    comparison['drift_difference'] = comparison['is_drifted_random'] != comparison['is_drifted_time']
    
    # 统计分析
    random_drift_count = comparison['is_drifted_random'].sum()
    time_drift_count = comparison['is_drifted_time'].sum()
    both_drift_count = (comparison['is_drifted_random'] & comparison['is_drifted_time']).sum()
    only_random_drift = (comparison['is_drifted_random'] & ~comparison['is_drifted_time']).sum()
    only_time_drift = (~comparison['is_drifted_random'] & comparison['is_drifted_time']).sum()
    
    logger.info("\n" + "="*80)
    logger.info("随机划分与时间顺序划分的分布漂移比较")
    logger.info("="*80)
    logger.info(f"共同特征数: {len(comparison)}")
    logger.info(f"随机划分中检测到的漂移特征数: {random_drift_count} ({random_drift_count/len(comparison)*100:.2f}%)")
    logger.info(f"时间顺序划分中检测到的漂移特征数: {time_drift_count} ({time_drift_count/len(comparison)*100:.2f}%)")
    logger.info(f"两种方法都检测到漂移的特征数: {both_drift_count} ({both_drift_count/len(comparison)*100:.2f}%)")
    logger.info(f"仅在随机划分中检测到漂移的特征数: {only_random_drift} ({only_random_drift/len(comparison)*100:.2f}%)")
    logger.info(f"仅在时间顺序划分中检测到漂移的特征数: {only_time_drift} ({only_time_drift/len(comparison)*100:.2f}%)")
    logger.info(f"平均KS统计量差异 (随机 - 时间): {comparison['ks_diff'].mean():.4f}")
    
    # 保存比较结果
    comparison_path = "experiments/results/random_vs_time_split_comparison.csv"
    comparison.to_csv(comparison_path, index=False)
    logger.info(f"比较结果保存至: {comparison_path}")
    
    # 可视化KS统计量差异
    plt.figure(figsize=(12, 6))
    plt.scatter(comparison['ks_statistic_time'], comparison['ks_statistic_random'], alpha=0.6)
    plt.plot([0, 1], [0, 1], 'r--')  # 对角线
    plt.xlabel('KS Statistic (Time-based Split)')
    plt.ylabel('KS Statistic (Random Split)')
    plt.title('Comparison of KS Statistics Between Random and Time-based Splits')
    plt.grid(True, alpha=0.3)
    
    # 添加特征标签（仅添加前10个差异最大的特征）
    top_diff_features = comparison.sort_values('ks_diff', key=lambda x: abs(x), ascending=False).head(10)
    for _, row in top_diff_features.iterrows():
        plt.annotate(row['feature'], 
                    (row['ks_statistic_time'], row['ks_statistic_random']),
                    textcoords="offset points",
                    xytext=(0, 10),
                    ha='center')
    
    comparison_plot_path = "experiments/plots/random_vs_time_split_ks_comparison.png"
    plt.savefig(comparison_plot_path, dpi=300, bbox_inches='tight')
    logger.info(f"比较图保存至: {comparison_plot_path}")
    
    return comparison

def random_split_and_analyze_data():
    """
    主函数：加载数据，随机划分数据集，并分析分布漂移
    """
    logger.info("="*80)
    logger.info("开始随机划分数据与分布漂移分析")
    logger.info("="*80)
    
    # 数据目录
    data_dir = "Data/row/Energy_Data/Contacting"
    output_dir = "Data/processed/lsmt/random_split"
    
    try:
        # 1. 加载数据
        logger.info("第一步：加载CSV数据")
        df = load_csv_data(data_dir)
        
        # 2. 处理时间戳
        logger.info("第二步：处理时间戳")
        df = preprocess_timestamp(df)
        
        # 3. 随机划分数据集
        logger.info("第三步：随机划分数据集")
        train_df, val_df, test_df = random_split_data(df)
        
        # 4. 保存数据集
        logger.info("第四步：保存数据集")
        train_path, val_path, test_path = save_split_datasets(train_df, val_df, test_df, output_dir)
        
        # 5. 分析分布漂移
        logger.info("第五步：分析验证集和测试集之间的分布漂移")
        drift_results = analyze_distribution_shift(val_path, test_path)
        
        # 6. 比较与时间顺序划分的差异（可选）
        logger.info("第六步：比较与时间顺序划分的差异")
        random_result_path = "experiments/results/random_split_covariate_shift_analysis.csv"
        comparison = compare_with_time_split(random_result_path)
        
        logger.info("="*80)
        logger.info("随机划分数据与分布漂移分析完成")
        logger.info("="*80)
        
        return drift_results
    
    except Exception as e:
        logger.error(f"处理过程中出错: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    random_split_and_analyze_data() 