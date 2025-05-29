"""
时间序列数据集划分与分布漂移分析

该脚本用于:
1. 读取Data/row/Energy_Data/Contacting下的CSV数据
2. 按时间从早到晚排序
3. 按75:15:15比例划分为训练集、验证集和测试集
4. 分析验证集与测试集之间的数据分布漂移情况
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

def preprocess_timestamp(df: pd.DataFrame, timestamp_col: str = 'TimeStamp'):
    """
    处理时间戳字段，确保其为日期时间格式
    
    Args:
        df: 数据帧
        timestamp_col: 时间戳列名
        
    Returns:
        pd.DataFrame: 处理后的数据帧
    """
    logger.info(f"处理时间戳字段: {timestamp_col}")
    
    if timestamp_col not in df.columns:
        logger.error(f"数据中不存在 {timestamp_col} 列")
        raise ValueError(f"Column {timestamp_col} not found in the dataframe")
    
    # 尝试将时间戳转换为datetime格式
    try:
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])
        logger.info(f"时间戳转换成功")
    except Exception as e:
        logger.error(f"时间戳转换失败: {str(e)}")
        raise
    
    return df

def sort_and_split_data(df: pd.DataFrame, timestamp_col: str = 'TimeStamp', 
                        train_ratio: float = 0.75, val_ratio: float = 0.15):
    """
    按时间戳排序并划分数据集
    
    Args:
        df: 数据帧
        timestamp_col: 时间戳列名
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        
    Returns:
        tuple: (train_df, val_df, test_df)
    """
    logger.info(f"按时间戳 {timestamp_col} 排序并划分数据集")
    
    # 按时间戳排序
    df = df.sort_values(by=timestamp_col)
    logger.info(f"数据已按时间从早到晚排序")
    
    # 计算划分点
    n = len(df)
    train_idx = int(n * train_ratio)
    val_idx = train_idx + int(n * val_ratio)
    
    # 划分数据集
    train_df = df.iloc[:train_idx].copy()
    val_df = df.iloc[train_idx:val_idx].copy()
    test_df = df.iloc[val_idx:].copy()
    
    # 记录划分结果
    logger.info(f"数据划分完成:")
    logger.info(f"训练集: {len(train_df)} 行 ({len(train_df)/n*100:.2f}%)")
    logger.info(f"验证集: {len(val_df)} 行 ({len(val_df)/n*100:.2f}%)")
    logger.info(f"测试集: {len(test_df)} 行 ({len(test_df)/n*100:.2f}%)")
    
    # 记录时间范围
    logger.info(f"训练集时间范围: {train_df[timestamp_col].min()} 至 {train_df[timestamp_col].max()}")
    logger.info(f"验证集时间范围: {val_df[timestamp_col].min()} 至 {val_df[timestamp_col].max()}")
    logger.info(f"测试集时间范围: {test_df[timestamp_col].min()} 至 {test_df[timestamp_col].max()}")
    
    return train_df, val_df, test_df

def save_split_datasets(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, 
                       output_dir: str = "Data/processed/lsmt/test_data_drift"):
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
    result_path = "experiments/results/val_test_covariate_shift_analysis.csv"
    df_diff.to_csv(result_path, index=False)
    logger.info(f"分析结果保存至: {result_path}")
    
    # 可视化
    logger.info("生成可视化图表...")
    plot_drifted_features(df_diff, sampled_val, sampled_test, top_n=5, 
                         save_dir="experiments/plots")
    visualize_feature_correlation(df_diff, save_dir="experiments/plots")
    
    # 结果摘要
    drifted_features = df_diff[df_diff['is_drifted']]
    
    logger.info("\n" + "="*80)
    logger.info("验证集与测试集分布漂移分析结果")
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

def split_and_analyze_contact_data():
    """
    主函数：加载接触区数据，按时间排序，划分数据集，并分析分布漂移
    """
    logger.info("="*80)
    logger.info("开始接触区数据处理与分布漂移分析")
    logger.info("="*80)
    
    # 数据目录
    data_dir = "Data/row/Energy_Data/Contacting"
    output_dir = "Data/processed/lsmt/test_data_drift"
    
    try:
        # 1. 加载数据
        logger.info("第一步：加载CSV数据")
        df = load_csv_data(data_dir)
        
        # 2. 处理时间戳
        logger.info("第二步：处理时间戳")
        df = preprocess_timestamp(df)
        
        # 3. 按时间排序和划分数据集
        logger.info("第三步：按时间排序和划分数据集")
        train_df, val_df, test_df = sort_and_split_data(df)
        
        # 4. 保存数据集
        logger.info("第四步：保存数据集")
        train_path, val_path, test_path = save_split_datasets(train_df, val_df, test_df, output_dir)
        


        
        # 5. 分析分布漂移
        logger.info("第五步：分析验证集和测试集之间的分布漂移")
        drift_results = analyze_distribution_shift(val_path, val_path)
        
        logger.info("="*80)
        logger.info("接触区数据处理与分布漂移分析完成")
        logger.info("="*80)
        
        return drift_results
    
    except Exception as e:
        logger.error(f"处理过程中出错: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    split_and_analyze_contact_data() 