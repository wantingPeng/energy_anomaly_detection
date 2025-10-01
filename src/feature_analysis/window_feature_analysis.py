#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
窗口统计特征计算、重要性分析和特征筛选

此脚本执行以下步骤：
1. 从parquet文件加载数据
2. 计算窗口统计特征（窗口大小30分钟，步长5分钟）
3. 分析特征重要性，找出前30个重要特征
4. 过滤数据，只保留重要特征和必要的元数据列

示例用法:
python -m src.feature_analysis.window_feature_analysis \
    --input_file Data/filtered_feature/top_features_contact_cleaned_1minut_20250928_172122.parquet \
    --output_dir experiments/window_features \
    --window_size 30 \
    --step_size 5 \
    --top_n 30

直接进行特征重要性分析示例:
python -m src.feature_analysis.window_feature_analysis \
    --load_features_from experiments/window_features/window_features.parquet \
    --output_dir experiments/window_features \
    --top_n 30
"""

import os
import sys
import pandas as pd
import numpy as np
import argparse
from datetime import datetime, timedelta
from pathlib import Path
import time

# Add the project root to the path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from utils.logger import logger
from src.preprocessing.energy.machine_learning.calculate_window_features import calculate_window_features
from tqdm import tqdm


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='计算窗口统计特征并筛选重要特征')
    parser.add_argument('--input_file', type=str,
                        default='Data/filtered_feature/top_features_Ring_cleaned_1minut_20250928_170147.parquet',
                        help='输入数据文件路径')
    parser.add_argument('--output_dir', type=str,
                        default='experiments/statistic_30_window_features_pcb',
                        help='输出目录')
    parser.add_argument('--window_size', type=int,
                        default=30,
                        help='窗口大小（分钟）')
    parser.add_argument('--step_size', type=int,
                        default=5,
                        help='窗口步长（分钟）')
    parser.add_argument('--top_n', type=int,
                        default=60,
                        help='要选择的顶级特征数量')
    parser.add_argument('--load_features_from', type=str,
                        default=None,
                        help='直接从已计算的窗口特征文件加载，跳过特征计算步骤')
    parser.add_argument('--max_rows', type=int,
                        default=None,
                        help='限制读取的数据行数，如：使用前1000行数据')
    return parser.parse_args()


def load_data(file_path, max_rows=None):
    """加载数据并进行基本处理"""
    logger.info(f"从 {file_path} 加载数据")
    try:
        if max_rows:
            logger.info(f"仅加载前 {max_rows} 行数据")
            df = pd.read_parquet(file_path).head(max_rows)
        else:
            df = pd.read_parquet(file_path)
            
        logger.info(f"加载数据形状: {df.shape}")
        
        # 确保TimeStamp列是datetime类型
        if 'TimeStamp' in df.columns:
            if not pd.api.types.is_datetime64_any_dtype(df['TimeStamp']):
                df['TimeStamp'] = pd.to_datetime(df['TimeStamp'])
            
            # 确保数据是按时间排序的
            df = df.sort_values('TimeStamp')
        else:
            logger.error("数据中没有找到TimeStamp列")
            raise ValueError("数据中没有TimeStamp列")
            
        return df
    except Exception as e:
        logger.error(f"加载数据时出错: {e}")
        raise


def calculate_window_features_for_dataframe(df, window_size_minutes, step_size_minutes):
    """
    根据指定的窗口大小和步长计算窗口统计特征
    
    Args:
        df: 包含时间序列数据的DataFrame，必须有TimeStamp列
        window_size_minutes: 窗口大小（分钟）
        step_size_minutes: 窗口步长（分钟）
        
    Returns:
        pd.DataFrame: 包含窗口统计特征的DataFrame
    """
    logger.info(f"计算窗口统计特征：窗口大小 = {window_size_minutes}分钟，步长 = {step_size_minutes}分钟")
    
    # 获取时间范围
    start_time = df['TimeStamp'].min()
    end_time = df['TimeStamp'].max()
    
    # 计算窗口边界
    window_size = timedelta(minutes=window_size_minutes)
    step_size = timedelta(minutes=step_size_minutes)
    
    # 创建窗口起始时间列表
    window_starts = []
    current_time = start_time
    while current_time + window_size <= end_time:
        window_starts.append(current_time)
        current_time += step_size
    
    logger.info(f"将计算 {len(window_starts)} 个窗口")
    
    # 为每个窗口计算特征
    window_features = []


    # 使用tqdm显示进度条
    for window_start in tqdm(window_starts, desc="计算窗口特征"):
        window_end = window_start + window_size
        # 获取当前窗口的数据
        window_data = df[(df['TimeStamp'] >= window_start) & (df['TimeStamp'] < window_end)].copy()
        
        # 确保窗口内有数据
        if len(window_data) == 30:
            # 添加窗口ID
            
            # 计算窗口统计特征
            window_stats = calculate_window_features(window_data)
            
            # 从原始窗口数据中获取异常标签
            if 'anomaly_label' in window_data.columns:
                # 如果窗口内有一个异常点，则整个窗口标记为异常
                window_stats['anomaly_label'] = int(window_data['anomaly_label'].max())
            
            window_features.append(window_stats)
                
    # 将所有窗口特征合并为一个DataFrame
    features_df = pd.DataFrame(window_features)
    logger.info(f"计算得到的窗口特征形状: {features_df.shape}")
    
    return features_df


def run_feature_importance_analysis(features_df, output_dir, top_n=30):
    """
    运行特征重要性分析，找出顶级特征
    
    Args:
        features_df: 包含特征的DataFrame
        output_dir: 输出目录
        top_n: 要选择的顶级特征数量
        
    Returns:
        list: 顶级特征名称列表
    """
    logger.info("运行特征重要性分析")
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存特征数据供后续分析
    features_path = os.path.join(output_dir, 'window_features.parquet')
    features_df.to_parquet(features_path)
    logger.info(f"窗口特征已保存至 {features_path}")
    
    # 运行特征重要性分析脚本
    from src.feature_analysis.feature_importance import (
        preprocess_data, analyze_correlation, analyze_mutual_information,
        analyze_tree_based_importance, analyze_pca, generate_summary_report
    )
    
    # 预处理数据
    features, target = preprocess_data(features_df)
    
    # 分析特征重要性
    corr_results = analyze_correlation(features, target, output_dir)
    mi_results = analyze_mutual_information(features, target, output_dir)
    rf_results = analyze_tree_based_importance(features, target, output_dir)
    pca_results = analyze_pca(features, output_dir)
    
    # 生成汇总报告
    summary = generate_summary_report(corr_results, mi_results, rf_results, pca_results, output_dir)
    
    # 保存顶级特征
    top_features_path = os.path.join(output_dir, 'top_features.txt')
    with open(top_features_path, 'w') as f:
        for feature, row in summary.head(top_n).iterrows():
            line = f"{feature}: Mean Rank {row['mean_rank']:.2f}"
            f.write(line + '\n')
    
    # 获取顶级特征名称
    top_features = summary.head(top_n).index.tolist()
    logger.info(f"已确定前 {top_n} 个重要特征: {top_features}")
    
    return top_features


def filter_features(features_df, top_features, output_dir):
    """
    过滤数据，只保留重要特征和必要的元数据列
    
    Args:
        features_df: 包含特征的DataFrame
        top_features: 要保留的特征名称列表
        output_dir: 输出目录
        
    Returns:
        pd.DataFrame: 过滤后的DataFrame
    """
    logger.info(f"过滤数据，只保留前 {len(top_features)} 个重要特征和必要元数据列")
    
    # 添加必要的元数据列
    required_columns = ["TimeStamp","anomaly_label"]
    columns_to_keep = list(set(required_columns + top_features))
    
    # 过滤列
    filtered_columns = [col for col in columns_to_keep if col in features_df.columns]
    filtered_df = features_df[filtered_columns].copy()
    
    logger.info(f"过滤后的数据形状: {filtered_df.shape}")
    
    # 保存过滤后的数据
    filtered_path = os.path.join(output_dir, f'filtered_window_features_{len(top_features)}.parquet')
    filtered_df.to_parquet(filtered_path)
    logger.info(f"过滤后的特征已保存至 {filtered_path}")
    
    return filtered_df


def load_features_and_continue(features_path, output_dir, top_n):
    """
    从已计算的特征文件加载数据，然后直接从特征重要性汇总文件中获取顶级特征并筛选
    
    Args:
        features_path: 特征文件路径
        output_dir: 输出目录
        top_n: 要选择的顶级特征数量
    """
    logger.info(f"从 {features_path} 加载预计算的窗口特征")
    features_df = pd.read_parquet(features_path)
    logger.info(f"加载的特征形状: {features_df.shape}")
    
    # 直接从特征重要性汇总文件中获取顶级特征
    summary_path = os.path.join(output_dir, 'feature_importance_summary.csv')
    logger.info(f"从 {summary_path} 加载特征重要性汇总")
    

    summary = pd.read_csv(summary_path, index_col=0)
    # 获取顶级特征名称
    top_features = summary.head(top_n).index.tolist()
    logger.info(f"已从汇总文件获取前 {top_n} 个重要特征: {top_features}")

    # 过滤数据，只保留重要特征
    filtered_df = filter_features(features_df, top_features[:top_n], output_dir)
    
    return filtered_df


def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()
    
    # 设置日志文件
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"experiments/logs/window_feature_analysis_{timestamp}.log"
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        # 检查是否直接从特征文件加载
        if args.load_features_from:
            # 记录参数
            logger.info(f"特征文件: {args.load_features_from}")
            logger.info(f"输出目录: {args.output_dir}")
            logger.info(f"顶级特征数量: {args.top_n}")
            
            # 从特征文件加载并继续
            filtered_df = load_features_and_continue(args.load_features_from, args.output_dir, args.top_n)
        else:
            # 记录参数
            logger.info(f"输入文件: {args.input_file}")
            logger.info(f"输出目录: {args.output_dir}")
            logger.info(f"窗口大小: {args.window_size}分钟")
            logger.info(f"窗口步长: {args.step_size}分钟")
            logger.info(f"顶级特征数量: {args.top_n}")
            if args.max_rows:
                logger.info(f"数据行数限制: {args.max_rows}行")
            
            # 加载数据
            df = load_data(args.input_file, args.max_rows)
            
            # 计算窗口特征
            start_time = time.time()
            features_df = calculate_window_features_for_dataframe(df, args.window_size, args.step_size)
            logger.info(f"窗口特征计算完成，耗时 {(time.time() - start_time):.2f} 秒")
            
            # 运行特征重要性分析
            top_features = run_feature_importance_analysis(features_df, args.output_dir, args.top_n)
            
            # 过滤数据，只保留重要特征
            filtered_df = filter_features(features_df, top_features[:args.top_n], args.output_dir)
        
        logger.info("处理完成！")
        
    except Exception as e:
        logger.error(f"处理过程中发生错误: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()