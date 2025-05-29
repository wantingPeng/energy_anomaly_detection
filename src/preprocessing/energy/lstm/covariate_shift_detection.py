"""
数据分布漂移检测脚本 (Covariate Shift Detection)

验证训练集和验证集之间是否存在数据分布漂移
"""

import os
import sys
import glob
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ks_2samp
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# 添加src目录到路径
sys.path.append(str(Path(__file__).parent.parent / 'src'))
from src.utils.logger import logger


def load_train_windows(folder_path: str) -> np.ndarray:
    """
    加载训练集的所有批次窗口数据并合并
    
    Args:
        folder_path: 训练集文件夹路径
        
    Returns:
        np.ndarray: 形状为 (N, num_features) 的数据，每个样本是窗口的时间平均
    """
    logger.info(f"加载训练集数据：{folder_path}")
    
    # 获取所有.pt文件
    pt_files = glob.glob(os.path.join(folder_path, "batch_*.pt"))
    pt_files.sort()  # 按文件名排序
    
    logger.info(f"找到 {len(pt_files)} 个训练批次文件")
    
    all_windows = []
    total_samples = 0
    
    for pt_file in tqdm(pt_files, desc="加载训练批次"):
        logger.info(f"加载 {os.path.basename(pt_file)}")
        
        data = torch.load(pt_file, map_location='cpu')
        windows = data['windows']  # shape: (batch_size, window_size, num_features)
        
        # 对时间维度求平均，降维到 (batch_size, num_features)
        windows_avg = windows.mean(dim=1).numpy()  # shape: (batch_size, num_features)
        all_windows.append(windows_avg)
        
        samples_count = windows_avg.shape[0]
        total_samples += samples_count
        logger.info(f"  - 样本数: {samples_count}, 特征数: {windows_avg.shape[1]}")
    
    # 合并所有批次
    combined_windows = np.vstack(all_windows)
    logger.info(f"训练集总样本数: {combined_windows.shape[0]}, 特征数: {combined_windows.shape[1]}")
    
    return combined_windows


def load_val_windows(file_path: str) -> np.ndarray:
    """
    加载验证集的窗口数据
    
    Args:
        file_path: 验证集文件路径
        
    Returns:
        np.ndarray: 形状为 (N, num_features) 的数据，每个样本是窗口的时间平均
    """
    logger.info(f"加载验证集数据：{file_path}")
    
    data = torch.load(file_path, map_location='cpu')
    windows = data['windows']  # shape: (batch_size, window_size, num_features)
    
    # 对时间维度求平均，降维到 (batch_size, num_features)
    windows_avg = windows.mean(dim=1).numpy()  # shape: (batch_size, num_features)
    
    logger.info(f"验证集样本数: {windows_avg.shape[0]}, 特征数: {windows_avg.shape[1]}")
    
    return windows_avg


def calculate_statistics(data: np.ndarray, dataset_name: str) -> pd.DataFrame:
    """
    计算数据集的统计信息
    
    Args:
        data: 数据数组 (N, num_features)
        dataset_name: 数据集名称
        
    Returns:
        pd.DataFrame: 统计信息表
    """
    logger.info(f"计算 {dataset_name} 的统计信息")
    
    stats = pd.DataFrame({
        'feature_idx': range(data.shape[1]),
        'mean': data.mean(axis=0),
        'std': data.std(axis=0),
        'min': data.min(axis=0),
        'max': data.max(axis=0),
        'median': np.median(data, axis=0),
        'q25': np.percentile(data, 25, axis=0),
        'q75': np.percentile(data, 75, axis=0)
    })
    
    return stats


def compare_distributions(train_windows: np.ndarray, val_windows: np.ndarray) -> pd.DataFrame:
    """
    使用KS检验比较训练集和验证集的特征分布
    
    Args:
        train_windows: 训练集数据 (N_train, num_features)
        val_windows: 验证集数据 (N_val, num_features)
        
    Returns:
        pd.DataFrame: 包含KS检验结果和统计差异的DataFrame
    """
    logger.info("开始分布比较分析")
    
    num_features = train_windows.shape[1]
    results = []
    
    # 计算统计信息
    train_stats = calculate_statistics(train_windows, "训练集")
    val_stats = calculate_statistics(val_windows, "验证集")
    
    logger.info("执行KS检验...")
    
    for i in tqdm(range(num_features), desc="特征分布比较"):
        train_feature = train_windows[:, i]
        val_feature = val_windows[:, i]
        
        # KS检验
        ks_stat, p_value = ks_2samp(train_feature, val_feature)
        
        # 统计差异
        mean_diff = abs(train_stats.loc[i, 'mean'] - val_stats.loc[i, 'mean'])
        std_diff = abs(train_stats.loc[i, 'std'] - val_stats.loc[i, 'std'])
        
        # 相对差异（避免除零）
        mean_rel_diff = mean_diff / (abs(train_stats.loc[i, 'mean']) + 1e-8)
        std_rel_diff = std_diff / (abs(train_stats.loc[i, 'std']) + 1e-8)
        
        results.append({
            'feature_idx': i,
            'ks_statistic': ks_stat,
            'p_value': p_value,
            'is_drifted': p_value < 0.05,  # 显著性水平 0.05
            'train_mean': train_stats.loc[i, 'mean'],
            'val_mean': val_stats.loc[i, 'mean'],
            'train_std': train_stats.loc[i, 'std'],
            'val_std': val_stats.loc[i, 'std'],
            'mean_diff': mean_diff,
            'std_diff': std_diff,
            'mean_rel_diff': mean_rel_diff,
            'std_rel_diff': std_rel_diff
        })
    
    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values('ks_statistic', ascending=False)
    
    # 统计漂移情况
    drifted_features = df_results[df_results['is_drifted']]
    logger.info(f"检测到 {len(drifted_features)} 个特征发生分布漂移 (p < 0.05)")
    logger.info(f"漂移特征占比: {len(drifted_features) / len(df_results) * 100:.2f}%")
    
    return df_results


def plot_drifted_features(df_diff: pd.DataFrame, train_windows: np.ndarray, 
                         val_windows: np.ndarray, top_n: int = 5, 
                         save_dir: str = "experiments/plots"):
    """
    绘制分布漂移最严重的特征的直方图对比
    
    Args:
        df_diff: 分布比较结果DataFrame
        train_windows: 训练集数据
        val_windows: 验证集数据
        top_n: 绘制前n个差异最大的特征
        save_dir: 图片保存目录
    """
    logger.info(f"绘制前{top_n}个分布差异最大的特征")
    
    os.makedirs(save_dir, exist_ok=True)
    
    # 选择前top_n个KS统计量最大的特征
    top_features = df_diff.head(top_n)
    
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(top_n, 1, figsize=(12, 4*top_n))
    if top_n == 1:
        axes = [axes]
    
    for idx, (_, row) in enumerate(top_features.iterrows()):
        feature_idx = int(row['feature_idx'])
        
        # 获取特征数据
        train_feature = train_windows[:, feature_idx]
        val_feature = val_windows[:, feature_idx]
        
        # 绘制直方图
        axes[idx].hist(train_feature, bins=50, alpha=0.7, label='Train', 
                      density=True, color='blue')
        axes[idx].hist(val_feature, bins=50, alpha=0.7, label='Val', 
                      density=True, color='red')
        
        # 添加统计线
        axes[idx].axvline(train_feature.mean(), color='blue', linestyle='--', 
                         label=f'Train Mean: {train_feature.mean():.3f}')
        axes[idx].axvline(val_feature.mean(), color='red', linestyle='--', 
                         label=f'Val Mean: {val_feature.mean():.3f}')
        
        axes[idx].set_title(f'Feature {feature_idx}: KS={row["ks_statistic"]:.4f}, '
                           f'p={row["p_value"]:.2e}')
        axes[idx].set_xlabel('Feature Value')
        axes[idx].set_ylabel('Density')
        axes[idx].legend()
        axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(save_dir, 'covariate_shift_top_features.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    logger.info(f"特征分布图保存至: {plot_path}")
    plt.show()
    
    # 绘制KS统计量分布
    plt.figure(figsize=(10, 6))
    plt.hist(df_diff['ks_statistic'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    plt.axvline(df_diff['ks_statistic'].mean(), color='red', linestyle='--', 
                label=f'Mean KS: {df_diff["ks_statistic"].mean():.4f}')
    plt.xlabel('KS Statistic')
    plt.ylabel('Number of Features')
    plt.title('Distribution of KS Statistics Across All Features')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    ks_dist_path = os.path.join(save_dir, 'ks_statistics_distribution.png')
    plt.savefig(ks_dist_path, dpi=300, bbox_inches='tight')
    logger.info(f"KS统计量分布图保存至: {ks_dist_path}")
    plt.show()


def main():
    """主函数：执行完整的数据分布漂移检测流程"""
    logger.info("开始数据分布漂移检测分析")
    
    # 数据路径
    train_folder = "Data/processed/lsmt/dataset/train/contact"
    val_file = "Data/processed/lsmt/dataset/val/contact/batch_0.pt"
    
    logger.info(f"训练集路径: {train_folder}")
    logger.info(f"验证集路径: {val_file}")
    
    # 1. 加载数据
    logger.info("=== 第一步：加载数据 ===")
    train_windows = load_train_windows(train_folder)
    val_windows = load_val_windows(val_file)
    
    # 2. 分布比较
    logger.info("=== 第二步：分布比较分析 ===")
    df_diff = compare_distributions(train_windows, val_windows)
    
    # 3. 保存结果
    logger.info("=== 第三步：保存分析结果 ===")
    os.makedirs("experiments/results", exist_ok=True)
    result_path = "experiments/results/covariate_shift_analysis.csv"
    df_diff.to_csv(result_path, index=False)
    logger.info(f"分析结果保存至: {result_path}")
    
    # 4. 打印关键结果
    logger.info("=== 第四步：分析结果摘要 ===")
    drifted_features = df_diff[df_diff['is_drifted']]
    
    print("\n" + "="*80)
    print("数据分布漂移检测结果")
    print("="*80)
    print(f"总特征数: {len(df_diff)}")
    print(f"漂移特征数: {len(drifted_features)} (p < 0.05)")
    print(f"漂移比例: {len(drifted_features) / len(df_diff) * 100:.2f}%")
    print(f"平均KS统计量: {df_diff['ks_statistic'].mean():.4f}")
    print(f"最大KS统计量: {df_diff['ks_statistic'].max():.4f}")
    
    if len(drifted_features) > 0:
        print(f"\n前10个漂移最严重的特征:")
        print(drifted_features[['feature_idx', 'ks_statistic', 'p_value', 
                               'mean_rel_diff', 'std_rel_diff']].head(10).to_string(index=False))
    else:
        print("\n未检测到显著的分布漂移")
    
    # 5. 可视化
    logger.info("=== 第五步：生成可视化图表 ===")
    plot_drifted_features(df_diff, train_windows, val_windows, top_n=5)
    
    logger.info("数据分布漂移检测分析完成！")
    return df_diff


if __name__ == "__main__":
    drift_report = main() 