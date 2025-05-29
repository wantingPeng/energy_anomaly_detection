"""
原始数据分布漂移检测脚本 (Raw Data Covariate Shift Detection)

验证标准化前的parquet数据分布漂移情况
数据路径: 
- 训练集: Data/processed/lsmt/spilt/contact/train.parquet
- 验证集: Data/processed/lsmt/spilt/contact/val.parquet
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ks_2samp
from tqdm import tqdm
import warnings
import time
from pathlib import Path

# 添加src目录到路径
sys.path.append(str(Path(__file__).parent.parent / 'src'))
from src.utils.logger import logger

# 禁用不必要的警告
warnings.filterwarnings('ignore')


def load_parquet_data(train_path: str, val_path: str):
    """
    加载训练集和验证集的parquet数据
    
    Args:
        train_path: 训练集parquet文件路径
        val_path: 验证集parquet文件路径
        
    Returns:
        tuple: (train_df, val_df)
    """
    logger.info(f"加载训练集数据: {train_path}")
    start_time = time.time()
    train_df = pd.read_parquet(train_path, engine='pyarrow')
    logger.info(f"训练集加载完成, 耗时: {time.time() - start_time:.2f}秒")
    logger.info(f"训练集形状: {train_df.shape}")
    
    logger.info(f"加载验证集数据: {val_path}")
    start_time = time.time()
    val_df = pd.read_parquet(val_path, engine='pyarrow')
    logger.info(f"验证集加载完成, 耗时: {time.time() - start_time:.2f}秒")
    logger.info(f"验证集形状: {val_df.shape}")
    
    return train_df, val_df


def preprocess_data(train_df: pd.DataFrame, val_df: pd.DataFrame):
    """
    数据预处理，获取特征列
    
    Args:
        train_df: 训练集DataFrame
        val_df: 验证集DataFrame
        
    Returns:
        tuple: (train_features_df, val_features_df, numeric_columns)
    """
    logger.info("数据预处理...")
    
    # 获取所有列名
    all_columns = train_df.columns.tolist()
    
    # 识别非特征列
    non_feature_columns = ['ID', 'TimeStamp', 'Station', 'component_type', 'segment_id']
    
    # 获取数值类型特征列
    numeric_columns = [col for col in all_columns if col not in non_feature_columns and 
                       pd.api.types.is_numeric_dtype(train_df[col].dtype)]
    
    logger.info(f"特征列数量: {len(numeric_columns)}")
    logger.info(f"特征列: {numeric_columns[:5]}...")
    
    # 检查两个数据集的列是否一致
    missing_cols = [col for col in numeric_columns if col not in val_df.columns]
    if missing_cols:
        logger.warning(f"验证集缺少以下列: {missing_cols}")
        numeric_columns = [col for col in numeric_columns if col not in missing_cols]
    
    # 提取特征列
    train_features_df = train_df[numeric_columns]
    val_features_df = val_df[numeric_columns]
    
    logger.info(f"预处理后 - 训练集形状: {train_features_df.shape}, 验证集形状: {val_features_df.shape}")
    
    return train_features_df, val_features_df, numeric_columns


def calculate_statistics(data: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    """
    计算数据集的统计信息
    
    Args:
        data: 数据DataFrame
        dataset_name: 数据集名称
        
    Returns:
        pd.DataFrame: 统计信息表
    """
    logger.info(f"计算 {dataset_name} 的统计信息")
    
    stats = pd.DataFrame({
        'feature': data.columns,
        'mean': data.mean(),
        'std': data.std(),
        'min': data.min(),
        'max': data.max(),
        'median': data.median(),
        'q25': data.quantile(0.25),
        'q75': data.quantile(0.75),
        'skew': data.skew()
    })
    
    return stats


def sample_data_for_analysis(train_df: pd.DataFrame, val_df: pd.DataFrame, max_samples: int = 100000):
    """
    对大型数据集进行采样以提高计算效率
    
    Args:
        train_df: 训练集DataFrame
        val_df: 验证集DataFrame
        max_samples: 最大样本数
        
    Returns:
        tuple: (sampled_train_df, sampled_val_df)
    """
    logger.info("对大型数据集进行采样以提高计算效率...")
    
    if len(train_df) > max_samples:
        train_sample_frac = max_samples / len(train_df)
        sampled_train_df = train_df.sample(frac=train_sample_frac, random_state=42)
        logger.info(f"训练集采样: {len(train_df)} -> {len(sampled_train_df)}")
    else:
        sampled_train_df = train_df
        logger.info(f"训练集样本数 {len(train_df)} 小于阈值，不进行采样")
    
    if len(val_df) > max_samples:
        val_sample_frac = max_samples / len(val_df)
        sampled_val_df = val_df.sample(frac=val_sample_frac, random_state=42)
        logger.info(f"验证集采样: {len(val_df)} -> {len(sampled_val_df)}")
    else:
        sampled_val_df = val_df
        logger.info(f"验证集样本数 {len(val_df)} 小于阈值，不进行采样")
    
    return sampled_train_df, sampled_val_df


def compare_distributions(train_df: pd.DataFrame, val_df: pd.DataFrame) -> pd.DataFrame:
    """
    使用KS检验比较训练集和验证集的特征分布
    
    Args:
        train_df: 训练集特征DataFrame
        val_df: 验证集特征DataFrame
        
    Returns:
        pd.DataFrame: 包含KS检验结果和统计差异的DataFrame
    """
    logger.info("开始分布比较分析")
    
    # 计算统计信息
    train_stats = calculate_statistics(train_df, "训练集")
    val_stats = calculate_statistics(val_df, "验证集")
    
    # 合并训练集和验证集统计信息用于比较
    stats_comparison = pd.merge(
        train_stats, val_stats,
        on='feature',
        suffixes=('_train', '_val')
    )
    
    # 计算统计量的相对差异
    stats_comparison['mean_rel_diff'] = np.abs(stats_comparison['mean_train'] - stats_comparison['mean_val']) / (np.abs(stats_comparison['mean_train']) + 1e-8)
    stats_comparison['std_rel_diff'] = np.abs(stats_comparison['std_train'] - stats_comparison['std_val']) / (np.abs(stats_comparison['std_train']) + 1e-8)
    
    logger.info("执行KS检验...")
    
    # 执行KS检验
    results = []
    for feature in tqdm(train_df.columns, desc="特征分布比较"):
        train_feature = train_df[feature].dropna().values
        val_feature = val_df[feature].dropna().values
        
        if len(train_feature) > 0 and len(val_feature) > 0:
            # KS检验
            ks_stat, p_value = ks_2samp(train_feature, val_feature)
            
            results.append({
                'feature': feature,
                'ks_statistic': ks_stat,
                'p_value': p_value,
                'is_drifted': p_value < 0.05  # 显著性水平 0.05
            })
        else:
            logger.warning(f"特征 {feature} 存在全部为缺失值的情况，跳过KS检验")
            results.append({
                'feature': feature,
                'ks_statistic': np.nan,
                'p_value': np.nan,
                'is_drifted': False
            })
    
    # 创建KS检验结果DataFrame
    ks_results = pd.DataFrame(results)
    
    # 合并KS检验结果和统计比较
    df_results = pd.merge(ks_results, stats_comparison, on='feature')
    
    # 按KS统计量排序
    df_results = df_results.sort_values('ks_statistic', ascending=False)
    
    # 统计漂移情况
    drifted_features = df_results[df_results['is_drifted']]
    logger.info(f"检测到 {len(drifted_features)} 个特征发生分布漂移 (p < 0.05)")
    logger.info(f"漂移特征占比: {len(drifted_features) / len(df_results) * 100:.2f}%")
    
    return df_results


def plot_drifted_features(df_diff: pd.DataFrame, train_df: pd.DataFrame, 
                         val_df: pd.DataFrame, top_n: int = 5, 
                         save_dir: str = "experiments/plots"):
    """
    绘制分布漂移最严重的特征的直方图对比
    
    Args:
        df_diff: 分布比较结果DataFrame
        train_df: 训练集特征DataFrame
        val_df: 验证集特征DataFrame
        top_n: 绘制前n个差异最大的特征
        save_dir: 图片保存目录
    """
    logger.info(f"绘制前{top_n}个分布差异最大的特征")
    
    os.makedirs(save_dir, exist_ok=True)
    
    # 选择前top_n个KS统计量最大的特征
    top_features = df_diff.head(top_n)
    
    # 设置绘图样式
    plt.style.use('default')
    fig, axes = plt.subplots(top_n, 1, figsize=(12, 4*top_n))
    if top_n == 1:
        axes = [axes]
    
    for idx, (_, row) in enumerate(top_features.iterrows()):
        feature = row['feature']
        
        # 获取特征数据
        train_feature = train_df[feature].dropna()
        val_feature = val_df[feature].dropna()
        
        # 绘制直方图
        sns.histplot(train_feature, ax=axes[idx], label='Train', 
                     kde=True, color='blue', alpha=0.5, stat='density')
        sns.histplot(val_feature, ax=axes[idx], label='Val', 
                     kde=True, color='red', alpha=0.5, stat='density')
        
        # 添加统计线
        axes[idx].axvline(train_feature.mean(), color='blue', linestyle='--', 
                         label=f'Train Mean: {train_feature.mean():.3f}')
        axes[idx].axvline(val_feature.mean(), color='red', linestyle='--', 
                         label=f'Val Mean: {val_feature.mean():.3f}')
        
        axes[idx].set_title(f'Feature: {feature}\nKS={row["ks_statistic"]:.4f}, p={row["p_value"]:.2e}')
        axes[idx].set_xlabel('Value')
        axes[idx].set_ylabel('Density')
        axes[idx].legend()
        axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(save_dir, 'raw_data_covariate_shift_top_features.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    logger.info(f"特征分布图保存至: {plot_path}")
    
    # 绘制KS统计量分布
    plt.figure(figsize=(10, 6))
    plt.hist(df_diff['ks_statistic'].dropna(), bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    plt.axvline(df_diff['ks_statistic'].mean(), color='red', linestyle='--', 
                label=f'Mean KS: {df_diff["ks_statistic"].mean():.4f}')
    plt.xlabel('KS Statistic')
    plt.ylabel('Number of Features')
    plt.title('Distribution of KS Statistics Across All Features')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    ks_dist_path = os.path.join(save_dir, 'raw_data_ks_statistics_distribution.png')
    plt.savefig(ks_dist_path, dpi=300, bbox_inches='tight')
    logger.info(f"KS统计量分布图保存至: {ks_dist_path}")


def visualize_feature_correlation(df_diff: pd.DataFrame, save_dir: str = "experiments/plots"):
    """
    可视化漂移统计量之间的相关性
    
    Args:
        df_diff: 分布比较结果DataFrame
        save_dir: 图片保存目录
    """
    logger.info("分析漂移统计量之间的相关性")
    
    # 提取相关列
    corr_columns = ['ks_statistic', 'mean_rel_diff', 'std_rel_diff', 'skew_train', 'skew_val']
    corr_df = df_diff[corr_columns].copy()
    
    # 计算相关性
    corr_matrix = corr_df.corr()
    
    # 绘制相关性热图
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt=".2f")
    plt.title('Correlation Between Drift Statistics')
    plt.tight_layout()
    
    corr_path = os.path.join(save_dir, 'raw_data_drift_statistics_correlation.png')
    plt.savefig(corr_path, dpi=300, bbox_inches='tight')
    logger.info(f"漂移统计量相关性图保存至: {corr_path}")


def main():
    """主函数：执行完整的原始数据分布漂移检测流程"""
    logger.info("开始原始数据分布漂移检测分析")
    
    # 数据路径
    train_path = "Data/processed/lsmt/spilt/pcb_cleaned/test.parquet"
    val_path = "Data/processed/lsmt/spilt/pcb_cleaned/val.parquet"
    
    # 1. 加载数据
    logger.info("=== 第一步：加载parquet数据 ===")
    train_df, val_df = load_parquet_data(train_path, val_path)
    
    # 2. 数据预处理
    logger.info("=== 第二步：数据预处理 ===")
    train_features, val_features, feature_columns = preprocess_data(train_df, val_df)
    
    # 3. 对大数据集采样以提高效率
    logger.info("=== 第三步：数据采样 ===")
    sampled_train, sampled_val = sample_data_for_analysis(train_features, val_features)
    
    # 4. 分布比较
    logger.info("=== 第四步：分布比较分析 ===")
    df_diff = compare_distributions(sampled_train, sampled_val)
    
    # 5. 保存结果
    logger.info("=== 第五步：保存分析结果 ===")
    os.makedirs("experiments/results", exist_ok=True)
    result_path = "experiments/results/raw_data_covariate_shift_analysis.csv"
    df_diff.to_csv(result_path, index=False)
    logger.info(f"分析结果保存至: {result_path}")
    
    # 6. 打印关键结果
    logger.info("=== 第六步：分析结果摘要 ===")
    drifted_features = df_diff[df_diff['is_drifted']]
    
    print("\n" + "="*80)
    print("原始数据分布漂移检测结果")
    print("="*80)
    print(f"总特征数: {len(df_diff)}")
    print(f"漂移特征数: {len(drifted_features)} (p < 0.05)")
    print(f"漂移比例: {len(drifted_features) / len(df_diff) * 100:.2f}%")
    print(f"平均KS统计量: {df_diff['ks_statistic'].mean():.4f}")
    print(f"最大KS统计量: {df_diff['ks_statistic'].max():.4f}")
    
    if len(drifted_features) > 0:
        print(f"\n前10个漂移最严重的特征:")
        print(drifted_features[['feature', 'ks_statistic', 'p_value', 
                               'mean_rel_diff', 'std_rel_diff']].head(10).to_string(index=False))
    else:
        print("\n未检测到显著的分布漂移")
    
    # 7. 可视化
    logger.info("=== 第七步：生成可视化图表 ===")
    plot_drifted_features(df_diff, sampled_train, sampled_val, top_n=5)
    visualize_feature_correlation(df_diff)
    
    logger.info("原始数据分布漂移检测分析完成！")
    return df_diff


if __name__ == "__main__":
    drift_report = main() 