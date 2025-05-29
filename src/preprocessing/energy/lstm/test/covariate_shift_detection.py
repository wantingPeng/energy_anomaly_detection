"""
数据分布漂移检测脚本 (Covariate Shift Detection)

使用训练时相同的DataLoader加载数据，验证训练集和验证集之间是否存在数据分布漂移
只分析 component='contact' 的情况
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ks_2samp
from tqdm import tqdm
import warnings
import yaml
from pathlib import Path
warnings.filterwarnings('ignore')

# 添加src目录到路径
sys.path.append(str(Path(__file__).parent.parent / 'src'))
from src.utils.logger import logger
from src.training.lsmt.dataloader_from_batches import get_component_dataloaders


def load_config(config_path: str = "configs/lstm_training.yaml") -> dict:
    """
    加载配置文件
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        dict: 配置字典
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"加载配置文件: {config_path}")
        return config
    except Exception as e:
        logger.error(f"配置文件加载失败: {config_path}, 错误: {str(e)}")
        raise


def extract_windows_from_dataloader(dataloader, dataset_name: str) -> np.ndarray:
    """
    从DataLoader中提取所有窗口数据并计算时间平均
    
    Args:
        dataloader: PyTorch DataLoader
        dataset_name: 数据集名称（用于日志）
        
    Returns:
        np.ndarray: 形状为 (N, num_features) 的数据，每个样本是窗口的时间平均
    """
    logger.info(f"从{dataset_name}的DataLoader中提取窗口数据")
    
    all_windows = []
    all_labels = []
    total_samples = 0
    
    # 设置为评估模式（不影响数据，但确保一致性）
    with torch.no_grad():
        for batch_idx, (windows, labels) in enumerate(tqdm(dataloader, desc=f"提取{dataset_name}数据")):
            # windows shape: (batch_size, window_size, num_features)
            # 对时间维度求平均，降维到 (batch_size, num_features)
            windows_avg = windows.mean(dim=1).numpy()  # (batch_size, num_features)
            
            all_windows.append(windows_avg)
            all_labels.append(labels.numpy())
            total_samples += windows_avg.shape[0]
            
            # 记录第一个batch的详细信息
            if batch_idx == 0:
                logger.info(f"  - 第一个batch: windows原始shape={windows.shape}, 平均后shape={windows_avg.shape}")
                logger.info(f"  - 特征数: {windows_avg.shape[1]}, 窗口长度: {windows.shape[1]}")
    
    # 合并所有批次
    combined_windows = np.vstack(all_windows)
    combined_labels = np.concatenate(all_labels)
    
    logger.info(f"{dataset_name}总样本数: {combined_windows.shape[0]}, 特征数: {combined_windows.shape[1]}")
    logger.info(f"{dataset_name}标签分布: 0类={np.sum(combined_labels==0)}, 1类={np.sum(combined_labels==1)}")
    
    return combined_windows, combined_labels


def load_train_val_data(config: dict, component_name: str = "contact"):
    """
    使用训练时相同的DataLoader加载训练集和验证集数据
    
    Args:
        config: 配置字典
        component_name: 组件名称
        
    Returns:
        tuple: (train_windows, train_labels, val_windows, val_labels)
    """
    logger.info(f"使用训练时相同的DataLoader加载 {component_name} 数据")
    
    # 获取配置参数
    data_config = config.get('data', {})
    train_config = config.get('training', {})
    
    train_data_dir = data_config.get('data_dir', 'Data/processed/lsmt/test/spilt_after_sliding/train')
    val_data_dir = data_config.get('val_data_dir', 'Data/processed/lsmt/test/spilt_after_sliding/val')
    batch_size = train_config.get('batch_size', 128)
    
    logger.info(f"训练集路径: {train_data_dir}")
    logger.info(f"验证集路径: {val_data_dir}")
    logger.info(f"批次大小: {batch_size}")
    
    # 加载训练集数据
    logger.info("=== 加载训练集数据 ===")
    train_windows, train_labels = None, None
    
    for comp_name, train_dataloader in get_component_dataloaders(
        component_names=[component_name],
        data_dir=train_data_dir,
        batch_size=batch_size,
        shuffle=True,  
        num_workers=4, 
        pin_memory=True
    ):
        logger.info(f"正在处理训练集组件: {comp_name}")
        train_windows, train_labels = extract_windows_from_dataloader(train_dataloader, f"训练集-{comp_name}")
        break  # 只处理第一个（也是唯一的）组件
    
    # 加载验证集数据
    logger.info("=== 加载验证集数据 ===")
    val_windows, val_labels = None, None
    
    for comp_name, val_dataloader in get_component_dataloaders(
        component_names=[component_name],
        data_dir=val_data_dir,
        batch_size=batch_size,
        shuffle=True,  
        num_workers=4, 
        pin_memory=True
    ):
        logger.info(f"正在处理验证集组件: {comp_name}")
        val_windows, val_labels = extract_windows_from_dataloader(val_dataloader, f"验证集-{comp_name}")
        break  # 只处理第一个（也是唯一的）组件
    
    if train_windows is None or val_windows is None:
        raise ValueError(f"无法加载组件 {component_name} 的数据")
    
    return train_windows, train_labels, val_windows, val_labels


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
                         save_dir: str = "experiments/reports/lsmt"):
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
    
    plt.style.use('default')  # 使用默认样式避免seaborn版本问题
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


def analyze_label_distribution(train_labels: np.ndarray, val_labels: np.ndarray):
    """
    分析训练集和验证集的标签分布
    
    Args:
        train_labels: 训练集标签
        val_labels: 验证集标签
    """
    logger.info("=== 标签分布分析 ===")
    
    # 计算标签分布
    train_pos = np.sum(train_labels == 1)
    train_neg = np.sum(train_labels == 0)
    val_pos = np.sum(val_labels == 1)
    val_neg = np.sum(val_labels == 0)
    
    train_pos_ratio = train_pos / len(train_labels)
    val_pos_ratio = val_pos / len(val_labels)
    
    print(f"\n训练集标签分布:")
    print(f"  正样本: {train_pos} ({train_pos_ratio:.3f})")
    print(f"  负样本: {train_neg} ({1-train_pos_ratio:.3f})")
    
    print(f"\n验证集标签分布:")
    print(f"  正样本: {val_pos} ({val_pos_ratio:.3f})")
    print(f"  负样本: {val_neg} ({1-val_pos_ratio:.3f})")
    
    print(f"\n标签分布差异:")
    print(f"  正样本比例差异: {abs(train_pos_ratio - val_pos_ratio):.3f}")
    
    # 统计检验
    from scipy.stats import chi2_contingency
    contingency_table = np.array([[train_pos, train_neg], [val_pos, val_neg]])
    chi2, p_value, _, _ = chi2_contingency(contingency_table)
    
    print(f"  卡方检验: chi2={chi2:.4f}, p={p_value:.4f}")
    if p_value < 0.05:
        print("  标签分布存在显著差异 (p < 0.05)")
    else:
        print("  标签分布无显著差异 (p >= 0.05)")


def main():
    """主函数：执行完整的数据分布漂移检测流程"""
    logger.info("开始基于训练DataLoader的数据分布漂移检测分析")
    
    # 1. 加载配置
    logger.info("=== 第一步：加载配置 ===")
    config = load_config()
    
    # 2. 加载数据
    logger.info("=== 第二步：使用训练DataLoader加载数据 ===")
    component_name = "contact"
    logger.info(f"分析组件: {component_name}")
    
    train_windows, train_labels, val_windows, val_labels = load_train_val_data(config, component_name)
    
    # 3. 标签分布分析
    analyze_label_distribution(train_labels, val_labels)
    
    # 4. 特征分布比较
    logger.info("=== 第三步：特征分布比较分析 ===")
    df_diff = compare_distributions(train_windows, val_windows)
    
    # 5. 保存结果
    logger.info("=== 第四步：保存分析结果 ===")
    os.makedirs("experiments/results", exist_ok=True)
    result_path = f"experiments/reports/lsmt/covariate_shift_analysis_{component_name}.csv"
    df_diff.to_csv(result_path, index=False)
    logger.info(f"分析结果保存至: {result_path}")
    
    # 6. 打印关键结果
    logger.info("=== 第五步：分析结果摘要 ===")
    drifted_features = df_diff[df_diff['is_drifted']]
    
    print("\n" + "="*80)
    print(f"数据分布漂移检测结果 - 组件: {component_name}")
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
        print("\n未检测到显著的特征分布漂移")
    
    # 7. 可视化
    logger.info("=== 第六步：生成可视化图表 ===")
    plot_drifted_features(df_diff, train_windows, val_windows, top_n=5)
    
    logger.info("数据分布漂移检测分析完成！")
    return df_diff


if __name__ == "__main__":
    drift_report = main() 