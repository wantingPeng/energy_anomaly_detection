"""
Simple Shuffle and Resplit Sliding Windows

This script:
1. Loads all .pt files from the contact directories in train, test, val sets
2. Uses DataLoader's shuffle=True to shuffle windows while maintaining window-label alignment
3. Resplits the data into new train, validation, and test sets (75%, 15%, 15%)
4. Saves the results to Data/processed/lsmt/test/spilt_after_sliding
"""

import os
import sys
import torch
import glob
import time
import gc
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset, random_split
from multiprocessing import Pool, cpu_count
import functools

# Add project root directory to Python path
sys.path.append(str(Path(__file__).parents[4]))
# Import logger
from src.utils.logger import logger
from src.utils.memory_left import log_memory

def load_component_datasets(base_dir="Data/processed/lsmt/dataset_1200s", component="contact", output_dir="Data/processed/lsmt/test/spilt_after_sliding_1200s/"):
    """
    Load all datasets from the specified component in train, test, and val sets
    and save to disk
    
    Args:
        base_dir: Base directory containing train, test, val subdirectories
        component: Component name (e.g., 'contact', 'pcb', 'ring')
        output_dir: Directory to save the combined dataset
        
    Returns:
        str: Path to saved combined dataset
    """
    logger.info(f"开始加载{component}组件的所有数据集")
    start_time = time.time()
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    combined_path = os.path.join(output_dir, "combined_dataset.pt")
    
    # Check if combined dataset already exists
    if os.path.exists(combined_path):
        logger.info(f"找到已存在的合并数据集: {combined_path}")
        return combined_path
    
    # Find all PT files in component directories
    all_windows = []
    all_labels = []
    total_windows = 0
    
    for split in ['test', 'train', 'val']:
        component_dir = os.path.join(base_dir, split, component)
        if not os.path.exists(component_dir):
            logger.warning(f"目录不存在: {component_dir}")
            continue
            
        pt_files = glob.glob(os.path.join(component_dir, "*.pt"))
        if not pt_files:
            logger.warning(f"在 {component_dir} 目录下未找到PT文件")
            continue
            
        logger.info(f"在 {split} 集合中找到 {len(pt_files)} 个PT文件")
        
        # Load each PT file and collect windows and labels
        for file_path in tqdm(pt_files, desc=f"Loading {split} {component} files"):
            try:
                # Load the PT file
                data = torch.load(file_path)
                
                # Check if data contains windows and labels
                if isinstance(data, dict) and 'windows' in data and 'labels' in data:
                    windows = data['windows']
                    labels = data['labels']
                    
                    # Add to lists
                    all_windows.append(windows)
                    all_labels.append(labels)
                    
                    total_windows += len(windows)
                    logger.info(f"成功加载 {file_path}, 窗口数: {len(windows)}")
                    
                    # Free memory immediately
                    del windows, labels, data
                    gc.collect()
                else:
                    logger.warning(f"文件格式不正确: {file_path}")
            except Exception as e:
                logger.error(f"加载 {file_path} 时出错: {str(e)}")
    
    if not all_windows:
        logger.error(f"没有找到任何有效的数据集")
        raise ValueError("No valid datasets found")
    
    # Concatenate all windows and labels
    logger.info(f"合并所有数据...")
    all_windows_tensor = torch.cat(all_windows, dim=0)
    all_labels_tensor = torch.cat(all_labels, dim=0)
    
    # Clear lists to free memory
    del all_windows, all_labels
    gc.collect()
    
    # Save combined dataset to disk
    logger.info(f"保存合并数据集到磁盘: {combined_path}")
    torch.save({
        'windows': all_windows_tensor,
        'labels': all_labels_tensor
    }, combined_path)
    
    # Clear tensors to free memory
    del all_windows_tensor, all_labels_tensor
    gc.collect()
    
    logger.info(f"成功加载并保存所有数据集, 总窗口数: {total_windows}, 耗时: {time.time() - start_time:.2f}秒")
    
    return combined_path

def shuffle_and_save_dataset(combined_dataset_path, output_dir="Data/processed/lsmt/test/spilt_after_sliding_1200s/", batch_size=128, num_workers=4):
    """
    Load combined dataset from disk, shuffle using DataLoader and save to disk in two batches
    
    Args:
        combined_dataset_path: Path to combined dataset
        output_dir: Output directory for shuffled data
        batch_size: Batch size for DataLoader
        num_workers: Number of workers for DataLoader
        
    Returns:
        list: Paths to saved shuffled data batches
    """
    logger.info(f"开始从磁盘加载数据、打乱并分批保存")
    start_time = time.time()
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    shuffled_dir = os.path.join(output_dir, "shuffled")
    os.makedirs(shuffled_dir, exist_ok=True)
    
    # Define paths for the two batches
    shuffled_data_paths = [
        os.path.join(shuffled_dir, "shuffled_data_batch1.pt"),
        os.path.join(shuffled_dir, "shuffled_data_batch2.pt")
    ]
    
    # Check if shuffled data already exists
    if os.path.exists(shuffled_data_paths[0]) and os.path.exists(shuffled_data_paths[1]):
        logger.info(f"找到已存在的打乱数据批次")
        return shuffled_data_paths
    
    # Load combined dataset
    log_memory("Before loading combined dataset")
    combined_data = torch.load(combined_dataset_path)
    windows = combined_data['windows']
    labels = combined_data['labels']
    log_memory("After loading combined dataset")
    
    # Get total dataset size
    total_size = len(windows)
    logger.info(f"总数据集大小: {total_size} 窗口")
    
    # Process in two batches to reduce memory usage
    for batch_idx in range(2):
        logger.info(f"处理第 {batch_idx+1}/2 批次的数据")
        
        # Calculate slice indices for this batch
        start_idx = batch_idx * (total_size // 2)
        end_idx = total_size if batch_idx == 1 else (batch_idx + 1) * (total_size // 2)
        batch_size_actual = end_idx - start_idx
        
        logger.info(f"当前批次处理窗口范围: {start_idx} 到 {end_idx} (共 {batch_size_actual} 个窗口)")
        
        # Slice the data for this batch
        batch_windows = windows[start_idx:end_idx]
        batch_labels = labels[start_idx:end_idx]
        
        # Create TensorDataset for this batch
        batch_dataset = TensorDataset(batch_windows, batch_labels)
        
        # Free original batch data
        del batch_windows, batch_labels
        gc.collect()
        log_memory(f"After creating TensorDataset for batch {batch_idx+1}")
        
        # Create DataLoader with shuffle=True
        log_memory(f"Before creating DataLoader for batch {batch_idx+1}")
        batch_loader = DataLoader(
            batch_dataset, 
            batch_size=batch_size, 
            shuffle=True,  # Shuffle this batch
            num_workers=num_workers,
            pin_memory=True
        )
        
        # Collect shuffled windows and labels for this batch
        shuffled_windows, shuffled_labels = [], []
        
        logger.info(f"收集并打乱第 {batch_idx+1} 批次数据...")
        for windows_batch, labels_batch in tqdm(batch_loader, desc=f"Shuffling batch {batch_idx+1}"):
            shuffled_windows.append(windows_batch)
            shuffled_labels.append(labels_batch)
        
        # Concatenate all windows and labels for this batch
        shuffled_windows = torch.cat(shuffled_windows, dim=0)
        shuffled_labels = torch.cat(shuffled_labels, dim=0)
        
        log_memory(f"After collecting batch {batch_idx+1}")
        
        # Clear memory
        del batch_dataset, batch_loader
        gc.collect()
        log_memory(f"After clearing dataset and loader for batch {batch_idx+1}")
        
        # Save this batch to disk
        logger.info(f"保存第 {batch_idx+1} 批次打乱的数据到: {shuffled_data_paths[batch_idx]}")
        torch.save({
            'windows': shuffled_windows,
            'labels': shuffled_labels,
            'start_idx': start_idx,
            'end_idx': end_idx
        }, shuffled_data_paths[batch_idx])
        
        # Clear memory again
        del shuffled_windows, shuffled_labels
        gc.collect()
        log_memory(f"After saving and clearing batch {batch_idx+1}")
    
    # Clear original data
    del windows, labels, combined_data
    gc.collect()
    log_memory("After clearing all original data")
    
    logger.info(f"数据集打乱并分批保存完成, 耗时: {time.time() - start_time:.2f}秒")
    
    return shuffled_data_paths

def split_saved_dataset(shuffled_data_paths, train_ratio=0.70, val_ratio=0.15, output_dir="Data/processed/lsmt/test/spilt_after_sliding_1200s/", component="contact"):
    """
    Load shuffled data batches from disk, split each batch into train, val, and test sets, and save them directly.
    
    Args:
        shuffled_data_paths: Paths to saved shuffled data batches
        train_ratio: Ratio for training set
        val_ratio: Ratio for validation set
        output_dir: Output directory
        component: Component name
    
    Returns:
        None
    """
    logger.info(f"从磁盘加载打乱的数据批次并进行划分")
    start_time = time.time()
    
    # Create output directories
    train_dir = os.path.join(output_dir, "train", component)
    val_dir = os.path.join(output_dir, "val", component)
    test_dir = os.path.join(output_dir, "test", component)
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    # Process each batch
    for batch_idx, path in enumerate(shuffled_data_paths):
        logger.info(f"处理第 {batch_idx+1}/{len(shuffled_data_paths)} 批次")
        log_memory(f"Before loading batch {batch_idx+1}")
        
        # Load batch
        batch_data = torch.load(path)
        batch_windows = batch_data['windows']
        batch_labels = batch_data['labels']
        batch_size = len(batch_windows)
        
        log_memory(f"After loading batch {batch_idx+1}")
        
        # Calculate split indices for this batch
        train_end = int(batch_size * train_ratio)
        val_end = train_end + int(batch_size * val_ratio)
        
        # Split the batch
        train_windows = batch_windows[:train_end]
        train_labels = batch_labels[:train_end]
        val_windows = batch_windows[train_end:val_end]
        val_labels = batch_labels[train_end:val_end]
        test_windows = batch_windows[val_end:]
        test_labels = batch_labels[val_end:]
        
        # Save each split
        save_batch((train_windows, train_labels, os.path.join(train_dir, f"batch_{batch_idx}.pt")))
        save_batch((val_windows, val_labels, os.path.join(val_dir, f"batch_{batch_idx}.pt")))
        save_batch((test_windows, test_labels, os.path.join(test_dir, f"batch_{batch_idx}.pt")))
        
        # Free memory
        del batch_data, batch_windows, batch_labels
        gc.collect()
        log_memory(f"After processing batch {batch_idx+1}")
    
    logger.info(f"数据集划分并保存完成, 耗时: {time.time() - start_time:.2f}秒")

def save_batch(args):
    """
    Save a batch of data to disk, following the format in save_results() from save_for_dataset.py
    
    Args:
        args: Tuple containing (batch_windows, batch_labels, batch_path)
        
    Returns:
        str: Path to saved file
    """
    batch_windows, batch_labels, batch_path = args
    
    # Convert to torch tensors if they aren't already
    windows_tensor = torch.FloatTensor(batch_windows) if not isinstance(batch_windows, torch.Tensor) else batch_windows
    labels_tensor = torch.FloatTensor(batch_labels) if not isinstance(batch_labels, torch.Tensor) else batch_labels
    
    # Save in the same format as save_for_dataset.py (but without parquet file)
    torch.save({
        'windows': windows_tensor,
        'labels': labels_tensor
    }, batch_path)
    
    # Free memory
    del batch_windows, batch_labels, windows_tensor, labels_tensor
    gc.collect()
    
    return batch_path

def main():
    """
    Main function to execute the data loading, shuffling, and splitting process
    """
    logger.info("开始执行简单数据打乱与重新划分任务")
    start_time = time.time()
    
    try:
        # Parameters
        base_dir = "Data/processed/lsmt/dataset_800s"
        output_dir = "Data/processed/lsmt/test/spilt_after_sliding_800s/"
        component = "contact"
        train_ratio = 0.70  # Updated to match the function definition
        val_ratio = 0.15
        
        # Initial memory usage
        log_memory("Initial state")
        
        # Step 1: Load datasets and save to disk
        combined_dataset_path = load_component_datasets(base_dir, component, output_dir)
        log_memory("After loading and saving datasets")
        
        # Step 2: Load combined dataset, shuffle and save to disk in two batches
        shuffled_data_paths = shuffle_and_save_dataset(combined_dataset_path, output_dir)
        log_memory("After shuffling and saving in batches")
        #shuffled_data_paths = [
        #    "Data/processed/lsmt/test/spilt_after_sliding_1200s/shuffled/shuffled_data_batch1.pt",
        #    "Data/processed/lsmt/test/spilt_after_sliding_1200s/shuffled/shuffled_data_batch2.pt"
        #]
        # Step 3: Load shuffled data batches and split
        split_saved_dataset(shuffled_data_paths, train_ratio, val_ratio, output_dir, component)
        log_memory("After splitting")
        
    except Exception as e:
        logger.error(f"执行过程中发生错误: {str(e)}")
        logger.exception("详细错误信息:")
        raise

if __name__ == "__main__":
    main() 