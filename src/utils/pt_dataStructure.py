import os
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from src.utils.logger import logger

def analyze_pt_files(base_dir: str = "Data/processed/lsmt/dataset/train") -> Dict[str, List[Dict[str, Any]]]:
    """
    Analyze PyTorch (.pt) files in the specified directory and its subdirectories.
    
    Args:
        base_dir: Base directory to look for .pt files
        
    Returns:
        A dictionary with subdirectory names as keys and lists of dictionaries containing 
        information about each .pt file as values
    """
    base_path = Path(base_dir)
    result = {}
    
    if not base_path.exists():
        logger.error(f"Directory {base_dir} does not exist")
        return result
    
    # Iterate through subdirectories
    for subdir in [d for d in base_path.iterdir() if d.is_dir()]:
        subdir_name = subdir.name
        result[subdir_name] = []
        
        logger.info(f"Analyzing .pt files in {subdir}")
        
        # Find all .pt files in the subdirectory
        pt_files = [f for f in subdir.glob("*.pt")]
        
        for pt_file in pt_files:
            try:
                # Load the .pt file
                data = torch.load(pt_file, map_location=torch.device('cpu'))
                
                file_info = {
                    "file_name": pt_file.name,
                    "file_path": str(pt_file),
                    "file_size_mb": pt_file.stat().st_size / (1024 * 1024)
                }
                
                # If data is a dict, extract information from each key
                if isinstance(data, dict):
                    file_info["type"] = "dict"
                    file_info["keys"] = list(data.keys())
                    file_info["data_shapes"] = {k: tuple(v.shape) if hasattr(v, 'shape') else str(type(v)) for k, v in data.items()}
                    
                    # Try to extract feature names if available
                    if 'feature_names' in data:
                        file_info["feature_names"] = data['feature_names']
                    elif hasattr(data.get('features', None), 'names'):
                        file_info["feature_names"] = data['features'].names
                    elif hasattr(data.get('columns', None), 'names'):
                        file_info["feature_names"] = data['columns'].names
                
                # If data is a tensor, extract its shape
                elif isinstance(data, torch.Tensor):
                    file_info["type"] = "tensor"
                    file_info["shape"] = tuple(data.shape)
                    file_info["dtype"] = str(data.dtype)
                
                # For other types, just record the type
                else:
                    file_info["type"] = str(type(data))
                
                result[subdir_name].append(file_info)
                logger.info(f"Analyzed {pt_file.name} - Shape information: {file_info.get('data_shapes', file_info.get('shape', 'Unknown'))}")
                
            except Exception as e:
                logger.error(f"Error analyzing {pt_file}: {e}")
                result[subdir_name].append({
                    "file_name": pt_file.name,
                    "file_path": str(pt_file),
                    "error": str(e)
                })
    
    return result

def print_pt_summary(analysis_result: Dict[str, List[Dict[str, Any]]]) -> None:
    """
    Print a summary of the analyzed PyTorch files
    
    Args:
        analysis_result: Result from analyze_pt_files
    """
    for subdir, files_info in analysis_result.items():
        logger.info(f"\n{'='*20} {subdir} {'='*20}")
        
        for file_info in files_info:
            logger.info(f"\nFile: {file_info['file_name']} ({file_info['file_size_mb']:.2f} MB)")
            
            if 'error' in file_info:
                logger.error(f"  Error: {file_info['error']}")
                continue
                
            if file_info['type'] == 'dict':
                logger.info(f"  Keys: {file_info['keys']}")
                logger.info(f"  Data shapes:")
                for k, shape in file_info['data_shapes'].items():
                    logger.info(f"    - {k}: {shape}")
                
                if 'feature_names' in file_info:
                    logger.info(f"  Feature names: {file_info['feature_names']}")
            
            elif file_info['type'] == 'tensor':
                logger.info(f"  Shape: {file_info['shape']}")
                logger.info(f"  Data type: {file_info['dtype']}")
            
            else:
                logger.info(f"  Type: {file_info['type']}")

def analyze_and_print_pt_structure(base_dir: str = "Data/processed/lsmt/test/spilt_after_sliding_1200s/train_down_25%_2") -> Dict[str, List[Dict[str, Any]]]:
    """
    Convenience function to analyze PyTorch files and print a summary
    
    Args:
        base_dir: Base directory to look for .pt files
        
    Returns:
        Analysis result dictionary
    """
    logger.info(f"Starting analysis of PyTorch files in {base_dir}")
    result = analyze_pt_files(base_dir)
    print_pt_summary(result)
    logger.info("Analysis complete")
    return result

if __name__ == "__main__":
    # Example usage
    analyze_and_print_pt_structure()
