#!/usr/bin/env python3
"""
简化的调试脚本：直接检查数据文件中的nan/inf问题
"""

import os
import numpy as np
import pickle
from pathlib import Path

def check_data_file(file_path, dataset_name):
    """检查数据文件中的nan/inf问题"""
    print(f"\n{'='*60}")
    print(f"Checking {dataset_name}: {file_path}")
    print(f"{'='*60}")
    
    if not os.path.exists(file_path):
        print(f"文件不存在: {file_path}")
        return
    
    # 加载数据
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
    except Exception as e:
        print(f"加载文件失败: {e}")
        return
    
    # 检查数据结构
    if isinstance(data, dict):
        print(f"数据字典包含键: {list(data.keys())}")
        
        # 检查每个键的内容
        for key in data.keys():
            if isinstance(data[key], (list, np.ndarray)):
                check_array_data(data[key], f"{dataset_name}['{key}']")
    
    elif isinstance(data, (list, np.ndarray)):
        check_array_data(data, dataset_name)
    
    else:
        print(f"未知数据格式: {type(data)}")

def check_array_data(data, name):
    """检查数组数据中的nan/inf"""
    if isinstance(data, list):
        # 如果是列表，检查每个元素
        print(f"\n检查 {name}: 共 {len(data)} 个样本")
        
        nan_samples = []
        inf_samples = []
        zero_samples = []
        extreme_samples = []
        
        for i, sample in enumerate(data):
            if isinstance(sample, (np.ndarray, list)):
                sample_arr = np.array(sample) if isinstance(sample, list) else sample
                
                # 检查NaN
                if np.isnan(sample_arr).any():
                    nan_count = np.isnan(sample_arr).sum()
                    nan_samples.append((i, nan_count, sample_arr.size))
                
                # 检查Inf
                if np.isinf(sample_arr).any():
                    inf_count = np.isinf(sample_arr).sum()
                    inf_samples.append((i, inf_count, sample_arr.size))
                
                # 检查全零
                if np.all(sample_arr == 0):
                    zero_samples.append(i)
                
                # 检查极值
                valid_mask = ~np.isnan(sample_arr) & ~np.isinf(sample_arr)
                if valid_mask.any():
                    max_val = np.abs(sample_arr[valid_mask]).max()
                    min_val = np.abs(sample_arr[valid_mask][sample_arr[valid_mask] != 0]).min() if (sample_arr[valid_mask] != 0).any() else 0
                    
                    if max_val > 1e10:
                        extreme_samples.append((i, 'max', max_val))
                    if min_val > 0 and min_val < 1e-10:
                        extreme_samples.append((i, 'min', min_val))
        
        # 打印统计
        if nan_samples:
            print(f"  ⚠️ NaN问题: {len(nan_samples)} 个样本")
            for idx, count, total in nan_samples[:5]:
                print(f"    样本 {idx}: {count}/{total} NaN值")
            if len(nan_samples) > 5:
                print(f"    ... 还有 {len(nan_samples)-5} 个样本")
        
        if inf_samples:
            print(f"  ⚠️ Inf问题: {len(inf_samples)} 个样本")
            for idx, count, total in inf_samples[:5]:
                print(f"    样本 {idx}: {count}/{total} Inf值")
            if len(inf_samples) > 5:
                print(f"    ... 还有 {len(inf_samples)-5} 个样本")
        
        if zero_samples:
            print(f"  ⚠️ 全零问题: {len(zero_samples)} 个样本")
            print(f"    样本索引: {zero_samples[:10]}{'...' if len(zero_samples) > 10 else ''}")
        
        if extreme_samples:
            print(f"  ⚠️ 极值问题: {len(extreme_samples)} 个样本")
            for idx, typ, val in extreme_samples[:5]:
                print(f"    样本 {idx}: {typ}={val:.2e}")
        
        if not (nan_samples or inf_samples or zero_samples or extreme_samples):
            print(f"  ✓ 数据正常，无NaN/Inf/异常值")
    
    elif isinstance(data, np.ndarray):
        # 如果是单个数组
        print(f"\n检查 {name}: 形状 {data.shape}")
        
        nan_count = np.isnan(data).sum()
        inf_count = np.isinf(data).sum()
        zero_count = (data == 0).sum()
        
        if nan_count > 0:
            print(f"  ⚠️ NaN值: {nan_count}/{data.size}")
        if inf_count > 0:
            print(f"  ⚠️ Inf值: {inf_count}/{data.size}")
        if zero_count == data.size:
            print(f"  ⚠️ 全零数组")
        
        if nan_count == 0 and inf_count == 0 and zero_count < data.size:
            print(f"  ✓ 数据正常")

def main():
    """主函数"""
    print("="*60)
    print("数据文件NaN/Inf检查工具")
    print("="*60)
    
    # 数据文件路径
    base_path = Path("/root/autodl-fs/lnsde-contiformer/data")
    
    datasets = [
        ("ASAS_fixed", base_path / "ASAS_fixed.pkl"),
        ("ASAS_original", base_path / "ASAS_original.pkl"),
        ("LINEAR_fixed", base_path / "LINEAR_fixed.pkl"),
        ("LINEAR_original", base_path / "LINEAR_original.pkl"),
        ("MACHO_fixed", base_path / "MACHO_fixed.pkl"),
        ("MACHO_original", base_path / "MACHO_original.pkl"),
    ]
    
    # 检查每个数据集
    for name, path in datasets:
        check_data_file(path, name)
    
    # 检查重采样数据
    print("\n" + "="*60)
    print("检查重采样数据")
    print("="*60)
    
    resampled_path = base_path / "resampled"
    if resampled_path.exists():
        for file in resampled_path.glob("*.pkl"):
            check_data_file(file, f"Resampled_{file.stem}")

if __name__ == "__main__":
    main()