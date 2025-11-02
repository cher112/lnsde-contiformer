#!/usr/bin/env python3
"""
自动检测并删除导致NaN的样本
"""

import pickle
import numpy as np
import torch
import os
import sys
sys.path.append('/root/autodl-tmp/lnsde-contiformer')

from collections import Counter


def analyze_nan_log():
    """分析nan_samples.log文件"""
    if not os.path.exists('nan_samples.log'):
        print("没有找到nan_samples.log文件")
        return {}
    
    # 统计每个样本出现NaN的次数
    nan_counts = {}
    with open('nan_samples.log', 'r') as f:
        for line in f:
            if 'Sample' in line:
                parts = line.strip().split(',')
                for part in parts:
                    if 'Sample' in part:
                        sample_idx = int(part.split()[-1].replace('caused', '').replace('NaN', '').strip())
                        batch_idx = int(parts[1].split()[-1])
                        
                        key = f"batch_{batch_idx}_sample_{sample_idx}"
                        nan_counts[key] = nan_counts.get(key, 0) + 1
    
    print(f"发现 {len(nan_counts)} 个导致NaN的样本")
    return nan_counts


def clean_dataset(dataset_name, remove_indices=None):
    """清理数据集，删除导致NaN的样本"""
    
    # 数据路径
    enhanced_path = f'/root/autodl-fs/lnsde-contiformer/data/{dataset_name}_enhanced.pkl'
    fixed_path = f'/root/autodl-fs/lnsde-contiformer/data/{dataset_name}_fixed.pkl'
    original_path = f'/root/autodl-fs/lnsde-contiformer/data/{dataset_name}_original.pkl'
    
    # 选择要清理的数据文件
    if os.path.exists(enhanced_path):
        data_path = enhanced_path
        output_path = enhanced_path.replace('.pkl', '_clean.pkl')
        print(f"清理增强数据集: {dataset_name}")
    elif os.path.exists(fixed_path):
        data_path = fixed_path
        output_path = fixed_path.replace('.pkl', '_clean.pkl')
        print(f"清理fixed数据集: {dataset_name}")
    else:
        data_path = original_path
        output_path = original_path.replace('.pkl', '_clean.pkl')
        print(f"清理原始数据集: {dataset_name}")
    
    # 加载数据
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    original_count = len(data)
    
    if remove_indices:
        # 删除指定索引的样本
        clean_data = []
        for i, sample in enumerate(data):
            if i not in remove_indices:
                clean_data.append(sample)
        print(f"删除了 {len(remove_indices)} 个指定样本")
    else:
        # 自动检测并删除有问题的样本
        clean_data = []
        removed_count = 0
        
        for i, sample in enumerate(data):
            # 检查数据质量
            has_issue = False
            
            # 1. 检查NaN或Inf
            if np.any(np.isnan(sample['mag'])) or np.any(np.isinf(sample['mag'])):
                has_issue = True
            if np.any(np.isnan(sample['errmag'])) or np.any(np.isinf(sample['errmag'])):
                has_issue = True
            
            # 2. 检查极端值
            if np.any(np.abs(sample['mag']) > 1e10):
                has_issue = True
            
            # 3. 检查误差值
            if np.all(sample['errmag'] == 0):  # 全零误差
                has_issue = True
            if np.any(sample['errmag'] < 0):  # 负误差
                has_issue = True
                
            # 4. 检查序列长度
            if len(sample['time']) < 10:  # 太短的序列
                has_issue = True
            
            if not has_issue:
                clean_data.append(sample)
            else:
                removed_count += 1
                if removed_count <= 5:  # 显示前5个被删除的样本
                    print(f"  删除样本 {i}: file_id={sample.get('file_id', 'unknown')}")
        
        print(f"自动删除了 {removed_count} 个有问题的样本")
    
    # 保存清理后的数据
    with open(output_path, 'wb') as f:
        pickle.dump(clean_data, f)
    
    # 统计信息
    final_count = len(clean_data)
    print(f"\n清理结果:")
    print(f"  原始样本数: {original_count}")
    print(f"  清理后样本数: {final_count}")
    print(f"  删除比例: {(1 - final_count/original_count)*100:.2f}%")
    
    # 类别分布
    labels = [s['label'] for s in clean_data]
    class_counts = Counter(labels)
    print(f"\n类别分布:")
    for label, count in sorted(class_counts.items()):
        print(f"  类别 {label}: {count} 样本")
    
    print(f"\n清理后的数据已保存到: {output_path}")
    return output_path


def verify_clean_data(data_path):
    """验证清理后的数据质量"""
    print(f"\n验证数据: {data_path}")
    
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    issues_found = 0
    
    for i, sample in enumerate(data):
        # 检查NaN/Inf
        if np.any(np.isnan(sample['mag'])) or np.any(np.isinf(sample['mag'])):
            print(f"  样本 {i} 仍有NaN/Inf在mag中")
            issues_found += 1
        if np.any(np.isnan(sample['errmag'])) or np.any(np.isinf(sample['errmag'])):
            print(f"  样本 {i} 仍有NaN/Inf在errmag中") 
            issues_found += 1
    
    if issues_found == 0:
        print("✓ 数据质量检查通过，无NaN/Inf值")
    else:
        print(f"✗ 发现 {issues_found} 个问题")
    
    return issues_found == 0


def main():
    """主函数"""
    print("="*70)
    print("数据集NaN样本清理工具")
    print("="*70)
    
    # 1. 分析日志文件
    print("\n1. 分析NaN日志...")
    nan_samples = analyze_nan_log()
    
    # 2. 清理各个数据集
    datasets = ['MACHO', 'LINEAR', 'ASAS']
    
    for dataset in datasets:
        print(f"\n2. 清理 {dataset} 数据集...")
        clean_path = clean_dataset(dataset)
        
        # 3. 验证清理后的数据
        verify_clean_data(clean_path)
    
    print("\n" + "="*70)
    print("清理完成！")
    print("\n使用清理后的数据:")
    print("1. 将_clean.pkl文件重命名为_enhanced.pkl")
    print("2. 或修改config.py使用_clean版本")
    print("="*70)


if __name__ == "__main__":
    main()