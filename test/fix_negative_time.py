#!/usr/bin/env python3
"""
修复数据集中的负时间值问题
将相位时间添加偏移量使其全部为正值，并确保时间序列递增排序
"""

import pickle
import numpy as np
import os
from typing import List, Dict
import argparse

def analyze_time_data(data_path: str) -> Dict:
    """分析数据中的时间分布情况"""
    print(f"分析数据文件: {data_path}")
    
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    min_times = []
    max_times = []
    negative_count = 0
    total_samples = len(data)
    
    print(f"总样本数: {total_samples}")
    
    for i, item in enumerate(data[:100]):  # 先检查前100个样本
        times = item['time']
        min_time = np.min(times)
        max_time = np.max(times)
        
        min_times.append(min_time)
        max_times.append(max_time)
        
        if min_time < 0:
            negative_count += 1
            
        if i < 5:  # 打印前5个样本的时间信息
            print(f"  样本 {i}: 时间范围 [{min_time:.6f}, {max_time:.6f}], 长度 {len(times)}")
    
    overall_min = np.min(min_times)
    overall_max = np.max(max_times)
    
    analysis_result = {
        'total_samples': total_samples,
        'negative_samples': negative_count,
        'overall_min_time': overall_min,
        'overall_max_time': overall_max,
        'min_times': min_times,
        'max_times': max_times
    }
    
    print(f"\n分析结果:")
    print(f"  包含负时间的样本: {negative_count}/100")
    print(f"  整体时间范围: [{overall_min:.6f}, {overall_max:.6f}]")
    print(f"  需要的偏移量: {-overall_min if overall_min < 0 else 0}")
    
    return analysis_result

def fix_negative_times(data_path: str, output_path: str = None) -> str:
    """修复负时间值问题"""
    print(f"正在修复数据文件: {data_path}")
    
    # 先分析数据
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    # 找到全局最小时间值
    global_min_time = float('inf')
    for item in data:
        times = item['time']
        min_time = np.min(times)
        if min_time < global_min_time:
            global_min_time = min_time
    
    print(f"全局最小时间: {global_min_time}")
    
    # 计算偏移量，确保所有时间都为正值
    offset = 0
    if global_min_time < 0:
        offset = -global_min_time + 1e-6  # 添加小量确保严格为正
        print(f"添加偏移量: {offset}")
    else:
        print("所有时间值已经为正，无需偏移")
    
    # 修复数据
    fixed_count = 0
    for i, item in enumerate(data):
        original_times = item['time'].copy()
        
        # 添加偏移量
        if offset > 0:
            item['time'] = item['time'] + offset
        
        # 确保时间序列按升序排列
        sort_indices = np.argsort(item['time'])
        
        # 检查是否需要排序
        if not np.array_equal(sort_indices, np.arange(len(sort_indices))):
            print(f"样本 {i} 需要重新排序")
            item['time'] = item['time'][sort_indices]
            item['mag'] = item['mag'][sort_indices]
            item['errmag'] = item['errmag'][sort_indices]
            fixed_count += 1
        elif offset > 0:
            fixed_count += 1
    
    print(f"修复了 {fixed_count} 个样本")
    
    # 保存修复后的数据
    if output_path is None:
        output_path = data_path.replace('.pkl', '_fixed.pkl')
    
    with open(output_path, 'wb') as f:
        pickle.dump(data, f)
    
    print(f"修复后的数据已保存到: {output_path}")
    
    # 验证修复结果
    verify_fixed_data(output_path)
    
    return output_path

def verify_fixed_data(data_path: str):
    """验证修复后的数据"""
    print(f"\n验证修复后的数据: {data_path}")
    
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    negative_count = 0
    unsorted_count = 0
    
    for i, item in enumerate(data[:100]):  # 检查前100个样本
        times = item['time']
        
        # 检查负时间
        if np.any(times < 0):
            negative_count += 1
            
        # 检查排序
        if not np.all(np.diff(times) >= 0):
            unsorted_count += 1
            
        if i < 3:  # 打印前3个样本验证
            min_time = np.min(times)
            max_time = np.max(times) 
            is_sorted = np.all(np.diff(times) >= 0)
            print(f"  样本 {i}: 时间范围 [{min_time:.6f}, {max_time:.6f}], 已排序: {is_sorted}")
    
    print(f"\n验证结果:")
    print(f"  包含负时间的样本: {negative_count}/100")
    print(f"  未排序的样本: {unsorted_count}/100")
    
    if negative_count == 0 and unsorted_count == 0:
        print("✅ 数据修复成功!")
    else:
        print("❌ 数据修复失败!")

def backup_original_files():
    """备份原始文件"""
    data_files = [
        '/root/autodl-tmp/lnsde-contiformer/data/ASAS_folded_512.pkl',
        '/root/autodl-tmp/lnsde-contiformer/data/LINEAR_folded_512.pkl', 
        '/root/autodl-tmp/lnsde-contiformer/data/MACHO_folded_512.pkl'
    ]
    
    for file_path in data_files:
        if os.path.exists(file_path):
            backup_path = file_path.replace('.pkl', '_backup.pkl')
            if not os.path.exists(backup_path):
                print(f"备份 {file_path} -> {backup_path}")
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)
                with open(backup_path, 'wb') as f:
                    pickle.dump(data, f)
            else:
                print(f"备份文件已存在: {backup_path}")

def main():
    parser = argparse.ArgumentParser(description='修复数据集中的负时间值问题')
    parser.add_argument('--action', choices=['analyze', 'fix', 'backup'], 
                       default='analyze', help='执行的操作')
    parser.add_argument('--data_path', type=str, 
                       help='数据文件路径 (用于analyze和fix)')
    parser.add_argument('--output_path', type=str,
                       help='输出文件路径 (用于fix)')
    
    args = parser.parse_args()
    
    if args.action == 'analyze':
        if args.data_path:
            analyze_time_data(args.data_path)
        else:
            # 分析所有数据文件
            data_files = [
                '/root/autodl-tmp/lnsde-contiformer/data/ASAS_folded_512.pkl',
                '/root/autodl-tmp/lnsde-contiformer/data/LINEAR_folded_512.pkl',
                '/root/autodl-tmp/lnsde-contiformer/data/MACHO_folded_512.pkl'
            ]
            for file_path in data_files:
                if os.path.exists(file_path):
                    print(f"\n{'='*50}")
                    analyze_time_data(file_path)
    
    elif args.action == 'fix':
        if args.data_path:
            fix_negative_times(args.data_path, args.output_path)
        else:
            # 修复所有数据文件
            data_files = [
                '/root/autodl-tmp/lnsde-contiformer/data/ASAS_folded_512.pkl',
                '/root/autodl-tmp/lnsde-contiformer/data/LINEAR_folded_512.pkl', 
                '/root/autodl-tmp/lnsde-contiformer/data/MACHO_folded_512.pkl'
            ]
            
            # 先备份
            backup_original_files()
            
            # 修复每个文件
            for file_path in data_files:
                if os.path.exists(file_path):
                    print(f"\n{'='*50}")
                    fix_negative_times(file_path, file_path)  # 直接覆盖原文件
    
    elif args.action == 'backup':
        backup_original_files()

if __name__ == "__main__":
    main()