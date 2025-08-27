#!/usr/bin/env python3
"""
正确修复数据时间序列问题
将backup数据复制到主数据文件，确保时间序列正确
"""

import pickle
import numpy as np
import shutil
from pathlib import Path

def fix_data_files():
    """正确修复数据文件"""
    datasets = [
        ('data/ASAS_folded_512_backup.pkl', 'data/ASAS_folded_512.pkl'),
        ('data/LINEAR_folded_512_backup.pkl', 'data/LINEAR_folded_512.pkl'),
        ('data/MACHO_folded_512_backup.pkl', 'data/MACHO_folded_512.pkl')
    ]
    
    for backup_path, main_path in datasets:
        backup_file = Path(backup_path)
        main_file = Path(main_path)
        
        if not backup_file.exists():
            print(f"警告: 备份文件不存在: {backup_path}")
            continue
            
        print(f"正在修复: {main_path}")
        
        # 加载backup数据
        with open(backup_path, 'rb') as f:
            backup_data = pickle.load(f)
            
        # 处理时间序列，确保SDE求解器兼容
        fixed_data = []
        for i, sample in enumerate(backup_data):
            fixed_sample = sample.copy()
            
            # 获取时间和mask
            times = fixed_sample['time'].copy()
            mask = fixed_sample['mask'].astype(bool)
            
            # 只处理有效时间点
            if np.sum(mask) > 0:
                valid_times = times[mask]
                
                # 如果有负时间值，进行偏移使其都为正
                if np.any(valid_times < 0):
                    # 计算需要的偏移量
                    min_time = np.min(valid_times)
                    offset = -min_time + 1e-6  # 确保最小时间为正数
                    
                    # 只偏移有效时间点
                    times[mask] = valid_times + offset
                    fixed_sample['time'] = times
                    
                # 检查修复后的时间序列
                fixed_valid_times = times[mask]
                if len(fixed_valid_times) > 1:
                    # 按时间排序有效数据点（在mask范围内）
                    valid_indices = np.where(mask)[0]
                    time_sort_order = np.argsort(fixed_valid_times)
                    
                    # 重新排列数据以保证时间递增
                    sorted_indices = valid_indices[time_sort_order]
                    
                    # 更新时间、星等、误差数据
                    sorted_times = times[sorted_indices]
                    sorted_mags = fixed_sample['mag'][sorted_indices]
                    sorted_errmags = fixed_sample['errmag'][sorted_indices]
                    
                    # 更新到原位置
                    times[mask] = sorted_times
                    fixed_sample['mag'][mask] = sorted_mags
                    fixed_sample['errmag'][mask] = sorted_errmags
                    fixed_sample['time'] = times
            
            fixed_data.append(fixed_sample)
            
            if (i + 1) % 500 == 0:
                print(f"  已处理 {i+1}/{len(backup_data)} 个样本")
        
        # 验证修复结果
        print(f"验证修复结果...")
        test_sample = fixed_data[0]
        test_times = test_sample['time']
        test_mask = test_sample['mask'].astype(bool)
        test_valid_times = test_times[test_mask]
        
        print(f"  有效时间点数: {len(test_valid_times)}")
        if len(test_valid_times) > 1:
            print(f"  时间范围: {np.min(test_valid_times):.6f} to {np.max(test_valid_times):.6f}")
            print(f"  时间严格递增: {np.all(np.diff(test_valid_times) > 0)}")
            print(f"  最小时间间隔: {np.min(np.diff(test_valid_times)):.8f}")
        
        # 保存修复后的数据
        with open(main_path, 'wb') as f:
            pickle.dump(fixed_data, f)
            
        print(f"✅ 修复完成: {main_path}")
        print()

if __name__ == "__main__":
    fix_data_files()
    print("所有数据文件修复完成！")