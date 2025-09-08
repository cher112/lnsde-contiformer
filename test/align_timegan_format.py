#!/usr/bin/env python3
"""
快速检查和对齐物理约束TimeGAN数据格式
确保与现有数据集格式完全一致，可直接用于main.py训练
"""

import pickle
import numpy as np
import sys

def check_data_format(file_path, dataset_name):
    """检查数据格式"""
    print(f"\n🔍 检查 {dataset_name} 数据格式:")
    print(f"文件路径: {file_path}")
    
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    
    print(f"样本数量: {len(data)}")
    print(f"数据类型: {type(data)}")
    
    if data:
        sample = data[0]
        print(f"样本键: {list(sample.keys())}")
        
        for key, value in sample.items():
            if hasattr(value, 'shape'):
                print(f"  {key}: {type(value).__name__} {value.shape} {value.dtype}")
            else:
                print(f"  {key}: {type(value).__name__} = {value}")
    
    return data

def check_format_consistency():
    """检查所有数据集格式一致性"""
    datasets = {
        'MACHO_original': '/root/autodl-fs/lnsde-contiformer/data/MACHO_original.pkl',
        'MACHO_timegan': '/root/autodl-fs/lnsde-contiformer/data/macho_resample_timegan.pkl',
        'MACHO_enhanced': '/root/autodl-fs/lnsde-contiformer/data/MACHO_enhanced.pkl'
    }
    
    all_data = {}
    
    for name, path in datasets.items():
        try:
            all_data[name] = check_data_format(path, name)
        except Exception as e:
            print(f"❌ 加载{name}失败: {e}")
    
    # 检查格式一致性
    print(f"\n📊 格式一致性检查:")
    
    if 'MACHO_original' in all_data and 'MACHO_timegan' in all_data:
        orig_sample = all_data['MACHO_original'][0]
        timegan_sample = all_data['MACHO_timegan'][0]
        
        print(f"原始数据键: {set(orig_sample.keys())}")
        print(f"TimeGAN数据键: {set(timegan_sample.keys())}")
        print(f"键差异: {set(orig_sample.keys()) ^ set(timegan_sample.keys())}")
        
        # 检查关键字段
        key_fields = ['time', 'mag', 'errmag', 'mask', 'label', 'period']
        format_match = True
        
        for key in key_fields:
            if key in orig_sample and key in timegan_sample:
                orig_val = orig_sample[key]
                timegan_val = timegan_sample[key]
                
                if hasattr(orig_val, 'dtype') and hasattr(timegan_val, 'dtype'):
                    if orig_val.dtype != timegan_val.dtype:
                        print(f"❌ {key}类型不匹配: {orig_val.dtype} vs {timegan_val.dtype}")
                        format_match = False
                    elif hasattr(orig_val, 'shape') and hasattr(timegan_val, 'shape'):
                        if orig_val.shape != timegan_val.shape:
                            print(f"❌ {key}形状不匹配: {orig_val.shape} vs {timegan_val.shape}")
                            format_match = False
        
        if format_match:
            print("✅ TimeGAN数据格式与原始数据完全一致")
        else:
            print("❌ 发现格式不匹配")
    
    return all_data

def create_aligned_timegan_data():
    """如果需要，创建格式对齐的TimeGAN数据"""
    print(f"\n🔧 检查是否需要格式对齐...")
    
    # 检查现有TimeGAN数据
    timegan_path = '/root/autodl-fs/lnsde-contiformer/data/macho_resample_timegan.pkl'
    original_path = '/root/autodl-fs/lnsde-contiformer/data/MACHO_original.pkl'
    
    with open(timegan_path, 'rb') as f:
        timegan_data = pickle.load(f)
    
    with open(original_path, 'rb') as f:
        original_data = pickle.load(f)
    
    if not timegan_data or not original_data:
        print("❌ 数据加载失败")
        return False
    
    orig_sample = original_data[0]
    timegan_sample = timegan_data[0]
    
    # 检查是否需要修正
    needs_fix = False
    issues = []
    
    # 检查序列长度
    if orig_sample['time'].shape != timegan_sample['time'].shape:
        issues.append(f"序列长度: {orig_sample['time'].shape} vs {timegan_sample['time'].shape}")
        needs_fix = True
    
    # 检查数据类型
    for key in ['time', 'mag', 'errmag']:
        if orig_sample[key].dtype != timegan_sample[key].dtype:
            issues.append(f"{key}数据类型: {orig_sample[key].dtype} vs {timegan_sample[key].dtype}")
            needs_fix = True
    
    if not needs_fix:
        print("✅ TimeGAN数据已经完全对齐，可直接使用")
        return True
    
    print(f"⚠️ 发现需要修正的问题:")
    for issue in issues:
        print(f"  - {issue}")
    
    # 修正数据格式
    print(f"🔧 开始修正数据格式...")
    fixed_data = []
    
    for i, sample in enumerate(timegan_data):
        fixed_sample = {}
        
        # 复制所有字段
        for key, value in sample.items():
            if key in ['time', 'mag', 'errmag', 'mask']:
                # 确保数组形状和类型一致
                if hasattr(value, 'shape'):
                    if key == 'mask':
                        fixed_sample[key] = value.astype(bool)
                    else:
                        fixed_sample[key] = value.astype(np.float64)
                else:
                    fixed_sample[key] = value
            else:
                fixed_sample[key] = value
        
        fixed_data.append(fixed_sample)
    
    # 保存修正后的数据
    fixed_path = '/root/autodl-fs/lnsde-contiformer/data/MACHO_timegan_aligned.pkl'
    with open(fixed_path, 'wb') as f:
        pickle.dump(fixed_data, f)
    
    print(f"✅ 修正后的数据已保存至: {fixed_path}")
    
    # 验证修正结果
    with open(fixed_path, 'rb') as f:
        verified_data = pickle.load(f)
    
    verified_sample = verified_data[0]
    print(f"\n📊 验证修正结果:")
    for key in ['time', 'mag', 'errmag', 'mask']:
        orig_info = f"{orig_sample[key].shape} {orig_sample[key].dtype}"
        fixed_info = f"{verified_sample[key].shape} {verified_sample[key].dtype}"
        match = "✅" if orig_info == fixed_info else "❌"
        print(f"  {key}: {orig_info} == {fixed_info} {match}")
    
    return True

def main():
    print("🔧 物理约束TimeGAN数据格式对齐工具")
    print("=" * 50)
    
    # 1. 检查格式一致性
    all_data = check_format_consistency()
    
    # 2. 如果需要，创建对齐版本
    success = create_aligned_timegan_data()
    
    if success:
        print(f"\n🎯 使用说明:")
        print(f"main.py训练时可以使用以下参数:")
        print(f"python main.py --dataset 3 --resampled_data_path /root/autodl-fs/lnsde-contiformer/data/MACHO_timegan_aligned.pkl")
        print(f"或者直接:")
        print(f"python main.py --dataset 3 --use_resampling")
    
    return success

if __name__ == "__main__":
    main()