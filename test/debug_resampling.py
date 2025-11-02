#!/usr/bin/env python3
"""
调试重采样脚本 - 单进程测试
"""

import os
import sys
import numpy as np
import torch
import pickle
from collections import Counter

# 添加项目路径
sys.path.append('/root/autodl-tmp/lnsde-contiformer')

from utils.resampling import HybridResampler

def load_dataset_simple(dataset_name):
    """简单加载数据集"""
    print(f"🔍 尝试加载 {dataset_name} 数据集...")
    
    data_dir = '/root/autodl-fs/lnsde-contiformer/data'
    file_path = os.path.join(data_dir, f'{dataset_name}_original.pkl')
    
    if not os.path.exists(file_path):
        print(f"❌ 文件不存在: {file_path}")
        return None
    
    print(f"📂 加载文件: {file_path}")
    
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    
    print(f"✅ 文件加载成功，数据类型: {type(data)}")
    
    if isinstance(data, list) and len(data) > 0:
        print(f"📊 数据格式: 列表，长度: {len(data)}")
        if isinstance(data[0], dict):
            print(f"🔧 样本格式: 字典，键: {list(data[0].keys())}")
            seq_len = len(data[0]['time'])
            print(f"⏱️  序列长度: {seq_len}")
            
            # 转换数据格式
            n_samples = min(100, len(data))  # 只测试前100个样本
            n_features = 3
            
            X = np.zeros((n_samples, seq_len, n_features))
            y = np.zeros(n_samples, dtype=int)
            times = np.zeros((n_samples, seq_len))
            masks = np.ones((n_samples, seq_len), dtype=bool)
            
            print(f"🔄 转换前{n_samples}个样本...")
            for i in range(n_samples):
                sample = data[i]
                X[i, :, 0] = sample['time']
                X[i, :, 1] = sample['mag'] 
                X[i, :, 2] = sample['errmag']
                y[i] = sample['label']
                times[i] = sample['time']
                if 'mask' in sample:
                    masks[i] = sample['mask']
            
            print(f"✅ 数据转换完成")
            print(f"   - X shape: {X.shape}")
            print(f"   - y shape: {y.shape}")
            print(f"   - 类别分布: {Counter(y)}")
            
            return X, y, times, masks
    
    print(f"❌ 不支持的数据格式: {type(data)}")
    return None

def test_single_dataset(dataset_name):
    """测试单个数据集的重采样"""
    print(f"\n{'='*60}")
    print(f"🧪 测试 {dataset_name} 数据集重采样")
    print(f"{'='*60}")
    
    # 1. 加载数据
    result = load_dataset_simple(dataset_name)
    if result is None:
        print(f"❌ {dataset_name} 数据加载失败")
        return
    
    X, y, times, masks = result
    
    # 2. 创建重采样器
    print(f"\n🔧 创建混合重采样器...")
    resampler = HybridResampler(
        smote_k_neighbors=3,  # 减少邻居数
        enn_n_neighbors=3,
        sampling_strategy='balanced',
        synthesis_mode='hybrid',
        noise_level=0.05,
        apply_enn=True,
        random_state=535411460
    )
    
    # 3. 执行重采样
    print(f"\n⚡ 开始重采样...")
    try:
        X_resampled, y_resampled, times_resampled, masks_resampled = resampler.fit_resample(
            X, y, times, masks
        )
        
        print(f"✅ 重采样成功!")
        print(f"   - 原始样本数: {len(y)}")
        print(f"   - 重采样后: {len(y_resampled)}")
        print(f"   - 原始分布: {Counter(y)}")
        print(f"   - 重采样分布: {Counter(y_resampled)}")
        
        # 4. 保存测试结果
        save_path = f'/root/autodl-fs/lnsde-contiformer/data/{dataset_name}_resampled_test.pkl'
        test_data = {
            'X': X_resampled,
            'y': y_resampled,
            'times': times_resampled,
            'masks': masks_resampled,
            'dataset': dataset_name,
            'test_mode': True
        }
        
        with open(save_path, 'wb') as f:
            pickle.dump(test_data, f)
        
        print(f"💾 测试结果保存至: {save_path}")
        
    except Exception as e:
        print(f"❌ 重采样失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # 测试单个数据集
    test_dataset = 'ASAS'
    test_single_dataset(test_dataset)