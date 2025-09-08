#!/usr/bin/env python3
"""
分析类别0的数据特征
"""

import os
import sys
import numpy as np
import pickle
import matplotlib.pyplot as plt
from collections import Counter

# 添加项目路径
sys.path.append('/root/autodl-tmp/lnsde-contiformer')
from utils.resampling import configure_chinese_font

configure_chinese_font()

def analyze_class_characteristics(dataset_name='ASAS'):
    """分析各个类别的特征"""
    
    # 加载原始数据
    data_dir = '/root/autodl-fs/lnsde-contiformer/data'
    original_path = os.path.join(data_dir, f'{dataset_name}_original.pkl')
    
    with open(original_path, 'rb') as f:
        original_data = pickle.load(f)
    
    # 转换数据格式
    n_samples = len(original_data)
    seq_len = len(original_data[0]['time'])
    
    X_orig = np.zeros((n_samples, seq_len, 3))
    y_orig = np.zeros(n_samples, dtype=int)
    
    for i, sample in enumerate(original_data):
        X_orig[i, :, 0] = sample['time']
        X_orig[i, :, 1] = sample['mag'] 
        X_orig[i, :, 2] = sample['errmag']
        y_orig[i] = sample['label']
    
    # 分析每个类别的统计特征
    class_counts = Counter(y_orig)
    print(f"\n{dataset_name} 数据集类别分析:")
    print("="*60)
    
    fig, axes = plt.subplots(len(class_counts), 2, figsize=(15, len(class_counts)*3))
    if len(class_counts) == 1:
        axes = axes.reshape(1, -1)
    
    for idx, cls in enumerate(sorted(class_counts.keys())):
        # 获取当前类别的数据
        cls_indices = np.where(y_orig == cls)[0]
        cls_data = X_orig[cls_indices]
        
        # 计算统计特征
        mag_data = cls_data[:, :, 1]  # magnitude
        
        mean_mag = np.mean(mag_data)
        std_mag = np.std(mag_data)
        min_mag = np.min(mag_data)
        max_mag = np.max(mag_data)
        range_mag = max_mag - min_mag
        
        # 计算每个样本的变化幅度
        sample_ranges = []
        for sample in mag_data:
            sample_ranges.append(np.max(sample) - np.min(sample))
        
        avg_range = np.mean(sample_ranges)
        
        print(f"类别 {cls} ({class_counts[cls]}个样本):")
        print(f"  平均星等: {mean_mag:.3f}")
        print(f"  标准差: {std_mag:.3f}")
        print(f"  总体范围: {min_mag:.3f} ~ {max_mag:.3f} (跨度: {range_mag:.3f})")
        print(f"  平均样本变化幅度: {avg_range:.3f}")
        print(f"  变化幅度标准差: {np.std(sample_ranges):.3f}")
        
        # 绘制原始样本
        ax1 = axes[idx, 0]
        
        # 随机选择几个样本来显示
        sample_indices = np.random.choice(cls_indices, min(5, len(cls_indices)), replace=False)
        
        for i, sample_idx in enumerate(sample_indices):
            t = np.arange(len(X_orig[sample_idx, :, 1]))
            ax1.plot(t, X_orig[sample_idx, :, 1], alpha=0.7, linewidth=1, 
                    label=f'样本{i+1}')
        
        ax1.set_title(f'类别{cls} - 原始样本 (变化幅度: {avg_range:.3f})')
        ax1.set_xlabel('时间步')
        ax1.set_ylabel('星等')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # 绘制变化幅度分布
        ax2 = axes[idx, 1]
        ax2.hist(sample_ranges, bins=20, alpha=0.7, edgecolor='black')
        ax2.axvline(avg_range, color='red', linestyle='--', 
                   label=f'平均幅度: {avg_range:.3f}')
        ax2.set_title(f'类别{cls} - 变化幅度分布')
        ax2.set_xlabel('单个样本的变化幅度')
        ax2.set_ylabel('频次')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    plt.suptitle(f'{dataset_name} 数据集各类别特征分析', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # 保存图片
    save_dir = f'/root/autodl-tmp/lnsde-contiformer/results/pics/{dataset_name}'
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'{dataset_name}_class_characteristics.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n💾 特征分析图已保存至: {save_path}")
    
    # 找出变化最小的类别
    classes_by_variation = []
    for cls in sorted(class_counts.keys()):
        cls_indices = np.where(y_orig == cls)[0]
        mag_data = X_orig[cls_indices, :, 1]
        sample_ranges = [np.max(sample) - np.min(sample) for sample in mag_data]
        avg_range = np.mean(sample_ranges)
        classes_by_variation.append((cls, avg_range))
    
    classes_by_variation.sort(key=lambda x: x[1])
    
    print(f"\n📊 各类别按变化幅度排序:")
    for cls, variation in classes_by_variation:
        print(f"  类别{cls}: 平均变化幅度 {variation:.3f}")
        if variation < 0.5:
            print(f"    ⚠️  类别{cls}变化很小，可能是恒星类型")
    
    return classes_by_variation

if __name__ == "__main__":
    # 分析ASAS数据集
    analyze_class_characteristics('ASAS')