#!/usr/bin/env python3
"""
分析LINEAR数据集结构，为物理约束TimeGAN优化做准备
重点分析类别1和2的特征，以便针对性地生成区分性强的合成数据
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import seaborn as sns

def analyze_linear_data():
    """分析LINEAR数据集的结构和类别分布"""
    print("🔍 分析LINEAR数据集...")
    
    # 加载数据
    data_path = '/root/autodl-fs/lnsde-contiformer/data/LINEAR_original.pkl'
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    print(f"总样本数: {len(data)}")
    
    # 检查第一个样本的结构
    sample = data[0]
    print(f"样本字段: {list(sample.keys())}")
    
    # 分析类别分布
    labels = [item['label'] for item in data]
    class_names = [item['class_name'] for item in data]
    
    label_counts = Counter(labels)
    class_name_counts = Counter(class_names)
    
    print(f"\n📊 类别分布（按标签）:")
    for label, count in sorted(label_counts.items()):
        print(f"  类别 {label}: {count} 样本")
    
    print(f"\n📊 类别分布（按名称）:")
    for name, count in sorted(class_name_counts.items()):
        print(f"  {name}: {count} 样本")
    
    # 特别分析类别1和类别2的特征
    print(f"\n🎯 重点分析类别1和类别2...")
    
    class1_samples = [item for item in data if item['label'] == 1]
    class2_samples = [item for item in data if item['label'] == 2]
    
    print(f"类别1样本数: {len(class1_samples)}")
    print(f"类别2样本数: {len(class2_samples)}")
    
    if class1_samples:
        print(f"类别1类名: {class1_samples[0]['class_name']}")
    if class2_samples:
        print(f"类别2类名: {class2_samples[0]['class_name']}")
    
    # 分析时间序列长度分布
    def analyze_class_features(samples, class_name):
        """分析特定类别的特征"""
        print(f"\n📈 {class_name} 特征分析:")
        
        lengths = []
        periods = []
        mag_ranges = []
        error_means = []
        
        for sample in samples[:10]:  # 分析前10个样本
            mask = sample['mask'].astype(bool)
            times = sample['time'][mask]
            mags = sample['mag'][mask]
            errors = sample['errmag'][mask]
            
            lengths.append(len(times))
            periods.append(sample['period'])
            mag_ranges.append(mags.max() - mags.min())
            error_means.append(errors.mean())
        
        print(f"  序列长度: {np.mean(lengths):.1f} ± {np.std(lengths):.1f}")
        print(f"  周期范围: [{np.min(periods):.3f}, {np.max(periods):.3f}]")
        print(f"  星等变化: {np.mean(mag_ranges):.3f} ± {np.std(mag_ranges):.3f}")
        print(f"  误差水平: {np.mean(error_means):.4f} ± {np.std(error_means):.4f}")
        
        return {
            'lengths': lengths,
            'periods': periods, 
            'mag_ranges': mag_ranges,
            'error_means': error_means
        }
    
    class1_features = analyze_class_features(class1_samples, "类别1")
    class2_features = analyze_class_features(class2_samples, "类别2")
    
    # 分析所有类别的典型特征，找出混淆原因
    print(f"\n🔍 分析所有类别特征对比...")
    all_class_features = {}
    
    for label in sorted(label_counts.keys()):
        samples = [item for item in data if item['label'] == label]
        class_name = samples[0]['class_name'] if samples else f"Class_{label}"
        
        periods = [s['period'] for s in samples[:20]]
        mag_ranges = []
        
        for sample in samples[:20]:
            mask = sample['mask'].astype(bool)
            if np.sum(mask) > 0:
                mags = sample['mag'][mask]
                mag_ranges.append(mags.max() - mags.min())
        
        all_class_features[label] = {
            'name': class_name,
            'count': len(samples),
            'period_mean': np.mean(periods) if periods else 0,
            'period_std': np.std(periods) if periods else 0,
            'mag_range_mean': np.mean(mag_ranges) if mag_ranges else 0,
            'mag_range_std': np.std(mag_ranges) if mag_ranges else 0
        }
    
    print(f"{'类别':<6} {'名称':<12} {'样本数':<8} {'周期均值':<10} {'周期标准差':<12} {'星等变化':<10} {'变化标准差':<10}")
    print("-" * 80)
    
    for label, features in all_class_features.items():
        print(f"{label:<6} {features['name']:<12} {features['count']:<8} "
              f"{features['period_mean']:<10.3f} {features['period_std']:<12.3f} "
              f"{features['mag_range_mean']:<10.3f} {features['mag_range_std']:<10.3f}")
    
    return data, all_class_features

def create_linear_timegan_strategy():
    """基于分析结果制定LINEAR TimeGAN策略"""
    print(f"\n🎯 制定LINEAR TimeGAN增强策略...")
    
    strategy = {
        'target_classes': [1, 2],  # 重点优化类别1和2
        'physics_constraints': {
            'enhanced_periodicity': 0.3,     # 加强周期约束
            'noise_reduction': 0.25,         # 加强去噪
            'class_separation': 0.2,         # 增加类别区分度
            'magnitude_consistency': 0.15    # 星等一致性
        },
        'generation_params': {
            'batch_size': 64,
            'n_epochs': 200,    # 增加训练轮数提高质量
            'lr': 0.0005       # 较小学习率确保稳定收敛
        }
    }
    
    print(f"策略配置:")
    print(f"  目标类别: {strategy['target_classes']}")
    print(f"  物理约束权重: {strategy['physics_constraints']}")
    print(f"  生成参数: {strategy['generation_params']}")
    
    return strategy

if __name__ == "__main__":
    data, features = analyze_linear_data()
    strategy = create_linear_timegan_strategy()
    
    print(f"\n✅ LINEAR数据集分析完成！")
    print(f"下一步: 应用物理约束TimeGAN生成针对类别1、2优化的合成数据")