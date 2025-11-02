#!/usr/bin/env python3
"""
可视化重采样效果 - 展示合成曲线与源样本对比
"""

import os
import sys
import numpy as np
import torch
import pickle
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from collections import Counter

# 添加项目路径
sys.path.append('/root/autodl-tmp/lnsde-contiformer')

from utils.resampling import configure_chinese_font

# 配置中文字体
configure_chinese_font()

def load_original_and_resampled_data(dataset_name):
    """加载原始和重采样数据"""
    data_dir = '/root/autodl-fs/lnsde-contiformer/data'
    
    # 加载原始数据
    original_path = os.path.join(data_dir, f'{dataset_name}_original.pkl')
    with open(original_path, 'rb') as f:
        original_data = pickle.load(f)
    
    # 转换原始数据格式
    if isinstance(original_data, list):
        n_samples = len(original_data)
        seq_len = len(original_data[0]['time'])
        n_features = 3
        
        X_orig = np.zeros((n_samples, seq_len, n_features))
        y_orig = np.zeros(n_samples, dtype=int)
        
        for i, sample in enumerate(original_data):
            X_orig[i, :, 0] = sample['time']
            X_orig[i, :, 1] = sample['mag'] 
            X_orig[i, :, 2] = sample['errmag']
            y_orig[i] = sample['label']
    
    # 加载重采样数据
    resampled_path = os.path.join(data_dir, f'{dataset_name}_resampled.pkl')
    with open(resampled_path, 'rb') as f:
        resampled_data = pickle.load(f)
    
    X_resampled = resampled_data['X']
    y_resampled = resampled_data['y']
    
    # 转换为numpy数组
    if torch.is_tensor(X_resampled):
        X_resampled = X_resampled.cpu().numpy()
    if torch.is_tensor(y_resampled):
        y_resampled = y_resampled.cpu().numpy()
    
    return X_orig, y_orig, X_resampled, y_resampled

def find_synthetic_samples(X_orig, y_orig, X_resampled, y_resampled):
    """识别合成样本（重采样数据中超出原始数据量的部分）"""
    original_counts = Counter(y_orig)
    synthetic_indices = {}
    
    current_idx = 0
    for cls in sorted(set(y_resampled)):
        # 找到当前类别在重采样数据中的索引
        cls_indices = np.where(y_resampled == cls)[0]
        
        # 原始数据中这个类别有多少样本
        original_count = original_counts.get(cls, 0)
        
        # 合成样本就是超出原始数量的那些
        if len(cls_indices) > original_count:
            synthetic_indices[cls] = cls_indices[original_count:]
        
    return synthetic_indices

def visualize_synthesis_comparison(dataset_name, save_dir='/root/autodl-tmp/lnsde-contiformer/results/pics'):
    """可视化合成样本与原始样本对比"""
    print(f"🎨 生成 {dataset_name} 数据集的合成效果可视化...")
    
    # 加载数据
    X_orig, y_orig, X_resampled, y_resampled = load_original_and_resampled_data(dataset_name)
    
    # 找到合成样本
    synthetic_indices = find_synthetic_samples(X_orig, y_orig, X_resampled, y_resampled)
    
    # 统计信息
    original_counts = Counter(y_orig)
    resampled_counts = Counter(y_resampled)
    
    print(f"📊 {dataset_name} 数据统计:")
    print(f"   原始分布: {dict(original_counts)}")
    print(f"   重采样分布: {dict(resampled_counts)}")
    
    # 创建保存目录
    dataset_save_dir = os.path.join(save_dir, dataset_name)
    os.makedirs(dataset_save_dir, exist_ok=True)
    
    # 选择要可视化的类别（选择有合成样本的类别）
    classes_to_visualize = list(synthetic_indices.keys())[:4]  # 最多4个类别
    
    if not classes_to_visualize:
        print(f"⚠️ {dataset_name} 没有找到合成样本")
        return
    
    # 创建子图
    n_classes = len(classes_to_visualize)
    n_examples = 3  # 每个类别显示3个对比例子
    
    fig, axes = plt.subplots(n_classes, n_examples, figsize=(n_examples*5, n_classes*4))
    if n_classes == 1:
        axes = axes.reshape(1, -1)
    if n_examples == 1:
        axes = axes.reshape(-1, 1)
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57', '#FF9FF3']
    
    for cls_idx, cls in enumerate(classes_to_visualize):
        # 获取原始样本（这个类别的前几个）
        orig_cls_indices = np.where(y_orig == cls)[0]
        
        # 获取合成样本索引
        synth_indices = synthetic_indices[cls]
        
        for ex_idx in range(min(n_examples, len(synth_indices))):
            ax = axes[cls_idx, ex_idx] if n_classes > 1 else axes[ex_idx]
            
            # 随机选择一个原始样本和一个合成样本
            if len(orig_cls_indices) > 0:
                orig_idx = np.random.choice(orig_cls_indices)
                orig_sample = X_orig[orig_idx]
            else:
                continue
                
            synth_idx = synth_indices[ex_idx % len(synth_indices)]
            synth_sample = X_resampled[synth_idx]
            
            # 绘制时间序列（只显示magnitude特征）
            seq_len = min(len(orig_sample), len(synth_sample))
            t = np.linspace(0, 1, seq_len)
            
            # 原始样本
            ax.plot(t, orig_sample[:seq_len, 1], 'o-', alpha=0.8, linewidth=2, 
                   color=colors[0], label='原始样本', markersize=3)
            
            # 合成样本
            ax.plot(t, synth_sample[:seq_len, 1], '^-', linewidth=2, 
                   color=colors[2], label='GPU合成样本', markersize=4, alpha=0.9)
            
            # 设置标题和标签
            ax.set_title(f'{dataset_name} - 类别{cls} - 样本{ex_idx+1}\\nGPU混合模式合成', 
                        fontsize=12, fontweight='bold')
            ax.set_xlabel('时间步')
            ax.set_ylabel('星等 (Magnitude)')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=10)
            
            # 添加统计信息
            orig_std = np.std(orig_sample[:, 1])
            synth_std = np.std(synth_sample[:, 1])
            orig_mean = np.mean(orig_sample[:, 1])
            synth_mean = np.mean(synth_sample[:, 1])
            
            stats_text = f'原始: μ={orig_mean:.3f}, σ={orig_std:.3f}\\n合成: μ={synth_mean:.3f}, σ={synth_std:.3f}'
            ax.text(0.02, 0.98, stats_text, 
                   transform=ax.transAxes, va='top', ha='left', 
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                   fontsize=9)
    
    # 总标题
    plt.suptitle(f'{dataset_name} 数据集 GPU加速混合重采样效果对比\\n'
                f'原始: {len(y_orig):,}样本 → 重采样: {len(y_resampled):,}样本 '
                f'(+{len(y_resampled)-len(y_orig):,})', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # 保存图片
    save_path = os.path.join(dataset_save_dir, f'{dataset_name}_gpu_synthesis_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"💾 {dataset_name} 合成对比图已保存至: {save_path}")
    return save_path

def create_distribution_comparison(dataset_name, save_dir='/root/autodl-tmp/lnsde-contiformer/results/pics'):
    """创建类别分布对比图"""
    print(f"📊 生成 {dataset_name} 类别分布对比图...")
    
    # 加载数据
    X_orig, y_orig, X_resampled, y_resampled = load_original_and_resampled_data(dataset_name)
    
    # 统计分布
    original_counts = Counter(y_orig)
    resampled_counts = Counter(y_resampled)
    
    # 创建对比图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 原始分布
    classes = sorted(set(list(original_counts.keys()) + list(resampled_counts.keys())))
    orig_values = [original_counts.get(cls, 0) for cls in classes]
    resampled_values = [resampled_counts.get(cls, 0) for cls in classes]
    
    x_pos = np.arange(len(classes))
    
    # 原始分布柱状图
    bars1 = ax1.bar(x_pos, orig_values, color='#FF6B6B', alpha=0.8, edgecolor='black')
    ax1.set_title(f'{dataset_name} 原始类别分布\\n总样本: {sum(orig_values):,}', 
                 fontsize=14, fontweight='bold')
    ax1.set_xlabel('类别', fontsize=12)
    ax1.set_ylabel('样本数', fontsize=12)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([f'类别{cls}' for cls in classes])
    ax1.grid(axis='y', alpha=0.3)
    
    # 添加数值标签
    for bar, value in zip(bars1, orig_values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(value)}', ha='center', va='bottom', fontsize=10)
    
    # 重采样分布柱状图
    bars2 = ax2.bar(x_pos, resampled_values, color='#4ECDC4', alpha=0.8, edgecolor='black')
    ax2.set_title(f'{dataset_name} GPU重采样后分布\\n总样本: {sum(resampled_values):,} '
                 f'(+{sum(resampled_values)-sum(orig_values):,})', 
                 fontsize=14, fontweight='bold')
    ax2.set_xlabel('类别', fontsize=12)
    ax2.set_ylabel('样本数', fontsize=12)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([f'类别{cls}' for cls in classes])
    ax2.grid(axis='y', alpha=0.3)
    
    # 添加数值标签
    for bar, value in zip(bars2, resampled_values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(value)}', ha='center', va='bottom', fontsize=10)
    
    # 计算不平衡率
    if min(orig_values) > 0:
        orig_imbalance = max(orig_values) / min(orig_values)
    else:
        orig_imbalance = float('inf')
        
    if min(resampled_values) > 0:
        resampled_imbalance = max(resampled_values) / min(resampled_values)
    else:
        resampled_imbalance = float('inf')
    
    # 添加不平衡率信息
    ax1.text(0.5, 0.95, f'不平衡率: {orig_imbalance:.2f}',
            transform=ax1.transAxes, ha='center', va='top',
            bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))
    
    ax2.text(0.5, 0.95, f'不平衡率: {resampled_imbalance:.2f}',
            transform=ax2.transAxes, ha='center', va='top',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
    
    plt.suptitle(f'{dataset_name} GPU加速混合重采样 - 类别分布对比', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # 保存图片
    dataset_save_dir = os.path.join(save_dir, dataset_name)
    os.makedirs(dataset_save_dir, exist_ok=True)
    save_path = os.path.join(dataset_save_dir, f'{dataset_name}_distribution_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"💾 {dataset_name} 分布对比图已保存至: {save_path}")
    return save_path

def visualize_all_datasets():
    """为所有数据集生成可视化"""
    datasets = ['ASAS', 'LINEAR', 'MACHO']
    
    print("🎨 开始生成所有数据集的可视化...")
    print("="*60)
    
    for dataset in datasets:
        try:
            print(f"\n🔄 处理 {dataset} 数据集...")
            
            # 生成合成样本对比图
            synthesis_path = visualize_synthesis_comparison(dataset)
            
            # 生成分布对比图
            distribution_path = create_distribution_comparison(dataset)
            
            print(f"✅ {dataset} 可视化完成!")
            
        except Exception as e:
            print(f"❌ {dataset} 可视化失败: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*60)
    print("🎉 所有数据集可视化完成！")
    print("📁 图片保存位置: /root/autodl-tmp/lnsde-contiformer/results/pics/")

if __name__ == "__main__":
    visualize_all_datasets()