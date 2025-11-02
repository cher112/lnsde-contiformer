#!/usr/bin/env python3
"""
快速重采样质量分析 - 专门分析MACHO数据集的重采样效果
包括：
1. mask使用统计
2. 有效数据曲线可视化
3. 时序特征保持性分析
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import torch
from collections import Counter
import pickle
from datetime import datetime

# 添加项目路径
sys.path.append('/root/autodl-tmp/lnsde-contiformer')

from utils import create_dataloaders, setup_dataset_mapping
from utils.resampling import HybridResampler, configure_chinese_font
import warnings
warnings.filterwarnings('ignore')

# 配置中文字体
configure_chinese_font()

def quick_analysis():
    """快速分析MACHO数据集"""
    # 设置输出目录
    output_dir = '/root/autodl-tmp/lnsde-contiformer/results/pics/MACHO'
    os.makedirs(output_dir, exist_ok=True)
    
    print("快速重采样质量分析 - MACHO数据集")
    print("="*60)
    
    # 创建虚拟args对象加载MACHO数据集
    class Args:
        def __init__(self):
            self.dataset = 3  # MACHO
            self.batch_size = 64
            self.seed = 42
            self.model_type = 2
            self.learning_rate = 1e-4
            self.weight_decay = 1e-4
            self.hidden_channels = 128
            self.contiformer_dim = 256
            self.n_heads = 8
            self.n_layers = 6
            self.dropout = 0.1
            self.use_sde = 1
            self.use_contiformer = 1
            self.use_cga = 1
            self.sde_config = 1
    
    args = Args()
    args = setup_dataset_mapping(args)
    
    print(f"加载数据集: {args.dataset_name}")
    
    # 加载数据 - 只取前500个样本进行快速分析
    train_loader, test_loader, num_classes = create_dataloaders(
        data_path=args.data_path,
        batch_size=args.batch_size,
        num_workers=4,
        random_seed=42
    )
    
    # 提取前3个batch的数据进行快速分析
    print("提取原始数据样本...")
    
    all_features = []
    all_labels = []
    all_times = []
    all_masks = []
    
    batch_count = 0
    for batch in train_loader:
        if batch_count >= 3:  # 只取前3个batch
            break
            
        # 处理数据格式
        if 'features' in batch:
            features = batch['features']
            times = features[:, :, 0]  # 提取时间维度
        elif 'x' in batch:
            x = batch['x']  # (batch, seq_len, 2) [time, mag]
            batch_size, seq_len = x.shape[0], x.shape[1]
            errmag = torch.zeros(batch_size, seq_len, 1)
            features = torch.cat([x, errmag], dim=2)  # (batch, seq_len, 3)
            times = x[:, :, 0]  # 提取时间维度
        else:
            continue
            
        all_features.append(features)
        all_labels.append(batch['labels'])
        all_times.append(times)
        all_masks.append(batch.get('mask', torch.ones_like(times, dtype=torch.bool)))
        
        batch_count += 1
    
    # 合并数据
    X_orig = torch.cat(all_features, dim=0)
    y_orig = torch.cat(all_labels, dim=0)
    times_orig = torch.cat(all_times, dim=0)
    masks_orig = torch.cat(all_masks, dim=0)
    
    print(f"原始数据统计:")
    print(f"  样本数: {len(y_orig)}")
    print(f"  类别分布: {dict(Counter(y_orig.cpu().numpy()))}")
    print(f"  序列长度: {X_orig.shape[1]}")
    
    # 执行重采样
    print("\n执行重采样...")
    resampler = HybridResampler(
        smote_k_neighbors=3,  # 减少邻居数加快速度
        enn_n_neighbors=3,
        sampling_strategy='balanced',
        apply_enn=False,  # 不进行欠采样
        random_state=42
    )
    
    X_resamp, y_resamp, times_resamp, masks_resamp = resampler.fit_resample(
        X_orig, y_orig, times_orig, masks_orig
    )
    
    print(f"重采样后数据统计:")
    print(f"  样本数: {len(y_resamp)}")
    print(f"  类别分布: {dict(Counter(y_resamp.cpu().numpy() if torch.is_tensor(y_resamp) else y_resamp))}")
    
    # 1. 可视化有效数据曲线对比
    print("\n生成有效数据曲线对比图...")
    visualize_effective_curves(X_orig, y_orig, times_orig, masks_orig,
                              X_resamp, y_resamp, times_resamp, masks_resamp,
                              output_dir)
    
    # 2. mask统计分析
    print("生成mask统计分析图...")
    analyze_mask_statistics(masks_orig, y_orig, masks_resamp, y_resamp, output_dir)
    
    # 3. 重采样分布图
    print("生成重采样分布图...")
    dist_path = os.path.join(output_dir, 'resampling_distribution.png')
    resampler.visualize_distribution(save_path=dist_path)
    
    print(f"\n分析完成！输出目录: {output_dir}")
    
    return output_dir

def visualize_effective_curves(X_orig, y_orig, times_orig, masks_orig,
                              X_resamp, y_resamp, times_resamp, masks_resamp,
                              output_dir):
    """可视化有效数据曲线"""
    unique_classes = torch.unique(y_orig).cpu().numpy()
    n_classes = len(unique_classes)
    
    # 设置颜色
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57']
    
    fig, axes = plt.subplots(n_classes, 4, figsize=(16, n_classes * 3))
    if n_classes == 1:
        axes = axes.reshape(1, -1)
    
    for row, cls in enumerate(unique_classes):
        color = colors[cls % len(colors)]
        
        # 原始数据：选择2个样本
        orig_indices = torch.where(y_orig == cls)[0][:2]
        for col in range(2):
            if col < len(orig_indices):
                idx = orig_indices[col]
                features = X_orig[idx]
                mask = masks_orig[idx]
                times = times_orig[idx]
                
                # 只绘制有效数据点
                valid_mask = mask.cpu().numpy()
                if valid_mask.sum() > 0:  # 确保有有效数据
                    valid_times = times[valid_mask].cpu().numpy()
                    valid_mags = features[:, 1][valid_mask].cpu().numpy()  # 星等
                    
                    axes[row, col].plot(valid_times, valid_mags, 'o-', 
                                       color=color, alpha=0.8, markersize=4, linewidth=1.5)
                    axes[row, col].set_title(f'原始 Class {cls} #{col+1}', fontsize=10, fontweight='bold')
                    axes[row, col].grid(True, alpha=0.3)
                    axes[row, col].set_xlabel('时间')
                    axes[row, col].set_ylabel('星等')
                    
                    # 添加有效数据点数量信息
                    axes[row, col].text(0.05, 0.95, f'有效点: {valid_mask.sum()}', 
                                       transform=axes[row, col].transAxes, 
                                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7),
                                       fontsize=8)
        
        # 重采样数据：选择2个样本
        resamp_indices = torch.where(y_resamp == cls)[0][:2]
        for col in range(2):
            if col < len(resamp_indices):
                idx = resamp_indices[col]
                features = X_resamp[idx]
                mask = masks_resamp[idx] if masks_resamp is not None else torch.ones_like(times_resamp[idx], dtype=torch.bool)
                times = times_resamp[idx] if times_resamp is not None else torch.linspace(0, 1, features.shape[0])
                
                # 处理tensor和numpy的兼容性
                if torch.is_tensor(mask):
                    valid_mask = mask.cpu().numpy()
                else:
                    valid_mask = mask
                    
                if valid_mask.sum() > 0:  # 确保有有效数据
                    if torch.is_tensor(times):
                        valid_times = times[valid_mask].cpu().numpy()
                    else:
                        valid_times = times[valid_mask]
                        
                    if torch.is_tensor(features):
                        valid_mags = features[:, 1][valid_mask].cpu().numpy()
                    else:
                        valid_mags = features[:, 1][valid_mask]
                    
                    axes[row, col + 2].plot(valid_times, valid_mags, 'o-', 
                                           color=color, alpha=0.8, markersize=4, linewidth=1.5)
                    axes[row, col + 2].set_title(f'重采样 Class {cls} #{col+1}', fontsize=10, fontweight='bold')
                    axes[row, col + 2].grid(True, alpha=0.3)
                    axes[row, col + 2].set_xlabel('时间')
                    axes[row, col + 2].set_ylabel('星等')
                    
                    # 添加有效数据点数量信息
                    axes[row, col + 2].text(0.05, 0.95, f'有效点: {valid_mask.sum()}', 
                                           transform=axes[row, col + 2].transAxes,
                                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7),
                                           fontsize=8)
    
    plt.suptitle('MACHO数据集 - 原始vs重采样有效数据曲线对比', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, 'effective_curves_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"有效数据曲线对比图已保存: {save_path}")
    return save_path

def analyze_mask_statistics(masks_orig, y_orig, masks_resamp, y_resamp, output_dir):
    """分析mask使用统计"""
    unique_classes = torch.unique(y_orig).cpu().numpy()
    
    # 计算每个类别的mask统计
    orig_stats = {}
    resamp_stats = {}
    
    for cls in unique_classes:
        # 原始数据统计
        cls_mask_orig = y_orig == cls
        cls_masks_orig = masks_orig[cls_mask_orig]
        
        total_positions = cls_masks_orig.numel()
        valid_positions = cls_masks_orig.sum().item()
        valid_ratio = valid_positions / total_positions if total_positions > 0 else 0
        seq_valid_lengths = cls_masks_orig.sum(dim=1).cpu().numpy()
        
        orig_stats[cls] = {
            'valid_ratio': valid_ratio,
            'total_samples': cls_mask_orig.sum().item(),
            'mean_valid_length': np.mean(seq_valid_lengths),
            'std_valid_length': np.std(seq_valid_lengths),
            'seq_lengths': seq_valid_lengths
        }
        
        # 重采样数据统计
        if masks_resamp is not None:
            cls_mask_resamp = y_resamp == cls
            cls_masks_resamp = masks_resamp[cls_mask_resamp]
            
            total_positions = cls_masks_resamp.numel()
            valid_positions = cls_masks_resamp.sum().item()
            valid_ratio = valid_positions / total_positions if total_positions > 0 else 0
            seq_valid_lengths = cls_masks_resamp.sum(dim=1).cpu().numpy() if torch.is_tensor(cls_masks_resamp) else cls_masks_resamp.sum(axis=1)
            
            resamp_stats[cls] = {
                'valid_ratio': valid_ratio,
                'total_samples': cls_mask_resamp.sum().item() if torch.is_tensor(cls_mask_resamp) else cls_mask_resamp.sum(),
                'mean_valid_length': np.mean(seq_valid_lengths),
                'std_valid_length': np.std(seq_valid_lengths),
                'seq_lengths': seq_valid_lengths
            }
        else:
            # 如果没有mask，假设所有数据都有效
            cls_count = (y_resamp == cls).sum().item() if torch.is_tensor(y_resamp) else (y_resamp == cls).sum()
            seq_len = masks_orig.shape[1]
            
            resamp_stats[cls] = {
                'valid_ratio': 1.0,
                'total_samples': cls_count,
                'mean_valid_length': seq_len,
                'std_valid_length': 0.0,
                'seq_lengths': np.full(cls_count, seq_len)
            }
    
    # 创建可视化
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # 1. 有效数据比例对比
    classes = [f'Class {cls}' for cls in sorted(unique_classes)]
    orig_ratios = [orig_stats[cls]['valid_ratio'] for cls in sorted(unique_classes)]
    resamp_ratios = [resamp_stats[cls]['valid_ratio'] for cls in sorted(unique_classes)]
    
    x = np.arange(len(classes))
    width = 0.35
    
    axes[0, 0].bar(x - width/2, orig_ratios, width, label='原始', alpha=0.8, color='#FF6B6B')
    axes[0, 0].bar(x + width/2, resamp_ratios, width, label='重采样', alpha=0.8, color='#4ECDC4')
    axes[0, 0].set_xlabel('类别')
    axes[0, 0].set_ylabel('有效数据比例')
    axes[0, 0].set_title('有效数据比例对比', fontweight='bold')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(classes)
    axes[0, 0].legend()
    axes[0, 0].grid(axis='y', alpha=0.3)
    
    # 2. 平均有效序列长度对比
    orig_lengths = [orig_stats[cls]['mean_valid_length'] for cls in sorted(unique_classes)]
    resamp_lengths = [resamp_stats[cls]['mean_valid_length'] for cls in sorted(unique_classes)]
    
    axes[0, 1].bar(x - width/2, orig_lengths, width, label='原始', alpha=0.8, color='#FF6B6B')
    axes[0, 1].bar(x + width/2, resamp_lengths, width, label='重采样', alpha=0.8, color='#4ECDC4')
    axes[0, 1].set_xlabel('类别')
    axes[0, 1].set_ylabel('平均有效长度')
    axes[0, 1].set_title('平均有效序列长度对比', fontweight='bold')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(classes)
    axes[0, 1].legend()
    axes[0, 1].grid(axis='y', alpha=0.3)
    
    # 3. 样本数量对比
    orig_counts = [orig_stats[cls]['total_samples'] for cls in sorted(unique_classes)]
    resamp_counts = [resamp_stats[cls]['total_samples'] for cls in sorted(unique_classes)]
    
    axes[1, 0].bar(x - width/2, orig_counts, width, label='原始', alpha=0.8, color='#FF6B6B')
    axes[1, 0].bar(x + width/2, resamp_counts, width, label='重采样', alpha=0.8, color='#4ECDC4')
    axes[1, 0].set_xlabel('类别')
    axes[1, 0].set_ylabel('样本数量')
    axes[1, 0].set_title('样本数量对比', fontweight='bold')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(classes)
    axes[1, 0].legend()
    axes[1, 0].grid(axis='y', alpha=0.3)
    
    # 4. 有效长度分布（选择样本最多的类别）
    max_samples_cls = max(unique_classes, key=lambda cls: orig_stats[cls]['total_samples'])
    
    orig_dist = orig_stats[max_samples_cls]['seq_lengths']
    resamp_dist = resamp_stats[max_samples_cls]['seq_lengths']
    
    axes[1, 1].hist(orig_dist, bins=20, alpha=0.7, label='原始', color='#FF6B6B', density=True)
    axes[1, 1].hist(resamp_dist, bins=20, alpha=0.7, label='重采样', color='#4ECDC4', density=True)
    axes[1, 1].set_xlabel('有效序列长度')
    axes[1, 1].set_ylabel('密度')
    axes[1, 1].set_title(f'Class {max_samples_cls} 有效长度分布', fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)
    
    plt.suptitle('MACHO数据集 - Mask使用统计分析', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, 'mask_statistics_analysis.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Mask统计分析图已保存: {save_path}")
    
    # 生成统计报告
    report_path = os.path.join(output_dir, 'mask_quality_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("MACHO数据集重采样Mask质量报告\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*60 + "\n\n")
        
        for cls in sorted(unique_classes):
            f.write(f"Class {cls}:\n")
            f.write(f"  原始数据:\n")
            f.write(f"    样本数: {orig_stats[cls]['total_samples']}\n")
            f.write(f"    有效数据比例: {orig_stats[cls]['valid_ratio']:.3f}\n")
            f.write(f"    平均有效长度: {orig_stats[cls]['mean_valid_length']:.1f}\n")
            f.write(f"    长度标准差: {orig_stats[cls]['std_valid_length']:.1f}\n")
            
            f.write(f"  重采样数据:\n")
            f.write(f"    样本数: {resamp_stats[cls]['total_samples']}\n")
            f.write(f"    有效数据比例: {resamp_stats[cls]['valid_ratio']:.3f}\n")
            f.write(f"    平均有效长度: {resamp_stats[cls]['mean_valid_length']:.1f}\n")
            f.write(f"    长度标准差: {resamp_stats[cls]['std_valid_length']:.1f}\n")
            f.write(f"  保持率: {resamp_stats[cls]['valid_ratio']/orig_stats[cls]['valid_ratio']:.3f}\n\n")
    
    print(f"质量报告已保存: {report_path}")
    return save_path

if __name__ == "__main__":
    quick_analysis()