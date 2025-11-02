#!/usr/bin/env python3
"""
对真实数据集执行混合重采样
生成平衡的数据集用于训练
"""

import sys
import os
sys.path.append('/root/autodl-tmp/lnsde-contiformer')

import torch
import numpy as np
from utils.resampling import HybridResampler, save_resampled_data
from utils import create_dataloaders
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm


def configure_chinese_font():
    """配置中文字体显示"""
    try:
        fm.fontManager.addfont('/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc')
        fm.fontManager.addfont('/usr/share/fonts/truetype/wqy/wqy-microhei.ttc')
    except:
        pass
    plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'WenQuanYi Micro Hei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False


def resample_dataset(dataset_name, batch_size=64, apply_enn=True):  # 默认启用ENN
    """
    对指定数据集执行重采样
    
    Args:
        dataset_name: 数据集名称 ('ASAS', 'LINEAR', 'MACHO')
        batch_size: 批大小
        apply_enn: 是否应用ENN清理（默认True，实现混合重采样）
    """
    print(f"\n处理数据集: {dataset_name}")
    print("="*60)
    
    # 确定数据路径 - 使用fixed版本
    data_path = f'/root/autodl-fs/lnsde-contiformer/data/{dataset_name}_folded_512_fixed.pkl'
    
    # 检查文件是否存在
    import os
    if not os.path.exists(data_path):
        print(f"数据文件不存在: {data_path}")
        return None, None
    
    # 加载数据
    train_loader, test_loader, num_classes = create_dataloaders(
        data_path=data_path,
        batch_size=batch_size,
        num_workers=4,
        random_seed=535411460
    )
    
    # 提取训练数据
    all_features = []
    all_labels = []
    all_times = []
    all_masks = []
    
    print("提取训练数据...")
    for batch_idx, batch in enumerate(train_loader):
        features = batch['features']  # (batch, seq_len, 3)
        labels = batch['labels']      # (batch,) - 注意是'labels'
        times = batch['times']        # (batch, seq_len)
        masks = batch['mask']         # (batch, seq_len) - 注意是'mask'不是'masks'
        
        all_features.append(features)
        all_labels.append(labels)
        all_times.append(times)
        all_masks.append(masks)
    
    # 合并所有批次
    X = torch.cat(all_features, dim=0)
    y = torch.cat(all_labels, dim=0)
    times = torch.cat(all_times, dim=0)
    masks = torch.cat(all_masks, dim=0)
    
    print(f"原始数据形状: X={X.shape}, y={y.shape}")
    
    # 创建重采样器
    resampler = HybridResampler(
        smote_k_neighbors=5,
        enn_n_neighbors=3,
        sampling_strategy='balanced',  # 完全平衡
        apply_enn=apply_enn,  # 启用ENN欠采样
        random_state=535411460
    )
    
    # 执行重采样
    X_resampled, y_resampled, times_resampled, masks_resampled = resampler.fit_resample(
        X, y, times, masks
    )
    
    print(f"重采样后数据形状: X={X_resampled.shape}, y={y_resampled.shape}")
    
    # 可视化分布
    configure_chinese_font()
    fig_path = f'/root/autodl-tmp/lnsde-contiformer/results/pics/{dataset_name}/resampling_distribution.png'
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)
    resampler.visualize_distribution(save_path=fig_path)
    
    # 保存重采样数据
    save_path = save_resampled_data(
        X_resampled, y_resampled, times_resampled, masks_resampled,
        dataset_name=dataset_name,
        save_dir='/root/autodl-fs/lnsde-contiformer/data/resampled'
    )
    
    return save_path, resampler.stats_


def resample_all_datasets():
    """对所有数据集执行重采样"""
    datasets = ['ASAS', 'LINEAR', 'MACHO']
    results = {}
    
    for dataset in datasets:
        try:
            save_path, stats = resample_dataset(
                dataset_name=dataset,
                batch_size=64,
                apply_enn=True  # 启用ENN欠采样
            )
            results[dataset] = {
                'save_path': save_path,
                'stats': stats,
                'success': True
            }
        except Exception as e:
            print(f"处理{dataset}时出错: {e}")
            results[dataset] = {
                'error': str(e),
                'success': False
            }
    
    # 生成汇总报告
    print("\n" + "="*60)
    print("重采样汇总报告")
    print("="*60)
    
    for dataset, result in results.items():
        print(f"\n{dataset}:")
        if result['success']:
            stats = result['stats']
            print(f"  原始分布: {stats.get('original', {})}")
            print(f"  最终分布: {stats.get('final', {})}")
            print(f"  保存路径: {result['save_path']}")
        else:
            print(f"  错误: {result['error']}")
    
    return results


def create_comparison_plot():
    """创建所有数据集的重采样对比图"""
    datasets = ['ASAS', 'LINEAR', 'MACHO']
    
    configure_chinese_font()
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    
    for idx, dataset in enumerate(datasets):
        try:
            # 重新采样以获取统计信息
            resampler = HybridResampler(
                smote_k_neighbors=5,
                enn_n_neighbors=3,
                sampling_strategy='balanced',
                apply_enn=True,  # 启用ENN
                random_state=535411460
            )
            
            # 加载数据 - 使用fixed版本
            data_path = f'/root/autodl-fs/lnsde-contiformer/data/{dataset}_folded_512_fixed.pkl'
            
            if not os.path.exists(data_path):
                print(f"数据文件不存在: {data_path}")
                continue
                
            train_loader, _, _ = create_dataloaders(
                data_path=data_path,
                batch_size=64,
                num_workers=4,
                random_seed=535411460
            )
            
            # 提取数据
            all_features = []
            all_labels = []
            
            for batch in train_loader:
                all_features.append(batch['features'])
                all_labels.append(batch['labels'])  # 注意是'labels'
                
            X = torch.cat(all_features, dim=0)
            y = torch.cat(all_labels, dim=0)
            
            # 执行重采样
            X_res, y_res, _, _ = resampler.fit_resample(X, y)
            
            # 绘制三个阶段
            stages = ['original', 'after_smote', 'final']
            titles = ['原始分布', 'SMOTE后', '最终分布']
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
            
            for j, (stage, title, color) in enumerate(zip(stages, titles, colors)):
                ax = axes[idx, j]
                if stage in resampler.stats_:
                    data = resampler.stats_[stage]
                    classes = list(data.keys())
                    counts = list(data.values())
                    
                    bars = ax.bar(classes, counts, color=color, alpha=0.7, edgecolor='black')
                    
                    if idx == 0:
                        ax.set_title(title, fontsize=12, fontweight='bold')
                    if j == 0:
                        ax.set_ylabel(f'{dataset}\n样本数', fontsize=11)
                    if idx == 2:
                        ax.set_xlabel('类别', fontsize=11)
                    
                    ax.grid(axis='y', alpha=0.3)
                    
                    # 添加数值标签
                    for bar, count in zip(bars, counts):
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height,
                               f'{int(count)}', ha='center', va='bottom', fontsize=9)
                    
                    # 计算不平衡率
                    if counts:
                        max_count = max(counts)
                        min_count = min(counts)
                        imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
                        ax.text(0.5, 0.95, f'IR: {imbalance_ratio:.2f}',
                               transform=ax.transAxes, ha='center', va='top',
                               fontsize=9, bbox=dict(boxstyle='round', 
                                                   facecolor='wheat', alpha=0.5))
                    
        except Exception as e:
            print(f"处理{dataset}时出错: {e}")
            for j in range(3):
                axes[idx, j].text(0.5, 0.5, f'Error: {str(e)[:20]}...',
                                ha='center', va='center',
                                transform=axes[idx, j].transAxes)
    
    plt.suptitle('三个数据集的混合重采样效果对比', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    save_path = '/root/autodl-tmp/lnsde-contiformer/results/pics/all_datasets_resampling.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n对比图已保存至: {save_path}")
    
    plt.show()
    
    return fig


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='执行数据集重采样')
    parser.add_argument('--dataset', type=str, default='all',
                       choices=['ASAS', 'LINEAR', 'MACHO', 'all'],
                       help='要重采样的数据集')
    parser.add_argument('--apply_enn', action='store_true',
                       help='是否应用ENN清理（可能过度欠采样）')
    parser.add_argument('--comparison', action='store_true',
                       help='生成对比图')
    
    args = parser.parse_args()
    
    if args.comparison:
        create_comparison_plot()
    elif args.dataset == 'all':
        resample_all_datasets()
    else:
        resample_dataset(
            dataset_name=args.dataset,
            apply_enn=args.apply_enn
        )
    
    print("\n重采样完成！")