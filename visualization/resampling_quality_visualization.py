#!/usr/bin/env python3
"""
重采样质量可视化脚本 - 展示每个类别每个数据集的重采样曲线质量
包括：
1. 原始vs重采样的有效数据曲线对比
2. 时序特征保持性分析
3. 类别间差异可视化
4. mask使用分析
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from collections import Counter, defaultdict
import torch
from datetime import datetime
import pickle
import pandas as pd

# 添加项目路径
sys.path.append('/root/autodl-tmp/lnsde-contiformer')

from utils import create_dataloaders, get_dataset_specific_params, setup_dataset_mapping
from utils.resampling import HybridResampler, configure_chinese_font
import warnings
warnings.filterwarnings('ignore')

# 配置中文字体
configure_chinese_font()

class ResamplingQualityVisualizer:
    """重采样质量可视化器"""
    
    def __init__(self, output_dir='/root/autodl-tmp/lnsde-contiformer/results/pics/resampling'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # 设置颜色方案
        self.colors = {
            0: '#FF6B6B',  # 红色
            1: '#4ECDC4',  # 青色  
            2: '#45B7D1',  # 蓝色
            3: '#96CEB4',  # 绿色
            4: '#FECA57',  # 黄色
            5: '#FF9FF3',  # 粉色
            6: '#54A0FF',  # 亮蓝色
            7: '#5F27CD',  # 紫色
        }
        
    def load_dataset(self, dataset_num, batch_size=64):
        """加载指定数据集"""
        # 创建虚拟args对象
        class Args:
            def __init__(self):
                self.dataset = dataset_num
                self.batch_size = batch_size
                self.seed = 42
                self.model_type = 2  # 添加缺少的属性
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
        
        # 创建数据加载器
        train_loader, test_loader, num_classes = create_dataloaders(
            data_path=args.data_path,
            batch_size=batch_size,
            num_workers=4,
            random_seed=42
        )
        
        return train_loader, test_loader, num_classes, args.dataset_name
    
    def extract_data_from_loader(self, loader):
        """从数据加载器中提取所有数据"""
        all_features = []
        all_labels = []
        all_times = []
        all_masks = []
        all_periods = []
        
        for batch in loader:
            # 处理不同的数据格式
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
                raise KeyError("批次数据中未找到'features'或'x'键")
            
            all_features.append(features)
            all_labels.append(batch['labels'])
            all_times.append(times)
            all_masks.append(batch.get('mask', torch.ones_like(times, dtype=torch.bool)))
            all_periods.append(batch.get('periods', torch.zeros(features.shape[0])))
        
        # 合并数据
        X = torch.cat(all_features, dim=0)
        y = torch.cat(all_labels, dim=0)
        times = torch.cat(all_times, dim=0)
        masks = torch.cat(all_masks, dim=0)
        periods = torch.cat(all_periods, dim=0)
        
        return X, y, times, masks, periods
    
    def analyze_effective_data_ratio(self, masks, y):
        """分析每个类别的有效数据比例"""
        class_mask_stats = {}
        unique_classes = torch.unique(y).cpu().numpy()
        
        for cls in unique_classes:
            cls_mask = y == cls
            cls_masks = masks[cls_mask]
            
            # 计算有效数据比例
            total_positions = cls_masks.numel()
            valid_positions = cls_masks.sum().item()
            valid_ratio = valid_positions / total_positions if total_positions > 0 else 0
            
            # 计算每个序列的有效长度分布
            seq_valid_lengths = cls_masks.sum(dim=1).cpu().numpy()
            
            class_mask_stats[cls] = {
                'valid_ratio': valid_ratio,
                'total_samples': cls_mask.sum().item(),
                'seq_valid_lengths': seq_valid_lengths,
                'mean_valid_length': np.mean(seq_valid_lengths),
                'std_valid_length': np.std(seq_valid_lengths)
            }
        
        return class_mask_stats
    
    def visualize_lightcurves_comparison(self, X_orig, y_orig, times_orig, masks_orig, 
                                       X_resamp, y_resamp, times_resamp, masks_resamp,
                                       dataset_name, num_samples=3):
        """可视化原始vs重采样的光变曲线对比"""
        unique_classes = torch.unique(y_orig).cpu().numpy()
        n_classes = len(unique_classes)
        
        fig, axes = plt.subplots(n_classes, num_samples * 2, 
                               figsize=(num_samples * 4, n_classes * 3))
        if n_classes == 1:
            axes = axes.reshape(1, -1)
        
        # 为每个类别选择样本
        for row, cls in enumerate(unique_classes):
            # 原始样本
            orig_indices = torch.where(y_orig == cls)[0][:num_samples]
            # 重采样样本（包括合成的）
            resamp_indices = torch.where(y_resamp == cls)[0][:num_samples]
            
            for col in range(num_samples):
                # 原始样本
                if col < len(orig_indices):
                    idx = orig_indices[col]
                    features = X_orig[idx]
                    mask = masks_orig[idx]
                    times = times_orig[idx]
                    
                    # 只绘制有效数据点
                    valid_mask = mask.cpu().numpy()
                    valid_times = times[valid_mask].cpu().numpy()
                    valid_mags = features[:, 1][valid_mask].cpu().numpy()  # 星等
                    
                    axes[row, col * 2].plot(valid_times, valid_mags, 'o-', 
                                          color=self.colors[cls % len(self.colors)], 
                                          alpha=0.7, markersize=3, linewidth=1)
                    axes[row, col * 2].set_title(f'原始 Class {cls} #{col+1}', fontsize=10)
                    axes[row, col * 2].grid(True, alpha=0.3)
                    axes[row, col * 2].set_xlabel('时间')
                    axes[row, col * 2].set_ylabel('星等')
                
                # 重采样样本
                if col < len(resamp_indices):
                    idx = resamp_indices[col]
                    features = X_resamp[idx]
                    mask = masks_resamp[idx] if masks_resamp is not None else torch.ones_like(times_resamp[idx], dtype=torch.bool)
                    times = times_resamp[idx] if times_resamp is not None else torch.linspace(0, 1, features.shape[0])
                    
                    # 只绘制有效数据点
                    if torch.is_tensor(mask):
                        valid_mask = mask.cpu().numpy()
                    else:
                        valid_mask = mask
                    
                    if torch.is_tensor(times):
                        valid_times = times[valid_mask].cpu().numpy()
                    else:
                        valid_times = times[valid_mask]
                        
                    if torch.is_tensor(features):
                        valid_mags = features[:, 1][valid_mask].cpu().numpy()
                    else:
                        valid_mags = features[:, 1][valid_mask]
                    
                    axes[row, col * 2 + 1].plot(valid_times, valid_mags, 'o-', 
                                               color=self.colors[cls % len(self.colors)], 
                                               alpha=0.7, markersize=3, linewidth=1)
                    axes[row, col * 2 + 1].set_title(f'重采样 Class {cls} #{col+1}', fontsize=10)
                    axes[row, col * 2 + 1].grid(True, alpha=0.3)
                    axes[row, col * 2 + 1].set_xlabel('时间')
                    axes[row, col * 2 + 1].set_ylabel('星等')
        
        plt.suptitle(f'{dataset_name} 原始vs重采样光变曲线对比', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        save_path = os.path.join(self.output_dir, f'{dataset_name}_lightcurves_comparison.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"光变曲线对比图已保存: {save_path}")
        return save_path
    
    def visualize_mask_statistics(self, masks_orig, y_orig, masks_resamp, y_resamp, dataset_name):
        """可视化mask使用统计"""
        # 分析原始数据
        orig_stats = self.analyze_effective_data_ratio(masks_orig, y_orig)
        
        # 分析重采样数据
        if masks_resamp is not None:
            resamp_stats = self.analyze_effective_data_ratio(masks_resamp, y_resamp)
        else:
            # 如果没有mask，假设所有数据都有效
            resamp_stats = {}
            unique_classes = torch.unique(y_resamp).cpu().numpy()
            for cls in unique_classes:
                cls_count = (y_resamp == cls).sum().item()
                resamp_stats[cls] = {
                    'valid_ratio': 1.0,
                    'total_samples': cls_count,
                    'seq_valid_lengths': np.full(cls_count, masks_orig.shape[1]),
                    'mean_valid_length': masks_orig.shape[1],
                    'std_valid_length': 0.0
                }
        
        # 创建可视化
        unique_classes = list(set(orig_stats.keys()) | set(resamp_stats.keys()))
        n_classes = len(unique_classes)
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # 1. 有效数据比例对比
        classes = []
        orig_ratios = []
        resamp_ratios = []
        
        for cls in sorted(unique_classes):
            classes.append(f'Class {cls}')
            orig_ratios.append(orig_stats.get(cls, {}).get('valid_ratio', 0))
            resamp_ratios.append(resamp_stats.get(cls, {}).get('valid_ratio', 0))
        
        x = np.arange(len(classes))
        width = 0.35
        
        axes[0, 0].bar(x - width/2, orig_ratios, width, label='原始', alpha=0.7, color='#FF6B6B')
        axes[0, 0].bar(x + width/2, resamp_ratios, width, label='重采样', alpha=0.7, color='#4ECDC4')
        axes[0, 0].set_xlabel('类别')
        axes[0, 0].set_ylabel('有效数据比例')
        axes[0, 0].set_title('有效数据比例对比')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(classes)
        axes[0, 0].legend()
        axes[0, 0].grid(axis='y', alpha=0.3)
        
        # 2. 平均有效序列长度对比
        orig_lengths = []
        resamp_lengths = []
        
        for cls in sorted(unique_classes):
            orig_lengths.append(orig_stats.get(cls, {}).get('mean_valid_length', 0))
            resamp_lengths.append(resamp_stats.get(cls, {}).get('mean_valid_length', 0))
        
        axes[0, 1].bar(x - width/2, orig_lengths, width, label='原始', alpha=0.7, color='#FF6B6B')
        axes[0, 1].bar(x + width/2, resamp_lengths, width, label='重采样', alpha=0.7, color='#4ECDC4')
        axes[0, 1].set_xlabel('类别')
        axes[0, 1].set_ylabel('平均有效长度')
        axes[0, 1].set_title('平均有效序列长度对比')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(classes)
        axes[0, 1].legend()
        axes[0, 1].grid(axis='y', alpha=0.3)
        
        # 3. 样本数量对比
        orig_counts = []
        resamp_counts = []
        
        for cls in sorted(unique_classes):
            orig_counts.append(orig_stats.get(cls, {}).get('total_samples', 0))
            resamp_counts.append(resamp_stats.get(cls, {}).get('total_samples', 0))
        
        axes[1, 0].bar(x - width/2, orig_counts, width, label='原始', alpha=0.7, color='#FF6B6B')
        axes[1, 0].bar(x + width/2, resamp_counts, width, label='重采样', alpha=0.7, color='#4ECDC4')
        axes[1, 0].set_xlabel('类别')
        axes[1, 0].set_ylabel('样本数量')
        axes[1, 0].set_title('样本数量对比')
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(classes)
        axes[1, 0].legend()
        axes[1, 0].grid(axis='y', alpha=0.3)
        
        # 4. 有效长度分布（选择一个代表性类别）
        if unique_classes:
            representative_class = unique_classes[0]
            
            orig_dist = orig_stats.get(representative_class, {}).get('seq_valid_lengths', [])
            resamp_dist = resamp_stats.get(representative_class, {}).get('seq_valid_lengths', [])
            
            if len(orig_dist) > 0:
                axes[1, 1].hist(orig_dist, bins=20, alpha=0.7, label='原始', color='#FF6B6B', density=True)
            if len(resamp_dist) > 0:
                axes[1, 1].hist(resamp_dist, bins=20, alpha=0.7, label='重采样', color='#4ECDC4', density=True)
            
            axes[1, 1].set_xlabel('有效序列长度')
            axes[1, 1].set_ylabel('密度')
            axes[1, 1].set_title(f'Class {representative_class} 有效长度分布')
            axes[1, 1].legend()
            axes[1, 1].grid(alpha=0.3)
        
        plt.suptitle(f'{dataset_name} Mask使用统计分析', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        save_path = os.path.join(self.output_dir, f'{dataset_name}_mask_statistics.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Mask统计图已保存: {save_path}")
        return save_path, orig_stats, resamp_stats
    
    def visualize_time_series_features(self, X_orig, y_orig, masks_orig, 
                                     X_resamp, y_resamp, masks_resamp, dataset_name):
        """可视化时序特征保持性"""
        unique_classes = torch.unique(y_orig).cpu().numpy()
        n_classes = len(unique_classes)
        
        fig, axes = plt.subplots(n_classes, 3, figsize=(15, n_classes * 4))
        if n_classes == 1:
            axes = axes.reshape(1, -1)
        
        for row, cls in enumerate(unique_classes):
            # 获取类别数据
            orig_indices = torch.where(y_orig == cls)[0]
            resamp_indices = torch.where(y_resamp == cls)[0]
            
            # 计算原始数据特征
            orig_features = X_orig[orig_indices]
            orig_masks = masks_orig[orig_indices]
            
            # 提取有效数据点的星等
            orig_mags = []
            for i in range(len(orig_indices)):
                mask = orig_masks[i]
                valid_mags = orig_features[i, mask, 1].cpu().numpy()  # 星等
                orig_mags.extend(valid_mags)
            orig_mags = np.array(orig_mags)
            
            # 计算重采样数据特征
            resamp_features = X_resamp[resamp_indices]
            if masks_resamp is not None:
                resamp_masks = masks_resamp[resamp_indices]
            else:
                resamp_masks = torch.ones_like(resamp_features[:, :, 0], dtype=torch.bool)
            
            resamp_mags = []
            for i in range(len(resamp_indices)):
                mask = resamp_masks[i]
                if torch.is_tensor(resamp_features):
                    valid_mags = resamp_features[i, mask, 1].cpu().numpy()
                else:
                    valid_mags = resamp_features[i, mask, 1]
                resamp_mags.extend(valid_mags)
            resamp_mags = np.array(resamp_mags)
            
            # 1. 星等分布对比
            axes[row, 0].hist(orig_mags, bins=30, alpha=0.7, label='原始', 
                             color=self.colors[cls % len(self.colors)], density=True)
            axes[row, 0].hist(resamp_mags, bins=30, alpha=0.7, label='重采样', 
                             color='gray', density=True)
            axes[row, 0].set_xlabel('星等')
            axes[row, 0].set_ylabel('密度')
            axes[row, 0].set_title(f'Class {cls} 星等分布')
            axes[row, 0].legend()
            axes[row, 0].grid(alpha=0.3)
            
            # 2. 统计量对比
            stats_names = ['均值', '标准差', '偏度', '峰度']
            orig_stats = [
                np.mean(orig_mags),
                np.std(orig_mags),
                self._safe_skewness(orig_mags),
                self._safe_kurtosis(orig_mags)
            ]
            resamp_stats = [
                np.mean(resamp_mags),
                np.std(resamp_mags),
                self._safe_skewness(resamp_mags),
                self._safe_kurtosis(resamp_mags)
            ]
            
            x = np.arange(len(stats_names))
            width = 0.35
            
            axes[row, 1].bar(x - width/2, orig_stats, width, label='原始', 
                           color=self.colors[cls % len(self.colors)], alpha=0.7)
            axes[row, 1].bar(x + width/2, resamp_stats, width, label='重采样', 
                           color='gray', alpha=0.7)
            axes[row, 1].set_xlabel('统计量')
            axes[row, 1].set_ylabel('值')
            axes[row, 1].set_title(f'Class {cls} 统计量对比')
            axes[row, 1].set_xticks(x)
            axes[row, 1].set_xticklabels(stats_names)
            axes[row, 1].legend()
            axes[row, 1].grid(axis='y', alpha=0.3)
            
            # 3. 变异性分析（星等变化范围）
            orig_ranges = []
            resamp_ranges = []
            
            # 计算每个序列的变异范围
            for i in range(min(100, len(orig_indices))):  # 限制样本数避免计算过慢
                mask = orig_masks[i]
                valid_mags = orig_features[i, mask, 1].cpu().numpy()
                if len(valid_mags) > 1:
                    orig_ranges.append(np.max(valid_mags) - np.min(valid_mags))
            
            for i in range(min(100, len(resamp_indices))):
                mask = resamp_masks[i]
                if torch.is_tensor(resamp_features):
                    valid_mags = resamp_features[i, mask, 1].cpu().numpy()
                else:
                    valid_mags = resamp_features[i, mask, 1]
                if len(valid_mags) > 1:
                    resamp_ranges.append(np.max(valid_mags) - np.min(valid_mags))
            
            if orig_ranges and resamp_ranges:
                axes[row, 2].boxplot([orig_ranges, resamp_ranges], labels=['原始', '重采样'])
                axes[row, 2].set_ylabel('星等变异范围')
                axes[row, 2].set_title(f'Class {cls} 变异性对比')
                axes[row, 2].grid(alpha=0.3)
        
        plt.suptitle(f'{dataset_name} 时序特征保持性分析', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        save_path = os.path.join(self.output_dir, f'{dataset_name}_time_series_features.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"时序特征分析图已保存: {save_path}")
        return save_path
    
    def _safe_skewness(self, data):
        """安全计算偏度"""
        try:
            from scipy.stats import skew
            return skew(data)
        except:
            return 0.0
    
    def _safe_kurtosis(self, data):
        """安全计算峰度"""
        try:
            from scipy.stats import kurtosis
            return kurtosis(data)
        except:
            return 0.0
    
    def analyze_dataset(self, dataset_num):
        """分析单个数据集的重采样质量"""
        print(f"\n{'='*60}")
        print(f"分析数据集 {dataset_num}")
        print(f"{'='*60}")
        
        # 加载数据集
        train_loader, test_loader, num_classes, dataset_name = self.load_dataset(dataset_num)
        
        # 提取原始数据
        print("提取原始数据...")
        X_orig, y_orig, times_orig, masks_orig, periods_orig = self.extract_data_from_loader(train_loader)
        
        print(f"原始数据统计:")
        print(f"  样本数: {len(y_orig)}")
        print(f"  类别分布: {dict(Counter(y_orig.cpu().numpy()))}")
        print(f"  序列长度: {X_orig.shape[1]}")
        print(f"  特征维度: {X_orig.shape[2]}")
        
        # 执行重采样
        print("\n执行重采样...")
        resampler = HybridResampler(
            smote_k_neighbors=5,
            enn_n_neighbors=3,
            sampling_strategy='balanced',
            apply_enn=False,  # 先不进行欠采样，保持更多样本
            random_state=42
        )
        
        X_resamp, y_resamp, times_resamp, masks_resamp = resampler.fit_resample(
            X_orig, y_orig, times_orig, masks_orig
        )
        
        print(f"重采样后数据统计:")
        print(f"  样本数: {len(y_resamp)}")
        print(f"  类别分布: {dict(Counter(y_resamp.cpu().numpy() if torch.is_tensor(y_resamp) else y_resamp))}")
        
        # 可视化分析
        print("\n生成可视化分析...")
        
        # 1. 光变曲线对比
        lc_path = self.visualize_lightcurves_comparison(
            X_orig, y_orig, times_orig, masks_orig,
            X_resamp, y_resamp, times_resamp, masks_resamp,
            dataset_name, num_samples=3
        )
        
        # 2. Mask统计分析
        mask_path, orig_stats, resamp_stats = self.visualize_mask_statistics(
            masks_orig, y_orig, masks_resamp, y_resamp, dataset_name
        )
        
        # 3. 时序特征保持性
        features_path = self.visualize_time_series_features(
            X_orig, y_orig, masks_orig,
            X_resamp, y_resamp, masks_resamp, dataset_name
        )
        
        # 4. 保存重采样分布图
        dist_path = os.path.join(self.output_dir, f'{dataset_name}_resampling_distribution.png')
        resampler.visualize_distribution(save_path=dist_path)
        
        # 生成质量报告
        report_path = self.generate_quality_report(
            dataset_name, orig_stats, resamp_stats, 
            len(y_orig), len(y_resamp), num_classes
        )
        
        print(f"\n{dataset_name} 分析完成!")
        print(f"输出文件:")
        print(f"  - 光变曲线对比: {lc_path}")
        print(f"  - Mask统计: {mask_path}")
        print(f"  - 时序特征: {features_path}")
        print(f"  - 分布对比: {dist_path}")
        print(f"  - 质量报告: {report_path}")
        
        return {
            'dataset_name': dataset_name,
            'paths': {
                'lightcurves': lc_path,
                'mask_stats': mask_path,
                'time_features': features_path,
                'distribution': dist_path,
                'report': report_path
            },
            'stats': {
                'original': orig_stats,
                'resampled': resamp_stats,
                'sample_counts': {
                    'original': len(y_orig),
                    'resampled': len(y_resamp)
                }
            }
        }
    
    def generate_quality_report(self, dataset_name, orig_stats, resamp_stats, 
                               orig_count, resamp_count, num_classes):
        """生成重采样质量报告"""
        report_path = os.path.join(self.output_dir, f'{dataset_name}_quality_report.txt')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"重采样质量报告 - {dataset_name}\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*80 + "\n\n")
            
            # 基本统计
            f.write("1. 基本统计\n")
            f.write("-"*40 + "\n")
            f.write(f"类别数量: {num_classes}\n")
            f.write(f"原始样本总数: {orig_count}\n")
            f.write(f"重采样后样本总数: {resamp_count}\n")
            f.write(f"增长倍数: {resamp_count/orig_count:.2f}x\n\n")
            
            # 按类别分析
            f.write("2. 按类别分析\n")
            f.write("-"*40 + "\n")
            all_classes = set(orig_stats.keys()) | set(resamp_stats.keys())
            
            for cls in sorted(all_classes):
                f.write(f"\nClass {cls}:\n")
                
                orig_data = orig_stats.get(cls, {})
                resamp_data = resamp_stats.get(cls, {})
                
                f.write(f"  样本数:\n")
                f.write(f"    原始: {orig_data.get('total_samples', 0)}\n")
                f.write(f"    重采样: {resamp_data.get('total_samples', 0)}\n")
                
                f.write(f"  有效数据比例:\n")
                f.write(f"    原始: {orig_data.get('valid_ratio', 0):.3f}\n")
                f.write(f"    重采样: {resamp_data.get('valid_ratio', 0):.3f}\n")
                
                f.write(f"  平均有效序列长度:\n")
                f.write(f"    原始: {orig_data.get('mean_valid_length', 0):.1f}\n")
                f.write(f"    重采样: {resamp_data.get('mean_valid_length', 0):.1f}\n")
                
                f.write(f"  序列长度标准差:\n")
                f.write(f"    原始: {orig_data.get('std_valid_length', 0):.1f}\n")
                f.write(f"    重采样: {resamp_data.get('std_valid_length', 0):.1f}\n")
            
            # 质量评估
            f.write("\n3. 质量评估\n")
            f.write("-"*40 + "\n")
            
            # 计算平衡性改进
            orig_counts = [orig_stats[cls]['total_samples'] for cls in sorted(orig_stats.keys())]
            resamp_counts = [resamp_stats[cls]['total_samples'] for cls in sorted(resamp_stats.keys())]
            
            orig_imbalance = max(orig_counts) / min(orig_counts) if min(orig_counts) > 0 else float('inf')
            resamp_imbalance = max(resamp_counts) / min(resamp_counts) if min(resamp_counts) > 0 else float('inf')
            
            f.write(f"类别不平衡率:\n")
            f.write(f"  重采样前: {orig_imbalance:.2f}\n")
            f.write(f"  重采样后: {resamp_imbalance:.2f}\n")
            f.write(f"  改进程度: {orig_imbalance/resamp_imbalance:.2f}x\n\n")
            
            # 有效数据保持性
            avg_orig_valid = np.mean([orig_stats[cls]['valid_ratio'] for cls in orig_stats])
            avg_resamp_valid = np.mean([resamp_stats[cls]['valid_ratio'] for cls in resamp_stats])
            
            f.write(f"有效数据保持性:\n")
            f.write(f"  原始平均有效比例: {avg_orig_valid:.3f}\n")
            f.write(f"  重采样平均有效比例: {avg_resamp_valid:.3f}\n")
            f.write(f"  保持率: {avg_resamp_valid/avg_orig_valid:.3f}\n\n")
            
            # 结论
            f.write("4. 结论\n")
            f.write("-"*40 + "\n")
            if resamp_imbalance < 1.5:
                f.write("✅ 类别平衡性: 优秀\n")
            elif resamp_imbalance < 2.0:
                f.write("✅ 类别平衡性: 良好\n")
            else:
                f.write("⚠️  类别平衡性: 需要改进\n")
                
            if avg_resamp_valid/avg_orig_valid > 0.8:
                f.write("✅ 有效数据保持: 优秀\n")
            elif avg_resamp_valid/avg_orig_valid > 0.6:
                f.write("✅ 有效数据保持: 良好\n")
            else:
                f.write("⚠️  有效数据保持: 需要改进\n")
        
        return report_path
    
    def analyze_all_datasets(self):
        """分析所有数据集"""
        datasets = [1, 2, 3]  # ASAS, LINEAR, MACHO
        results = {}
        
        print("开始分析所有数据集的重采样质量...")
        
        for dataset_num in datasets:
            try:
                result = self.analyze_dataset(dataset_num)
                results[result['dataset_name']] = result
            except Exception as e:
                print(f"分析数据集 {dataset_num} 时出错: {e}")
                continue
        
        # 生成汇总报告
        summary_path = self.generate_summary_report(results)
        
        print(f"\n{'='*80}")
        print("所有数据集分析完成!")
        print(f"汇总报告: {summary_path}")
        print(f"详细结果目录: {self.output_dir}")
        print(f"{'='*80}")
        
        return results
    
    def generate_summary_report(self, results):
        """生成汇总报告"""
        summary_path = os.path.join(self.output_dir, 'summary_report.txt')
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("重采样质量分析汇总报告\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*80 + "\n\n")
            
            for dataset_name, result in results.items():
                f.write(f"{dataset_name}:\n")
                f.write("-"*40 + "\n")
                
                stats = result['stats']
                orig_count = stats['sample_counts']['original']
                resamp_count = stats['sample_counts']['resampled']
                
                f.write(f"  样本数: {orig_count} → {resamp_count} ({resamp_count/orig_count:.2f}x)\n")
                
                # 计算平衡性改进
                orig_stats = stats['original']
                resamp_stats = stats['resampled']
                
                orig_counts = [orig_stats[cls]['total_samples'] for cls in orig_stats]
                resamp_counts = [resamp_stats[cls]['total_samples'] for cls in resamp_stats]
                
                orig_imbalance = max(orig_counts) / min(orig_counts) if min(orig_counts) > 0 else float('inf')
                resamp_imbalance = max(resamp_counts) / min(resamp_counts) if min(resamp_counts) > 0 else float('inf')
                
                f.write(f"  不平衡率: {orig_imbalance:.2f} → {resamp_imbalance:.2f}\n")
                
                # 有效数据保持
                avg_orig_valid = np.mean([orig_stats[cls]['valid_ratio'] for cls in orig_stats])
                avg_resamp_valid = np.mean([resamp_stats[cls]['valid_ratio'] for cls in resamp_stats])
                f.write(f"  有效数据保持率: {avg_resamp_valid/avg_orig_valid:.3f}\n")
                
                f.write("\n")
        
        return summary_path


def main():
    """主函数"""
    visualizer = ResamplingQualityVisualizer()
    
    # 分析所有数据集
    results = visualizer.analyze_all_datasets()
    
    print("\n重采样质量分析完成！")
    print("可查看生成的可视化文件了解详细情况。")


if __name__ == "__main__":
    main()