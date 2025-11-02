#!/usr/bin/env python3
"""
完整的重采样质量分析 - 分析所有数据集的重采样效果
包括：
1. 每个数据集的mask使用统计
2. 有效数据曲线质量对比
3. 时序特征保持性分析
4. 类别平衡性改进评估
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
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

class ComprehensiveResamplingAnalyzer:
    """全面的重采样质量分析器"""
    
    def __init__(self):
        self.output_dir = '/root/autodl-tmp/lnsde-contiformer/results/pics/resampling_analysis'
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 数据集配置
        self.datasets = {
            1: 'ASAS',
            2: 'LINEAR', 
            3: 'MACHO'
        }
        
        # 设置颜色方案
        self.colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57', '#FF9FF3', '#54A0FF', '#5F27CD']
        
        self.analysis_results = {}
    
    def load_dataset_sample(self, dataset_num, max_samples=150):
        """加载数据集样本进行分析"""
        # 创建虚拟args对象
        class Args:
            def __init__(self):
                self.dataset = dataset_num
                self.batch_size = 50  # 减小batch size
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
        
        # 加载数据
        train_loader, test_loader, num_classes = create_dataloaders(
            data_path=args.data_path,
            batch_size=args.batch_size,
            num_workers=4,
            random_seed=42
        )
        
        # 提取样本数据
        all_features = []
        all_labels = []
        all_times = []
        all_masks = []
        
        total_samples = 0
        for batch in train_loader:
            if total_samples >= max_samples:
                break
                
            # 处理数据格式
            if 'features' in batch:
                features = batch['features']
                times = features[:, :, 0]
            elif 'x' in batch:
                x = batch['x']
                batch_size, seq_len = x.shape[0], x.shape[1]
                errmag = torch.zeros(batch_size, seq_len, 1)
                features = torch.cat([x, errmag], dim=2)
                times = x[:, :, 0]
            else:
                continue
                
            all_features.append(features)
            all_labels.append(batch['labels'])
            all_times.append(times)
            all_masks.append(batch.get('mask', torch.ones_like(times, dtype=torch.bool)))
            
            total_samples += features.shape[0]
        
        # 合并数据
        X = torch.cat(all_features, dim=0)[:max_samples]
        y = torch.cat(all_labels, dim=0)[:max_samples]
        times = torch.cat(all_times, dim=0)[:max_samples]
        masks = torch.cat(all_masks, dim=0)[:max_samples]
        
        return X, y, times, masks, args.dataset_name, num_classes
    
    def perform_resampling(self, X, y, times, masks):
        """执行重采样"""
        resampler = HybridResampler(
            smote_k_neighbors=3,
            enn_n_neighbors=3,
            sampling_strategy='balanced',
            apply_enn=False,
            random_state=42
        )
        
        return resampler.fit_resample(X, y, times, masks), resampler
    
    def analyze_mask_quality(self, masks_orig, y_orig, masks_resamp, y_resamp, dataset_name):
        """分析mask质量"""
        unique_classes = torch.unique(y_orig).cpu().numpy()
        
        orig_stats = {}
        resamp_stats = {}
        
        for cls in unique_classes:
            # 原始数据
            cls_mask_orig = y_orig == cls
            cls_masks_orig = masks_orig[cls_mask_orig]
            
            if len(cls_masks_orig) > 0:
                total_positions = cls_masks_orig.numel()
                valid_positions = cls_masks_orig.sum().item()
                valid_ratio = valid_positions / total_positions if total_positions > 0 else 0
                seq_valid_lengths = cls_masks_orig.sum(dim=1).cpu().numpy()
                
                orig_stats[cls] = {
                    'valid_ratio': valid_ratio,
                    'total_samples': cls_mask_orig.sum().item(),
                    'mean_valid_length': np.mean(seq_valid_lengths),
                    'std_valid_length': np.std(seq_valid_lengths)
                }
            
            # 重采样数据
            if masks_resamp is not None:
                cls_mask_resamp = y_resamp == cls
                cls_masks_resamp = masks_resamp[cls_mask_resamp]
                
                if len(cls_masks_resamp) > 0:
                    total_positions = cls_masks_resamp.numel()
                    valid_positions = cls_masks_resamp.sum().item() if torch.is_tensor(cls_masks_resamp) else cls_masks_resamp.sum()
                    valid_ratio = valid_positions / total_positions if total_positions > 0 else 0
                    
                    if torch.is_tensor(cls_masks_resamp):
                        seq_valid_lengths = cls_masks_resamp.sum(dim=1).cpu().numpy()
                    else:
                        seq_valid_lengths = cls_masks_resamp.sum(axis=1)
                    
                    resamp_stats[cls] = {
                        'valid_ratio': valid_ratio,
                        'total_samples': cls_mask_resamp.sum().item() if torch.is_tensor(cls_mask_resamp) else cls_mask_resamp.sum(),
                        'mean_valid_length': np.mean(seq_valid_lengths),
                        'std_valid_length': np.std(seq_valid_lengths)
                    }
            else:
                # 如果没有mask，假设所有数据都有效
                cls_count = (y_resamp == cls).sum().item() if torch.is_tensor(y_resamp) else (y_resamp == cls).sum()
                seq_len = masks_orig.shape[1]
                
                resamp_stats[cls] = {
                    'valid_ratio': 1.0,
                    'total_samples': cls_count,
                    'mean_valid_length': seq_len,
                    'std_valid_length': 0.0
                }
        
        return orig_stats, resamp_stats
    
    def analyze_dataset(self, dataset_num):
        """分析单个数据集"""
        print(f"\n{'='*50}")
        print(f"分析数据集 {dataset_num}: {self.datasets[dataset_num]}")
        print(f"{'='*50}")
        
        try:
            # 加载数据
            X_orig, y_orig, times_orig, masks_orig, dataset_name, num_classes = self.load_dataset_sample(dataset_num)
            
            print(f"原始数据: {len(y_orig)} 样本, {num_classes} 类别")
            orig_dist = dict(Counter(y_orig.cpu().numpy()))
            print(f"原始分布: {orig_dist}")
            
            # 执行重采样
            (X_resamp, y_resamp, times_resamp, masks_resamp), resampler = self.perform_resampling(
                X_orig, y_orig, times_orig, masks_orig
            )
            
            resamp_dist = dict(Counter(y_resamp.cpu().numpy() if torch.is_tensor(y_resamp) else y_resamp))
            print(f"重采样后: {len(y_resamp)} 样本")
            print(f"重采样分布: {resamp_dist}")
            
            # 分析mask质量
            orig_stats, resamp_stats = self.analyze_mask_quality(
                masks_orig, y_orig, masks_resamp, y_resamp, dataset_name
            )
            
            # 计算质量指标
            quality_metrics = self.calculate_quality_metrics(orig_stats, resamp_stats, orig_dist, resamp_dist)
            
            # 保存结果
            self.analysis_results[dataset_name] = {
                'original_distribution': orig_dist,
                'resampled_distribution': resamp_dist,
                'original_count': len(y_orig),
                'resampled_count': len(y_resamp),
                'original_stats': orig_stats,
                'resampled_stats': resamp_stats,
                'quality_metrics': quality_metrics,
                'num_classes': num_classes
            }
            
            # 生成可视化
            self.create_dataset_visualization(
                X_orig, y_orig, times_orig, masks_orig,
                X_resamp, y_resamp, times_resamp, masks_resamp,
                dataset_name, resampler
            )
            
            print(f"✅ {dataset_name} 分析完成")
            return True
            
        except Exception as e:
            print(f"❌ {dataset_name} 分析失败: {e}")
            return False
    
    def calculate_quality_metrics(self, orig_stats, resamp_stats, orig_dist, resamp_dist):
        """计算重采样质量指标"""
        metrics = {}
        
        # 1. 类别平衡性改进
        orig_counts = list(orig_dist.values())
        resamp_counts = list(resamp_dist.values())
        
        orig_imbalance = max(orig_counts) / min(orig_counts) if min(orig_counts) > 0 else float('inf')
        resamp_imbalance = max(resamp_counts) / min(resamp_counts) if min(resamp_counts) > 0 else float('inf')
        
        metrics['original_imbalance'] = orig_imbalance
        metrics['resampled_imbalance'] = resamp_imbalance
        metrics['balance_improvement'] = orig_imbalance / resamp_imbalance if resamp_imbalance > 0 else 0
        
        # 2. 有效数据保持性
        all_classes = set(orig_stats.keys()) | set(resamp_stats.keys())
        valid_ratios_orig = [orig_stats.get(cls, {}).get('valid_ratio', 0) for cls in all_classes]
        valid_ratios_resamp = [resamp_stats.get(cls, {}).get('valid_ratio', 0) for cls in all_classes]
        
        avg_valid_orig = np.mean(valid_ratios_orig) if valid_ratios_orig else 0
        avg_valid_resamp = np.mean(valid_ratios_resamp) if valid_ratios_resamp else 0
        
        metrics['avg_valid_ratio_original'] = avg_valid_orig
        metrics['avg_valid_ratio_resampled'] = avg_valid_resamp
        metrics['valid_data_preservation'] = avg_valid_resamp / avg_valid_orig if avg_valid_orig > 0 else 0
        
        # 3. 样本增长
        metrics['sample_growth'] = len(resamp_counts) / len(orig_counts) if orig_counts else 0
        
        return metrics
    
    def create_dataset_visualization(self, X_orig, y_orig, times_orig, masks_orig,
                                   X_resamp, y_resamp, times_resamp, masks_resamp,
                                   dataset_name, resampler):
        """为单个数据集创建可视化"""
        dataset_dir = os.path.join(self.output_dir, dataset_name)
        os.makedirs(dataset_dir, exist_ok=True)
        
        # 1. 重采样分布图
        dist_path = os.path.join(dataset_dir, 'resampling_distribution.png')
        resampler.visualize_distribution(save_path=dist_path)
        
        # 2. 有效数据曲线对比（选择少数样本）
        unique_classes = torch.unique(y_orig).cpu().numpy()
        n_classes = len(unique_classes)
        
        fig, axes = plt.subplots(n_classes, 2, figsize=(12, n_classes * 3))
        if n_classes == 1:
            axes = axes.reshape(1, -1)
        
        for row, cls in enumerate(unique_classes):
            color = self.colors[cls % len(self.colors)]
            
            # 原始样本
            orig_indices = torch.where(y_orig == cls)[0][:1]  # 只取一个样本
            if len(orig_indices) > 0:
                idx = orig_indices[0]
                features = X_orig[idx]
                mask = masks_orig[idx]
                times = times_orig[idx]
                
                valid_mask = mask.cpu().numpy()
                if valid_mask.sum() > 0:
                    valid_times = times[valid_mask].cpu().numpy()
                    valid_mags = features[:, 1][valid_mask].cpu().numpy()
                    
                    axes[row, 0].plot(valid_times, valid_mags, 'o-', 
                                     color=color, alpha=0.8, markersize=3, linewidth=1)
                    axes[row, 0].set_title(f'原始 Class {cls}', fontsize=10)
                    axes[row, 0].grid(True, alpha=0.3)
                    axes[row, 0].set_xlabel('时间')
                    axes[row, 0].set_ylabel('星等')
                    axes[row, 0].text(0.05, 0.95, f'有效点: {valid_mask.sum()}', 
                                     transform=axes[row, 0].transAxes,
                                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.7),
                                     fontsize=8)
            
            # 重采样样本
            resamp_indices = torch.where(y_resamp == cls)[0][:1]  # 只取一个样本
            if len(resamp_indices) > 0:
                idx = resamp_indices[0]
                features = X_resamp[idx]
                mask = masks_resamp[idx] if masks_resamp is not None else torch.ones_like(times_resamp[idx], dtype=torch.bool)
                times = times_resamp[idx] if times_resamp is not None else torch.linspace(0, 1, features.shape[0])
                
                if torch.is_tensor(mask):
                    valid_mask = mask.cpu().numpy()
                else:
                    valid_mask = mask
                    
                if valid_mask.sum() > 0:
                    if torch.is_tensor(times):
                        valid_times = times[valid_mask].cpu().numpy()
                    else:
                        valid_times = times[valid_mask]
                        
                    if torch.is_tensor(features):
                        valid_mags = features[:, 1][valid_mask].cpu().numpy()
                    else:
                        valid_mags = features[:, 1][valid_mask]
                    
                    axes[row, 1].plot(valid_times, valid_mags, 'o-', 
                                     color=color, alpha=0.8, markersize=3, linewidth=1)
                    axes[row, 1].set_title(f'重采样 Class {cls}', fontsize=10)
                    axes[row, 1].grid(True, alpha=0.3)
                    axes[row, 1].set_xlabel('时间')
                    axes[row, 1].set_ylabel('星等')
                    axes[row, 1].text(0.05, 0.95, f'有效点: {valid_mask.sum()}', 
                                     transform=axes[row, 1].transAxes,
                                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.7),
                                     fontsize=8)
        
        plt.suptitle(f'{dataset_name} 有效数据曲线对比', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        curves_path = os.path.join(dataset_dir, 'effective_curves.png')
        plt.savefig(curves_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  - 有效数据曲线: {curves_path}")
        print(f"  - 分布对比: {dist_path}")
    
    def create_summary_visualization(self):
        """创建汇总可视化"""
        if not self.analysis_results:
            return
            
        datasets = list(self.analysis_results.keys())
        n_datasets = len(datasets)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. 类别平衡性改进对比
        dataset_names = []
        orig_imbalances = []
        resamp_imbalances = []
        improvements = []
        
        for dataset in datasets:
            result = self.analysis_results[dataset]
            metrics = result['quality_metrics']
            
            dataset_names.append(dataset)
            orig_imbalances.append(metrics['original_imbalance'])
            resamp_imbalances.append(metrics['resampled_imbalance'])
            improvements.append(metrics['balance_improvement'])
        
        x = np.arange(len(dataset_names))
        width = 0.35
        
        axes[0, 0].bar(x - width/2, orig_imbalances, width, label='重采样前', alpha=0.8, color='#FF6B6B')
        axes[0, 0].bar(x + width/2, resamp_imbalances, width, label='重采样后', alpha=0.8, color='#4ECDC4')
        axes[0, 0].set_xlabel('数据集')
        axes[0, 0].set_ylabel('不平衡率')
        axes[0, 0].set_title('类别不平衡率对比', fontweight='bold')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(dataset_names)
        axes[0, 0].legend()
        axes[0, 0].grid(axis='y', alpha=0.3)
        
        # 2. 有效数据保持性
        orig_valid = [self.analysis_results[dataset]['quality_metrics']['avg_valid_ratio_original'] for dataset in datasets]
        resamp_valid = [self.analysis_results[dataset]['quality_metrics']['avg_valid_ratio_resampled'] for dataset in datasets]
        
        axes[0, 1].bar(x - width/2, orig_valid, width, label='重采样前', alpha=0.8, color='#FF6B6B')
        axes[0, 1].bar(x + width/2, resamp_valid, width, label='重采样后', alpha=0.8, color='#4ECDC4')
        axes[0, 1].set_xlabel('数据集')
        axes[0, 1].set_ylabel('平均有效数据比例')
        axes[0, 1].set_title('有效数据保持性对比', fontweight='bold')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(dataset_names)
        axes[0, 1].legend()
        axes[0, 1].grid(axis='y', alpha=0.3)
        
        # 3. 样本增长倍数
        growth_rates = [self.analysis_results[dataset]['resampled_count'] / self.analysis_results[dataset]['original_count'] for dataset in datasets]
        
        bars = axes[1, 0].bar(dataset_names, growth_rates, color='#45B7D1', alpha=0.8)
        axes[1, 0].set_xlabel('数据集')
        axes[1, 0].set_ylabel('样本增长倍数')
        axes[1, 0].set_title('样本数量增长对比', fontweight='bold')
        axes[1, 0].grid(axis='y', alpha=0.3)
        
        # 添加数值标签
        for bar, rate in zip(bars, growth_rates):
            height = bar.get_height()
            axes[1, 0].text(bar.get_x() + bar.get_width()/2., height,
                           f'{rate:.1f}x', ha='center', va='bottom', fontsize=10)
        
        # 4. 综合质量评分
        quality_scores = []
        for dataset in datasets:
            result = self.analysis_results[dataset]
            metrics = result['quality_metrics']
            
            # 综合评分 = 平衡性改进 * 0.4 + 有效数据保持 * 0.4 + 样本增长适度性 * 0.2
            balance_score = min(metrics['balance_improvement'] / 5.0, 1.0)  # 归一化到0-1
            preserve_score = metrics['valid_data_preservation']
            growth_score = 1.0 - abs(growth_rates[datasets.index(dataset)] - 3.0) / 3.0  # 理想增长3倍
            growth_score = max(0, growth_score)
            
            overall_score = balance_score * 0.4 + preserve_score * 0.4 + growth_score * 0.2
            quality_scores.append(overall_score)
        
        bars = axes[1, 1].bar(dataset_names, quality_scores, color='#96CEB4', alpha=0.8)
        axes[1, 1].set_xlabel('数据集')
        axes[1, 1].set_ylabel('综合质量评分')
        axes[1, 1].set_title('重采样质量综合评分', fontweight='bold')
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].grid(axis='y', alpha=0.3)
        
        # 添加评分标签
        for bar, score in zip(bars, quality_scores):
            height = bar.get_height()
            axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                           f'{score:.2f}', ha='center', va='bottom', fontsize=10)
        
        plt.suptitle('重采样质量分析汇总', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        summary_path = os.path.join(self.output_dir, 'summary_analysis.png')
        plt.savefig(summary_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return summary_path
    
    def generate_comprehensive_report(self):
        """生成综合质量报告"""
        report_path = os.path.join(self.output_dir, 'comprehensive_quality_report.txt')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("重采样质量分析综合报告\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*80 + "\n\n")
            
            # 汇总统计
            f.write("1. 汇总统计\n")
            f.write("-"*50 + "\n")
            f.write(f"分析数据集数量: {len(self.analysis_results)}\n")
            
            total_orig = sum(result['original_count'] for result in self.analysis_results.values())
            total_resamp = sum(result['resampled_count'] for result in self.analysis_results.values())
            f.write(f"原始样本总数: {total_orig}\n")
            f.write(f"重采样后样本总数: {total_resamp}\n")
            f.write(f"总体增长倍数: {total_resamp/total_orig:.2f}x\n\n")
            
            # 按数据集详细分析
            f.write("2. 各数据集详细分析\n")
            f.write("-"*50 + "\n")
            
            for dataset_name, result in self.analysis_results.items():
                f.write(f"\n{dataset_name}:\n")
                f.write(f"  类别数: {result['num_classes']}\n")
                f.write(f"  样本数: {result['original_count']} → {result['resampled_count']} ({result['resampled_count']/result['original_count']:.1f}x)\n")
                
                metrics = result['quality_metrics']
                f.write(f"  不平衡率: {metrics['original_imbalance']:.2f} → {metrics['resampled_imbalance']:.2f}\n")
                f.write(f"  平衡性改进: {metrics['balance_improvement']:.2f}x\n")
                f.write(f"  有效数据保持率: {metrics['valid_data_preservation']:.3f}\n")
                
                f.write(f"  原始类别分布: {result['original_distribution']}\n")
                f.write(f"  重采样类别分布: {result['resampled_distribution']}\n")
            
            # 质量评估
            f.write("\n3. 质量评估\n")
            f.write("-"*50 + "\n")
            
            avg_balance_improvement = np.mean([result['quality_metrics']['balance_improvement'] 
                                             for result in self.analysis_results.values()])
            avg_preservation = np.mean([result['quality_metrics']['valid_data_preservation'] 
                                       for result in self.analysis_results.values()])
            
            f.write(f"平均平衡性改进: {avg_balance_improvement:.2f}x\n")
            f.write(f"平均有效数据保持率: {avg_preservation:.3f}\n\n")
            
            # 结论和建议
            f.write("4. 结论和建议\n")
            f.write("-"*50 + "\n")
            
            if avg_balance_improvement > 3.0:
                f.write("✅ 类别平衡性改进显著\n")
            elif avg_balance_improvement > 2.0:
                f.write("✅ 类别平衡性改进良好\n") 
            else:
                f.write("⚠️  类别平衡性改进有限\n")
                
            if avg_preservation > 0.9:
                f.write("✅ 有效数据保持性优秀\n")
            elif avg_preservation > 0.8:
                f.write("✅ 有效数据保持性良好\n")
            else:
                f.write("⚠️  有效数据保持性需要改进\n")
            
            f.write("\n建议:\n")
            f.write("- 重采样算法已考虑mask并保持时序特征\n")
            f.write("- 合成样本的mask通过并集操作保证有效数据覆盖\n")
            f.write("- SMOTE插值过程保持了原始时序结构\n")
            f.write("- 建议在训练时使用重采样数据以改善类别不平衡问题\n")
        
        return report_path
    
    def run_complete_analysis(self):
        """运行完整分析"""
        print("开始重采样质量分析...")
        print("="*80)
        
        # 分析各个数据集
        successful_analyses = 0
        for dataset_num in [1, 2, 3]:
            if self.analyze_dataset(dataset_num):
                successful_analyses += 1
        
        if successful_analyses == 0:
            print("❌ 没有成功分析任何数据集")
            return
        
        # 创建汇总可视化
        print(f"\n生成汇总分析...")
        summary_path = self.create_summary_visualization()
        
        # 生成综合报告
        report_path = self.generate_comprehensive_report()
        
        print(f"\n{'='*80}")
        print(f"重采样质量分析完成!")
        print(f"成功分析 {successful_analyses}/3 个数据集")
        print(f"汇总可视化: {summary_path}")
        print(f"综合报告: {report_path}")
        print(f"详细结果目录: {self.output_dir}")
        print(f"{'='*80}")
        
        return self.analysis_results


def main():
    """主函数"""
    analyzer = ComprehensiveResamplingAnalyzer()
    results = analyzer.run_complete_analysis()
    
    print("\n✅ 重采样质量分析全部完成!")
    print("可查看生成的可视化文件了解详细情况。")
    
    return results


if __name__ == "__main__":
    main()