#!/usr/bin/env python3
"""
重采样光变曲线对比可视化
为每个类别创建源数据vs重采样数据的光变曲线对比图
每个类别显示多条光变曲线，源数据和重采样数据分别放在不同子图中
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# 添加项目路径
sys.path.append('/root/autodl-tmp/lnsde-contiformer')

from utils import create_dataloaders, setup_dataset_mapping
from utils.resampling import HybridResampler, configure_chinese_font

# 配置中文字体
configure_chinese_font()

class LightCurveComparisonVisualizer:
    """光变曲线对比可视化器"""
    
    def __init__(self, output_dir='/root/autodl-tmp/lnsde-contiformer/results/pics/lightcurve_comparison'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # 设置颜色方案 - 为每个类别分配颜色
        self.class_colors = {
            0: '#FF6B6B',  # 红色
            1: '#4ECDC4',  # 青色  
            2: '#45B7D1',  # 蓝色
            3: '#96CEB4',  # 绿色
            4: '#FECA57',  # 黄色
            5: '#FF9FF3',  # 粉色
            6: '#54A0FF',  # 亮蓝色
            7: '#5F27CD',  # 紫色
            8: '#00D2D3',  # 青蓝色
            9: '#FF9F43',  # 橙色
        }
        
        # 数据集配置
        self.datasets = {
            1: 'ASAS',
            2: 'LINEAR', 
            3: 'MACHO'
        }
        
    def load_dataset_sample(self, dataset_num, max_samples=200):
        """加载数据集样本"""
        class Args:
            def __init__(self):
                self.dataset = dataset_num
                self.batch_size = 50
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
        
        X_resamp, y_resamp, times_resamp, masks_resamp = resampler.fit_resample(X, y, times, masks)
        return X_resamp, y_resamp, times_resamp, masks_resamp, resampler
    
    def plot_lightcurves_for_class(self, class_id, X_orig, y_orig, times_orig, masks_orig,
                                  X_resamp, y_resamp, times_resamp, masks_resamp,
                                  dataset_name, n_curves=5):
        """为特定类别绘制光变曲线对比"""
        
        # 获取该类别的原始样本
        orig_indices = torch.where(y_orig == class_id)[0]
        n_orig_curves = min(n_curves, len(orig_indices))
        
        # 获取该类别的重采样样本
        if torch.is_tensor(y_resamp):
            resamp_indices = torch.where(y_resamp == class_id)[0]
        else:
            resamp_indices = np.where(y_resamp == class_id)[0]
            resamp_indices = torch.from_numpy(resamp_indices)
        n_resamp_curves = min(n_curves, len(resamp_indices))
        
        if n_orig_curves == 0 and n_resamp_curves == 0:
            return None
        
        # 创建子图 - 左侧源数据，右侧重采样数据
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        color = self.class_colors.get(class_id, '#333333')
        
        # 绘制原始数据曲线
        if n_orig_curves > 0:
            for i in range(n_orig_curves):
                idx = orig_indices[i]
                features = X_orig[idx]
                mask = masks_orig[idx]
                times = times_orig[idx]
                
                # 提取有效数据点
                valid_mask = mask.cpu().numpy()
                if valid_mask.sum() > 0:
                    valid_times = times[valid_mask].cpu().numpy()
                    valid_mags = features[:, 1][valid_mask].cpu().numpy()  # 星等
                    valid_errors = features[:, 2][valid_mask].cpu().numpy() if features.shape[2] > 2 else None  # 误差
                    
                    # 绘制光变曲线
                    alpha = 0.8 - i * 0.1  # 透明度递减
                    ax1.plot(valid_times, valid_mags, 'o-', 
                            color=color, alpha=alpha, markersize=3, linewidth=1.5,
                            label=f'样本 {i+1} ({valid_mask.sum()}点)')
                    
                    # 如果有误差信息，添加误差棒
                    if valid_errors is not None and np.any(valid_errors > 0):
                        ax1.errorbar(valid_times, valid_mags, yerr=valid_errors, 
                                   color=color, alpha=alpha*0.5, fmt='none', capsize=2)
        
        ax1.set_title(f'源数据 - Class {class_id} ({n_orig_curves}条曲线)', fontsize=12, fontweight='bold')
        ax1.set_xlabel('时间')
        ax1.set_ylabel('星等')
        ax1.grid(True, alpha=0.3)
        ax1.invert_yaxis()  # 星等越小越亮，所以倒转y轴
        if n_orig_curves > 0:
            ax1.legend(loc='upper right', fontsize=8)
        
        # 绘制重采样数据曲线
        if n_resamp_curves > 0:
            for i in range(n_resamp_curves):
                idx = resamp_indices[i]
                
                # 处理tensor和numpy兼容性
                if torch.is_tensor(X_resamp):
                    features = X_resamp[idx]
                else:
                    features = torch.from_numpy(X_resamp[idx])
                
                if masks_resamp is not None:
                    if torch.is_tensor(masks_resamp):
                        mask = masks_resamp[idx]
                    else:
                        mask = torch.from_numpy(masks_resamp[idx])
                else:
                    mask = torch.ones(features.shape[0], dtype=torch.bool)
                
                if times_resamp is not None:
                    if torch.is_tensor(times_resamp):
                        times = times_resamp[idx]
                    else:
                        times = torch.from_numpy(times_resamp[idx])
                else:
                    times = torch.linspace(0, 1, features.shape[0])
                
                # 提取有效数据点
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
                        valid_errors = features[:, 2][valid_mask].cpu().numpy() if features.shape[1] > 2 else None
                    else:
                        valid_mags = features[:, 1][valid_mask]
                        valid_errors = features[:, 2][valid_mask] if features.shape[1] > 2 else None
                    
                    # 绘制光变曲线
                    alpha = 0.8 - i * 0.1
                    ax2.plot(valid_times, valid_mags, 'o-', 
                            color=color, alpha=alpha, markersize=3, linewidth=1.5,
                            label=f'样本 {i+1} ({valid_mask.sum()}点)')
                    
                    # 添加误差棒
                    if valid_errors is not None and np.any(valid_errors > 0):
                        ax2.errorbar(valid_times, valid_mags, yerr=valid_errors, 
                                   color=color, alpha=alpha*0.5, fmt='none', capsize=2)
        
        ax2.set_title(f'重采样数据 - Class {class_id} ({n_resamp_curves}条曲线)', fontsize=12, fontweight='bold')
        ax2.set_xlabel('时间')
        ax2.set_ylabel('星等')
        ax2.grid(True, alpha=0.3)
        ax2.invert_yaxis()
        if n_resamp_curves > 0:
            ax2.legend(loc='upper right', fontsize=8)
        
        plt.suptitle(f'{dataset_name} - Class {class_id} 光变曲线对比', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # 保存图片
        save_path = os.path.join(self.output_dir, f'{dataset_name}_class_{class_id}_comparison.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def create_class_overview(self, X_orig, y_orig, times_orig, masks_orig,
                            X_resamp, y_resamp, times_resamp, masks_resamp,
                            dataset_name, n_curves_per_class=3):
        """创建所有类别的概览图"""
        unique_classes = torch.unique(y_orig).cpu().numpy()
        n_classes = len(unique_classes)
        
        # 创建大图 - 每行一个类别，两列（源数据vs重采样数据）
        fig, axes = plt.subplots(n_classes, 2, figsize=(16, n_classes * 4))
        if n_classes == 1:
            axes = axes.reshape(1, -1)
        
        for row, class_id in enumerate(unique_classes):
            color = self.class_colors.get(class_id, '#333333')
            
            # 左侧：源数据
            orig_indices = torch.where(y_orig == class_id)[0][:n_curves_per_class]
            
            for i, idx in enumerate(orig_indices):
                features = X_orig[idx]
                mask = masks_orig[idx]
                times = times_orig[idx]
                
                valid_mask = mask.cpu().numpy()
                if valid_mask.sum() > 0:
                    valid_times = times[valid_mask].cpu().numpy()
                    valid_mags = features[:, 1][valid_mask].cpu().numpy()
                    
                    alpha = 0.9 - i * 0.2
                    axes[row, 0].plot(valid_times, valid_mags, 'o-', 
                                     color=color, alpha=alpha, markersize=2, linewidth=1,
                                     label=f'#{i+1}')
            
            axes[row, 0].set_title(f'源数据 - Class {class_id}', fontsize=10, fontweight='bold')
            axes[row, 0].set_xlabel('时间')
            axes[row, 0].set_ylabel('星等')
            axes[row, 0].grid(True, alpha=0.3)
            axes[row, 0].invert_yaxis()
            if len(orig_indices) > 0:
                axes[row, 0].legend(loc='upper right', fontsize=8)
            
            # 右侧：重采样数据
            if torch.is_tensor(y_resamp):
                resamp_indices = torch.where(y_resamp == class_id)[0][:n_curves_per_class]
            else:
                resamp_indices = np.where(y_resamp == class_id)[0][:n_curves_per_class]
                resamp_indices = torch.from_numpy(resamp_indices)
            
            for i, idx in enumerate(resamp_indices):
                # 处理数据兼容性
                if torch.is_tensor(X_resamp):
                    features = X_resamp[idx]
                else:
                    features = torch.from_numpy(X_resamp[idx])
                
                if masks_resamp is not None:
                    if torch.is_tensor(masks_resamp):
                        mask = masks_resamp[idx]
                    else:
                        mask = torch.from_numpy(masks_resamp[idx])
                else:
                    mask = torch.ones(features.shape[0], dtype=torch.bool)
                
                if times_resamp is not None:
                    if torch.is_tensor(times_resamp):
                        times = times_resamp[idx]
                    else:
                        times = torch.from_numpy(times_resamp[idx])
                else:
                    times = torch.linspace(0, 1, features.shape[0])
                
                # 提取有效数据
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
                    
                    alpha = 0.9 - i * 0.2
                    axes[row, 1].plot(valid_times, valid_mags, 'o-', 
                                     color=color, alpha=alpha, markersize=2, linewidth=1,
                                     label=f'#{i+1}')
            
            axes[row, 1].set_title(f'重采样数据 - Class {class_id}', fontsize=10, fontweight='bold')
            axes[row, 1].set_xlabel('时间')
            axes[row, 1].set_ylabel('星等')
            axes[row, 1].grid(True, alpha=0.3)
            axes[row, 1].invert_yaxis()
            if len(resamp_indices) > 0:
                axes[row, 1].legend(loc='upper right', fontsize=8)
        
        plt.suptitle(f'{dataset_name} 所有类别光变曲线对比概览', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # 保存概览图
        overview_path = os.path.join(self.output_dir, f'{dataset_name}_all_classes_overview.png')
        plt.savefig(overview_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return overview_path
    
    def analyze_dataset(self, dataset_num):
        """分析单个数据集"""
        print(f"\n{'='*50}")
        print(f"分析数据集 {dataset_num}: {self.datasets[dataset_num]}")
        print(f"{'='*50}")
        
        try:
            # 加载数据
            X_orig, y_orig, times_orig, masks_orig, dataset_name, num_classes = self.load_dataset_sample(dataset_num)
            
            print(f"加载完成: {len(y_orig)} 样本, {num_classes} 类别")
            orig_dist = dict(Counter(y_orig.cpu().numpy()))
            print(f"原始分布: {orig_dist}")
            
            # 执行重采样
            print("执行重采样...")
            X_resamp, y_resamp, times_resamp, masks_resamp, resampler = self.perform_resampling(
                X_orig, y_orig, times_orig, masks_orig
            )
            
            resamp_dist = dict(Counter(y_resamp.cpu().numpy() if torch.is_tensor(y_resamp) else y_resamp))
            print(f"重采样后: {len(y_resamp)} 样本")
            print(f"重采样分布: {resamp_dist}")
            
            # 创建输出目录
            dataset_dir = os.path.join(self.output_dir, dataset_name)
            os.makedirs(dataset_dir, exist_ok=True)
            
            # 创建所有类别概览图
            print("生成所有类别概览图...")
            overview_path = self.create_class_overview(
                X_orig, y_orig, times_orig, masks_orig,
                X_resamp, y_resamp, times_resamp, masks_resamp,
                dataset_name, n_curves_per_class=3
            )
            
            # 为每个类别创建详细对比图
            unique_classes = torch.unique(y_orig).cpu().numpy()
            class_paths = []
            
            print("为每个类别生成详细对比图...")
            for class_id in unique_classes:
                class_path = self.plot_lightcurves_for_class(
                    class_id, X_orig, y_orig, times_orig, masks_orig,
                    X_resamp, y_resamp, times_resamp, masks_resamp,
                    dataset_name, n_curves=5
                )
                if class_path:
                    class_paths.append(class_path)
                    print(f"  - Class {class_id}: {class_path}")
            
            # 生成重采样分布图
            dist_path = os.path.join(dataset_dir, 'resampling_distribution.png')
            resampler.visualize_distribution(save_path=dist_path)
            
            print(f"✅ {dataset_name} 完成")
            print(f"  - 概览图: {overview_path}")
            print(f"  - 类别详细图: {len(class_paths)} 个")
            print(f"  - 分布对比: {dist_path}")
            
            return True
            
        except Exception as e:
            print(f"❌ {dataset_name} 分析失败: {e}")
            return False
    
    def analyze_all_datasets(self):
        """分析所有数据集"""
        print("开始光变曲线对比可视化分析...")
        print("="*80)
        
        successful = 0
        for dataset_num in [1, 2, 3]:
            if self.analyze_dataset(dataset_num):
                successful += 1
        
        print(f"\n{'='*80}")
        print(f"光变曲线对比可视化完成!")
        print(f"成功分析 {successful}/3 个数据集")
        print(f"输出目录: {self.output_dir}")
        print(f"{'='*80}")


def main():
    """主函数"""
    visualizer = LightCurveComparisonVisualizer()
    visualizer.analyze_all_datasets()
    
    print("\n✅ 光变曲线对比可视化全部完成!")
    print("每个类别的源数据和重采样数据曲线已分别保存。")


if __name__ == "__main__":
    main()