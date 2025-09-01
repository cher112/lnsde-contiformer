"""
混合重采样模块 - 智能处理类别不平衡
包含改进的SMOTE（少数类过采样）和ENN（多数类欠采样）
专门针对时间序列数据进行优化
"""

import torch
import numpy as np
from sklearn.neighbors import NearestNeighbors
from collections import Counter
import pickle
import os
from typing import Tuple, Dict, Optional, List
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns

# 配置中文字体
def configure_chinese_font():
    """配置中文字体显示"""
    try:
        # 添加字体到matplotlib管理器
        fm.fontManager.addfont('/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc')
        fm.fontManager.addfont('/usr/share/fonts/truetype/wqy/wqy-microhei.ttc')
    except:
        pass
    
    # 设置中文字体优先级列表
    plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'WenQuanYi Micro Hei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    return True

# 初始化时配置字体
configure_chinese_font()
from datetime import datetime


class TimeSeriesSMOTE:
    """
    时间序列SMOTE - 对少数类进行智能过采样
    保持时间连贯性的同时生成合成样本
    """
    
    def __init__(self, 
                 k_neighbors=5,
                 sampling_strategy='auto',
                 random_state=42):
        """
        Args:
            k_neighbors: 用于SMOTE的邻居数
            sampling_strategy: 采样策略
                - 'auto': 自动平衡到多数类数量
                - float: 少数类相对多数类的比例
                - dict: 每个类的目标样本数
            random_state: 随机种子
        """
        self.k_neighbors = k_neighbors
        self.sampling_strategy = sampling_strategy
        self.random_state = random_state
        np.random.seed(random_state)
        
    def fit_resample(self, X, y, times=None, masks=None):
        """
        执行时间序列SMOTE重采样
        
        Args:
            X: (n_samples, seq_len, n_features) 时间序列数据
            y: (n_samples,) 标签
            times: (n_samples, seq_len) 时间戳
            masks: (n_samples, seq_len) 有效位置掩码
            
        Returns:
            X_resampled, y_resampled, times_resampled, masks_resampled
        """
        # 统计类别分布
        class_counts = Counter(y.tolist() if torch.is_tensor(y) else y)
        n_classes = len(class_counts)
        majority_class = max(class_counts, key=class_counts.get)
        majority_count = class_counts[majority_class]
        
        print(f"原始类别分布: {class_counts}")
        
        # 确定每个类的目标样本数
        if self.sampling_strategy == 'auto':
            # 自动策略：所有类都平衡到多数类数量
            target_counts = {cls: majority_count for cls in class_counts}
        elif isinstance(self.sampling_strategy, float):
            # 比例策略
            target_counts = {}
            for cls in class_counts:
                if cls == majority_class:
                    target_counts[cls] = class_counts[cls]
                else:
                    target_counts[cls] = int(majority_count * self.sampling_strategy)
        else:
            target_counts = self.sampling_strategy
            
        print(f"目标类别分布: {target_counts}")
        
        # 准备输出列表
        X_list, y_list, times_list, masks_list = [], [], [], []
        
        # 处理每个类
        for cls in class_counts:
            # 获取当前类的索引
            if torch.is_tensor(y):
                cls_indices = (y == cls).nonzero(as_tuple=True)[0].cpu().numpy()
            else:
                cls_indices = np.where(y == cls)[0]
            
            # 当前类的数据
            X_cls = X[cls_indices] if torch.is_tensor(X) else X[cls_indices]
            n_samples = len(cls_indices)
            n_synthetic = target_counts[cls] - n_samples
            
            # 添加原始样本
            X_list.append(X_cls)
            y_list.extend([cls] * n_samples)
            
            if times is not None:
                times_cls = times[cls_indices]
                times_list.append(times_cls)
                
            if masks is not None:
                masks_cls = masks[cls_indices]
                masks_list.append(masks_cls)
            
            # 如果需要生成合成样本
            if n_synthetic > 0:
                print(f"为类别 {cls} 生成 {n_synthetic} 个合成样本")
                
                # 将时间序列展平用于KNN
                if torch.is_tensor(X_cls):
                    X_cls_flat = X_cls.reshape(n_samples, -1).cpu().numpy()
                else:
                    X_cls_flat = X_cls.reshape(n_samples, -1)
                
                # 构建KNN
                k = min(self.k_neighbors, n_samples - 1)
                if k > 0:
                    nn = NearestNeighbors(n_neighbors=k+1, metric='euclidean')
                    nn.fit(X_cls_flat)
                    
                    # 生成合成样本
                    synthetic_samples = []
                    synthetic_times = [] if times is not None else None
                    synthetic_masks = [] if masks is not None else None
                    
                    for _ in range(n_synthetic):
                        # 随机选择一个样本
                        idx = np.random.randint(n_samples)
                        
                        # 找到最近邻
                        neighbors = nn.kneighbors([X_cls_flat[idx]], return_distance=False)[0][1:]
                        
                        # 随机选择一个邻居
                        nn_idx = neighbors[np.random.randint(len(neighbors))]
                        
                        # 生成插值系数（时间感知）
                        lambda_val = np.random.beta(2, 2)  # Beta分布，倾向于中间值
                        
                        # 插值生成新样本
                        if torch.is_tensor(X_cls):
                            sample1 = X_cls[idx].cpu().numpy()
                            sample2 = X_cls[nn_idx].cpu().numpy()
                        else:
                            sample1 = X_cls[idx]
                            sample2 = X_cls[nn_idx]
                            
                        synthetic_sample = sample1 + lambda_val * (sample2 - sample1)
                        synthetic_samples.append(synthetic_sample)
                        
                        # 插值时间戳
                        if times is not None:
                            if torch.is_tensor(times_cls):
                                time1 = times_cls[idx].cpu().numpy()
                                time2 = times_cls[nn_idx].cpu().numpy()
                            else:
                                time1 = times_cls[idx]
                                time2 = times_cls[nn_idx]
                            synthetic_time = time1 + lambda_val * (time2 - time1)
                            synthetic_times.append(synthetic_time)
                        
                        # 合并掩码（取并集）
                        if masks is not None:
                            if torch.is_tensor(masks_cls):
                                mask1 = masks_cls[idx].cpu().numpy()
                                mask2 = masks_cls[nn_idx].cpu().numpy()
                            else:
                                mask1 = masks_cls[idx]
                                mask2 = masks_cls[nn_idx]
                            synthetic_mask = mask1 | mask2
                            synthetic_masks.append(synthetic_mask)
                    
                    # 添加合成样本
                    if synthetic_samples:
                        X_list.append(np.array(synthetic_samples))
                        y_list.extend([cls] * n_synthetic)
                        
                        if synthetic_times:
                            times_list.append(np.array(synthetic_times))
                        if synthetic_masks:
                            masks_list.append(np.array(synthetic_masks))
        
        # 合并所有数据
        if torch.is_tensor(X):
            X_resampled = torch.cat([torch.tensor(x) if not torch.is_tensor(x) else x 
                                    for x in X_list], dim=0)
            y_resampled = torch.tensor(y_list)
        else:
            X_resampled = np.concatenate(X_list, axis=0)
            y_resampled = np.array(y_list)
            
        times_resampled = None
        if times is not None:
            if torch.is_tensor(times):
                times_resampled = torch.cat([torch.tensor(t) if not torch.is_tensor(t) else t 
                                            for t in times_list], dim=0)
            else:
                times_resampled = np.concatenate(times_list, axis=0)
                
        masks_resampled = None
        if masks is not None:
            if torch.is_tensor(masks):
                masks_resampled = torch.cat([torch.tensor(m) if not torch.is_tensor(m) else m 
                                            for m in masks_list], dim=0)
            else:
                masks_resampled = np.concatenate(masks_list, axis=0)
        
        return X_resampled, y_resampled, times_resampled, masks_resampled


class EditedNearestNeighbors:
    """
    编辑最近邻 (ENN) - 清理多数类中的噪声样本
    温和版本：只删除明显的噪声点
    """
    
    def __init__(self, 
                 n_neighbors=3,
                 kind_sel='mode',
                 max_removal_ratio=0.2):  # 最多删除20%的样本
        """
        Args:
            n_neighbors: 用于判断的邻居数
            kind_sel: 选择标准 ('mode' 或 'all')
            max_removal_ratio: 最大删除比例
        """
        self.n_neighbors = n_neighbors
        self.kind_sel = kind_sel
        self.max_removal_ratio = max_removal_ratio
        
    def fit_resample(self, X, y, times=None, masks=None):
        """
        执行ENN清理
        
        Args:
            X: (n_samples, seq_len, n_features) 时间序列数据
            y: (n_samples,) 标签
            times: 可选的时间戳
            masks: 可选的掩码
            
        Returns:
            清理后的数据
        """
        from collections import Counter  # 移到函数开始处
        
        # 统计类别分布
        class_counts = Counter(y.tolist() if torch.is_tensor(y) else y)
        print(f"ENN清理前类别分布: {class_counts}")
        
        # 展平数据用于KNN
        n_samples = len(y)
        if torch.is_tensor(X):
            X_flat = X.reshape(n_samples, -1).cpu().numpy()
            y_np = y.cpu().numpy() if torch.is_tensor(y) else y
        else:
            X_flat = X.reshape(n_samples, -1)
            y_np = y
            
        # 构建KNN
        nn = NearestNeighbors(n_neighbors=self.n_neighbors + 1)
        nn.fit(X_flat)
        
        # 找出每个样本的邻居
        neighbors = nn.kneighbors(X_flat, return_distance=False)[:, 1:]
        
        # 决定保留哪些样本
        keep_mask = np.ones(n_samples, dtype=bool)
        removal_candidates = []
        
        for i in range(n_samples):
            neighbor_labels = y_np[neighbors[i]]
            
            if self.kind_sel == 'mode':
                # 如果大多数邻居的类别与当前样本不同，则标记为候选删除
                mode_label = Counter(neighbor_labels).most_common(1)[0][0]
                if mode_label != y_np[i]:
                    removal_candidates.append((i, 1))  # 权重为1
            elif self.kind_sel == 'all':
                # 如果所有邻居的类别都与当前样本不同，则标记为候选删除
                if not np.any(neighbor_labels == y_np[i]):
                    removal_candidates.append((i, 2))  # 权重为2，优先删除
        
        # 限制删除数量
        max_removals = int(n_samples * self.max_removal_ratio)
        if len(removal_candidates) > max_removals:
            # 按权重排序，优先删除更明显的噪声
            removal_candidates.sort(key=lambda x: x[1], reverse=True)
            removal_candidates = removal_candidates[:max_removals]
        
        # 应用删除
        for idx, _ in removal_candidates:
            keep_mask[idx] = False
        
        # 应用掩码
        X_cleaned = X[keep_mask]
        y_cleaned = y[keep_mask] if torch.is_tensor(y) else y[keep_mask]
        
        times_cleaned = times[keep_mask] if times is not None else None
        masks_cleaned = masks[keep_mask] if masks is not None else None
        
        # 统计清理后的分布
        class_counts_after = Counter(y_cleaned.tolist() if torch.is_tensor(y_cleaned) else y_cleaned)
        print(f"ENN清理后类别分布: {class_counts_after}")
        print(f"共删除 {n_samples - sum(keep_mask)} 个样本")
        
        return X_cleaned, y_cleaned, times_cleaned, masks_cleaned


class HybridResampler:
    """
    混合重采样器 - 结合SMOTE和ENN
    """
    
    def __init__(self,
                 smote_k_neighbors=5,
                 enn_n_neighbors=3,
                 sampling_strategy='balanced',
                 apply_enn=True,
                 random_state=42):
        """
        Args:
            smote_k_neighbors: SMOTE的邻居数
            enn_n_neighbors: ENN的邻居数
            sampling_strategy: 采样策略
                - 'balanced': 完全平衡（所有类数量相同）
                - 'auto': 自动平衡到多数类
                - float: 少数类相对多数类的比例
            apply_enn: 是否应用ENN清理
            random_state: 随机种子
        """
        self.sampling_strategy = sampling_strategy
        self.apply_enn = apply_enn
        self.random_state = random_state
        
        # 初始化SMOTE
        self.smote = TimeSeriesSMOTE(
            k_neighbors=smote_k_neighbors,
            sampling_strategy='auto' if sampling_strategy == 'balanced' else sampling_strategy,
            random_state=random_state
        )
        
        # 初始化ENN
        self.enn = EditedNearestNeighbors(n_neighbors=enn_n_neighbors)
        
        # 统计信息
        self.stats_ = {}
        
    def fit_resample(self, X, y, times=None, masks=None):
        """
        执行混合重采样
        """
        # 记录原始分布
        original_counts = Counter(y.tolist() if torch.is_tensor(y) else y)
        self.stats_['original'] = dict(original_counts)
        
        print("\n" + "="*60)
        print("开始混合重采样")
        print("="*60)
        
        # Step 1: SMOTE过采样
        print("\nStep 1: SMOTE过采样")
        X_smote, y_smote, times_smote, masks_smote = self.smote.fit_resample(
            X, y, times, masks
        )
        smote_counts = Counter(y_smote.tolist() if torch.is_tensor(y_smote) else y_smote)
        self.stats_['after_smote'] = dict(smote_counts)
        
        # Step 2: ENN清理（可选）
        if self.apply_enn:
            print("\nStep 2: ENN清理")
            X_final, y_final, times_final, masks_final = self.enn.fit_resample(
                X_smote, y_smote, times_smote, masks_smote
            )
        else:
            X_final, y_final = X_smote, y_smote
            times_final, masks_final = times_smote, masks_smote
            
        final_counts = Counter(y_final.tolist() if torch.is_tensor(y_final) else y_final)
        self.stats_['final'] = dict(final_counts)
        
        print("\n重采样完成！")
        print(f"原始总样本数: {len(y)}")
        print(f"最终总样本数: {len(y_final)}")
        print("="*60 + "\n")
        
        return X_final, y_final, times_final, masks_final
    
    def visualize_distribution(self, save_path=None):
        """
        可视化重采样前后的类别分布
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        stages = ['original', 'after_smote', 'final']
        titles = ['原始分布', 'SMOTE后', '最终分布(ENN后)']
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        
        for ax, stage, title, color in zip(axes, stages, titles, colors):
            if stage in self.stats_:
                data = self.stats_[stage]
                classes = list(data.keys())
                counts = list(data.values())
                
                bars = ax.bar(classes, counts, color=color, alpha=0.7, edgecolor='black')
                ax.set_title(title, fontsize=14, fontweight='bold')
                ax.set_xlabel('类别', fontsize=12)
                ax.set_ylabel('样本数', fontsize=12)
                ax.grid(axis='y', alpha=0.3)
                
                # 添加数值标签
                for bar, count in zip(bars, counts):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{int(count)}', ha='center', va='bottom', fontsize=10)
                
                # 计算不平衡率
                max_count = max(counts)
                min_count = min(counts)
                imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
                ax.text(0.5, 0.95, f'不平衡率: {imbalance_ratio:.2f}',
                       transform=ax.transAxes, ha='center', va='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.suptitle('混合重采样效果分析', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"分布图已保存至: {save_path}")
        
        plt.show()
        
        return fig


def save_resampled_data(X, y, times, masks, dataset_name, save_dir='/root/autodl-fs/lnsde-contiformer/data/resampled'):
    """
    保存重采样后的数据
    
    Args:
        X: 重采样后的特征
        y: 重采样后的标签
        times: 重采样后的时间戳
        masks: 重采样后的掩码
        dataset_name: 数据集名称
        save_dir: 保存目录
    """
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 生成时间戳
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 保存路径
    save_path = os.path.join(save_dir, f'{dataset_name}_resampled_{timestamp}.pkl')
    
    # 保存数据
    data = {
        'X': X,
        'y': y,
        'times': times,
        'masks': masks,
        'dataset': dataset_name,
        'timestamp': timestamp,
        'distribution': dict(Counter(y.tolist() if torch.is_tensor(y) else y))
    }
    
    with open(save_path, 'wb') as f:
        pickle.dump(data, f)
    
    print(f"重采样数据已保存至: {save_path}")
    
    # 保存统计信息
    stats_path = os.path.join(save_dir, f'{dataset_name}_stats_{timestamp}.txt')
    with open(stats_path, 'w') as f:
        f.write(f"数据集: {dataset_name}\n")
        f.write(f"时间戳: {timestamp}\n")
        f.write(f"总样本数: {len(y)}\n")
        f.write(f"类别分布: {data['distribution']}\n")
        f.write(f"特征形状: {X.shape}\n")
        if times is not None:
            f.write(f"时间戳形状: {times.shape}\n")
        if masks is not None:
            f.write(f"掩码形状: {masks.shape}\n")
    
    print(f"统计信息已保存至: {stats_path}")
    
    return save_path


def load_resampled_data(file_path):
    """
    加载重采样后的数据
    
    Args:
        file_path: 数据文件路径
        
    Returns:
        包含重采样数据的字典
    """
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    
    print(f"已加载重采样数据:")
    print(f"  数据集: {data['dataset']}")
    print(f"  时间戳: {data['timestamp']}")
    print(f"  总样本数: {len(data['y'])}")
    print(f"  类别分布: {data['distribution']}")
    
    return data


# 测试函数
if __name__ == "__main__":
    # 创建模拟数据进行测试
    np.random.seed(42)
    
    # 模拟不平衡的时间序列数据
    n_majority = 1000
    n_minority1 = 100
    n_minority2 = 50
    seq_len = 100
    n_features = 3
    
    # 生成数据
    X_maj = np.random.randn(n_majority, seq_len, n_features)
    X_min1 = np.random.randn(n_minority1, seq_len, n_features) + 2
    X_min2 = np.random.randn(n_minority2, seq_len, n_features) - 2
    
    X = np.concatenate([X_maj, X_min1, X_min2], axis=0)
    y = np.concatenate([
        np.zeros(n_majority, dtype=int),
        np.ones(n_minority1, dtype=int),
        np.ones(n_minority2, dtype=int) * 2
    ])
    
    # 生成时间戳
    times = np.tile(np.linspace(0, 1, seq_len), (len(y), 1))
    
    # 生成掩码
    masks = np.ones((len(y), seq_len), dtype=bool)
    
    print("测试混合重采样器...")
    
    # 创建重采样器
    resampler = HybridResampler(
        smote_k_neighbors=5,
        enn_n_neighbors=3,
        sampling_strategy='balanced',
        apply_enn=True
    )
    
    # 执行重采样
    X_resampled, y_resampled, times_resampled, masks_resampled = resampler.fit_resample(
        X, y, times, masks
    )
    
    # 可视化
    resampler.visualize_distribution(save_path='/root/autodl-tmp/lnsde-contiformer/results/pics/resampling_test.png')
    
    # 保存数据
    save_path = save_resampled_data(
        X_resampled, y_resampled, times_resampled, masks_resampled,
        dataset_name='test'
    )
    
    print("\n测试完成！")