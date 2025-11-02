#!/usr/bin/env python3
"""
快速测试ENN清理效果
"""
import sys
sys.path.append('/root/autodl-tmp/lnsde-contiformer')
import torch
import numpy as np
from utils.resampling import HybridResampler

# 创建简单的测试数据
np.random.seed(535411460)
torch.manual_seed(535411460)

# 创建不平衡数据
n_samples_per_class = [100, 50, 25]  # 类别0:100, 类别1:50, 类别2:25
seq_len = 50
n_features = 3

X_list = []
y_list = []

for class_id, n_samples in enumerate(n_samples_per_class):
    # 为每个类生成不同的数据分布
    class_data = np.random.randn(n_samples, seq_len, n_features) + class_id * 2
    # 添加一些噪声样本（交叉在其他类的区域）
    if class_id == 0:  # 在类别0中添加一些像类别1的样本
        noise_samples = 10
        class_data[-noise_samples:] = np.random.randn(noise_samples, seq_len, n_features) + 1 * 2
    
    X_list.append(class_data)
    y_list.extend([class_id] * n_samples)

X = np.concatenate(X_list, axis=0)
y = np.array(y_list)

print(f"原始数据: {len(y)} 样本")
print(f"类别分布: {dict(zip(*np.unique(y, return_counts=True)))}")

# 创建重采样器
resampler = HybridResampler(
    smote_k_neighbors=3,
    enn_n_neighbors=3,
    sampling_strategy='balanced',
    apply_enn=True,
    random_state=535411460
)

# 执行重采样
X_resampled, y_resampled, _, _ = resampler.fit_resample(X, y)

print(f"\n最终结果:")
print(f"  原始分布: {resampler.stats_['original']}")
print(f"  SMOTE后: {resampler.stats_['after_smote']}")  
print(f"  最终分布: {resampler.stats_['final']}")

if 'after_smote' in resampler.stats_ and 'final' in resampler.stats_:
    smote_total = sum(resampler.stats_['after_smote'].values())
    final_total = sum(resampler.stats_['final'].values())
    removed = smote_total - final_total
    print(f"\nENN删除了 {removed} 个样本 ({removed/smote_total*100:.1f}%)")
else:
    print("\n统计信息不完整")