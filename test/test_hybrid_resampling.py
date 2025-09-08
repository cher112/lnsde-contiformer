#!/usr/bin/env python3
"""
测试混合重采样（SMOTE+ENN）的效果
"""

import sys
sys.path.append('/root/autodl-tmp/lnsde-contiformer')
import torch
import warnings
warnings.filterwarnings('ignore')

from utils.resampling import HybridResampler, configure_chinese_font
from utils import create_dataloaders

# 配置中文字体
configure_chinese_font()

# 加载LINEAR数据
data_path = '/root/autodl-fs/lnsde-contiformer/data/LINEAR_folded_512_fixed.pkl'
train_loader, test_loader, num_classes = create_dataloaders(
    data_path=data_path,
    batch_size=64,
    num_workers=0,
    random_seed=535411460
)

# 提取训练数据
all_features = []
all_labels = []
all_times = []
all_masks = []

print("提取训练数据...")
for batch in train_loader:
    all_features.append(batch['features'])
    all_labels.append(batch['labels'])
    all_times.append(batch['times'])
    all_masks.append(batch['mask'])

X = torch.cat(all_features, dim=0)
y = torch.cat(all_labels, dim=0)
times = torch.cat(all_times, dim=0)
masks = torch.cat(all_masks, dim=0)

print(f"原始数据形状: X={X.shape}, y={y.shape}")

# 创建重采样器 - 显式启用ENN
resampler = HybridResampler(
    smote_k_neighbors=5,
    enn_n_neighbors=3,
    sampling_strategy='balanced',
    apply_enn=True,  # 显式启用ENN
    random_state=535411460
)

# 执行重采样
X_resampled, y_resampled, times_resampled, masks_resampled = resampler.fit_resample(
    X, y, times, masks
)

print(f"\n重采样后数据形状: X={X_resampled.shape}, y={y_resampled.shape}")
print(f"\n重采样统计:")
print(f"  原始分布: {resampler.stats_['original']}")
print(f"  SMOTE后: {resampler.stats_['after_smote']}")
print(f"  最终分布: {resampler.stats_['final']}")

# 可视化
import os
fig_path = '/root/autodl-tmp/lnsde-contiformer/results/pics/LINEAR/test_hybrid_resampling.png'
os.makedirs(os.path.dirname(fig_path), exist_ok=True)
resampler.visualize_distribution(save_path=fig_path)

print(f"\n可视化已保存至: {fig_path}")