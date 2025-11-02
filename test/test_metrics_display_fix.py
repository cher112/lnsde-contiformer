#!/usr/bin/env python3
"""
测试F1和Recall显示修复效果
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils.logging_utils import print_epoch_summary

# 模拟训练指标
train_metrics = {
    'f1_score': 0.8234,      # 82.34%
    'recall': 0.8156,        # 81.56%
    'macro_f1': 0.7654,
    'weighted_f1': 0.8234,
    'macro_recall': 0.7543,
    'weighted_recall': 0.8156
}

val_metrics = {
    'f1_score': 0.7845,      # 78.45%
    'recall': 0.7892,        # 78.92%
    'macro_f1': 0.7234,
    'weighted_f1': 0.7845,
    'macro_recall': 0.7123,
    'weighted_recall': 0.7892
}

print("=" * 60)
print("测试显示修复效果")
print("=" * 60)

print_epoch_summary(
    epoch=1,
    total_epochs=100,
    train_loss=0.4567,
    train_acc=82.34,      # 准确率已经是百分比
    val_loss=0.5234,
    val_acc=78.92,        # 准确率已经是百分比
    train_metrics=train_metrics,
    val_metrics=val_metrics,
    lr=1e-5
)

print("\n预期结果:")
print("• 训练准确率: 82.34% ≈ 加权F1: 82.3% ≈ 加权Recall: 81.6%")
print("• 验证准确率: 78.92% ≈ 加权F1: 78.5% ≈ 加权Recall: 78.9%")
print("• 现在F1和Recall应该显示为百分比格式")