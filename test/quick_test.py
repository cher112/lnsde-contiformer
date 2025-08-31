#!/usr/bin/env python3
"""
快速测试单个模型
"""

import torch
import numpy as np
import pickle
from sklearn.metrics import accuracy_score, f1_score, recall_score
import sys
import os

sys.path.append('/root/autodl-tmp/lnsde-contiformer')

def quick_test():
    # 加载数据
    print("加载ASAS数据...")
    with open('/root/autodl-tmp/db/ASAS/ASAS_processed.pkl', 'rb') as f:
        data = pickle.load(f)
    
    features = []
    labels = []
    for item in data:
        if 'fit_lc' in item and 'label' in item:
            features.append(item['fit_lc'][:3])  # 只取前3个特征
            labels.append(item['label'])
    
    features = np.array(features, dtype=np.float32)
    labels = np.array(labels, dtype=np.int64)
    
    print(f"数据形状: features={features.shape}, labels={labels.shape}")
    print(f"标签分布: {np.bincount(labels)}")
    
    # 加载模型
    model_path = '/autodl-fs/data/lnsde-contiformer/results/20250828/ASAS/2116/models/ASAS_linear_noise_best.pth'
    checkpoint = torch.load(model_path, map_location='cpu')
    state_dict = checkpoint['model_state_dict']
    
    num_classes = state_dict['classifier.3.weight'].shape[0]
    print(f"模型类别数: {num_classes}")
    
    # 创建模型
    from models import LinearNoiseSDEContiformer
    model = LinearNoiseSDEContiformer(
        input_dim=3,
        hidden_channels=128,
        contiformer_dim=128,
        n_heads=8,
        n_layers=6,
        num_classes=num_classes,
        dropout=0.1
    )
    
    model.load_state_dict(state_dict)
    model.eval()
    
    # 转换数据
    features = torch.tensor(features, dtype=torch.float32)
    
    # 预测
    print("开始预测...")
    with torch.no_grad():
        outputs = model(features)
        predictions = torch.argmax(outputs, dim=1).numpy()
    
    # 计算指标
    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='weighted')
    recall = recall_score(labels, predictions, average='weighted')
    
    print(f"\nASAS模型在ASAS数据上的结果:")
    print(f"样本数: {len(labels)}")
    print(f"准确率: {acc:.4f}")
    print(f"加权F1: {f1:.4f}")
    print(f"加权召回率: {recall:.4f}")

if __name__ == "__main__":
    quick_test()