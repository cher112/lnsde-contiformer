#!/usr/bin/env python3
"""
测试数据加载器的输出格式
"""

import sys
sys.path.append('/root/autodl-tmp/lnsde-contiformer')

from utils import create_dataloaders

# 加载数据
data_path = '/root/autodl-fs/lnsde-contiformer/data/LINEAR_folded_512_fixed.pkl'
train_loader, test_loader, num_classes = create_dataloaders(
    data_path=data_path,
    batch_size=2,
    num_workers=0,
    random_seed=535411460
)

# 获取一个批次
for batch in train_loader:
    print("Batch keys:", batch.keys())
    for key, value in batch.items():
        if hasattr(value, 'shape'):
            print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
        else:
            print(f"  {key}: type={type(value)}, value={value}")
    break