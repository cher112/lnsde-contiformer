#!/usr/bin/env python3
"""
诊断训练速度问题
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import torch
from models.linear_noise_sde import LinearNoiseSDEContiformer
from utils import create_dataloaders

print("="*60)
print("训练速度诊断")
print("="*60)

# 1. 测试数据加载速度
print("\n1. 数据加载速度测试...")
data_path = '/root/autodl-fs/lnsde-contiformer/data/MACHO_original.pkl'
start = time.time()
train_loader, test_loader, num_classes = create_dataloaders(
    data_path=data_path,
    batch_size=8,
    num_workers=16,
    pin_memory=True,
    persistent_workers=True
)
print(f"数据加载器创建: {time.time()-start:.2f}秒")

# 获取一个batch
start = time.time()
batch = next(iter(train_loader))
print(f"获取一个batch: {time.time()-start:.2f}秒")
print(f"Batch大小: {batch['features'].shape}")

# 2. 测试不同配置的模型速度
device = torch.device('cuda:0')
features = batch['features'].to(device)
mask = batch['mask'].to(device) if 'mask' in batch else None

configs = [
    ("基础模型", dict(use_sde=False, use_contiformer=False, use_cga=False)),
    ("仅SDE", dict(use_sde=True, use_contiformer=False, use_cga=False)),
    ("SDE+ContiFormer", dict(use_sde=True, use_contiformer=True, use_cga=False)),
    ("完整模型", dict(use_sde=True, use_contiformer=True, use_cga=True)),
]

print("\n2. 模型前向传播速度测试...")
for name, config in configs:
    model = LinearNoiseSDEContiformer(**config).to(device)
    model.eval()
    
    # 预热
    with torch.no_grad():
        _ = model(features, mask)
    
    # 测试10次前向传播
    torch.cuda.synchronize()
    start = time.time()
    with torch.no_grad():
        for _ in range(10):
            logits = model(features, mask)
    torch.cuda.synchronize()
    elapsed = time.time() - start
    
    print(f"{name}: {elapsed/10:.3f}秒/batch")

# 3. 分析SDE求解瓶颈
print("\n3. SDE求解分析...")
model = LinearNoiseSDEContiformer(use_sde=True, use_contiformer=False).to(device)
model.eval()

# 测试不同的SDE配置
sde_configs = [
    ("euler dt=0.01", dict(sde_method='euler', dt=0.01)),
    ("euler dt=0.05", dict(sde_method='euler', dt=0.05)),
    ("euler dt=0.1", dict(sde_method='euler', dt=0.1)),
    ("midpoint dt=0.01", dict(sde_method='midpoint', dt=0.01)),
]

for name, sde_config in sde_configs:
    model.dt = sde_config['dt']
    model.sde_method = sde_config['sde_method']
    
    torch.cuda.synchronize()
    start = time.time()
    with torch.no_grad():
        for _ in range(5):
            logits = model(features, mask)
    torch.cuda.synchronize()
    elapsed = time.time() - start
    
    print(f"{name}: {elapsed/5:.3f}秒/batch")

print("\n4. 瓶颈分析总结:")
print("- 如果数据加载>0.5秒: 数据IO瓶颈")
print("- 如果SDE模型比基础模型慢10倍以上: SDE求解瓶颈")
print("- 如果GPU利用率<50%: CPU瓶颈或数据传输瓶颈")
print("="*60)