#!/usr/bin/env python3
"""
诊断梯度裁剪效果和内存泄漏问题
"""

import sys
sys.path.append('/root/autodl-tmp/lnsde-contiformer')

import torch
import torch.nn as nn
import numpy as np
import gc
import psutil
import time
from torch.cuda.amp import autocast, GradScaler

# 导入项目模块
from utils.dataloader import create_dataloaders
from utils.config import Config
from models.contiformer import ContiFormer
from models.linear_noise_sde import LinearNoiseSDE


def monitor_memory():
    """监控内存使用"""
    # GPU内存
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        max_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        gpu_usage = (allocated / max_mem) * 100
        print(f"GPU内存: {allocated:.2f}GB/{max_mem:.2f}GB ({gpu_usage:.1f}%)")
    
    # 系统内存
    memory = psutil.virtual_memory()
    print(f"系统内存: {memory.used/1024**3:.2f}GB/{memory.total/1024**3:.2f}GB ({memory.percent:.1f}%)")


def check_gradient_health(model):
    """检查梯度健康状况"""
    total_params = 0
    nan_params = 0
    inf_params = 0
    zero_params = 0
    large_params = 0
    
    grad_norms = []
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad = param.grad.data
            total_params += grad.numel()
            
            # 统计异常梯度
            nan_count = torch.isnan(grad).sum().item()
            inf_count = torch.isinf(grad).sum().item()
            zero_count = (grad == 0).sum().item()
            large_count = (torch.abs(grad) > 10).sum().item()
            
            nan_params += nan_count
            inf_params += inf_count
            zero_params += zero_count
            large_params += large_count
            
            # 计算梯度范数
            grad_norm = torch.norm(grad).item()
            grad_norms.append((name, grad_norm))
    
    print(f"\n梯度统计:")
    print(f"  总参数: {total_params}")
    print(f"  NaN梯度: {nan_params} ({nan_params/total_params*100:.3f}%)")
    print(f"  Inf梯度: {inf_params} ({inf_params/total_params*100:.3f}%)")
    print(f"  零梯度: {zero_params} ({zero_params/total_params*100:.3f}%)")
    print(f"  大梯度(>10): {large_params} ({large_params/total_params*100:.3f}%)")
    
    # 显示前5个最大梯度范数
    grad_norms.sort(key=lambda x: x[1], reverse=True)
    print("\n最大梯度范数:")
    for name, norm in grad_norms[:5]:
        print(f"  {name}: {norm:.6f}")
    
    return nan_params > 0 or inf_params > 0


def test_gradient_clipping():
    """测试梯度裁剪效果"""
    print("=" * 60)
    print("测试梯度裁剪效果")
    print("=" * 60)
    
    # 配置
    config = Config()
    config.dataset = 'MACHO'
    config.use_sde = True
    config.use_contiformer = True
    config.use_cga = True
    config.batch_size = 8  # 小批次用于测试
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 数据加载器
    train_loader, val_loader, _ = create_dataloaders(config)
    print(f"数据集大小: train={len(train_loader)}, val={len(val_loader)}")
    
    # 模型
    if config.use_contiformer:
        model = ContiFormer(config).to(device)
    else:
        model = LinearNoiseSDE(config).to(device)
    
    # 优化器和损失
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler()
    
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters())}")
    
    # 测试不同的梯度裁剪值
    clip_values = [0.1, 0.5, 1.0, 2.0, 5.0]
    
    for clip_val in clip_values:
        print(f"\n" + "="*40)
        print(f"测试梯度裁剪值: {clip_val}")
        print("="*40)
        
        model.train()
        batch_count = 0
        memory_leak_detected = False
        initial_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        
        # 测试几个批次
        for batch_idx, batch in enumerate(train_loader):
            if batch_idx >= 5:  # 只测试5个批次
                break
                
            batch_count += 1
            print(f"\n批次 {batch_idx + 1}/5:")
            
            try:
                # 前向传播
                if 'features' in batch:
                    features = batch['features'].to(device)
                elif 'x' in batch:
                    features = torch.cat([
                        batch['x'].to(device), 
                        torch.zeros(batch['x'].shape[0], batch['x'].shape[1], 1, device=device)
                    ], dim=2)
                else:
                    continue
                
                labels = batch['labels'].to(device)
                mask = batch.get('mask', torch.ones_like(features[:, :, 0])).to(device)
                
                print(f"  输入形状: {features.shape}")
                monitor_memory()
                
                # 前向传播
                with autocast():
                    outputs = model(features, mask)
                    loss = criterion(outputs, labels)
                
                print(f"  Loss: {loss.item():.6f}")
                
                # 反向传播
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                
                # 检查梯度健康状况
                has_bad_grad = check_gradient_health(model)
                
                if has_bad_grad:
                    print("  ❌ 发现异常梯度，跳过优化步骤")
                    optimizer.zero_grad()
                else:
                    print("  ✅ 梯度正常")
                    # 应用梯度裁剪
                    scaler.unscale_(optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_val)
                    print(f"  梯度范数 (裁剪前): {grad_norm:.6f}")
                    print(f"  梯度裁剪值: {clip_val}")
                    
                    scaler.step(optimizer)
                    scaler.update()
                
                # 检查内存泄漏
                current_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
                memory_growth = (current_memory - initial_memory) / 1024**3
                if memory_growth > 0.5:  # 超过500MB增长
                    print(f"  ⚠️ 检测到内存泄漏: +{memory_growth:.2f}GB")
                    memory_leak_detected = True
                
                # 清理
                del outputs, loss
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                
            except Exception as e:
                print(f"  ❌ 批次{batch_idx}失败: {str(e)}")
                continue
        
        print(f"\n梯度裁剪值 {clip_val} 测试结果:")
        print(f"  成功批次: {batch_count}/5")
        print(f"  内存泄漏: {'是' if memory_leak_detected else '否'}")
        
        # 强制清理
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def identify_memory_leak_source():
    """识别内存泄漏源头"""
    print("\n" + "="*60)
    print("识别内存泄漏源头")
    print("="*60)
    
    config = Config()
    config.dataset = 'MACHO'
    config.batch_size = 4
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 测试各个组件
    components = [
        ('基础模型', {'use_sde': False, 'use_contiformer': False, 'use_cga': False}),
        ('SDE组件', {'use_sde': True, 'use_contiformer': False, 'use_cga': False}),
        ('ContiFormer组件', {'use_sde': False, 'use_contiformer': True, 'use_cga': False}),
        ('CGA组件', {'use_sde': False, 'use_contiformer': False, 'use_cga': True}),
        ('完整模型', {'use_sde': True, 'use_contiformer': True, 'use_cga': True}),
    ]
    
    for name, settings in components:
        print(f"\n测试组件: {name}")
        print("-" * 30)
        
        # 更新配置
        for key, value in settings.items():
            setattr(config, key, value)
        
        try:
            # 获取数据
            train_loader, _, _ = create_dataloaders(config)
            batch = next(iter(train_loader))
            
            # 创建模型
            if config.use_contiformer:
                model = ContiFormer(config).to(device)
            else:
                model = LinearNoiseSDE(config).to(device)
            
            print(f"  模型参数: {sum(p.numel() for p in model.parameters())}")
            
            initial_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
            
            # 多次前向传播测试
            for i in range(10):
                # 准备输入
                if 'features' in batch:
                    features = batch['features'].to(device)
                else:
                    features = torch.cat([
                        batch['x'].to(device), 
                        torch.zeros(batch['x'].shape[0], batch['x'].shape[1], 1, device=device)
                    ], dim=2)
                
                mask = batch.get('mask', torch.ones_like(features[:, :, 0])).to(device)
                
                # 前向传播
                with torch.no_grad():
                    outputs = model(features, mask)
                
                # 检查内存增长
                current_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
                memory_growth = (current_memory - initial_memory) / 1024**2  # MB
                
                if i == 0:
                    first_pass_memory = current_memory
                elif i == 9:
                    print(f"  内存增长: {memory_growth:.1f}MB (10次前向传播)")
                    if memory_growth > 100:  # 超过100MB
                        print(f"  ⚠️ {name} 可能存在内存泄漏")
                
                del outputs
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"  ❌ 测试{name}失败: {str(e)}")
            continue


def main():
    print("梯度裁剪和内存泄漏诊断工具")
    print("=" * 60)
    
    # 系统信息
    print("系统信息:")
    print(f"  PyTorch版本: {torch.__version__}")
    print(f"  CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA版本: {torch.version.cuda}")
        print(f"  GPU型号: {torch.cuda.get_device_name(0)}")
        print(f"  GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    
    monitor_memory()
    
    # 1. 测试梯度裁剪
    test_gradient_clipping()
    
    # 2. 识别内存泄漏
    identify_memory_leak_source()
    
    print("\n" + "="*60)
    print("诊断完成")
    print("="*60)


if __name__ == "__main__":
    main()