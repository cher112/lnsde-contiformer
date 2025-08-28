"""
MACHO训练速度测试脚本 - 基准版本（无时间间隔优化）
"""

import torch
import numpy as np
import time
import sys
import os
# 添加项目根目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models.linear_noise_sde import LinearNoiseSDEContiformer
from utils.dataloader import create_dataloaders

def test_macho_baseline():
    """测试MACHO数据集训练速度 - 基准版本"""
    print('=== MACHO速度测试（基准版本，无优化）===')
    
    print("1. 测试数据加载速度...")
    start_time = time.time()
    try:
        train_loader, val_loader, num_classes = create_dataloaders(
            'data/MACHO_folded_512.pkl',
            batch_size=8,  # 小批次测试
            train_ratio=0.8,
            test_ratio=0.2,
            num_workers=2  # 减少worker数量
        )
        load_time = time.time() - start_time
        print(f"   ✓ 数据加载时间: {load_time:.2f}秒")
        print(f"   ✓ 训练集批次数: {len(train_loader)}")
        print(f"   ✓ 类别数: {num_classes}")
    except Exception as e:
        print(f"   ❌ 数据加载失败: {e}")
        return
    
    print("\n2. 测试模型创建速度（基准版本）...")
    start_time = time.time()
    try:
        model = LinearNoiseSDEContiformer(
            input_dim=3,
            num_classes=num_classes,
            hidden_channels=64,
            contiformer_dim=128,
            n_heads=8,
            n_layers=4,
            dropout=0.1,
            sde_method='milstein',
            dt=0.025,
            rtol=1e-5,
            atol=1e-6,
            enable_gradient_detach=True,
            detach_interval=10,
            debug_mode=False,  # 关闭调试模式避免输出干扰
            min_time_interval=0.0  # 无时间间隔限制，基准版本
        )
        model_time = time.time() - start_time
        print(f"   ✓ 模型创建时间: {model_time:.2f}秒")
        print(f"   ✓ 模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    except Exception as e:
        print(f"   ❌ 模型创建失败: {e}")
        return
    
    print("\n3. 测试单批次前向传播速度（基准版本）...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    batch_times = []
    try:
        for i, batch in enumerate(train_loader):
            if i >= 1:  # 只测试1个批次即可
                break
                
            print(f"\n   --- 批次 {i+1} ---")
            
            # 数据移动到设备
            x = batch['x'].to(device)
            mask = batch['mask'].to(device)
            
            batch_size, seq_len = x.shape[0], x.shape[1]
            print(f"   批次大小: {batch_size}, 序列长度: {seq_len}")
            
            # 将2维x转换为3维features [time, mag, errmag] 
            errmag = torch.zeros(batch_size, seq_len, 1, device=device)
            features = torch.cat([x, errmag], dim=2)
            
            # 计时前向传播
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            start_time = time.time()
            
            with torch.no_grad():
                logits, sde_features = model(features, mask)
                
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            batch_time = time.time() - start_time
            batch_times.append(batch_time)
            
            print(f"   前向传播时间: {batch_time:.3f}秒")
            print(f"   输出形状: {logits.shape}, SDE特征形状: {sde_features.shape}")
    
    except Exception as e:
        print(f"   ❌ 前向传播失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print(f"\n=== 总结（基准版本）===")
    if batch_times:
        avg_batch_time = np.mean(batch_times)
        print(f"平均批次时间: {avg_batch_time:.3f}秒")
        print(f"预估每epoch时间: {avg_batch_time * len(train_loader) / 60:.1f}分钟")
        
        return avg_batch_time
    return None

if __name__ == '__main__':
    test_macho_baseline()