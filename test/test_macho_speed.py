"""
MACHO训练速度测试脚本
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

def test_macho_speed():
    """测试MACHO数据集训练速度"""
    print('=== MACHO速度测试 ===')
    
    # 检查数据集大小
    try:
        print("1. 检查MACHO数据集大小...")
        import pickle
        with open('data/MACHO_folded_512.pkl', 'rb') as f:
            data = pickle.load(f)
        print(f"   MACHO数据集包含 {len(data)} 个样本")
        
        # 检查样本数据结构
        if len(data) > 0:
            sample = data[0]
            print(f"   样本数据键: {list(sample.keys())}")
            if 'time' in sample:
                print(f"   时间序列长度: {len(sample['time'])}")
                print(f"   有效时间点数: {sample['mask'].sum() if 'mask' in sample else 'N/A'}")
    except Exception as e:
        print(f"   ❌ 数据集加载失败: {e}")
        return
    
    print("\n2. 测试数据加载速度...")
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
        print(f"   ✓ 验证集批次数: {len(val_loader)}")
        print(f"   ✓ 类别数: {num_classes}")
    except Exception as e:
        print(f"   ❌ 数据加载失败: {e}")
        return
    
    print("\n3. 测试模型创建速度...")
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
            dt=0.025,  # 使用较大步长测试
            rtol=1e-5,
            atol=1e-6,
            enable_gradient_detach=True,
            detach_interval=10,
            debug_mode=True,  # 启用调试模式
            min_time_interval=0.003  # 跳过75%的密集时间点
        )
        model_time = time.time() - start_time
        print(f"   ✓ 模型创建时间: {model_time:.2f}秒")
        print(f"   ✓ 模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    except Exception as e:
        print(f"   ❌ 模型创建失败: {e}")
        return
    
    print("\n4. 测试单批次前向传播速度...")
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
    
    print(f"\n=== 总结 ===")
    if batch_times:
        avg_batch_time = np.mean(batch_times)
        print(f"平均批次时间: {avg_batch_time:.3f}秒")
        print(f"预估每epoch时间: {avg_batch_time * len(train_loader) / 60:.1f}分钟")
        
        # 性能分析
        if avg_batch_time > 10:
            print("⚠️  性能问题：批次时间过长！")
            print("建议检查：")
            print("- SDE求解参数是否过于严格")
            print("- 是否有大量SDE求解失败")
            print("- 梯度断开机制是否合理")
        elif avg_batch_time > 5:
            print("⚠️  性能较慢，可以优化")
        else:
            print("✅ 性能正常")

if __name__ == '__main__':
    test_macho_speed()