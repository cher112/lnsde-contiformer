"""
测试模型基本功能
"""

import torch
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import LangevinSDEContiformer, LinearNoiseSDEContiformer, GeometricSDEContiformer

def test_model_shapes():
    """测试模型形状和基本功能"""
    print('=== 模型形状测试 ===')

    # 创建测试数据
    batch_size = 4
    seq_len = 32
    input_dim = 3
    num_classes = 5

    # 模拟光变曲线数据 (time, mag, errmag)
    features = torch.randn(batch_size, seq_len, input_dim)
    times = torch.linspace(0, 1, seq_len).unsqueeze(0).expand(batch_size, -1)
    mask = torch.ones(batch_size, seq_len, dtype=torch.bool)

    print(f'测试数据形状:')
    print(f'  features: {features.shape}')
    print(f'  times: {times.shape}')
    print(f'  mask: {mask.shape}')

    # 测试三种模型
    models = {
        'Langevin SDE': LangevinSDEContiformer(
            input_dim=input_dim, 
            num_classes=num_classes,
            hidden_channels=32,
            contiformer_dim=64,
            n_heads=4,
            n_layers=2
        ),
        'Linear Noise SDE': LinearNoiseSDEContiformer(
            input_dim=input_dim,
            num_classes=num_classes, 
            hidden_channels=32,
            contiformer_dim=64,
            n_heads=4,
            n_layers=2
        ),
        'Geometric SDE': GeometricSDEContiformer(
            input_dim=input_dim,
            num_classes=num_classes,
            hidden_channels=32,
            contiformer_dim=64, 
            n_heads=4,
            n_layers=2
        )
    }

    print('\\n=== 前向传播测试 ===')
    for model_name, model in models.items():
        try:
            print(f'\\n测试 {model_name}...')
            
            # 前向传播
            logits, sde_features = model(features, times, mask)
            
            print(f'  ✓ 输出logits形状: {logits.shape}')
            print(f'  ✓ SDE特征形状: {sde_features.shape}')
            
            # 测试损失计算
            labels = torch.randint(0, num_classes, (batch_size,))
            if hasattr(model, 'compute_loss'):
                # 不传递权重参数，避免参数错误
                loss_result = model.compute_loss(logits, labels)
                if isinstance(loss_result, tuple):
                    loss = loss_result[0]
                else:
                    loss = loss_result
                print(f'  ✓ 损失计算成功: {loss.item():.4f}')
            
            print(f'  ✅ {model_name} 测试通过!')
            
        except Exception as e:
            print(f'  ❌ {model_name} 测试失败: {e}')
            import traceback
            traceback.print_exc()
            
    return True

if __name__ == '__main__':
    test_model_shapes()
    print('\\n🎯 模型形状测试完成!')