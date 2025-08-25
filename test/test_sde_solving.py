"""
测试SDE求解功能
"""

import torch
import numpy as np
import sys
sys.path.append('..')
sys.path.append('/root/autodl-tmp/lnsde+contiformer')

from models.base_sde import BaseSDEModel
from models.langevin_sde import LangevinSDE
from models.linear_noise_sde import LinearNoiseSDE  
from models.geometric_sde import GeometricSDE

def test_sde_solving():
    """测试SDE求解功能"""
    print('=== SDE求解测试 ===')
    
    batch_size = 4
    hidden_channels = 32
    
    # 创建测试数据
    y0 = torch.randn(batch_size, hidden_channels)
    times = torch.tensor([0.0, 0.1])
    
    print(f'初始状态形状: {y0.shape}')
    print(f'时间序列: {times}')
    
    # 测试三种SDE
    sde_models = {
        'Langevin SDE': LangevinSDE(3, hidden_channels, hidden_channels),
        'Linear Noise SDE': LinearNoiseSDE(3, hidden_channels, hidden_channels),
        'Geometric SDE': GeometricSDE(3, hidden_channels, hidden_channels)
    }
    
    for sde_name, sde_model in sde_models.items():
        print(f'\\n测试 {sde_name}...')
        try:
            import torchsde
            
            # 测试漂移和扩散函数
            t = torch.tensor([0.05])
            drift = sde_model.f(t, y0)
            diffusion = sde_model.g(t, y0)
            
            print(f'  ✓ 漂移函数输出形状: {drift.shape}')
            print(f'  ✓ 扩散函数输出形状: {diffusion.shape}')
            
            # 测试SDE求解
            ys = torchsde.sdeint(
                sde=sde_model,
                y0=y0,
                ts=times,
                method='euler',
                dt=0.01
            )
            
            print(f'  ✓ SDE求解成功，输出形状: {ys.shape}')
            print(f'  ✓ 最终状态形状: {ys[-1].shape}')
            print(f'  ✅ {sde_name} 测试通过!')
            
        except Exception as e:
            print(f'  ❌ {sde_name} 测试失败: {e}')
            import traceback
            traceback.print_exc()

if __name__ == '__main__':
    test_sde_solving()
    print('\\n🎯 SDE求解测试完成!')