#!/usr/bin/env python3
"""
验证所有SDE模型集成测试
测试Langevin, LinearNoise, Geometric三种SDE与ContiFormer + CGA的集成
"""

import torch
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, '/root/autodl-tmp/lnsde-contiformer')

from models import (
    LangevinSDEContiformer,
    LinearNoiseSDEContiformer, 
    GeometricSDEContiformer
)

def test_model_integration():
    """测试所有模型的集成功能"""
    print("=== SDE模型集成测试 ===")
    
    # 测试数据
    batch_size = 4
    seq_len = 100
    input_dim = 3
    num_classes = 7
    
    # 创建测试数据
    time_series = torch.randn(batch_size, seq_len, input_dim)
    time_series[:, :, 0] = torch.sort(torch.rand(batch_size, seq_len), dim=1)[0]  # 确保时间递增
    
    # 创建mask (模拟不规则时间序列)
    mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
    for i in range(batch_size):
        # 随机设置一些位置为False
        invalid_positions = torch.randperm(seq_len)[:seq_len//4]
        mask[i, invalid_positions] = False
    
    print(f"测试数据形状: {time_series.shape}")
    print(f"Mask形状: {mask.shape}")
    print(f"有效数据比例: {mask.float().mean():.2f}")
    
    # 测试配置
    model_configs = {
        'basic': {
            'use_sde': True,
            'use_contiformer': True,  
            'use_cga': False
        },
        'with_cga': {
            'use_sde': True,
            'use_contiformer': True,
            'use_cga': True
        },
        'no_sde': {
            'use_sde': False,
            'use_contiformer': True,
            'use_cga': False
        },
        'no_contiformer': {
            'use_sde': True,
            'use_contiformer': False,
            'use_cga': False
        },
        'minimal': {
            'use_sde': False,
            'use_contiformer': False,
            'use_cga': False
        }
    }
    
    # 测试所有模型类型
    model_classes = {
        'Langevin': LangevinSDEContiformer,
        'LinearNoise': LinearNoiseSDEContiformer,
        'Geometric': GeometricSDEContiformer
    }
    
    results = {}
    
    for model_name, ModelClass in model_classes.items():
        print(f"\n--- 测试 {model_name} SDE ---")
        results[model_name] = {}
        
        for config_name, config in model_configs.items():
            print(f"  配置: {config_name} {config}")
            
            try:
                # 创建模型
                model = ModelClass(
                    input_dim=input_dim,
                    num_classes=num_classes,
                    hidden_channels=32,  # 小一点减少计算量
                    contiformer_dim=64,
                    n_heads=4,
                    n_layers=2,
                    dropout=0.1,
                    dt=0.1,  # 大步长快速测试
                    rtol=1e-3,
                    atol=1e-4,
                    **config
                )
                
                model.eval()
                
                # 前向传播测试
                with torch.no_grad():
                    logits = model(time_series, mask)
                
                # 验证输出
                assert logits.shape == (batch_size, num_classes), f"输出形状错误: {logits.shape}"
                assert not torch.isnan(logits).any(), "输出包含NaN"
                assert not torch.isinf(logits).any(), "输出包含Inf"
                
                # 检查softmax概率
                probs = torch.softmax(logits, dim=1)
                assert torch.allclose(probs.sum(dim=1), torch.ones(batch_size), atol=1e-6), "概率不为1"
                
                results[model_name][config_name] = {
                    'success': True,
                    'output_shape': logits.shape,
                    'max_prob': probs.max().item(),
                    'min_prob': probs.min().item()
                }
                
                print(f"    ✓ 成功! 输出形状: {logits.shape}")
                
            except Exception as e:
                print(f"    ✗ 失败: {e}")
                results[model_name][config_name] = {
                    'success': False,
                    'error': str(e)
                }
    
    # 打印测试结果摘要
    print(f"\n=== 测试结果摘要 ===")
    
    for model_name, model_results in results.items():
        print(f"\n{model_name} SDE:")
        for config_name, result in model_results.items():
            status = "✓" if result['success'] else "✗"
            print(f"  {config_name:15s}: {status}")
            if not result['success']:
                print(f"    错误: {result['error']}")
    
    # 统计成功率
    total_tests = sum(len(model_results) for model_results in results.values())
    successful_tests = sum(
        sum(1 for result in model_results.values() if result['success'])
        for model_results in results.values()
    )
    
    print(f"\n总体成功率: {successful_tests}/{total_tests} ({100*successful_tests/total_tests:.1f}%)")
    
    return results

def test_consistency():
    """测试模型一致性 - 确保相同配置产生相同输出形状"""
    print("\n=== 模型一致性测试 ===")
    
    batch_size = 2
    seq_len = 50
    input_dim = 3
    num_classes = 7
    
    time_series = torch.randn(batch_size, seq_len, input_dim)
    time_series[:, :, 0] = torch.sort(torch.rand(batch_size, seq_len), dim=1)[0]
    mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
    
    # 测试相同配置下的输出一致性
    config = {
        'input_dim': input_dim,
        'num_classes': num_classes,
        'hidden_channels': 32,
        'contiformer_dim': 64,
        'use_sde': True,
        'use_contiformer': True,
        'use_cga': False
    }
    
    models = {
        'Langevin': LangevinSDEContiformer(**config),
        'LinearNoise': LinearNoiseSDEContiformer(**config),
        'Geometric': GeometricSDEContiformer(**config)
    }
    
    outputs = {}
    for name, model in models.items():
        model.eval()
        with torch.no_grad():
            logits = model(time_series, mask)
        outputs[name] = logits
        print(f"{name:12s}: {logits.shape}, 范围: [{logits.min():.3f}, {logits.max():.3f}]")
    
    # 验证所有输出形状相同
    shapes = [output.shape for output in outputs.values()]
    assert all(shape == shapes[0] for shape in shapes), f"输出形状不一致: {shapes}"
    
    print("✓ 所有模型输出形状一致")

if __name__ == "__main__":
    try:
        # 运行测试
        results = test_model_integration()
        test_consistency()
        
        print("\n🎉 所有测试完成!")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()