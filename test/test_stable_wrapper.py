#!/usr/bin/env python3
"""
测试模型包装器的nan/inf修复效果
"""

import sys
import torch
import numpy as np
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from models import LinearNoiseSDEContiformer
from utils.model_wrapper import StableModelWrapper, NumericallyStableTrainer
import torch.optim as optim


def test_stable_wrapper():
    """测试稳定包装器"""
    print("="*60)
    print("测试模型包装器")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}\n")
    
    # 创建原始模型
    model = LinearNoiseSDEContiformer(
        input_dim=3,
        num_classes=8,
        hidden_channels=64,
        contiformer_dim=128,
        n_heads=8,
        n_layers=4,
        dropout=0.1,
        use_sde=True,
        use_contiformer=True,
        use_cga=True,
        cga_group_dim=16,
        cga_heads=4,
        cga_temperature=1.0,
        cga_gate_threshold=0.5
    ).to(device)
    
    # 包装模型
    wrapped_model = StableModelWrapper(model)
    wrapped_model.eval()
    
    # 测试不同输入
    test_cases = [
        ("正常输入", torch.randn(4, 200, 3)),
        ("包含NaN", torch.randn(4, 200, 3)),
        ("包含Inf", torch.randn(4, 200, 3)),
        ("极小值", torch.randn(4, 200, 3) * 1e-10),
        ("极大值", torch.randn(4, 200, 3) * 1e10),
        ("混合问题", torch.randn(4, 200, 3))
    ]
    
    # 添加问题数据
    test_cases[1][1][0, :10, :] = float('nan')  # 添加NaN
    test_cases[2][1][0, :10, :] = float('inf')  # 添加Inf
    test_cases[5][1][0, :50, :] = float('nan')  # 混合NaN
    test_cases[5][1][1, :50, :] = float('inf')  # 混合Inf
    test_cases[5][1][2, :50, :] = 1e15  # 极大值
    
    for name, test_input in test_cases:
        print(f"\n测试: {name}")
        test_input = test_input.to(device)
        
        # 检查输入
        has_nan = torch.isnan(test_input).any()
        has_inf = torch.isinf(test_input).any()
        print(f"  输入: NaN={has_nan}, Inf={has_inf}")
        
        if not has_nan and not has_inf:
            print(f"  输入范围: [{test_input.min():.2e}, {test_input.max():.2e}]")
        
        # 前向传播
        with torch.no_grad():
            try:
                output = wrapped_model(test_input)
                
                # 检查输出
                out_has_nan = torch.isnan(output).any()
                out_has_inf = torch.isinf(output).any()
                
                if out_has_nan or out_has_inf:
                    print(f"  ❌ 输出仍有问题: NaN={out_has_nan}, Inf={out_has_inf}")
                else:
                    print(f"  ✓ 输出正常")
                    print(f"    形状: {output.shape}")
                    print(f"    范围: [{output.min():.4f}, {output.max():.4f}]")
                    
            except Exception as e:
                print(f"  ❌ 错误: {e}")


def test_stable_training():
    """测试稳定训练"""
    print("\n" + "="*60)
    print("测试稳定训练")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建模型和训练器
    model = LinearNoiseSDEContiformer(
        input_dim=3,
        num_classes=8,
        hidden_channels=64,
        contiformer_dim=128,
        n_heads=8,
        n_layers=4,
        dropout=0.1,
        use_sde=True,
        use_contiformer=True,
        use_cga=False  # 暂时关闭CGA以减少复杂性
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    trainer = NumericallyStableTrainer(model, optimizer, device)
    
    # 创建测试数据
    batch_data = torch.randn(8, 200, 3)
    batch_labels = torch.randint(0, 8, (8,))
    
    # 添加一些问题数据
    batch_data[0, :10, :] = float('nan')
    batch_data[1, :10, :] = 1e10
    
    print("\n训练5个步骤...")
    for step in range(5):
        loss, outputs = trainer.train_step(batch_data, batch_labels)
        
        print(f"\n步骤 {step+1}:")
        print(f"  损失: {loss:.4f}")
        print(f"  输出形状: {outputs.shape}")
        
        # 检查输出
        if torch.isnan(outputs).any():
            print(f"  ⚠️ 输出包含NaN")
        elif torch.isinf(outputs).any():
            print(f"  ⚠️ 输出包含Inf")
        else:
            print(f"  ✓ 输出正常")


def main():
    print("="*60)
    print("NaN/Inf修复测试")
    print("="*60)
    
    # 测试包装器
    test_stable_wrapper()
    
    # 测试稳定训练
    test_stable_training()
    
    print("\n" + "="*60)
    print("测试完成")
    print("="*60)
    print("\n建议：在main.py中使用StableModelWrapper包装模型")
    print("from utils.model_wrapper import StableModelWrapper")
    print("model = StableModelWrapper(original_model)")


if __name__ == "__main__":
    main()