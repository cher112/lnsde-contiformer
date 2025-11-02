#!/usr/bin/env python3
"""
测试CGA（类别感知分组注意力）模块的功能
验证与Contiformer串联的效果
"""

import torch
import torch.nn as nn
import sys
import os
sys.path.append('/root/autodl-tmp/lnsde-contiformer')

from models.cga_module import CategoryAwareGroupedAttention, CGAClassifier
from models.contiformer import ContiFormerModule
from models.linear_noise_sde import LinearNoiseSDEContiformer


def test_cga_module():
    """测试独立的CGA模块"""
    print("=" * 60)
    print("测试1: 独立CGA模块")
    print("=" * 60)
    
    # 设置参数
    batch_size = 4
    seq_len = 100
    input_dim = 128  # Contiformer输出维度
    num_classes = 3
    
    # 创建CGA模块
    cga = CategoryAwareGroupedAttention(
        input_dim=input_dim,
        num_classes=num_classes,
        group_dim=64,
        n_heads=4,
        temperature=0.1,
        gate_threshold=0.5
    )
    
    # 创建测试输入
    x = torch.randn(batch_size, seq_len, input_dim)
    mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
    mask[:, seq_len//2:] = False  # 后半部分无效
    
    # 前向传播
    output, class_representations = cga(x, mask)
    
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    print(f"类别表示形状: {class_representations.shape}")
    print(f"输出维度保持: {output.shape == x.shape}")
    
    # 测试注意力权重获取
    attn_weights = cga.get_class_attention_weights(x, class_idx=0, mask=mask)
    print(f"注意力权重形状: {attn_weights.shape}")
    
    print("\n✓ CGA模块测试通过\n")


def test_cga_classifier():
    """测试CGA分类器"""
    print("=" * 60)
    print("测试2: CGA分类器")
    print("=" * 60)
    
    batch_size = 4
    seq_len = 100
    input_dim = 128
    num_classes = 3
    
    # 创建CGA分类器
    cga_classifier = CGAClassifier(
        input_dim=input_dim,
        num_classes=num_classes,
        cga_config={
            'group_dim': 64,
            'n_heads': 4,
            'temperature': 0.1,
            'gate_threshold': 0.5
        },
        dropout=0.1
    )
    
    # 测试输入
    x = torch.randn(batch_size, seq_len, input_dim)
    mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
    
    # 前向传播
    logits, cga_features, class_representations = cga_classifier(x, mask)
    
    print(f"输入形状: {x.shape}")
    print(f"Logits形状: {logits.shape}")
    print(f"CGA特征形状: {cga_features.shape}")
    print(f"类别表示形状: {class_representations.shape}")
    
    # 验证输出
    assert logits.shape == (batch_size, num_classes), "Logits形状错误"
    assert cga_features.shape == x.shape, "CGA特征形状错误"
    
    print("\n✓ CGA分类器测试通过\n")


def test_contiformer_cga_integration():
    """测试Contiformer + CGA集成"""
    print("=" * 60)
    print("测试3: Contiformer + CGA集成")
    print("=" * 60)
    
    batch_size = 2
    seq_len = 50
    input_dim = 3  # [time, mag, errmag]
    hidden_dim = 128
    contiformer_dim = 128
    num_classes = 3
    
    # 创建完整模型（仅Contiformer + CGA，不使用SDE）
    model = LinearNoiseSDEContiformer(
        input_dim=input_dim,
        hidden_channels=hidden_dim,
        num_classes=num_classes,
        contiformer_dim=contiformer_dim,
        n_heads=8,
        n_layers=2,
        use_sde=0,  # 不使用SDE
        use_contiformer=1,  # 使用Contiformer
        use_cga=1,  # 使用CGA
        cga_config={
            'group_dim': 64,
            'n_heads': 4,
            'temperature': 0.1,
            'gate_threshold': 0.5
        }
    )
    
    # 创建测试数据
    time_series = torch.randn(batch_size, seq_len, input_dim)
    times = torch.linspace(0, 1, seq_len).unsqueeze(0).repeat(batch_size, 1)
    mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
    
    # 前向传播
    logits, features = model(time_series, times, mask)
    
    print(f"输入时间序列形状: {time_series.shape}")
    print(f"时间戳形状: {times.shape}")
    print(f"输出logits形状: {logits.shape}")
    print(f"特征形状: {features.shape}")
    
    # 验证输出
    assert logits.shape == (batch_size, num_classes), "Logits形状错误"
    
    print("\n✓ Contiformer + CGA集成测试通过\n")


def test_full_model_with_cga():
    """测试完整模型（LNSDE + Contiformer + CGA）"""
    print("=" * 60)
    print("测试4: 完整模型（LNSDE + Contiformer + CGA）")
    print("=" * 60)
    
    batch_size = 2
    seq_len = 30  # 减少序列长度以加快SDE求解
    input_dim = 3
    hidden_dim = 64
    contiformer_dim = 64
    num_classes = 3
    
    # 创建完整模型
    model = LinearNoiseSDEContiformer(
        input_dim=input_dim,
        hidden_channels=hidden_dim,
        num_classes=num_classes,
        contiformer_dim=contiformer_dim,
        n_heads=4,
        n_layers=2,
        use_sde=1,  # 使用SDE
        use_contiformer=1,  # 使用Contiformer  
        use_cga=1,  # 使用CGA
        cga_config={
            'group_dim': 32,
            'n_heads': 2,
            'temperature': 0.1,
            'gate_threshold': 0.5
        },
        dt=0.01,
        sde_method='euler'
    )
    
    # 创建测试数据
    time_series = torch.randn(batch_size, seq_len, input_dim)
    times = torch.linspace(0.1, 1.0, seq_len).unsqueeze(0).repeat(batch_size, 1)
    mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
    
    # 前向传播
    print("执行前向传播...")
    logits, features = model(time_series, times, mask)
    
    print(f"输入时间序列形状: {time_series.shape}")
    print(f"输出logits形状: {logits.shape}")
    print(f"特征形状: {features.shape}")
    
    # 验证输出
    assert logits.shape == (batch_size, num_classes), "Logits形状错误"
    
    print("\n✓ 完整模型测试通过\n")


def test_gradient_flow():
    """测试梯度流动"""
    print("=" * 60)
    print("测试5: 梯度流动测试")
    print("=" * 60)
    
    batch_size = 2
    seq_len = 20
    input_dim = 3
    num_classes = 3
    
    # 创建模型
    model = LinearNoiseSDEContiformer(
        input_dim=input_dim,
        hidden_channels=32,
        num_classes=num_classes,
        contiformer_dim=32,
        n_heads=2,
        n_layers=1,
        use_sde=0,  # 不使用SDE以加快测试
        use_contiformer=1,
        use_cga=1,
        cga_config={
            'group_dim': 16,
            'n_heads': 2,
            'temperature': 0.1,
            'gate_threshold': 0.5
        }
    )
    
    # 创建测试数据
    time_series = torch.randn(batch_size, seq_len, input_dim, requires_grad=True)
    times = torch.linspace(0, 1, seq_len).unsqueeze(0).repeat(batch_size, 1)
    mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
    targets = torch.randint(0, num_classes, (batch_size,))
    
    # 前向传播
    logits, features = model(time_series, times, mask)
    
    # 计算损失
    criterion = nn.CrossEntropyLoss()
    loss = criterion(logits, targets)
    
    # 反向传播
    loss.backward()
    
    # 检查梯度
    assert time_series.grad is not None, "输入梯度为None"
    assert not torch.isnan(time_series.grad).any(), "输入梯度包含NaN"
    
    # 检查模型参数梯度
    for name, param in model.named_parameters():
        if param.grad is not None:
            assert not torch.isnan(param.grad).any(), f"参数{name}的梯度包含NaN"
    
    print(f"损失值: {loss.item():.4f}")
    print(f"输入梯度范数: {time_series.grad.norm().item():.4f}")
    
    print("\n✓ 梯度流动测试通过\n")


def main():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("开始CGA模块测试")
    print("=" * 60 + "\n")
    
    # 设置随机种子
    torch.manual_seed(535411460)
    
    try:
        # 运行各项测试
        test_cga_module()
        test_cga_classifier()
        test_contiformer_cga_integration()
        test_full_model_with_cga()
        test_gradient_flow()
        
        print("\n" + "=" * 60)
        print("所有测试通过！✓")
        print("CGA模块已成功集成到LNSDE-Contiformer架构中")
        print("=" * 60 + "\n")
        
    except Exception as e:
        print(f"\n测试失败: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())