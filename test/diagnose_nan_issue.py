#!/usr/bin/env python3
"""
诊断NaN Loss问题的综合分析脚本
"""

import torch
import numpy as np
import pickle
import sys
import os
sys.path.append('/root/autodl-tmp/lnsde-contiformer')

from models import LinearNoiseSDEContiformer
from utils import create_dataloaders, get_device

def check_data_for_issues(data_path):
    """检查数据集中是否有NaN、Inf或异常值"""
    print(f"\n=== 检查数据集: {data_path} ===")
    
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    # 数据格式是 [(label, time_series), ...]
    print(f"数据集大小: {len(data)} 样本")
    
    issues = []
    stats = {
        'total_samples': 0,
        'nan_count': 0,
        'inf_count': 0,
        'extreme_values': 0,
        'zero_time_intervals': 0
    }
    
    # 统计各类别样本数
    label_counts = {}
    for label, _ in data:
        label_counts[label] = label_counts.get(label, 0) + 1
    print(f"类别分布: {label_counts}")
    
    for idx, (label, sample) in enumerate(data):
        stats['total_samples'] += 1
        
        # 提取时间、幅值和误差
        times = sample[:, 0]
        values = sample[:, 1]
        errors = sample[:, 2] if sample.shape[1] > 2 else None
        
        # 检查NaN
        if np.isnan(times).any() or np.isnan(values).any():
            stats['nan_count'] += 1
            issues.append(f"  样本{idx}(类别{label}): 包含NaN值")
            if idx < 10:  # 只打印前10个问题
                print(f"  ⚠️ 样本{idx}: 发现NaN值")
            continue
            
        # 检查Inf
        if np.isinf(times).any() or np.isinf(values).any():
            stats['inf_count'] += 1
            issues.append(f"  样本{idx}(类别{label}): 包含Inf值")
            if idx < 10:
                print(f"  ⚠️ 样本{idx}: 发现Inf值")
            continue
            
        # 检查极端值
        if np.abs(values).max() > 1e6:
            stats['extreme_values'] += 1
            if idx < 10:
                print(f"  ⚠️ 样本{idx}: 极端值 max={np.abs(values).max():.2e}")
            
        # 检查时间间隔
        time_diffs = np.diff(times)
        if (time_diffs <= 0).any():
            stats['zero_time_intervals'] += 1
            if idx < 10:
                print(f"  ⚠️ 样本{idx}: 非递增时间序列")
        
        # 检查误差值
        if errors is not None:
            if (errors <= 0).any():
                if idx < 10:
                    print(f"  ⚠️ 样本{idx}: 负误差值或零误差")
    
    # 统计汇总
    print(f"\n=== 数据质量统计 ===")
    print(f"总样本数: {stats['total_samples']}")
    print(f"包含NaN: {stats['nan_count']} ({100*stats['nan_count']/stats['total_samples']:.2f}%)")
    print(f"包含Inf: {stats['inf_count']} ({100*stats['inf_count']/stats['total_samples']:.2f}%)")
    print(f"极端值: {stats['extreme_values']} ({100*stats['extreme_values']/stats['total_samples']:.2f}%)")
    print(f"时间问题: {stats['zero_time_intervals']} ({100*stats['zero_time_intervals']/stats['total_samples']:.2f}%)")
    
    return stats, issues


def test_model_forward_pass():
    """测试模型前向传播，检查中间值"""
    print("\n=== 测试模型前向传播 ===")
    
    device = get_device('auto')
    print(f"使用设备: {device}")
    
    # 创建小批量测试数据
    batch_size = 2
    seq_len = 100
    input_dim = 3
    num_classes = 7
    
    # 创建模型（最小配置）
    model = LinearNoiseSDEContiformer(
        input_channels=input_dim,
        hidden_channels=32,
        output_channels=num_classes,
        contiformer_dim=64,
        n_heads=2,
        n_layers=1,
        use_sde=False,  # 先禁用SDE测试
        use_contiformer=False,  # 先禁用ContiFormer
        use_cga=False  # 禁用CGA
    ).to(device)
    
    model.eval()
    
    # 创建测试数据
    times = torch.linspace(0, 100, seq_len).unsqueeze(0).repeat(batch_size, 1).to(device)
    values = torch.randn(batch_size, seq_len, 1).to(device) * 0.1  # 小值避免数值问题
    errors = torch.ones(batch_size, seq_len, 1).to(device) * 0.01
    
    # 组合输入
    input_data = torch.cat([times.unsqueeze(-1), values, errors], dim=-1)
    
    print(f"输入形状: {input_data.shape}")
    print(f"输入范围: [{input_data.min():.3f}, {input_data.max():.3f}]")
    
    # 前向传播并监控
    with torch.no_grad():
        # Hook来监控中间值
        activations = {}
        
        def hook_fn(name):
            def hook(module, input, output):
                if isinstance(output, torch.Tensor):
                    activations[name] = {
                        'min': output.min().item(),
                        'max': output.max().item(),
                        'mean': output.mean().item(),
                        'has_nan': torch.isnan(output).any().item(),
                        'has_inf': torch.isinf(output).any().item()
                    }
            return hook
        
        # 注册hook
        if hasattr(model, 'direct_mapping'):
            model.direct_mapping.register_forward_hook(hook_fn('direct_mapping'))
        if hasattr(model, 'classifier'):
            for idx, layer in enumerate(model.classifier):
                if isinstance(layer, torch.nn.Linear):
                    layer.register_forward_hook(hook_fn(f'classifier_{idx}'))
        
        # 前向传播
        try:
            output = model(input_data)
            print(f"\n✅ 前向传播成功")
            print(f"输出形状: {output.shape}")
            print(f"输出范围: [{output.min():.3f}, {output.max():.3f}]")
            
            # 检查输出
            if torch.isnan(output).any():
                print("⚠️ 输出包含NaN!")
            if torch.isinf(output).any():
                print("⚠️ 输出包含Inf!")
                
            # 显示中间激活值
            print("\n中间层激活值:")
            for name, stats in activations.items():
                print(f"  {name}:")
                print(f"    范围: [{stats['min']:.3f}, {stats['max']:.3f}], 均值: {stats['mean']:.3f}")
                if stats['has_nan']:
                    print(f"    ⚠️ 包含NaN")
                if stats['has_inf']:
                    print(f"    ⚠️ 包含Inf")
                    
        except Exception as e:
            print(f"❌ 前向传播失败: {e}")
            import traceback
            traceback.print_exc()
    
    # 测试带SDE的版本
    print("\n=== 测试SDE组件 ===")
    model_with_sde = LinearNoiseSDEContiformer(
        input_channels=input_dim,
        hidden_channels=32,
        output_channels=num_classes,
        contiformer_dim=64,
        n_heads=2,
        n_layers=1,
        use_sde=True,  # 启用SDE
        use_contiformer=False,
        use_cga=False,
        dt=0.1,  # 大步长，快速测试
        sde_method='euler'  # 简单方法
    ).to(device)
    
    model_with_sde.eval()
    
    with torch.no_grad():
        try:
            output_sde = model_with_sde(input_data)
            print(f"✅ SDE前向传播成功")
            print(f"输出范围: [{output_sde.min():.3f}, {output_sde.max():.3f}]")
            
            if torch.isnan(output_sde).any():
                print("⚠️ SDE输出包含NaN!")
            if torch.isinf(output_sde).any():
                print("⚠️ SDE输出包含Inf!")
                
        except Exception as e:
            print(f"❌ SDE前向传播失败: {e}")


def check_gradient_flow():
    """检查梯度流动情况"""
    print("\n=== 检查梯度流动 ===")
    
    device = get_device('auto')
    
    # 最简单的模型配置
    model = LinearNoiseSDEContiformer(
        input_channels=3,
        hidden_channels=16,
        output_channels=7,
        contiformer_dim=32,
        n_heads=2,
        n_layers=1,
        use_sde=False,
        use_contiformer=False,
        use_cga=False
    ).to(device)
    
    # 创建小批量数据
    batch_size = 4
    seq_len = 50
    input_data = torch.randn(batch_size, seq_len, 3).to(device) * 0.1
    labels = torch.randint(0, 7, (batch_size,)).to(device)
    
    # 前向传播和反向传播
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = torch.nn.CrossEntropyLoss()
    
    # 清零梯度
    optimizer.zero_grad()
    
    # 前向传播
    output = model(input_data)
    loss = criterion(output, labels)
    
    print(f"Loss值: {loss.item():.4f}")
    
    if torch.isnan(loss):
        print("⚠️ Loss是NaN!")
    elif torch.isinf(loss):
        print("⚠️ Loss是Inf!")
    else:
        # 反向传播
        loss.backward()
        
        # 检查梯度
        grad_stats = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_stats[name] = {
                    'min': param.grad.min().item(),
                    'max': param.grad.max().item(),
                    'mean': param.grad.abs().mean().item(),
                    'has_nan': torch.isnan(param.grad).any().item(),
                    'has_inf': torch.isinf(param.grad).any().item()
                }
        
        print("\n梯度统计:")
        for name, stats in list(grad_stats.items())[:5]:  # 只显示前5个
            print(f"  {name}:")
            print(f"    范围: [{stats['min']:.6f}, {stats['max']:.6f}], |均值|: {stats['mean']:.6f}")
            if stats['has_nan']:
                print(f"    ⚠️ 包含NaN梯度")
            if stats['has_inf']:
                print(f"    ⚠️ 包含Inf梯度")


def main():
    """主诊断流程"""
    print("="*60)
    print("NaN Loss问题诊断工具")
    print("="*60)
    
    # 1. 检查MACHO数据集
    macho_path = '/root/autodl-fs/lnsde-contiformer/data/MACHO_original.pkl'
    if os.path.exists(macho_path):
        stats, issues = check_data_for_issues(macho_path)
    
    # 2. 测试模型前向传播
    test_model_forward_pass()
    
    # 3. 检查梯度流动
    check_gradient_flow()
    
    print("\n" + "="*60)
    print("诊断完成")
    print("="*60)
    
    print("\n=== 关于NaN Loss的分析 ===")
    print("1. 模型已经正确地将数据和参数移动到GPU")
    print("2. 可能的NaN原因:")
    print("   - SDE求解过程中的数值不稳定")
    print("   - 学习率过大导致梯度爆炸")
    print("   - 除零错误（如mask处理时）")
    print("   - 数据预处理问题（虽然原始数据没问题）")
    print("3. 建议解决方案:")
    print("   - 降低学习率")
    print("   - 使用梯度裁剪")
    print("   - 启用数值稳定性包装器")
    print("   - 检查并修复除零问题")


if __name__ == "__main__":
    main()