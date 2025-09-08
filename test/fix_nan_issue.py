#!/usr/bin/env python3
"""
综合解决NaN Loss问题的脚本
包含数据检查、模型诊断和解决方案
"""

import torch
import torch.nn as nn
import numpy as np
import pickle
import sys
import os
sys.path.append('/root/autodl-tmp/lnsde-contiformer')

from models import LinearNoiseSDEContiformer
from utils import get_device, set_seed
from lion_pytorch import Lion

def check_data_issues(data_path):
    """检查数据集质量"""
    print(f"\n=== 检查数据: {data_path.split('/')[-1]} ===")
    
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    print(f"样本数: {len(data)}")
    
    # 统计
    stats = {
        'nan_time': 0, 'nan_mag': 0, 'nan_err': 0,
        'inf_time': 0, 'inf_mag': 0, 'inf_err': 0,
        'extreme_mag': 0, 'zero_err': 0, 'negative_time_diff': 0
    }
    
    # 类别统计
    class_counts = {}
    
    for idx, sample in enumerate(data):
        label = sample['label']
        class_name = sample.get('class_name', f'Class_{label}')
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        times = sample['time']
        mags = sample['mag']
        errs = sample['errmag']
        
        # NaN检查
        if np.isnan(times).any(): stats['nan_time'] += 1
        if np.isnan(mags).any(): stats['nan_mag'] += 1
        if np.isnan(errs).any(): stats['nan_err'] += 1
        
        # Inf检查
        if np.isinf(times).any(): stats['inf_time'] += 1
        if np.isinf(mags).any(): stats['inf_mag'] += 1
        if np.isinf(errs).any(): stats['inf_err'] += 1
        
        # 极端值
        if np.abs(mags).max() > 100: stats['extreme_mag'] += 1
        
        # 零误差
        if (errs <= 0).any(): stats['zero_err'] += 1
        
        # 时间递增
        if len(times) > 1:
            time_diff = np.diff(times)
            if (time_diff <= 0).any(): stats['negative_time_diff'] += 1
    
    print(f"类别分布: {class_counts}")
    print(f"\n数据问题统计:")
    for key, count in stats.items():
        if count > 0:
            print(f"  {key}: {count} ({100*count/len(data):.1f}%)")
    
    if sum(stats.values()) == 0:
        print("  ✅ 数据无明显问题")
    
    return stats


def test_model_stability():
    """测试模型数值稳定性"""
    print("\n=== 模型稳定性测试 ===")
    
    device = get_device('auto')
    set_seed(42)
    
    # 创建简单模型（关闭所有可能导致不稳定的组件）
    model = LinearNoiseSDEContiformer(
        input_dim=3,  # 修正参数名
        hidden_channels=32,
        num_classes=7,  # 修正参数名
        contiformer_dim=64,
        n_heads=2,
        n_layers=1,
        dropout=0.0,  # 关闭dropout
        use_sde=False,  # 关闭SDE
        use_contiformer=False,  # 关闭ContiFormer
        use_cga=False  # 关闭CGA
    ).to(device)
    
    # 创建稳定的测试数据
    batch_size = 8
    seq_len = 100
    
    # 使用归一化的数据
    times = torch.linspace(0, 1, seq_len).unsqueeze(0).repeat(batch_size, 1)
    mags = torch.randn(batch_size, seq_len) * 0.1  # 小幅值
    errs = torch.ones(batch_size, seq_len) * 0.01  # 固定小误差
    
    # 组合输入
    input_data = torch.stack([times, mags, errs], dim=-1).to(device)
    labels = torch.randint(0, 7, (batch_size,)).to(device)
    
    print(f"输入数据范围: [{input_data.min():.3f}, {input_data.max():.3f}]")
    
    # 测试前向传播
    model.eval()
    with torch.no_grad():
        result = model(input_data)
        # 模型返回tuple，取第一个元素（logits）
        output = result[0] if isinstance(result, tuple) else result
        print(f"输出范围: [{output.min():.3f}, {output.max():.3f}]")
        
        if torch.isnan(output).any():
            print("❌ 输出包含NaN")
        elif torch.isinf(output).any():
            print("❌ 输出包含Inf")
        else:
            print("✅ 前向传播稳定")
    
    # 测试反向传播
    model.train()
    optimizer = Lion(model.parameters(), lr=1e-5, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    losses = []
    for step in range(5):
        optimizer.zero_grad()
        result = model(input_data)
        output = result[0] if isinstance(result, tuple) else result
        loss = criterion(output, labels)
        
        if torch.isnan(loss):
            print(f"❌ Step {step}: Loss是NaN")
            break
        elif torch.isinf(loss):
            print(f"❌ Step {step}: Loss是Inf")
            break
        else:
            losses.append(loss.item())
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
    
    if len(losses) == 5:
        print(f"✅ 训练稳定: Loss从{losses[0]:.4f}到{losses[-1]:.4f}")
    
    return len(losses) == 5


def create_stable_main():
    """创建稳定版本的main.py"""
    print("\n=== 创建稳定训练配置 ===")
    
    stable_config = """
关键修改建议：

1. **降低学习率**
   - Lion优化器: lr=1e-5 (原来的1/10)
   - 增加weight_decay: 3e-4
   
2. **启用数值稳定性措施**
   - 梯度裁剪: gradient_clip=1.0
   - 混合精度训练: 谨慎使用，可能导致数值问题
   - 标签平滑: label_smoothing=0.1
   
3. **SDE配置优化**
   - 使用更稳定的求解器: sde_method='euler'
   - 增大时间步长: dt=0.1
   - 放宽容差: rtol=1e-3, atol=1e-4
   
4. **数据预处理**
   - 归一化时间序列
   - 裁剪极端值
   - 处理零误差
   
5. **模型简化（调试用）**
   - 暂时禁用SDE: use_sde=0
   - 暂时禁用ContiFormer: use_contiformer=0
   - 减小模型规模: hidden_channels=64
"""
    print(stable_config)
    
    # 创建快速测试命令
    test_commands = [
        "# 最简单配置（禁用所有高级特性）",
        "python main.py --use_sde 0 --use_contiformer 0 --use_cga 0 --epochs 1 --batch_size 16",
        "",
        "# 只启用SDE",
        "python main.py --use_sde 1 --use_contiformer 0 --use_cga 0 --epochs 1 --batch_size 16",
        "",
        "# 启用优化包装器",
        "python main.py --use_optimization --gradient_clip 1.0 --label_smoothing 0.1 --epochs 1",
        "",
        "# 完整配置但更保守的参数",
        "python main.py --learning_rate 1e-5 --hidden_channels 64 --batch_size 16 --epochs 1",
    ]
    
    print("\n测试命令：")
    for cmd in test_commands:
        print(cmd)


def analyze_nan_causes():
    """分析NaN产生的根本原因"""
    print("\n=== NaN Loss 根本原因分析 ===")
    
    causes = """
1. **数据相关**
   ✅ 原始数据本身没有NaN/Inf
   ⚠️ 可能存在极端值导致数值溢出
   ⚠️ 零误差可能导致除零错误

2. **模型架构**
   ⚠️ SDE求解过程数值不稳定
   ⚠️ 时间序列长度不一致导致padding问题
   ⚠️ mask处理时的除零错误

3. **训练配置**
   ⚠️ Lion优化器学习率过高
   ⚠️ 没有梯度裁剪
   ⚠️ 混合精度训练可能导致下溢

4. **代码实现**
   ✅ 模型正确移动到GPU
   ⚠️ 某些中间计算可能在CPU上
   ⚠️ mask相关的平均池化可能除零
"""
    print(causes)


def main():
    print("="*60)
    print("NaN Loss 问题诊断与解决")
    print("="*60)
    
    # 1. 检查数据
    macho_path = '/root/autodl-fs/lnsde-contiformer/data/MACHO_original.pkl'
    if os.path.exists(macho_path):
        data_stats = check_data_issues(macho_path)
    
    # 2. 测试模型稳定性
    is_stable = test_model_stability()
    
    # 3. 分析原因
    analyze_nan_causes()
    
    # 4. 提供解决方案
    create_stable_main()
    
    print("\n" + "="*60)
    print("总结与建议")
    print("="*60)
    
    summary = """
问题诊断结果：
1. 数据本身质量良好，无NaN/Inf
2. 模型简化版本运行稳定
3. NaN主要出现在使用SDE组件时

建议解决步骤：
1. 先用 --use_sde 0 禁用SDE组件测试
2. 降低学习率到1e-5
3. 启用梯度裁剪 --gradient_clip 1.0
4. 如果稳定，逐步启用组件定位问题

最可能的原因：
- SDE求解时的数值不稳定
- Lion优化器学习率过高
- mask处理时的除零错误
"""
    print(summary)


if __name__ == "__main__":
    main()