#!/usr/bin/env python3
"""
修复NaN损失问题的改进方案
主要针对Lion优化器的数值稳定性
"""

import torch
import torch.nn as nn
from pathlib import Path

def create_stable_optimizer_config():
    """
    创建稳定的优化器配置
    """
    return {
        # 方案1：使用更保守的Lion配置
        'lion_stable': {
            'optimizer_type': 'Lion',
            'base_lr_multiplier': 0.05,  # 进一步降低基础学习率
            'weight_decay_multiplier': 2.0,  # 适度权重衰减
            'scheduler_type': 'ReduceLROnPlateau',  # 替换为基于性能的调度器
            'scheduler_config': {
                'mode': 'max',
                'factor': 0.5,
                'patience': 5,
                'threshold': 0.001,
                'min_lr': 1e-8
            },
            'gradient_clipping': 0.5,  # 更强的梯度裁剪
        },
        
        # 方案2：混合调度器
        'lion_hybrid': {
            'optimizer_type': 'Lion',
            'base_lr_multiplier': 0.08,
            'weight_decay_multiplier': 2.5,
            'scheduler_type': 'WarmupCosine',  # 预热+余弦
            'warmup_epochs': 10,
            'scheduler_config': {
                'T_max': 200,
                'eta_min': 1e-7,
                'warmup_start_lr': 1e-8
            },
            'gradient_clipping': 0.8,
        },
        
        # 方案3：AdamW备选（稳定性优先）
        'adamw_stable': {
            'optimizer_type': 'AdamW',
            'base_lr_multiplier': 1.0,  # AdamW可以用原始学习率
            'weight_decay_multiplier': 1.0,
            'scheduler_type': 'ReduceLROnPlateau',
            'scheduler_config': {
                'mode': 'max',
                'factor': 0.7,
                'patience': 8,
                'threshold': 0.005
            },
            'gradient_clipping': 1.0,
        }
    }

def create_stable_sde_config():
    """
    创建数值稳定的SDE配置
    """
    return {
        'sde_solver': 'euler',  # 使用更稳定的求解器
        'dt': 0.01,  # 更小的时间步长
        'rtol': 1e-4,  # 更严格的相对容差
        'atol': 1e-5,  # 更严格的绝对容差
        'min_diffusion': 0.01,  # 更小的最小扩散
        'max_diffusion': 1.0,   # 添加最大扩散限制
        'stability_check': True,  # 启用稳定性检查
    }

def enhance_numerical_stability():
    """
    数值稳定性增强建议
    """
    improvements = {
        'model_modifications': [
            "在SDE扩散函数中添加梯度裁剪",
            "使用LayerNorm替代BatchNorm", 
            "在关键计算中添加数值保护",
            "限制SDE扩散系数的范围"
        ],
        
        'training_modifications': [
            "添加损失值监控和早停",
            "实现梯度监控和异常检测",
            "使用混合精度训练但禁用梯度缩放",
            "添加学习率预热阶段"
        ],
        
        'monitoring_suggestions': [
            "监控每个epoch的梯度范数",
            "记录SDE求解步骤的稳定性指标", 
            "跟踪各层激活值的统计信息",
            "设置损失阈值自动停止"
        ]
    }
    return improvements

if __name__ == "__main__":
    # 输出配置建议
    print("=== NaN损失修复建议 ===")
    
    optimizer_configs = create_stable_optimizer_config()
    sde_config = create_stable_sde_config()
    stability_enhancements = enhance_numerical_stability()
    
    print("\n1. 优化器配置选项:")
    for name, config in optimizer_configs.items():
        print(f"\n{name}:")
        for key, value in config.items():
            print(f"  {key}: {value}")
    
    print(f"\n2. SDE配置建议:")
    for key, value in sde_config.items():
        print(f"  {key}: {value}")
    
    print(f"\n3. 数值稳定性增强:")
    for category, items in stability_enhancements.items():
        print(f"\n{category}:")
        for item in items:
            print(f"  - {item}")