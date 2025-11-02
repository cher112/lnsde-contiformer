#!/usr/bin/env python3
"""
调试脚本：逐个样本检查每个数据集中导致nan/inf的具体样本
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import argparse

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent))

from utils.dataloader import create_dataloaders
from utils.model_utils import create_model
import warnings
warnings.filterwarnings('ignore')

def check_tensor_validity(tensor, name="tensor"):
    """检查张量的有效性"""
    if tensor is None:
        return f"{name} is None"
    
    issues = []
    if torch.isnan(tensor).any():
        nan_count = torch.isnan(tensor).sum().item()
        nan_positions = torch.where(torch.isnan(tensor))
        issues.append(f"NaN values: {nan_count}/{tensor.numel()} at positions {nan_positions[0][:5].tolist() if len(nan_positions[0]) > 0 else []}")
    
    if torch.isinf(tensor).any():
        inf_count = torch.isinf(tensor).sum().item()
        inf_positions = torch.where(torch.isinf(tensor))
        issues.append(f"Inf values: {inf_count}/{tensor.numel()} at positions {inf_positions[0][:5].tolist() if len(inf_positions[0]) > 0 else []}")
    
    # 检查数值范围
    if tensor.numel() > 0:
        valid_mask = ~torch.isnan(tensor) & ~torch.isinf(tensor)
        if valid_mask.any():
            min_val = tensor[valid_mask].min().item()
            max_val = tensor[valid_mask].max().item()
            mean_val = tensor[valid_mask].mean().item()
            
            if abs(min_val) > 1e10 or abs(max_val) > 1e10:
                issues.append(f"Extreme values: min={min_val:.2e}, max={max_val:.2e}")
    
    return issues if issues else None

def test_data_sample(data, label, idx, dataset_name):
    """测试数据样本本身"""
    issues = []
    
    # 检查数据
    data_issues = check_tensor_validity(data, f"data")
    if data_issues:
        issues.extend(data_issues)
    
    # 检查是否全零
    if (data == 0).all():
        issues.append("All zeros in data")
    
    # 检查是否有极小值
    if data.numel() > 0:
        abs_data = torch.abs(data)
        non_zero_mask = abs_data > 0
        if non_zero_mask.any():
            min_non_zero = abs_data[non_zero_mask].min().item()
            if min_non_zero < 1e-10:
                issues.append(f"Very small values: {min_non_zero:.2e}")
    
    if issues:
        print(f"\n[{dataset_name}] Sample {idx} (Label: {label.item()}) data issues:")
        for issue in issues:
            print(f"  - {issue}")
        
        # 打印详细统计
        print(f"  Stats:")
        print(f"    Shape: {data.shape}")
        if not torch.isnan(data).all() and not torch.isinf(data).all():
            valid_mask = ~torch.isnan(data) & ~torch.isinf(data)
            if valid_mask.any():
                print(f"    Min: {data[valid_mask].min().item():.6f}")
                print(f"    Max: {data[valid_mask].max().item():.6f}")
                print(f"    Mean: {data[valid_mask].mean().item():.6f}")
                print(f"    Std: {data[valid_mask].std().item():.6f}")
        
        zero_count = (data == 0).sum().item()
        print(f"    Zero values: {zero_count}/{data.numel()} ({100*zero_count/data.numel():.1f}%)")
        
        return True
    
    return False

def test_model_forward(model, data, label, idx, dataset_name):
    """测试模型前向传播"""
    model.eval()
    issues = {}
    
    with torch.no_grad():
        try:
            # 记录中间输出
            intermediate_outputs = {}
            
            # Hook来捕获中间层输出
            def get_activation(name):
                def hook(model, input, output):
                    intermediate_outputs[name] = output.detach()
                return hook
            
            hooks = []
            
            # 注册hooks到关键层
            if hasattr(model, 'embedding'):
                hook = model.embedding.register_forward_hook(get_activation('embedding'))
                hooks.append(hook)
            
            # 对Contiformer的每一层注册hook
            if hasattr(model, 'contiformer') and hasattr(model.contiformer, 'layers'):
                for i, layer in enumerate(model.contiformer.layers):
                    hook = layer.register_forward_hook(get_activation(f'contiformer_layer_{i}'))
                    hooks.append(hook)
            
            # LNSDE hook
            if hasattr(model, 'lnsde'):
                hook = model.lnsde.register_forward_hook(get_activation('lnsde'))
                hooks.append(hook)
            
            # CGA hook  
            if hasattr(model, 'cga') and model.use_cga:
                hook = model.cga.register_forward_hook(get_activation('cga'))
                hooks.append(hook)
            
            # 前向传播
            output = model(data.unsqueeze(0))
            
            # 检查输出
            output_issues = check_tensor_validity(output, "output")
            if output_issues:
                issues['output'] = output_issues
            
            # 检查中间输出
            for name, activation in intermediate_outputs.items():
                activation_issues = check_tensor_validity(activation, name)
                if activation_issues:
                    issues[name] = activation_issues
            
            # 移除hooks
            for hook in hooks:
                hook.remove()
            
        except Exception as e:
            issues['error'] = f"Forward pass error: {str(e)}"
    
    if issues:
        print(f"\n[{dataset_name}] Sample {idx} model forward issues:")
        for layer, issue in issues.items():
            print(f"  - {layer}: {issue}")
        return True
    
    return False

def test_dataset(dataset_name, dataset_id, max_samples=100):
    """测试数据集"""
    print(f"\n{'='*70}")
    print(f"Testing {dataset_name} Dataset (ID: {dataset_id})")
    print(f"{'='*70}")
    
    # 创建参数
    parser = argparse.ArgumentParser()
    # 数据集参数
    parser.add_argument('--dataset', type=int, default=dataset_id)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--use_original', action='store_true', default=False)
    parser.add_argument('--use_resampling', action='store_true', default=False)
    parser.add_argument('--resample_on_fly', action='store_true', default=False)
    parser.add_argument('--resampled_data_path', type=str, default=None)
    parser.add_argument('--pin_memory', action='store_true', default=False)
    parser.add_argument('--prefetch_factor', type=int, default=2)
    
    # 模型参数
    parser.add_argument('--model_type', type=int, default=2)  # LinearNoiseSDEContiformer
    parser.add_argument('--hidden_channels', type=int, default=64)
    parser.add_argument('--contiformer_dim', type=int, default=128)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--n_layers', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--temperature', type=float, default=0.07)
    parser.add_argument('--use_sde', type=int, default=1)
    parser.add_argument('--use_contiformer', type=int, default=1)
    parser.add_argument('--use_cga', type=int, default=1)
    parser.add_argument('--cga_group_dim', type=int, default=16)
    parser.add_argument('--cga_heads', type=int, default=4)
    parser.add_argument('--cga_temperature', type=float, default=1.0)
    parser.add_argument('--cga_gate_threshold', type=float, default=0.5)
    parser.add_argument('--sde_config', type=int, default=1)
    
    args = parser.parse_args([])
    args.dataset = dataset_id
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 创建数据加载器
    try:
        train_loader, val_loader, test_loader, num_classes = create_dataloaders(args)
        print(f"Successfully loaded dataset with {num_classes} classes")
        print(f"Train samples: {len(train_loader.dataset)}")
        print(f"Val samples: {len(val_loader.dataset)}")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        import traceback
        traceback.print_exc()
        return [], []
    
    # 获取数据集特定配置
    from utils.config import get_dataset_specific_params
    dataset_config = get_dataset_specific_params(dataset_id, args)
    
    # 创建模型
    model = create_model(args.model_type, num_classes, args, dataset_config).to(args.device)
    
    print(f"Model created successfully on {args.device}")
    
    # 测试训练集样本
    print(f"\n--- Checking Training Set (first {max_samples} samples) ---")
    train_data_issues = []
    train_model_issues = []
    
    for idx, (data, label) in enumerate(train_loader):
        if idx >= max_samples:
            break
        
        data = data.to(args.device)
        label = label.to(args.device)
        
        # 先检查数据本身
        if test_data_sample(data.squeeze(0), label, idx, f"{dataset_name}_train"):
            train_data_issues.append(idx)
        
        # 再测试模型前向传播
        if not idx in train_data_issues:  # 只对数据正常的样本测试模型
            if test_model_forward(model, data.squeeze(0), label, idx, f"{dataset_name}_train"):
                train_model_issues.append(idx)
        
        if (idx + 1) % 50 == 0:
            print(f"  Progress: {idx + 1}/{max_samples} samples checked")
    
    # 测试验证集样本
    print(f"\n--- Checking Validation Set (first {max_samples} samples) ---")
    val_data_issues = []
    val_model_issues = []
    
    for idx, (data, label) in enumerate(val_loader):
        if idx >= max_samples:
            break
        
        data = data.to(args.device)
        label = label.to(args.device)
        
        if test_data_sample(data.squeeze(0), label, idx, f"{dataset_name}_val"):
            val_data_issues.append(idx)
        
        if not idx in val_data_issues:
            if test_model_forward(model, data.squeeze(0), label, idx, f"{dataset_name}_val"):
                val_model_issues.append(idx)
        
        if (idx + 1) % 50 == 0:
            print(f"  Progress: {idx + 1}/{max_samples} samples checked")
    
    # 打印总结
    print(f"\n{'='*50}")
    print(f"{dataset_name} Dataset Summary:")
    print(f"{'='*50}")
    print(f"Training Set:")
    print(f"  Data issues: {len(train_data_issues)} samples")
    if train_data_issues:
        print(f"    Indices: {train_data_issues[:10]}{'...' if len(train_data_issues) > 10 else ''}")
    print(f"  Model forward issues: {len(train_model_issues)} samples")
    if train_model_issues:
        print(f"    Indices: {train_model_issues[:10]}{'...' if len(train_model_issues) > 10 else ''}")
    
    print(f"\nValidation Set:")
    print(f"  Data issues: {len(val_data_issues)} samples")
    if val_data_issues:
        print(f"    Indices: {val_data_issues[:10]}{'...' if len(val_data_issues) > 10 else ''}")
    print(f"  Model forward issues: {len(val_model_issues)} samples")
    if val_model_issues:
        print(f"    Indices: {val_model_issues[:10]}{'...' if len(val_model_issues) > 10 else ''}")
    
    return {
        'train_data': train_data_issues,
        'train_model': train_model_issues,
        'val_data': val_data_issues,
        'val_model': val_model_issues
    }

def main():
    """主函数"""
    print("="*70)
    print("NaN/Inf Sample Detection Tool")
    print("="*70)
    
    datasets = [
        ('ASAS', 1),
        ('LINEAR', 2),
        ('MACHO', 3)
    ]
    
    all_results = {}
    
    # 测试每个数据集
    for dataset_name, dataset_id in datasets:
        results = test_dataset(dataset_name, dataset_id, max_samples=200)
        all_results[dataset_name] = results
        
        # 如果发现问题，打印警告
        total_issues = (len(results['train_data']) + len(results['train_model']) + 
                       len(results['val_data']) + len(results['val_model']))
        if total_issues > 0:
            print(f"\n⚠️  WARNING: Found {total_issues} total issues in {dataset_name} dataset!")
    
    # 最终总结
    print("\n" + "="*70)
    print("FINAL REPORT")
    print("="*70)
    
    for dataset_name, results in all_results.items():
        total = (len(results['train_data']) + len(results['train_model']) + 
                len(results['val_data']) + len(results['val_model']))
        
        print(f"\n{dataset_name} Dataset:")
        print(f"  Training data issues: {len(results['train_data'])}")
        print(f"  Training model issues: {len(results['train_model'])}")
        print(f"  Validation data issues: {len(results['val_data'])}")
        print(f"  Validation model issues: {len(results['val_model'])}")
        print(f"  Total issues: {total}")
        
        if total == 0:
            print(f"  ✓ No NaN/Inf issues detected")
        else:
            print(f"  ⚠️ Issues detected - needs investigation")

if __name__ == "__main__":
    main()