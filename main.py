#!/usr/bin/env python3
"""
Neural SDE + ContiFormer Training Script
与当前架构匹配的主入口文件
"""

import argparse
import os
import torch
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# 导入优化器
import torch

# 导入模型
from models import (
    LangevinSDEContiformer,
    LinearNoiseSDEContiformer, 
    GeometricSDEContiformer
)

# 导入工具模块
from utils import (
    create_dataloaders, set_seed, get_device, clear_gpu_memory,
    get_dataset_specific_params, setup_sde_config, setup_dataset_mapping,
    create_model, load_model_checkpoint, setup_logging,
    TrainingManager
)
from utils.path_manager import get_timestamp_path


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Neural SDE + ContiFormer Training')
    
    # 模型选择
    parser.add_argument('--model_type', type=int, default=1,
                       choices=[1, 2, 3],
                       help='SDE模型类型 (1:langevin, 2:linear_noise, 3:geometric)')
    
    # 模型加载选项
    parser.add_argument('--load_model', type=int, default=2,
                       choices=[0, 1, 2],
                       help='模型加载选项 (0:不加载, 1:加载最新, 2:加载最优)')
    
    # 数据相关 - 使用数字代表数据集
    parser.add_argument('--dataset', type=int, default=1,
                       choices=[1, 2, 3],
                       help='数据集选择: 1=ASAS, 2=LINEAR, 3=MACHO')
    parser.add_argument('--use_original', action='store_true', default=False,
                       help='使用原始完整数据集')
    # 数据增强参数
    parser.add_argument('--use_enhanced', action='store_true', default=False,
                       help='使用增强数据集(enhanced)，包含恢复的高质量样本和SMOTE生成样本')
    parser.add_argument('--use_resampling', action='store_true', default=False,
                       help='是否使用重采样数据训练')
    parser.add_argument('--resampled_data_path', type=str, default=None,
                       help='重采样数据文件路径（不指定则自动查找最新）')
    
    # 训练参数 - 基于数据特征优化的默认值
    parser.add_argument('--batch_size', type=int, default=64, help='批大小（适中以平衡内存和效率）')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--learning_rate', type=float, default=1e-5, help='学习率（降低以提高稳定性）')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='权重衰减（增强正则化）')
    
    # 模型架构参数 
    parser.add_argument('--hidden_channels', type=int, default=128, help='SDE隐藏维度（增大）')
    parser.add_argument('--contiformer_dim', type=int, default=256, help='ContiFormer维度（增大）')
    parser.add_argument('--n_heads', type=int, default=8, help='注意力头数')
    parser.add_argument('--n_layers', type=int, default=6, help='编码器层数')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout率')
    
    # 梯度累积参数
    parser.add_argument('--gradient_accumulation_steps', type=int, default=2, help='梯度累积步数（有效批次=32）')
    
    # 优化选项 - 基于数据特征的平衡参数
    parser.add_argument('--gradient_clip', type=float, default=5.0,
                       help='梯度裁剪值（适度裁剪，平衡稳定性与训练效率）')
    parser.add_argument('--label_smoothing', type=float, default=0.1,
                       help='标签平滑系数')
    
    # SDE配置
    parser.add_argument('--sde_config', type=int, default=1, choices=[1, 2, 3],
                       help='SDE配置: 1=准确率优先（最稳定）, 2=平衡, 3=时间优先')
    
    # 组件开关选项
    parser.add_argument('--use_sde', type=int, default=1, choices=[0, 1],
                       help='是否使用SDE组件 (0=不使用, 1=使用)')
    parser.add_argument('--use_contiformer', type=int, default=1, choices=[0, 1], 
                       help='是否使用ContiFormer组件 (0=不使用, 1=使用)')
    parser.add_argument('--use_cga', type=int, default=1, choices=[0, 1],
                       help='是否使用CGA模块 (0=不使用, 1=使用)')
    
    # CGA配置参数
    parser.add_argument('--cga_group_dim', type=int, default=64,
                       help='CGA中每个类别组的表示维度')
    parser.add_argument('--cga_heads', type=int, default=4,
                       help='CGA的注意力头数')
    parser.add_argument('--cga_temperature', type=float, default=0.5,
                       help='CGA语义相似度温度参数（增高避免过度集中）')
    parser.add_argument('--cga_gate_threshold', type=float, default=0.3,
                       help='CGA门控阈值（降低以增加稳定性）')
    
    # GPU优化参数
    parser.add_argument('--use_gradient_checkpoint', action='store_true', default=True,
                       help='启用梯度检查点以节省内存')
    
    # 注意：当use_sde=0且use_contiformer=0时，模型将只使用基础特征编码器和分类器
    
    # Linear Noise SDE特有参数
    parser.add_argument('--enable_gradient_detach', action='store_true', default=True,
                       help='是否启用每N步梯度断开（防止RecursionError）')
    parser.add_argument('--detach_interval', type=int, default=50,
                       help='梯度断开间隔步数（更频繁以增加稳定性）')
    
    # 损失函数参数
    parser.add_argument('--temperature', type=float, default=None,
                       help='温度缩放参数（None时使用数据集默认值）')
    parser.add_argument('--focal_gamma', type=float, default=None,
                       help='Focal Loss gamma参数（None时使用数据集默认值）')
    parser.add_argument('--focal_alpha', type=float, default=None,
                       help='Focal Loss alpha参数')
    parser.add_argument('--min_time_interval', type=float, default=None,
                       help='最小时间间隔（跳过小于此间隔的数据点）')
    
    # 系统参数
    parser.add_argument('--device', type=str, default='auto', 
                       help='计算设备 (auto/cpu/cuda/cuda:0/cuda:1/cuda:2 等)')
    parser.add_argument('--gpu_id', type=int, default=-1, 
                       help='指定GPU ID (-1为自动选择空闲GPU，>=0为指定GPU)')
    parser.add_argument('--num_workers', type=int, default=16, help='数据加载进程数（增加以减少数据加载瓶颈）')
    parser.add_argument('--pin_memory', action='store_true', default=True, help='使用固定内存加速数据传输')
    parser.add_argument('--prefetch_factor', type=int, default=4, help='预取批次数量')
    parser.add_argument('--no_amp', action='store_true', default=False, 
                       help='禁用混合精度训练 (默认启用AMP)')
    parser.add_argument('--seed', type=int, default=None, help='随机种子 (None=自动生成随机种子)')
    
    # 保存和日志参数 - 使用标准化路径管理
    parser.add_argument('--base_dir', type=str, default='/root/autodl-tmp/lnsde-contiformer/results', help='结果保存基目录')
    parser.add_argument('--save_interval', type=int, default=10, help='模型保存间隔(epochs)')
    
    return parser.parse_args()


def generate_random_seed():
    """生成真正随机的种子"""
    import os
    import time
    import hashlib
    import random
    
    # 使用多个随机源生成种子
    current_time = time.time_ns()
    system_random = os.urandom(16)
    python_random = random.random()
    
    # 组合随机源
    combined = f"{current_time}_{system_random.hex()}_{python_random}"
    
    # 使用hash生成稳定的整数种子
    hash_obj = hashlib.sha256(combined.encode())
    seed = int(hash_obj.hexdigest()[:8], 16) % (2**31 - 1)
    
    return seed


def main():
    """主函数"""
    # 0. 设置环境变量以增强稳定性
    import os
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # 可选：有助于调试但会降低性能
    
    # 1. 解析参数和基础设置
    args = parse_args()
    
    # 设置混合精度训练标志 - 默认启用，除非指定--no_amp
    args.use_amp = not args.no_amp
    
    # 处理随机种子：如果为None则自动生成
    if args.seed is None:
        args.seed = generate_random_seed()
        print(f"🎲 自动生成随机种子: {args.seed}")
    else:
        print(f"🔧 使用指定种子: {args.seed}")
    
    set_seed(args.seed)
    device = get_device(args.device, getattr(args, 'gpu_id', -1))
    
    # 2. 配置设置
    args = setup_dataset_mapping(args)
    dataset_config = get_dataset_specific_params(args.dataset, args)
    sde_config = setup_sde_config(args.sde_config, args)
    
    # 将sde_config参数合并到args中，以便模型创建时使用
    for key, value in sde_config.items():
        setattr(args, key, value)
    
    # 3. 创建标准化目录结构
    print("=== 设置输出目录 ===")
    timestamp_path = get_timestamp_path(args.base_dir, args.dataset_name, create_dirs=True)
    print(f"输出目录: {timestamp_path}")
    
    # 将标准化路径传递给后续使用
    args.save_dir = timestamp_path
    args.log_dir = timestamp_path
    
    # 4. 数据加载
    print("=== 数据加载 ===")
    
    # 检查是否使用重采样数据
    if args.use_resampling:
        print("使用重采样数据训练")
        
        if args.resampled_data_path is None:
            # 自动查找重采样数据文件 - 优先查找TimeGAN数据
            if args.dataset_name == 'MACHO_TimeGAN':
                # TimeGAN重采样数据
                resampled_file = '/root/autodl-fs/lnsde-contiformer/data/macho_resample_timegan.pkl'
            else:
                # 传统重采样数据
                resampled_file = f'/root/autodl-fs/lnsde-contiformer/data/{args.dataset_name}_resampled.pkl'
            
            if os.path.exists(resampled_file):
                args.resampled_data_path = resampled_file
                print(f"找到重采样数据: {args.resampled_data_path}")
            else:
                raise ValueError(f"未找到{args.dataset_name}的重采样数据文件: {resampled_file}")
        
        # 直接使用重采样数据路径，格式已经完全兼容
        print(f"加载重采样数据: {args.resampled_data_path}")
        train_loader, test_loader, num_classes = create_dataloaders(
            data_path=args.resampled_data_path,
            batch_size=args.batch_size, 
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            prefetch_factor=args.prefetch_factor,
            persistent_workers=True,
            random_seed=args.seed
        )
        
    else:
        # 使用原始数据
        train_loader, test_loader, num_classes = create_dataloaders(
            data_path=args.data_path, 
            batch_size=args.batch_size, 
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            prefetch_factor=args.prefetch_factor,
            persistent_workers=True,
            random_seed=args.seed
        )
    # 由于当前实现只有train和test，将test_loader用作val_loader
    val_loader = test_loader
    print(f"类别数量: {num_classes}")
    
    # 5. 模型创建
    print("=== 模型创建 ===")
    model = create_model(args.model_type, num_classes, args, dataset_config)
    model = model.to(device)
    
    # 显示模型参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
    
    # 显示混合精度训练状态
    if args.use_amp and device.type == 'cuda':
        print("🚀 混合精度训练: 已启用 (AMP)")
    else:
        if device.type != 'cuda':
            print("⚠️  混合精度训练: 已禁用 (需要CUDA设备)")
        else:
            print("🐢 混合精度训练: 已禁用 (全精度训练模式)")
    
    # 6. 优化器和学习率调度器 - 使用AdamW替代Lion
    # AdamW对学习率更稳定，适合milstein方法的长时间训练
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,  # AdamW可以使用正常学习率
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),  # 标准Adam参数
        eps=1e-8
    )
    print(f"优化器: AdamW (lr={args.learning_rate:.1e}, wd={args.weight_decay:.1e})")
    
    # 使用ReduceLROnPlateau而不是CosineAnnealing，更稳定
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=10, min_lr=1e-8
    )
    
    # 损失函数 - 使用模型内部的compute_loss方法
    criterion = torch.nn.CrossEntropyLoss()
    
    # 7. 模型加载
    print("=== 检查已有模型 ===")
    best_val_acc, start_epoch = load_model_checkpoint(model, optimizer, args, args.load_model)
    
    # 8. 设置日志
    print("=== 设置日志 ===")
    log_path, log_data = setup_logging(args.log_dir, args.dataset_name, args.model_type, args.sde_config, args)
    
    # 记录完整实验配置到日志中
    print(f"📝 实验配置已记录")
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(f"\n{'='*60}\n")
        f.write(f"                    实验配置\n")
        f.write(f"{'='*60}\n")
        
        # 基础配置
        f.write(f"【基础设置】\n")
        f.write(f"  随机种子: {args.seed}\n")
        f.write(f"  数据集: {args.dataset_name}\n")
        f.write(f"  数据版本: {'增强数据(enhanced)' if args.use_enhanced else ('原始完整数据(original)' if args.use_original else '修复后数据(fixed)')}\n")
        f.write(f"  使用重采样: {'是' if args.use_resampling else '否'}\n")
        if args.use_resampling and hasattr(args, 'resampled_data_path'):
            f.write(f"  重采样数据: {args.resampled_data_path}\n")
        f.write(f"\n")
        
        # 模型配置
        model_type_names = {1: 'Langevin SDE', 2: 'Linear Noise SDE', 3: 'Geometric SDE'}
        f.write(f"【模型配置】\n")
        f.write(f"  模型类型: {model_type_names.get(args.model_type, f'Type-{args.model_type}')}\n")
        f.write(f"  SDE组件: {'启用' if args.use_sde else '禁用'}\n")
        f.write(f"  ContiFormer: {'启用' if args.use_contiformer else '禁用'}\n")
        f.write(f"  CGA模块: {'启用' if args.use_cga else '禁用'}\n")
        f.write(f"\n")
        
        # 模型架构参数
        f.write(f"【架构参数】\n")
        f.write(f"  隐藏通道数: {args.hidden_channels}\n")
        f.write(f"  ContiFormer维度: {args.contiformer_dim}\n")
        f.write(f"  注意力头数: {args.n_heads}\n")
        f.write(f"  编码器层数: {args.n_layers}\n")
        f.write(f"  Dropout率: {args.dropout}\n")
        f.write(f"\n")
        
        # SDE配置
        sde_config_names = {1: '准确率优先', 2: '平衡', 3: '时间优先'}
        f.write(f"【SDE设置】\n")
        f.write(f"  配置方案: {sde_config_names.get(args.sde_config, f'Config-{args.sde_config}')}\n")
        f.write(f"  求解方法: {args.sde_method}\n")
        f.write(f"  时间步长: {args.dt}\n")
        f.write(f"  相对容差: {args.rtol}\n")
        f.write(f"  绝对容差: {args.atol}\n")
        f.write(f"\n")
        
        # 训练配置
        f.write(f"【训练配置】\n")
        f.write(f"  批大小: {args.batch_size}\n")
        f.write(f"  训练轮数: {args.epochs}\n")
        f.write(f"  学习率: {args.learning_rate:.1e}\n")
        f.write(f"  权重衰减: {args.weight_decay:.1e}\n")
        f.write(f"  梯度累积步数: {args.gradient_accumulation_steps}\n")
        f.write(f"  混合精度(AMP): {'启用' if args.use_amp else '禁用'}\n")
        f.write(f"\n")
        
        # 系统配置
        f.write(f"【系统配置】\n")
        f.write(f"  设备: {device}\n")
        f.write(f"  数据加载进程: {args.num_workers}\n")
        f.write(f"  固定内存: {'启用' if args.pin_memory else '禁用'}\n")
        f.write(f"  预取批次数: {args.prefetch_factor}\n")
        
        f.write(f"{'='*60}\n\n")
    
    # 9. 开始训练 - 使用默认训练管理器
    training_manager = TrainingManager(
        model, train_loader, val_loader, optimizer, criterion, device, args, dataset_config, scheduler
    )
    
    final_best_acc = training_manager.run_training(log_path, log_data, best_val_acc, start_epoch)
    
    # 10. 清理GPU内存
    clear_gpu_memory()
    
    print(f"训练完成！最佳验证准确率: {final_best_acc:.4f}%")


if __name__ == "__main__":
    main()