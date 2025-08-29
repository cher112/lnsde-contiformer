#!/usr/bin/env python3
"""
Neural SDE + ContiFormer Training Script
与当前架构匹配的主入口文件
"""

import argparse
import os
import torch
import warnings
warnings.filterwarnings('ignore')

# Lion优化器
from lion_pytorch import Lion

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
    train_epoch, validate_epoch, calculate_class_accuracy, TrainingManager
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
    parser.add_argument('--load_model', type=int, default=1,
                       choices=[0, 1, 2],
                       help='模型加载选项 (0:不加载, 1:加载最新, 2:加载最优)')
    
    # 数据相关 - 使用数字代表数据集
    parser.add_argument('--dataset', type=int, default=2,
                       choices=[1, 2, 3],
                       help='数据集选择: 1=ASAS, 2=LINEAR, 3=MACHO')
    
    parser.add_argument('--use_backup_data', action='store_true', default=False,
                       help='是否使用backup数据文件（推荐，避免SDE时间序列问题）')
    
    # 训练参数 - 准确率优先设置
    parser.add_argument('--batch_size', type=int, default=64, help='批大小（优化GPU利用率）')
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='权重衰减')
    
    # 模型架构参数 
    parser.add_argument('--hidden_channels', type=int, default=128, help='SDE隐藏维度（增大）')
    parser.add_argument('--contiformer_dim', type=int, default=128, help='ContiFormer维度（增大）')
    parser.add_argument('--n_heads', type=int, default=8, help='注意力头数')
    parser.add_argument('--n_layers', type=int, default=6, help='编码器层数')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout率')
    
    # 梯度累积参数
    parser.add_argument('--gradient_accumulation_steps', type=int, default=2, help='梯度累积步数（减少以提高GPU利用率）')
    
    # SDE配置
    parser.add_argument('--sde_config', type=int, default=1, choices=[1, 2, 3],
                       help='SDE配置: 1=准确率优先, 2=平衡, 3=时间优先')
    
    # Linear Noise SDE特有参数
    parser.add_argument('--enable_gradient_detach', action='store_true',
                       help='是否启用每N步梯度断开（防止RecursionError）')
    parser.add_argument('--detach_interval', type=int, default=50,
                       help='梯度断开间隔步数')
    
    # 损失函数参数
    parser.add_argument('--temperature', type=float, default=None,
                       help='温度缩放参数（None时使用数据集默认值）')
    parser.add_argument('--focal_gamma', type=float, default=None,
                       help='Focal Loss gamma参数（None时使用数据集默认值）')
    parser.add_argument('--focal_alpha', type=float, default=None,
                       help='Focal Loss alpha参数')
    
    # 系统参数
    parser.add_argument('--device', type=str, default='auto', 
                       help='计算设备 (auto/cpu/cuda/cuda:0/cuda:1/cuda:2 等)')
    parser.add_argument('--gpu_id', type=int, default=-1, 
                       help='指定GPU ID (-1为自动选择空闲GPU，>=0为指定GPU)')
    parser.add_argument('--num_workers', type=int, default=8, help='数据加载进程数（提高数据加载速度）')
    parser.add_argument('--use_amp', action='store_true', default=True, help='启用混合精度训练（提高GPU利用率）')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    
    # 保存和日志参数 - 使用标准化路径管理
    parser.add_argument('--base_dir', type=str, default='./results', help='结果保存基目录')
    parser.add_argument('--save_interval', type=int, default=10, help='模型保存间隔(epochs)')
    
    return parser.parse_args()


def main():
    """主函数"""
    # 1. 解析参数和基础设置
    args = parse_args()
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
    train_loader, test_loader, num_classes = create_dataloaders(
        data_path=args.data_path, 
        batch_size=args.batch_size, 
        num_workers=args.num_workers,
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
    
    # 6. 优化器和学习率调度器
    optimizer = Lion(
        model.parameters(),
        lr=args.learning_rate * 0.3,  # Lion通常用更小的学习率
        weight_decay=args.weight_decay * 10  # Lion可以用更大的weight_decay
    )
    print(f"优化器: Lion (lr={args.learning_rate * 0.3:.1e}, wd={args.weight_decay * 10:.1e})")
    
    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-7
    )
    
    # 损失函数 - 使用模型内部的compute_loss方法
    criterion = torch.nn.CrossEntropyLoss()
    
    # 7. 模型加载
    print("=== 检查已有模型 ===")
    best_val_acc, start_epoch = load_model_checkpoint(model, optimizer, args, args.load_model)
    
    # 8. 设置日志
    print("=== 设置日志 ===")
    log_path, log_data = setup_logging(args.log_dir, args.dataset_name, args.model_type, args.sde_config)
    
    # 9. 开始训练
    training_manager = TrainingManager(
        model, train_loader, val_loader, optimizer, criterion, device, args, dataset_config, scheduler
    )
    
    final_best_acc = training_manager.run_training(log_path, log_data, best_val_acc, start_epoch)
    
    # 10. 清理GPU内存
    clear_gpu_memory()
    
    print(f"训练完成！最佳验证准确率: {final_best_acc:.4f}%")


if __name__ == "__main__":
    main()