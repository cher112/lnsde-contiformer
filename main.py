#!/usr/bin/env python3
"""
Neural SDE + ContiFormer Training Script
重构版本 - 简洁的主入口文件
"""

import argparse
import os
import torch
from torch.optim import AdamW
from lion_pytorch import Lion

# 导入所有必要的工具模块
from utils import (
    set_seed, get_device, clear_gpu_memory,
    setup_dataset_mapping, get_dataset_specific_params, setup_sde_config,
    create_dataloaders, create_model, load_model_checkpoint,
    setup_logging, TrainingManager
)
from utils.loss import FocalLoss


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Neural SDE + ContiFormer Training')
    
    # 模型选择
    parser.add_argument('--model_type', type=int, default=2,
                       choices=[1, 2, 3],
                       help='SDE模型类型 (1:langevin, 2:linear_noise, 3:geometric)')
    
    # 模型加载选项
    parser.add_argument('--load_model', type=int, default=2,
                       choices=[0, 1, 2],
                       help='模型加载选项 (0:不加载, 1:加载最新, 2:加载最优)')
    
    # 数据相关 - 使用数字代表数据集
    parser.add_argument('--dataset', type=int, default=2,
                       choices=[1, 2, 3],
                       help='数据集选择: 1=ASAS, 2=LINEAR, 3=MACHO')
    
    parser.add_argument('--use_backup_data', action='store_true', default=False,
                       help='是否使用backup数据文件（推荐，避免SDE时间序列问题）')
    
    # 训练参数 - 准确率优先设置
    parser.add_argument('--batch_size', type=int, default=32, help='批大小（准确率优先设置）')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='权重衰减')
    
    # 模型架构参数 - 增大以提高准确率
    parser.add_argument('--hidden_channels', type=int, default=128, help='SDE隐藏维度')
    parser.add_argument('--contiformer_dim', type=int, default=256, help='ContiFormer维度')
    parser.add_argument('--n_heads', type=int, default=16, help='注意力头数（增大）')
    parser.add_argument('--n_layers', type=int, default=8, help='编码器层数（增大）')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout率')
    
    # SDE配置
    parser.add_argument('--sde_config', type=int, default=1, choices=[1, 2, 3],
                       help='SDE配置: 1=准确率优先, 2=平衡, 3=时间优先')
    
    # Linear Noise SDE特有参数
    parser.add_argument('--enable_gradient_detach', action='store_true',
                       help='是否启用每N步梯度断开（防止RecursionError）')
    parser.add_argument('--detach_interval', type=int, default=10,
                       help='梯度断开间隔步数')
    
    # 损失函数参数
    parser.add_argument('--temperature', type=float, default=None,
                       help='温度缩放参数（None时使用数据集默认值）')
    parser.add_argument('--focal_gamma', type=float, default=None,
                       help='Focal Loss gamma参数（None时使用数据集默认值）')
    parser.add_argument('--focal_alpha', type=float, default=None,
                       help='Focal Loss alpha参数')
    
    # 系统参数
    parser.add_argument('--device', type=str, default='auto', help='计算设备 (auto/cpu/cuda)')
    parser.add_argument('--num_workers', type=int, default=8, help='数据加载进程数')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    
    # 保存和日志参数
    parser.add_argument('--save_dir', type=str, default='./results/checkpoints', help='模型保存目录')
    parser.add_argument('--log_dir', type=str, default='./results/logs', help='日志保存目录')
    parser.add_argument('--save_interval', type=int, default=20, help='模型保存间隔(epochs)')
    
    return parser.parse_args()


def main():
    """主函数 - 整合训练流程"""
    # 1. 解析参数和基础设置
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)
    
    # 2. 配置设置
    args = setup_dataset_mapping(args)
    dataset_config = get_dataset_specific_params(args.dataset, args)
    sde_config = setup_sde_config(args.sde_config, args)
    
    # 3. 创建目录
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # 4. 数据加载
    print("=== 数据加载 ===")
    train_loader, val_loader, test_loader, num_classes = create_dataloaders(
        data_path=args.data_path, 
        batch_size=args.batch_size, 
        num_workers=args.num_workers
    )
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
    
    # 6. 损失函数和优化器
    criterion = torch.nn.CrossEntropyLoss()  # 使用CrossEntropyLoss而不是FocalLoss
    print(f"损失函数: CrossEntropyLoss")
    
    # 使用Lion优化器（更好的性能）
    optimizer = Lion(
        model.parameters(),
        lr=args.learning_rate * 0.3,  # Lion通常用更小的学习率
        weight_decay=args.weight_decay * 10  # Lion可以用更大的weight_decay
    )
    print(f"优化器: Lion (lr={args.learning_rate * 0.3:.1e}, wd={args.weight_decay * 10:.1e})")
    
    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20, T_mult=2, eta_min=1e-6
    )
    
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
    
    # 10. 保存训练历史
    import json
    history_path = os.path.join(
        args.save_dir,
        f"{args.dataset_name}_{args.model_type}_history.json"
    )
    with open(history_path, 'w', encoding='utf-8') as f:
        json.dump(log_data, f, ensure_ascii=False, indent=2)
    
    print(f"训练历史已保存: {history_path}")
    print(f"最终最佳验证准确率: {final_best_acc:.4f}%")
    
    # 清理GPU内存
    clear_gpu_memory()


if __name__ == "__main__":
    main()