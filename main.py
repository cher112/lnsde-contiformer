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
from utils.resampling import HybridResampler, load_resampled_data


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
    
    # 重采样参数
    parser.add_argument('--use_resampling', action='store_true', default=False,
                       help='是否使用重采样数据训练')
    parser.add_argument('--resampled_data_path', type=str, default=None,
                       help='重采样数据文件路径（不指定则自动查找最新）')
    parser.add_argument('--resample_on_fly', action='store_true', default=False,
                       help='实时生成重采样数据（而非加载预生成的）')
    
    # 训练参数 - 准确率优先设置
    parser.add_argument('--batch_size', type=int, default=64, help='批大小（优化GPU利用率）')
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='权重衰减')
    
    # 模型架构参数 
    parser.add_argument('--hidden_channels', type=int, default=128, help='SDE隐藏维度（增大）')
    parser.add_argument('--contiformer_dim', type=int, default=256, help='ContiFormer维度（增大）')
    parser.add_argument('--n_heads', type=int, default=8, help='注意力头数')
    parser.add_argument('--n_layers', type=int, default=6, help='编码器层数')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout率')
    
    # 梯度累积参数
    parser.add_argument('--gradient_accumulation_steps', type=int, default=2, help='梯度累积步数（减少以提高GPU利用率）')
    
    # SDE配置
    parser.add_argument('--sde_config', type=int, default=1, choices=[1, 2, 3],
                       help='SDE配置: 1=准确率优先, 2=平衡, 3=时间优先')
    
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
    parser.add_argument('--cga_temperature', type=float, default=0.1,
                       help='CGA语义相似度温度参数')
    parser.add_argument('--cga_gate_threshold', type=float, default=0.5,
                       help='CGA门控阈值')
    
    # 注意：当use_sde=0且use_contiformer=0时，模型将只使用基础特征编码器和分类器
    
    # Linear Noise SDE特有参数
    parser.add_argument('--enable_gradient_detach', action='store_true',
                       help='是否启用每N步梯度断开（防止RecursionError）')
    parser.add_argument('--detach_interval', type=int, default=100,
                       help='梯度断开间隔步数')
    
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
    parser.add_argument('--num_workers', type=int, default=8, help='数据加载进程数（提高数据加载速度）')
    parser.add_argument('--use_amp', action='store_true', default=True, help='启用混合精度训练（提高GPU利用率）')
    parser.add_argument('--seed', type=int, default=123, help='随机种子')
    
    # 保存和日志参数 - 使用标准化路径管理
    parser.add_argument('--base_dir', type=str, default='/root/autodl-tmp/lnsde-contiformer/results', help='结果保存基目录')
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
    
    # 检查是否使用重采样数据
    if args.use_resampling:
        print("使用重采样数据训练")
        
        if args.resample_on_fly:
            # 实时生成重采样数据
            print("实时生成重采样数据...")
            train_loader, test_loader, num_classes = create_dataloaders(
                data_path=args.data_path, 
                batch_size=args.batch_size, 
                num_workers=args.num_workers,
                random_seed=args.seed
            )
            
            # 提取所有训练数据
            all_features = []
            all_labels = []
            all_times = []
            all_masks = []
            
            for batch in train_loader:
                all_features.append(batch['features'])
                all_labels.append(batch['labels'])
                all_times.append(batch['times'])
                all_masks.append(batch['masks'])
            
            X = torch.cat(all_features, dim=0)
            y = torch.cat(all_labels, dim=0)
            times = torch.cat(all_times, dim=0)
            masks = torch.cat(all_masks, dim=0)
            
            # 执行重采样
            resampler = HybridResampler(
                smote_k_neighbors=5,
                enn_n_neighbors=3,
                sampling_strategy='balanced',
                apply_enn=False,  # 不过度欠采样
                random_state=args.seed
            )
            
            X_resampled, y_resampled, times_resampled, masks_resampled = resampler.fit_resample(
                X, y, times, masks
            )
            
            # 创建重采样后的数据加载器
            from torch.utils.data import TensorDataset, DataLoader
            resampled_dataset = TensorDataset(X_resampled, y_resampled, times_resampled, masks_resampled)
            train_loader = DataLoader(
                resampled_dataset,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.num_workers
            )
            
            # 可视化重采样效果
            import os
            fig_path = os.path.join(args.save_dir, 'resampling_distribution.png')
            resampler.visualize_distribution(save_path=fig_path)
            
        else:
            # 加载预生成的重采样数据
            if args.resampled_data_path is None:
                # 自动查找最新的重采样数据
                import glob
                import os
                
                resampled_dir = '/root/autodl-fs/lnsde-contiformer/data/resampled'
                pattern = f'{args.dataset_name}_resampled_*.pkl'
                files = glob.glob(os.path.join(resampled_dir, pattern))
                
                if not files:
                    raise ValueError(f"未找到{args.dataset_name}的重采样数据，请先运行resample_datasets.py生成")
                
                # 选择最新的文件
                args.resampled_data_path = max(files, key=os.path.getctime)
                print(f"自动选择最新重采样数据: {args.resampled_data_path}")
            
            # 加载重采样数据
            resampled_data = load_resampled_data(args.resampled_data_path)
            
            # 创建数据加载器 - 修复重采样数据格式
            from torch.utils.data import Dataset, DataLoader
            
            class ResampledDataset(Dataset):
                def __init__(self, X, y, times, masks):
                    self.X = X
                    self.y = y
                    self.times = times
                    self.masks = masks
                
                def __len__(self):
                    return len(self.y)
                
                def __getitem__(self, idx):
                    return {
                        'features': self.X[idx],
                        'labels': self.y[idx],
                        'times': self.times[idx],
                        'mask': self.masks[idx]
                    }
            
            X = torch.tensor(resampled_data['X']) if not torch.is_tensor(resampled_data['X']) else resampled_data['X']
            y = torch.tensor(resampled_data['y']) if not torch.is_tensor(resampled_data['y']) else resampled_data['y']
            times = torch.tensor(resampled_data['times']) if resampled_data['times'] is not None and not torch.is_tensor(resampled_data['times']) else resampled_data['times']
            masks = torch.tensor(resampled_data['masks']) if resampled_data['masks'] is not None and not torch.is_tensor(resampled_data['masks']) else resampled_data['masks']
            
            # 分割训练集和测试集（80:20）
            n_samples = len(y)
            n_train = int(0.8 * n_samples)
            
            indices = torch.randperm(n_samples)
            train_indices = indices[:n_train]
            test_indices = indices[n_train:]
            
            # 创建数据集
            train_dataset = ResampledDataset(
                X[train_indices],
                y[train_indices], 
                times[train_indices] if times is not None else torch.zeros(n_train, X.shape[1]),
                masks[train_indices] if masks is not None else torch.ones(n_train, X.shape[1], dtype=torch.bool)
            )
            
            test_dataset = ResampledDataset(
                X[test_indices],
                y[test_indices],
                times[test_indices] if times is not None else torch.zeros(len(test_indices), X.shape[1]),
                masks[test_indices] if masks is not None else torch.ones(len(test_indices), X.shape[1], dtype=torch.bool)
            )
            
            # 创建数据加载器
            train_loader = DataLoader(
                train_dataset,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.num_workers,
                collate_fn=lambda batch: {
                    'features': torch.stack([item['features'] for item in batch]),
                    'labels': torch.stack([item['labels'] for item in batch]),
                    'times': torch.stack([item['times'] for item in batch]),
                    'mask': torch.stack([item['mask'] for item in batch])
                }
            )
            
            test_loader = DataLoader(
                test_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                collate_fn=lambda batch: {
                    'features': torch.stack([item['features'] for item in batch]),
                    'labels': torch.stack([item['labels'] for item in batch]),
                    'times': torch.stack([item['times'] for item in batch]),
                    'mask': torch.stack([item['mask'] for item in batch])
                }
            )
            
            num_classes = len(resampled_data['distribution'])
            
            print(f"加载重采样数据完成:")
            print(f"  训练集: {n_train} 样本")
            print(f"  测试集: {n_samples - n_train} 样本")
            print(f"  类别分布: {resampled_data['distribution']}")
    else:
        # 使用原始数据
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
    
    # 6. 优化器和学习率调度器 - 防止Lion崩塌到单一类别
    # 优化参数：更小的学习率，适中的weight decay
    optimizer = Lion(
        model.parameters(),
        lr=args.learning_rate * 0.1,  # 进一步降低学习率，防止崩塌
        weight_decay=args.weight_decay * 3  # 适度的weight decay，避免过度正则化
    )
    print(f"优化器: Lion (lr={args.learning_rate * 0.1:.1e}, wd={args.weight_decay * 3:.1e})")
    
    # 使用Cosine退火调度器，更平滑的学习率衰减
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-7
    )
    
    # 损失函数 - 使用模型内部的compute_loss方法
    criterion = torch.nn.CrossEntropyLoss()
    
    # 7. 模型加载
    print("=== 检查已有模型 ===")
    best_val_acc, start_epoch = load_model_checkpoint(model, optimizer, args, args.load_model)
    
    # 8. 设置日志
    print("=== 设置日志 ===")
    log_path, log_data = setup_logging(args.log_dir, args.dataset_name, args.model_type, args.sde_config, args)
    
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