"""
模型相关工具函数
"""

import os
import torch
from datetime import datetime
from models.langevin_sde import LangevinSDEContiformer
from models.linear_noise_sde import LinearNoiseSDEContiformer
from models.geometric_sde import GeometricSDEContiformer


def setup_model_save_paths(save_dir, dataset_name, model_type):
    """设置模型保存路径 - 新格式: checkpoints/日期/数据集/"""
    date_str = datetime.now().strftime("%Y%m%d")
    
    # 创建保存目录: checkpoints/日期/数据集/
    date_save_dir = os.path.join(save_dir, date_str)
    dataset_save_dir = os.path.join(date_save_dir, dataset_name)
    os.makedirs(dataset_save_dir, exist_ok=True)
    
    print(f"模型保存目录: {dataset_save_dir}")
    return dataset_save_dir


def create_model(model_type, num_classes, args, dataset_config):
    """创建指定类型的模型"""
    model_configs = {
        'input_dim': 3,  # time, mag, errmag（需要保持3维）
        'num_classes': num_classes,
        'hidden_channels': args.hidden_channels,
        'contiformer_dim': args.contiformer_dim,
        'n_heads': args.n_heads,
        'n_layers': args.n_layers,
        'dropout': args.dropout,
        'sde_method': args.sde_method,
        'dt': args.dt,
        'rtol': args.rtol,
        'atol': args.atol
    }
    
    # 为Linear Noise SDE添加梯度管理参数
    if model_type == 'linear_noise':
        model_configs.update({
            'enable_gradient_detach': dataset_config['enable_gradient_detach'],
            'detach_interval': args.detach_interval
        })
    
    if model_type == 'langevin':
        model = LangevinSDEContiformer(**model_configs)
        print("创建Langevin-type SDE + ContiFormer模型")
        
    elif model_type == 'linear_noise':
        model = LinearNoiseSDEContiformer(**model_configs)
        print(f"创建Linear Noise SDE + ContiFormer模型")
        print(f"  - 梯度断开: {dataset_config['enable_gradient_detach']}")
        print(f"  - 断开间隔: {args.detach_interval}")
        
    elif model_type == 'geometric':
        model = GeometricSDEContiformer(**model_configs)
        print("创建Geometric SDE + ContiFormer模型")
        
    else:
        raise ValueError(f"未知模型类型: {model_type}")
    
    return model


def save_checkpoint(model, optimizer, epoch, loss, accuracy, save_path):
    """保存模型检查点"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'accuracy': accuracy,
    }
    torch.save(checkpoint, save_path)
    print(f"模型已保存: {save_path}")


def load_model_checkpoint(model, optimizer, args, load_type):
    """根据选项加载模型检查点 - 支持新的路径格式"""
    if load_type == 0:  # 不加载
        print("不加载任何预训练模型，从头开始训练")
        return 0.0, 0
    
    # 搜索所有可能的日期目录
    potential_paths = []
    
    if load_type == 1:  # 加载最新
        # 在checkpoints目录下搜索所有日期目录中的epoch模型
        for root, dirs, files in os.walk(args.save_dir):
            for file in files:
                if (file.startswith(f"{args.dataset_name}_{args.model_type}_epoch_") and 
                    file.endswith('.pth')):
                    full_path = os.path.join(root, file)
                    # 提取epoch号
                    try:
                        epoch_num = int(file.split('_epoch_')[1].split('.')[0])
                        potential_paths.append((epoch_num, full_path))
                    except:
                        continue
        
        if not potential_paths:
            print("未找到任何epoch模型文件，从头开始训练")
            return 0.0, 0
        
        # 选择最高epoch的模型
        potential_paths.sort(reverse=True)  # 按epoch降序排序
        latest_path = potential_paths[0][1]
        print(f"加载最新模型: {latest_path}")
        checkpoint_path = latest_path
        
    elif load_type == 2:  # 加载最好
        # 搜索所有日期目录中的best模型
        for root, dirs, files in os.walk(args.save_dir):
            for file in files:
                if file == f"{args.dataset_name}_{args.model_type}_best.pth":
                    full_path = os.path.join(root, file)
                    potential_paths.append(full_path)
        
        if not potential_paths:
            print(f"未找到最佳模型")
            return 0.0, 0
        
        # 选择最新的best模型（如果有多个）
        potential_paths.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        best_path = potential_paths[0]
        print(f"加载最佳模型: {best_path}")
        checkpoint_path = best_path
    
    # 加载检查点
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # 验证模型参数是否匹配
        saved_params = checkpoint.get('model_params', {})
        current_params = {
            'hidden_channels': args.hidden_channels,
            'contiformer_dim': args.contiformer_dim,
            'n_heads': args.n_heads,
            'n_layers': args.n_layers,
            'sde_method': args.sde_method,
            'dt': args.dt,
        }
        
        # 检查关键参数是否匹配
        param_mismatch = False
        for key, value in current_params.items():
            if key in saved_params and saved_params[key] != value:
                print(f"参数不匹配 {key}: 当前={value}, 保存={saved_params[key]}")
                param_mismatch = True
        
        if param_mismatch:
            print("模型参数不匹配，从头开始训练")
            return 0.0, 0
        
        # 加载模型状态
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✓ 模型状态已加载")
        
        # 加载优化器状态（仅对epoch模型）
        if load_type == 1 and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print(f"✓ 优化器状态已加载")
        
        # 返回相关信息
        best_val_acc = checkpoint.get('best_val_accuracy', checkpoint.get('accuracy', 0.0))
        start_epoch = checkpoint.get('epoch', 0) + 1 if load_type == 1 else checkpoint.get('epoch', 0) + 1
        
        print(f"✓ 加载完成 - 最佳验证精度: {best_val_acc:.4f}, 开始轮次: {start_epoch}")
        return best_val_acc, start_epoch
        
    except Exception as e:
        print(f"加载模型失败: {str(e)}")
        print("从头开始训练")
        return 0.0, 0