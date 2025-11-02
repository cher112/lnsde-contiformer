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
        'atol': args.atol,
        'use_sde': args.use_sde,
        'use_contiformer': args.use_contiformer,
        'use_cga': args.use_cga,
        'sde_solve_mode': getattr(args, 'sde_solve_mode', 0)  # 默认0=逐步求解
    }
    
    # 添加CGA参数
    if args.use_cga:
        model_configs.update({
            'cga_group_dim': args.cga_group_dim,
            'cga_heads': args.cga_heads,
            'cga_temperature': args.cga_temperature,
            'cga_gate_threshold': args.cga_gate_threshold
        })
    
    # 添加GPU优化参数
    model_configs.update({
        'use_gradient_checkpoint': getattr(args, 'use_gradient_checkpoint', False)
    })
    
    # 为Linear Noise SDE添加梯度管理参数
    if model_type == 'linear_noise':
        model_configs.update({
            'enable_gradient_detach': dataset_config['enable_gradient_detach'],
            'detach_interval': args.detach_interval
        })
    
    if model_type == 'langevin':
        model = LangevinSDEContiformer(**model_configs)
        print("创建Langevin-type SDE + ContiFormer模型")
        if args.use_sde == 0:
            print("  - SDE组件: 已禁用")
        if args.use_contiformer == 0:
            print("  - ContiFormer组件: 已禁用")
        if args.use_cga == 1:
            print("  - CGA组件: 已启用")
        else:
            print("  - CGA组件: 已禁用")
        
    elif model_type == 'linear_noise':
        model = LinearNoiseSDEContiformer(**model_configs)
        print(f"创建Linear Noise SDE + ContiFormer模型")
        if args.use_sde == 0:
            print("  - SDE组件: 已禁用")
        else:
            sde_mode_desc = "逐步求解" if getattr(args, 'sde_solve_mode', 0) == 0 else "一次性求解整个轨迹"
            print(f"  - SDE求解模式: {sde_mode_desc}")
        if args.use_contiformer == 0:
            print("  - ContiFormer组件: 已禁用")
        if args.use_cga == 1:
            print("  - CGA组件: 已启用")
            print(f"    * 组维度: {args.cga_group_dim}")
            print(f"    * 注意力头数: {args.cga_heads}")
            print(f"    * 温度参数: {args.cga_temperature}")
        else:
            print("  - CGA组件: 已禁用")
        print(f"  - 梯度断开: {dataset_config['enable_gradient_detach']}")
        print(f"  - 断开间隔: {args.detach_interval}")
        
    elif model_type == 'geometric':
        model = GeometricSDEContiformer(**model_configs)
        print("创建Geometric SDE + ContiFormer模型")
        if args.use_sde == 0:
            print("  - SDE组件: 已禁用")
        if args.use_contiformer == 0:
            print("  - ContiFormer组件: 已禁用")
        if args.use_cga == 1:
            print("  - CGA组件: 已启用")
        else:
            print("  - CGA组件: 已禁用")
        
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
    """根据选项加载模型检查点 - 使用标准化路径结构"""
    if load_type == 0:  # 不加载
        print("不加载任何预训练模型，从头开始训练")
        return 0.0, 0
    
    # 搜索所有可能的路径
    potential_paths = []
    
    if load_type == 1:  # 加载最新
        # 在results目录下递归搜索epoch模型
        for root, _, files in os.walk("./results"):
            for file in files:
                if (file.startswith(f"{args.dataset_name}_{args.model_type}") and 
                    "_epoch" in file and file.endswith('.pth') and "best" not in file):
                    full_path = os.path.join(root, file)
                    # 提取epoch号
                    try:
                        if "_epoch_" in file:
                            epoch_num = int(file.split('_epoch_')[1].split('.')[0])
                        else:
                            # 新格式: epochN.pth
                            epoch_part = file.split('_epoch')[1].split('.')[0]
                            epoch_num = int(epoch_part)
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
        # 递归搜索best模型 - 支持新旧文件名格式
        for root, _, files in os.walk("./results"):
            for file in files:
                # 旧格式: ASAS_linear_noise_best.pth
                # 新格式: ASAS_linear_noise_sde1_cf1_cga1_hc128_cd256_lr1e-04_bs50_best.pth
                if (file.startswith(f"{args.dataset_name}_{args.model_type}") and 
                    file.endswith('_best.pth')):
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
        if saved_params:  # 只有当保存了参数信息时才检查
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
                print("⚠️ 模型参数不匹配，但尝试部分加载（架构兼容模式）")
                # 不直接返回，继续尝试加载模型状态
        
        # 加载模型状态 - 处理架构不匹配的情况
        try:
            model.load_state_dict(checkpoint['model_state_dict'], strict=True)
            print(f"✓ 模型状态已完整加载")
        except RuntimeError as e:
            if "Missing key(s)" in str(e) or "Unexpected key(s)" in str(e):
                print(f"⚠️ 模型架构不匹配，尝试部分加载...")
                print(f"错误详情: {e}")
                
                # 尝试部分加载
                try:
                    missing_keys, unexpected_keys = model.load_state_dict(
                        checkpoint['model_state_dict'], strict=False
                    )
                    
                    if missing_keys:
                        print(f"⚠️ 缺失的参数将使用随机初始化: {len(missing_keys)} 个参数")
                        for key in missing_keys[:5]:  # 只显示前5个
                            print(f"  - {key}")
                        if len(missing_keys) > 5:
                            print(f"  - ... 还有 {len(missing_keys) - 5} 个参数")
                    
                    if unexpected_keys:
                        print(f"⚠️ 忽略的参数: {len(unexpected_keys)} 个")
                        for key in unexpected_keys[:5]:  # 只显示前5个
                            print(f"  - {key}")
                        if len(unexpected_keys) > 5:
                            print(f"  - ... 还有 {len(unexpected_keys) - 5} 个参数")
                    
                    print(f"✓ 模型状态已部分加载（架构兼容模式）")
                except Exception as e2:
                    print(f"❌ 部分加载也失败: {e2}")
                    print("从头开始训练")
                    return 0.0, 0
            else:
                raise e
        
        # 尝试加载优化器状态（仅对继续训练有效）
        if load_type == 1 and 'optimizer_state_dict' in checkpoint:
            try:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                print(f"✓ 优化器状态已加载")
            except Exception as e:
                print(f"⚠️ 优化器加载失败（可能是优化器类型不匹配）: {e}")
        
        # 返回相关信息
        best_val_acc = checkpoint.get('best_val_accuracy', checkpoint.get('accuracy', 0.0))
        loaded_epoch = checkpoint.get('epoch', 0)
        start_epoch = loaded_epoch + 1 if load_type == 1 else 0
        
        print(f"✓ 加载完成 - 已完成epoch: {loaded_epoch}, 最佳验证精度: {best_val_acc:.4f}, 继续从epoch: {start_epoch}")
        return best_val_acc, start_epoch
        
    except Exception as e:
        print(f"加载模型失败: {str(e)}")
        print("从头开始训练")
        return 0.0, 0