"""
Main training script for Neural SDE + ContiFormer models
支持三种SDE架构的统一训练入口
"""

import argparse
import torch
import torch.nn as nn
import numpy as np
import os
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')

# 添加混合精度训练支持
from torch.amp import autocast, GradScaler

# Lion优化器
from lion_pytorch import Lion

# 设置路径
import sys
sys.path.append('/root/autodl-tmp/torchsde')
sys.path.append('/root/autodl-tmp/PhysioPro')

from models import (
    LangevinSDEContiformer,
    LinearNoiseSDEContiformer,
    GeometricSDEContiformer
)
from utils import create_dataloaders


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Neural SDE + ContiFormer Training')
    
    # 模型选择
    parser.add_argument('--model_type', type=str, default='linear_noise',
                       choices=['langevin', 'linear_noise', 'geometric'],
                       help='SDE模型类型')
    
    # 数据相关 - 使用数字代表数据集
    parser.add_argument('--dataset', type=int, default=2,
                       choices=[1, 2, 3],
                       help='数据集选择: 1=ASAS, 2=LINEAR, 3=MACHO')
    
    # 训练参数 - 准确率优先设置
    parser.add_argument('--batch_size', type=int, default=32, help='批大小（准确率优先设置）')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='权重衰减')
    
    # 模型参数 - 增大模型以充分利用GPU
    parser.add_argument('--hidden_channels', type=int, default=128, help='SDE隐藏维度（增大）')
    parser.add_argument('--contiformer_dim', type=int, default=256, help='ContiFormer维度（增大）')
    parser.add_argument('--n_heads', type=int, default=16, help='注意力头数（增大）')
    parser.add_argument('--n_layers', type=int, default=6, help='编码器层数（增大）')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout率')
    
    # SDE求解参数配置
    parser.add_argument('--sde_config', type=int, default=1,
                       choices=[1, 2, 3],
                       help='SDE配置: 1=准确率优先, 2=平衡, 3=时间优先')
    
    # SDE梯度管理参数
    parser.add_argument('--enable_gradient_detach', action='store_true', default=False,
                       help='是否启用每N步梯度断开（防止RecursionError）')
    parser.add_argument('--detach_interval', type=int, default=10,
                       help='梯度断开间隔步数')
    
    # 损失函数参数 - 数据集特定
    parser.add_argument('--temperature', type=float, default=None,
                       help='温度缩放参数（None时使用数据集默认值）')
    parser.add_argument('--focal_gamma', type=float, default=None,
                       help='Focal Loss gamma参数（None时使用数据集默认值）')
    parser.add_argument('--focal_alpha', type=float, default=1.0,
                       help='Focal Loss alpha参数')
    
    # 训练设置 - 提高数据加载效率
    parser.add_argument('--device', type=str, default='auto',
                       help='计算设备 (auto/cpu/cuda)')
    parser.add_argument('--num_workers', type=int, default=8, help='数据加载进程数（增大）')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    
    # 保存和日志
    parser.add_argument('--save_dir', type=str, default='./checkpoints',
                       help='模型保存目录')
    parser.add_argument('--log_dir', type=str, default='./results/logs',
                       help='日志保存目录')
    parser.add_argument('--save_interval', type=int, default=10,
                       help='模型保存间隔(epochs)')
    
    return parser.parse_args()


def get_dataset_specific_params(dataset_id, args):
    """
    获取数据集特定的超参数
    根据数据集特点优化温度和Focal Loss参数
    """
    dataset_configs = {
        1: {  # ASAS
            'name': 'ASAS',
            'temperature': 1.0,      # 标准温度
            'focal_gamma': 2.0,      # 标准gamma
            'enable_gradient_detach': False,  # ASAS表现较好，不需要梯度断开
            'data_path': './data/ASAS_folded_512.pkl'
        },
        2: {  # LINEAR  
            'name': 'LINEAR',
            'temperature': 0.8,      # 略低温度
            'focal_gamma': 3.0,      # 适中gamma
            'enable_gradient_detach': False,  # LINEAR表现中等
            'data_path': './data/LINEAR_folded_512.pkl'
        },
        3: {  # MACHO - 专门优化
            'name': 'MACHO', 
            'temperature': 1.5,      # 提高温度，降低过度自信
            'focal_gamma': 2.0,      # 降低gamma，减少对简单样本的过度惩罚
            'enable_gradient_detach': True,   # MACHO容易RecursionError
            'data_path': './data/MACHO_folded_512.pkl'
        }
    }
    
    config = dataset_configs[dataset_id]
    
    # 使用命令行参数覆盖默认值（如果提供）
    if args.temperature is not None:
        config['temperature'] = args.temperature
    if args.focal_gamma is not None:
        config['focal_gamma'] = args.focal_gamma
    if hasattr(args, 'enable_gradient_detach') and args.enable_gradient_detach:
        config['enable_gradient_detach'] = True
    
    print(f"=== {config['name']} 数据集配置 ===")
    print(f"温度参数: {config['temperature']}")
    print(f"Focal Gamma: {config['focal_gamma']}")
    print(f"梯度断开: {config['enable_gradient_detach']}")
    
    return config


def set_seed(seed):
    """设置随机种子"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def clear_gpu_memory():
    """清空GPU内存占用"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def get_device(device_arg):
    """获取计算设备"""
    if device_arg == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print(f"使用GPU: {torch.cuda.get_device_name()}")
        else:
            device = torch.device('cpu')
            print("使用CPU")
    else:
        device = torch.device(device_arg)
        print(f"使用指定设备: {device}")
    
    return device


def create_model(model_type, num_classes, args, dataset_config):
    """创建指定类型的模型"""
    model_configs = {
        'input_dim': 3,  # time, mag, errmag
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


def train_epoch(model, train_loader, optimizer, criterion, device, model_type, dataset_config, scaler=None):
    """训练一个epoch - 支持混合精度训练"""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, batch in enumerate(train_loader):
        # 检查数据格式并适应不同的输入接口
        if 'x' in batch and 'time_steps' in batch and 'attn_mask' in batch:
            # 新的elses兼容格式
            x = batch['x'].to(device, non_blocking=True)          # (batch, seq_len, 2) [time, mag]
            time_steps = batch['time_steps'].to(device, non_blocking=True)  # (batch, seq_len)
            attn_mask = batch['attn_mask'].to(device, non_blocking=True)    # (batch, seq_len)
            periods = batch['periods'].to(device, non_blocking=True)        # (batch, 1)
            labels = batch['labels'].to(device, non_blocking=True)          # (batch,)
            
            # 将x转换为features格式 [time, mag, errmag] 用于现有模型
            # 假设errmag暂时设为零向量
            batch_size, seq_len = x.shape[0], x.shape[1]
            errmag = torch.zeros(batch_size, seq_len, 1, device=device)
            features = torch.cat([x, errmag], dim=2)  # (batch, seq_len, 3)
            
            # 修复时间序列：只传递有效的时间点给SDE求解器
            # 使用mask来截取有效时间，避免padding的零值
            times = time_steps
            mask = attn_mask.bool()  # 确保mask是bool类型
            
            # 对于SDE求解，我们需要确保每个样本的时间序列都是严格递增的
            # 在模型内部处理这个问题，这里先保持原格式
            
        else:
            # 原有格式兼容
            features = batch['features'].to(device, non_blocking=True)  # (batch, seq_len, input_dim)
            times = batch['times'].to(device, non_blocking=True)        # (batch, seq_len)
            mask = batch['mask'].to(device, non_blocking=True)          # (batch, seq_len)
            labels = batch['labels'].to(device, non_blocking=True)      # (batch,)
        
        # 前向传播
        optimizer.zero_grad()
        
        try:
            # 使用混合精度
            if scaler is not None:
                with autocast(device_type='cuda'):
                    if model_type == 'geometric':
                        # Geometric SDE支持返回稳定性信息
                        logits, sde_features = model(features, times, mask)
                        loss = model.compute_loss(
                            logits, labels, sde_features,
                            focal_gamma=dataset_config['focal_gamma'],
                            temperature=dataset_config['temperature']
                        )[0]
                    else:
                        # Langevin 和 Linear Noise SDE
                        logits, sde_features = model(features, times, mask)
                        loss = model.compute_loss(
                            logits, labels, sde_features,
                            focal_gamma=dataset_config['focal_gamma'],
                            temperature=dataset_config['temperature']
                        )
                        if isinstance(loss, tuple):
                            loss = loss[0]  # 取总损失
                            
                # 反向传播（混合精度）
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                
            else:
                # 标准精度训练
                if model_type == 'geometric':
                    logits, sde_features = model(features, times, mask)
                    loss = model.compute_loss(
                        logits, labels, sde_features,
                        focal_gamma=dataset_config['focal_gamma'],
                        temperature=dataset_config['temperature']
                    )[0]
                else:
                    logits, sde_features = model(features, times, mask)
                    loss = model.compute_loss(
                        logits, labels, sde_features,
                        focal_gamma=dataset_config['focal_gamma'],
                        temperature=dataset_config['temperature']
                    )
                    if isinstance(loss, tuple):
                        loss = loss[0]
                        
                # 反向传播（标准精度）
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            
        except Exception as e:
            print(f"前向传播失败 (batch {batch_idx}): {e}")
            continue
        
        # 统计
        total_loss += loss.item()
        _, predicted = torch.max(logits.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        if batch_idx % 5 == 0:  # 更频繁的输出
            print(f'Batch [{batch_idx}/{len(train_loader)}], '
                  f'Loss: {loss.item():.4f}, '
                  f'Acc: {100.*correct/total:.2f}%')
            
            # 每5个batch清理一次GPU内存
            if batch_idx % 20 == 0 and batch_idx > 0:
                torch.cuda.empty_cache()
    
    avg_loss = total_loss / len(train_loader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy


def calculate_class_accuracy(predictions, labels):
    """计算每类的准确率"""
    import numpy as np
    from sklearn.metrics import confusion_matrix, classification_report
    
    predictions = np.array(predictions)
    labels = np.array(labels)
    
    # 获取唯一类别
    unique_classes = np.unique(labels)
    class_accuracy = {}
    
    for class_id in unique_classes:
        # 找到该类别的所有样本
        class_mask = (labels == class_id)
        class_predictions = predictions[class_mask]
        class_labels = labels[class_mask]
        
        # 计算该类别的准确率
        if len(class_labels) > 0:
            accuracy = np.sum(class_predictions == class_labels) / len(class_labels)
            class_accuracy[int(class_id)] = accuracy * 100
        else:
            class_accuracy[int(class_id)] = 0.0
    
    return class_accuracy


def setup_logging(log_dir, dataset_name, model_type, sde_config):
    """设置日志文件"""
    import os
    from datetime import datetime
    import json
    
    os.makedirs(log_dir, exist_ok=True)
    
    # 生成日志文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"{dataset_name}_{model_type}_config{sde_config}_{timestamp}.log"
    log_path = os.path.join(log_dir, log_filename)
    
    # 初始化日志数据结构
    log_data = {
        'experiment_info': {
            'dataset': dataset_name,
            'model_type': model_type,
            'sde_config': sde_config,
            'start_time': timestamp,
            'log_file': log_filename
        },
        'training_history': {
            'epochs': [],
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': [],
            'class_accuracy_history': [],
            'learning_rates': []
        },
        'best_metrics': {
            'best_val_accuracy': 0.0,
            'best_epoch': 0,
            'best_class_accuracy': {}
        }
    }
    
    return log_path, log_data


def update_log(log_path, log_data, epoch, train_loss, train_acc, val_loss, val_acc, 
               class_accuracy, learning_rate, is_best=False):
    """更新日志"""
    import json
    from datetime import datetime
    
    # 更新训练历史
    log_data['training_history']['epochs'].append(epoch)
    log_data['training_history']['train_loss'].append(float(train_loss))
    log_data['training_history']['train_accuracy'].append(float(train_acc))
    log_data['training_history']['val_loss'].append(float(val_loss))
    log_data['training_history']['val_accuracy'].append(float(val_acc))
    log_data['training_history']['class_accuracy_history'].append(class_accuracy)
    log_data['training_history']['learning_rates'].append(float(learning_rate))
    
    # 更新最佳指标
    if is_best:
        log_data['best_metrics']['best_val_accuracy'] = float(val_acc)
        log_data['best_metrics']['best_epoch'] = epoch
        log_data['best_metrics']['best_class_accuracy'] = class_accuracy.copy()
    
    # 添加当前时间戳
    log_data['last_update'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # 保存到文件
    with open(log_path, 'w', encoding='utf-8') as f:
        json.dump(log_data, f, indent=2, ensure_ascii=False)
    
    return log_data


def print_epoch_summary(epoch, total_epochs, train_loss, train_acc, val_loss, val_acc, 
                       class_accuracy, learning_rate, is_best=False):
    """打印epoch总结，包含每类准确率"""
    print(f"\nEpoch [{epoch+1}/{total_epochs}] Summary:")
    print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
    print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
    print(f"  Learning Rate: {learning_rate:.2e}")
    
    if is_best:
        print(f"  🎉 New Best Validation Accuracy!")
    
    # 打印每类准确率（简化版）
    class_accs_list = [f"{class_accuracy[class_id]:.1f}%" 
                       for class_id in sorted(class_accuracy.keys())]
    print(f"  Class Acc: [{', '.join(class_accs_list)}]")
    
    print("-" * 80)


def validate_epoch(model, val_loader, criterion, device, model_type, dataset_config):
    """验证一个epoch - 优化数据传输并返回每类准确率"""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    # 统计每类的预测情况
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in val_loader:
            # 检查数据格式并适应不同的输入接口
            if 'x' in batch and 'time_steps' in batch and 'attn_mask' in batch:
                # 新的elses兼容格式
                x = batch['x'].to(device, non_blocking=True)
                time_steps = batch['time_steps'].to(device, non_blocking=True)
                attn_mask = batch['attn_mask'].to(device, non_blocking=True)
                periods = batch['periods'].to(device, non_blocking=True)
                labels = batch['labels'].to(device, non_blocking=True)
                
                # 转换格式
                batch_size, seq_len = x.shape[0], x.shape[1]
                errmag = torch.zeros(batch_size, seq_len, 1, device=device)
                features = torch.cat([x, errmag], dim=2)
                times = time_steps
                mask = attn_mask.bool()  # 确保mask是bool类型
                
            else:
                # 原有格式兼容
                features = batch['features'].to(device, non_blocking=True)
                times = batch['times'].to(device, non_blocking=True)
                mask = batch['mask'].to(device, non_blocking=True)
                labels = batch['labels'].to(device, non_blocking=True)
            
            try:
                if model_type == 'geometric':
                    logits, sde_features = model(features, times, mask)
                    loss = model.compute_loss(
                        logits, labels, sde_features,
                        focal_gamma=dataset_config['focal_gamma'],
                        temperature=dataset_config['temperature']
                    )[0]
                else:
                    logits, sde_features = model(features, times, mask)
                    loss = model.compute_loss(
                        logits, labels, sde_features,
                        focal_gamma=dataset_config['focal_gamma'],
                        temperature=dataset_config['temperature']
                    )
                    if isinstance(loss, tuple):
                        loss = loss[0]
                        
            except Exception as e:
                print(f"验证失败: {e}")
                continue
            
            total_loss += loss.item()
            _, predicted = torch.max(logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # 收集预测结果用于计算每类准确率
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(val_loader)
    accuracy = 100. * correct / total
    
    # 计算每类准确率
    class_accuracy = calculate_class_accuracy(all_predictions, all_labels)
    
    return avg_loss, accuracy, class_accuracy


def save_checkpoint(model, optimizer, epoch, loss, accuracy, save_path):
    """保存检查点"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'accuracy': accuracy,
        # 保存关键参数用于验证
        'model_params': {
            'hidden_channels': getattr(model.sde, 'hidden_channels', None) if hasattr(model, 'sde') else None,
            'contiformer_dim': getattr(model.contiformer, 'd_model', None) if hasattr(model, 'contiformer') else None,
            'n_heads': getattr(model.contiformer, 'n_heads', None) if hasattr(model, 'contiformer') else None,
            'n_layers': getattr(model.contiformer, 'n_layers', None) if hasattr(model, 'contiformer') else None,
            'sde_method': getattr(model.sde, 'method', None) if hasattr(model, 'sde') else None,
            'dt': getattr(model.sde, 'dt', None) if hasattr(model, 'sde') else None,
        }
    }
    torch.save(checkpoint, save_path)
    print(f"模型已保存: {save_path}")


def load_best_model_if_exists(model, optimizer, args):
    """检查并加载最佳模型（如果存在且参数匹配）"""
    best_model_path = os.path.join(
        args.save_dir,
        f"{args.dataset_name}_{args.model_type}_best.pth"
    )
    
    if not os.path.exists(best_model_path):
        print(f"未找到已保存的最佳模型: {best_model_path}")
        return 0.0, 0  # 返回初始best_val_acc和start_epoch
    
    try:
        checkpoint = torch.load(best_model_path, map_location='cpu')
        
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
        param_mismatch = []
        for key, current_val in current_params.items():
            saved_val = saved_params.get(key)
            if saved_val is not None and saved_val != current_val:
                param_mismatch.append(f"{key}: 当前={current_val}, 已保存={saved_val}")
        
        if param_mismatch:
            print(f"模型参数不匹配，无法加载已保存模型:")
            for mismatch in param_mismatch:
                print(f"  - {mismatch}")
            print("将从头开始训练...")
            return 0.0, 0
        
        # 参数匹配，加载模型
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        best_val_acc = checkpoint.get('accuracy', 0.0)
        epoch = checkpoint.get('epoch', 0)
        
        print(f"✅ 成功加载最佳模型:")
        print(f"  - 路径: {best_model_path}")
        print(f"  - 轮次: {epoch}")
        print(f"  - 最佳验证准确率: {best_val_acc:.4f}")
        print(f"  - 验证损失: {checkpoint.get('loss', 'N/A')}")
        
        return best_val_acc, epoch + 1  # 从下一个epoch开始
        
    except Exception as e:
        print(f"❌ 加载模型时出错: {e}")
        print("将从头开始训练...")
        return 0.0, 0


def main():
    args = parse_args()
    
    # 根据数字参数设置数据路径和数据集名称
    dataset_mapping = {
        1: ('data/ASAS_folded_512.pkl', 'ASAS'),
        2: ('data/LINEAR_folded_512.pkl', 'LINEAR'),
        3: ('data/MACHO_folded_512.pkl', 'MACHO')
    }
    
    args.data_path, args.dataset_name = dataset_mapping[args.dataset]
    print(f"选择数据集: {args.dataset_name} ({args.data_path})")
    
    # 获取数据集特定的配置
    dataset_config = get_dataset_specific_params(args.dataset, args)
    
    # 根据SDE配置设置求解参数
    sde_config_mapping = {
        1: {  # 准确率优先
            'sde_method': 'milstein',
            'dt': 0.01,
            'rtol': 1e-6,
            'atol': 1e-7
        },
        2: {  # 平衡
            'sde_method': 'heun',
            'dt': 0.05,
            'rtol': 1e-4,
            'atol': 1e-5
        },
        3: {  # 时间优先
            'sde_method': 'euler',
            'dt': 0.1,
            'rtol': 1e-2,
            'atol': 1e-3
        }
    }
    
    sde_params = sde_config_mapping[args.sde_config]
    args.sde_method = sde_params['sde_method']
    args.dt = sde_params['dt']
    args.rtol = sde_params['rtol']
    args.atol = sde_params['atol']
    
    config_names = {1: '准确率优先', 2: '平衡', 3: '时间优先'}
    print(f"SDE配置: {config_names[args.sde_config]} (方法:{args.sde_method}, dt:{args.dt}, rtol:{args.rtol}, atol:{args.atol})")
    
    # 清空GPU内存（训练开始前）
    clear_gpu_memory()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 获取设备
    device = get_device(args.device)
    
    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # 加载数据
    print("=== 数据加载 ===")
    train_loader, val_loader, test_loader, class_weights = create_dataloaders(
        data_path=args.data_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        random_seed=args.seed
    )
    
    # 获取类别数量
    num_classes = len(class_weights)
    print(f"类别数量: {num_classes}")
    
    # 创建模型
    print("=== 模型创建 ===")
    model = create_model(args.model_type, num_classes, args, dataset_config)
    model = model.to(device)
    
    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
    print(f"损失函数: Focal Loss (gamma={dataset_config['focal_gamma']}, temp={dataset_config['temperature']})")
    print(f"优化器: Lion (lr={args.learning_rate * 0.3:.1e}, wd={args.weight_decay * 10:.1e})")
    
    # 创建优化器和损失函数 - 使用Lion优化器
    optimizer = Lion(model.parameters(), 
                     lr=args.learning_rate * 0.3,  # Lion通常用更小的学习率
                     weight_decay=args.weight_decay * 10)  # Lion可以用更大的weight_decay
    
    # 禁用混合精度训练，SDE模型数值范围可能导致溢出
    scaler = None  # GradScaler('cuda') if device.type == 'cuda' else None
    
    # 重新启用类别权重，模型内部已修复梯度问题
    criterion = nn.CrossEntropyLoss()  # 主要使用模型内部的compute_loss
    
    # 学习率调度器 - 使用CosineAnnealingWarmRestarts突破收敛瓶颈
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20, T_mult=2, eta_min=1e-6
    )
    
    # 尝试加载已有的最佳模型
    print("=== 检查已有模型 ===")
    best_val_acc, start_epoch = load_best_model_if_exists(model, optimizer, args)
    
    # 设置日志
    print("=== 设置日志 ===")
    log_path, log_data = setup_logging(args.log_dir, args.dataset_name, args.model_type, args.sde_config)
    print(f"日志文件: {log_path}")
    
    # 训练循环
    print("=== 开始训练 ===")
    if start_epoch > 0:
        print(f"从第 {start_epoch+1} 轮继续训练...")
    else:
        print("从头开始训练...")
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    for epoch in range(start_epoch, args.epochs):
        print(f"Epoch [{epoch+1}/{args.epochs}]")
        
        # 每个epoch开始前清理GPU内存
        if epoch > 0:  # 第一个epoch已经在main开始时清理过
            clear_gpu_memory()
        
        # 训练
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device, args.model_type, dataset_config, scaler
        )
        
        # 验证
        val_loss, val_acc, class_accuracy = validate_epoch(
            model, val_loader, criterion, device, args.model_type, dataset_config
        )
        
        # 更新学习率 - CosineAnnealingWarmRestarts按epoch调度
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # 记录
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        # 检查是否为最佳模型
        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
        
        # 打印详细的epoch总结
        print_epoch_summary(epoch, args.epochs, train_loss, train_acc, val_loss, val_acc,
                          class_accuracy, current_lr, is_best)
        
        # 更新日志
        log_data = update_log(log_path, log_data, epoch, train_loss, train_acc, val_loss, val_acc,
                            class_accuracy, current_lr, is_best)
        
        # 检查分类分布，避免全部预测为一个类别
        if epoch == 0:  # 第一个epoch后检查
            model.eval()
            with torch.no_grad():
                pred_counts = torch.zeros(num_classes)
                total_samples = 0
                
                for batch in val_loader:
                    features = batch['features'].to(device, non_blocking=True)
                    times = batch['times'].to(device, non_blocking=True) 
                    mask = batch['mask'].to(device, non_blocking=True)
                    
                    try:
                        if args.model_type == 'geometric':
                            logits, _ = model(features, times, mask)
                        else:
                            logits, _ = model(features, times, mask)
                        
                        _, predicted = torch.max(logits.data, 1)
                        for pred in predicted:
                            pred_counts[pred.item()] += 1
                        total_samples += predicted.size(0)
                    except:
                        continue
                
                pred_ratios = pred_counts / total_samples
                print(f"预测分布: {[f'{ratio:.2f}' for ratio in pred_ratios]}")
                
                # 如果95%以上的预测都是同一个类别，发出警告
                max_ratio = pred_ratios.max().item()
                if max_ratio > 0.95:
                    print(f"警告: {max_ratio*100:.1f}% 的预测集中在类别 {pred_ratios.argmax().item()}!")
                    print("可能存在类别不平衡或模型收敛问题")
        
        # 保存最佳模型
        if is_best:
            best_model_path = os.path.join(
                args.save_dir, 
                f"{args.dataset_name}_{args.model_type}_best.pth"
            )
            save_checkpoint(model, optimizer, epoch, val_loss, val_acc, best_model_path)
        
        # 定期保存
        if (epoch + 1) % args.save_interval == 0:
            checkpoint_path = os.path.join(
                args.save_dir,
                f"{args.dataset_name}_{args.model_type}_epoch_{epoch+1}.pth"
            )
            save_checkpoint(model, optimizer, epoch, val_loss, val_acc, checkpoint_path)
    
    print(f"训练完成! 最佳验证准确率: {best_val_acc:.2f}%")
    
    # 训练结束后最终清理GPU内存
    clear_gpu_memory()
    
    # 保存训练历史
    history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accs': train_accs,
        'val_accs': val_accs,
        'best_val_acc': best_val_acc,
        'args': vars(args)
    }
    
    history_path = os.path.join(
        args.log_dir, 
        f"{args.dataset_name}_{args.model_type}_history.json"
    )
    
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"训练历史已保存: {history_path}")


if __name__ == '__main__':
    main()