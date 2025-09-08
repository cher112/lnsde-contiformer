"""
日志相关工具函数
"""

import os
import json
import time
from datetime import datetime
from .path_manager import get_log_path


def setup_logging(timestamp_dir, dataset_name, model_type, sde_config, args=None):
    """设置日志记录 - 使用新的时间戳目录结构"""
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    date_str = now.strftime("%Y%m%d")
    
    # 构建包含模型参数的日志文件名
    if args is not None:
        # 基础模型类型映射
        base_model_map = {1: 'langevin', 2: 'linear_noise', 3: 'geometric'}
        base_model = base_model_map.get(model_type, f'model{model_type}')
        
        # 组件开关状态
        use_sde = getattr(args, 'use_sde', 1)
        use_contiformer = getattr(args, 'use_contiformer', 1)
        
        # 根据组件开关组合确定完整模型类型
        if use_sde and use_contiformer:
            model_name = f"{base_model}_sde_cf"  # 完整模型
        elif use_sde and not use_contiformer:
            model_name = f"{base_model}_sde_only"  # 只有SDE
        elif not use_sde and use_contiformer:
            model_name = "contiformer_only"   # 只有ContiFormer，不需要SDE类型
        else:
            model_name = "baseline"  # 基础模型，不需要SDE类型
        
        # 关键参数信息
        lr = getattr(args, 'learning_rate', 1e-4)
        batch_size = getattr(args, 'batch_size', 64)
        hidden_channels = getattr(args, 'hidden_channels', 128)
        contiformer_dim = getattr(args, 'contiformer_dim', 128)
        
        # 构建详细文件名
        filename = (f"{dataset_name}_{model_name}_config{sde_config}"
                   f"_lr{lr:.0e}_bs{batch_size}_hc{hidden_channels}_cd{contiformer_dim}.log")
    else:
        # 保持原有格式作为后备
        filename = f"{dataset_name}_{model_type}_config{sde_config}.log"
    
    # 使用新的路径管理获取日志路径
    log_path = os.path.join(timestamp_dir, "logs", filename)
    
    # 确保目录存在
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    
    # 初始化日志数据
    log_data = {
        'dataset': dataset_name,
        'model_type': model_type,
        'sde_config': sde_config,
        'start_time': timestamp,
        'date': date_str,
        'best_epoch': 0,  # 当前最佳epoch
        'best_val_acc': 0.0,  # 当前最佳验证准确率
        'best_timestamp': None,  # 最佳准确率达成时间
        'epochs': []
    }
    
    print(f"日志文件: {log_path}")
    return log_path, log_data


def update_log(log_path, log_data, epoch, train_loss, train_acc, val_loss, val_acc, 
               class_accuracies=None, total_time=None, lr=None, train_metrics=None, val_metrics=None, 
               is_best=False):
    """更新日志数据"""
    epoch_data = {
        'epoch': epoch,
        'train_loss': float(train_loss),
        'train_acc': float(train_acc),
        'val_loss': float(val_loss),
        'val_acc': float(val_acc),
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'is_best': is_best  # 标记是否为最佳epoch
    }
    
    # 添加训练额外指标 (加权平均)
    if train_metrics is not None:
        epoch_data['train_f1'] = float(train_metrics['f1_score'])  # 加权平均F1
        epoch_data['train_recall'] = float(train_metrics['recall'])  # 加权平均Recall
    
    # 添加验证额外指标 (加权平均)
    if val_metrics is not None:
        epoch_data['val_f1'] = float(val_metrics['f1_score'])  # 加权平均F1
        epoch_data['val_recall'] = float(val_metrics['recall'])  # 加权平均Recall
        
        # 每个epoch都保存混淆矩阵（如果存在）
        if 'confusion_matrix' in val_metrics and val_metrics['confusion_matrix'] is not None:
            cm = val_metrics['confusion_matrix']
            # 对于numpy数组，检查是否有元素且不全为0
            if hasattr(cm, 'size') and cm.size > 0:
                epoch_data['confusion_matrix'] = cm.tolist()  # 转为列表便于JSON序列化
    
    # 每个epoch都记录class_accuracies（如果存在）
    if class_accuracies is not None:
        epoch_data['class_accuracies'] = {k: float(v) for k, v in class_accuracies.items()}
    
    if total_time is not None:
        epoch_data['epoch_time'] = float(total_time)
        
    if lr is not None:
        epoch_data['learning_rate'] = float(lr)
    
    # 更新最佳epoch记录
    if is_best:
        log_data['best_epoch'] = epoch
        log_data['best_val_acc'] = float(val_acc)
        log_data['best_timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    log_data['epochs'].append(epoch_data)
    
    # 确保日志目录存在
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    
    # 保存日志文件
    with open(log_path, 'w', encoding='utf-8') as f:
        json.dump(log_data, f, ensure_ascii=False, indent=2)


def print_epoch_summary(epoch, total_epochs, train_loss, train_acc, val_loss, val_acc, 
                       class_accuracies=None, epoch_time=None, lr=None, train_metrics=None, val_metrics=None, best_epoch=None):
    """打印训练轮次总结"""
    print(f"\nEpoch [{epoch}/{total_epochs}] 总结:")
    
    # 训练指标
    train_info = f"  训练 - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%"
    if train_metrics:
        train_info += f", 加权F1: {train_metrics['f1_score']*100:.1f}%, 加权Recall: {train_metrics['recall']*100:.1f}%"
    print(train_info)
    
    # 验证指标
    val_info = f"  验证 - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%"
    if val_metrics:
        val_info += f", 加权F1: {val_metrics['f1_score']*100:.1f}%, 加权Recall: {val_metrics['recall']*100:.1f}%"
    print(val_info)
    
    # 显示最佳epoch信息
    if best_epoch is not None and best_epoch > 0:
        if epoch == best_epoch:
            print(f"  🎉 新的最佳验证准确率: {val_acc:.2f}% (Epoch {best_epoch})")
        else:
            print(f"  📊 当前最佳: Epoch {best_epoch}")
    
    if lr is not None:
        print(f"  学习率: {lr:.2e}")
    
    if epoch_time is not None:
        print(f"  耗时: {epoch_time:.1f}s")
    
    if val_metrics and 'confusion_matrix' in val_metrics and val_metrics['confusion_matrix'] is not None:
        confusion_matrix = val_metrics['confusion_matrix']
        # 确保是有效的矩阵
        if hasattr(confusion_matrix, 'shape') and confusion_matrix.size > 0:
            print("  混淆矩阵:")
            num_classes = confusion_matrix.shape[0]
            
            # 直接打印n*n矩阵，每行缩进4个空格
            for i in range(num_classes):
                print("    ", end="")
                for j in range(num_classes):
                    print(f"{confusion_matrix[i, j]:>4}", end=" ")
                print()
    elif class_accuracies is not None:
        # 备用显示：如果没有混淆矩阵，仍显示各类别准确率
        print("  各类别准确率:")
        for class_name, acc in class_accuracies.items():
            print(f"    {class_name}: {acc:.2f}%")
    
    print("-" * 80)