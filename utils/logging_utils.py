"""
日志相关工具函数
"""

import os
import json
import time
from datetime import datetime
from .path_manager import get_log_path


def setup_logging(timestamp_dir, dataset_name, model_type, sde_config):
    """设置日志记录 - 使用新的时间戳目录结构"""
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    date_str = now.strftime("%Y%m%d")
    
    # 使用新的路径管理获取日志路径
    log_path = os.path.join(timestamp_dir, "logs", f"{dataset_name}_{model_type}_config{sde_config}.log")
    
    # 确保目录存在
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    
    # 初始化日志数据
    log_data = {
        'dataset': dataset_name,
        'model_type': model_type,
        'sde_config': sde_config,
        'start_time': timestamp,
        'date': date_str,
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
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # 添加训练额外指标
    if train_metrics is not None:
        epoch_data['train_f1'] = float(train_metrics['f1_score'])
        epoch_data['train_recall'] = float(train_metrics['recall'])
    
    # 添加验证额外指标
    if val_metrics is not None:
        epoch_data['val_f1'] = float(val_metrics['f1_score'])
        epoch_data['val_recall'] = float(val_metrics['recall'])
        
        # 如果有混淆矩阵，也保存它
        if 'confusion_matrix' in val_metrics and val_metrics['confusion_matrix']:
            epoch_data['confusion_matrix'] = val_metrics['confusion_matrix']
    
    # 只在最优批次记录class_accuracies
    if class_accuracies is not None and is_best:
        epoch_data['class_accuracies'] = {k: float(v) for k, v in class_accuracies.items()}
    
    if total_time is not None:
        epoch_data['epoch_time'] = float(total_time)
        
    if lr is not None:
        epoch_data['learning_rate'] = float(lr)
    
    log_data['epochs'].append(epoch_data)
    
    # 保存日志文件
    with open(log_path, 'w', encoding='utf-8') as f:
        json.dump(log_data, f, ensure_ascii=False, indent=2)


def print_epoch_summary(epoch, total_epochs, train_loss, train_acc, val_loss, val_acc, 
                       class_accuracies=None, epoch_time=None, lr=None, train_metrics=None, val_metrics=None):
    """打印训练轮次总结"""
    print(f"\nEpoch [{epoch}/{total_epochs}] 总结:")
    
    # 训练指标
    train_info = f"  训练 - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%"
    if train_metrics:
        train_info += f", F1: {train_metrics['f1_score']:.1f}, Recall: {train_metrics['recall']:.1f}"
    print(train_info)
    
    # 验证指标
    val_info = f"  验证 - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%"
    if val_metrics:
        val_info += f", F1: {val_metrics['f1_score']:.1f}, Recall: {val_metrics['recall']:.1f}"
    print(val_info)
    
    if lr is not None:
        print(f"  学习率: {lr:.2e}")
    
    if epoch_time is not None:
        print(f"  耗时: {epoch_time:.1f}s")
    
    if class_accuracies is not None:
        print("  各类别准确率:")
        for class_name, acc in class_accuracies.items():
            print(f"    {class_name}: {acc:.2f}%")
    
    print("-" * 80)