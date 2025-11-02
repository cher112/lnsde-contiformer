#!/usr/bin/env python3
"""
训练可视化工具模块
用于生成训练过程中各个指标的可视化图表
"""

import json
import matplotlib
import matplotlib.font_manager as fm
import numpy as np
import os
from datetime import datetime
import warnings

def configure_chinese_font():
    """配置中文字体显示 - 在导入pyplot之前调用"""
    try:
        # 字体文件路径
        font_path = '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc'
        
        if os.path.exists(font_path):
            # 创建FontProperties对象
            prop = fm.FontProperties(fname=font_path)
            
            # 获取matplotlib识别的确切字体名称
            font_name = prop.get_name()
            
            # 添加字体到管理器
            fm.fontManager.addfont(font_path)
            
            # 设置matplotlib的rcParams - 关键：在导入pyplot之前
            matplotlib.rcParams['font.sans-serif'] = [font_name, 'DejaVu Sans', 'Arial']
            matplotlib.rcParams['axes.unicode_minus'] = False
            
            print(f"✓ 中文字体配置成功: {font_name}")
            return True
        else:
            print(f"✗ 字体文件不存在: {font_path}")
            # 使用默认英文字体
            matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']
            matplotlib.rcParams['axes.unicode_minus'] = False
            return False
            
    except Exception as e:
        print(f"✗ 中文字体配置失败: {e}")
        # 使用默认英文字体
        matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']
        matplotlib.rcParams['axes.unicode_minus'] = False
        return False

# 在导入pyplot之前配置字体
configure_chinese_font()

# 现在导入pyplot
import matplotlib.pyplot as plt

def extract_metrics_from_log(log_file):
    """从日志文件中提取训练指标"""
    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        epochs_data = data.get('epochs', [])
        if not epochs_data:
            print(f"警告: 日志文件 {log_file} 中没有找到epochs数据")
            return None
        
        # 提取各项指标
        metrics = {
            'epochs': [],
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'train_f1': [],
            'val_f1': [],
            'train_recall': [],
            'val_recall': [],
            'learning_rate': [],
            'epoch_time': []
        }
        
        for epoch_data in epochs_data:
            metrics['epochs'].append(epoch_data.get('epoch', 0))
            metrics['train_loss'].append(epoch_data.get('train_loss', 0))
            metrics['val_loss'].append(epoch_data.get('val_loss', 0))
            metrics['train_acc'].append(epoch_data.get('train_acc', 0))
            metrics['val_acc'].append(epoch_data.get('val_acc', 0))
            metrics['train_f1'].append(epoch_data.get('train_f1', 0))
            metrics['val_f1'].append(epoch_data.get('val_f1', 0))
            metrics['train_recall'].append(epoch_data.get('train_recall', 0))
            metrics['val_recall'].append(epoch_data.get('val_recall', 0))
            metrics['learning_rate'].append(epoch_data.get('learning_rate', 0))
            metrics['epoch_time'].append(epoch_data.get('epoch_time', 0))
        
        return metrics
        
    except Exception as e:
        print(f"提取指标失败: {e}")
        return None

def create_metric_plot(epochs, train_values, val_values, metric_name, ylabel, save_path, use_chinese=True):
    """创建单个指标的训练曲线图"""
    plt.figure(figsize=(12, 8))
    
    # 根据是否支持中文选择标签
    if use_chinese:
        train_label = f'训练 {metric_name}'
        val_label = f'验证 {metric_name}'
        xlabel = '训练轮次 (Epoch)'
        title_prefix = '训练曲线'
        best_train_text = '最佳训练'
        best_val_text = '最佳验证'
    else:
        train_label = f'Train {metric_name}'
        val_label = f'Val {metric_name}'
        xlabel = 'Epoch'
        title_prefix = 'Training Curve'
        best_train_text = 'Best Train'
        best_val_text = 'Best Val'
    
    plt.plot(epochs, train_values, 'b-', linewidth=2, label=train_label, marker='o', markersize=4)
    plt.plot(epochs, val_values, 'r-', linewidth=2, label=val_label, marker='s', markersize=4)
    
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.title(f'{metric_name} {title_prefix}', fontsize=16, fontweight='bold')
    
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    
    # 添加最佳值标注
    if 'loss' in metric_name.lower():
        # 对于loss，找到最小值
        best_train_idx = np.argmin(train_values)
        best_val_idx = np.argmin(val_values)
        best_train_val = train_values[best_train_idx]
        best_val_val = val_values[best_val_idx]
    else:
        # 对于acc、f1、recall，找到最大值
        best_train_idx = np.argmax(train_values)
        best_val_idx = np.argmax(val_values)
        best_train_val = train_values[best_train_idx]
        best_val_val = val_values[best_val_idx]
    
    # 标注最佳点
    plt.annotate(f'{best_train_text}: {best_train_val:.4f}', 
                xy=(epochs[best_train_idx], best_train_val),
                xytext=(10, 10), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.7),
                arrowprops=dict(arrowstyle='->', color='blue'),
                fontsize=10)
    
    plt.annotate(f'{best_val_text}: {best_val_val:.4f}', 
                xy=(epochs[best_val_idx], best_val_val),
                xytext=(10, -20), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightcoral', alpha=0.7),
                arrowprops=dict(arrowstyle='->', color='red'),
                fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return save_path

def create_learning_rate_plot(epochs, lr_values, save_path, use_chinese=True):
    """创建学习率变化曲线图"""
    plt.figure(figsize=(12, 6))
    
    if use_chinese:
        label = '学习率'
        xlabel = '训练轮次 (Epoch)'
        ylabel = '学习率'
        title = '学习率变化曲线'
    else:
        label = 'Learning Rate'
        xlabel = 'Epoch'
        ylabel = 'Learning Rate'
        title = 'Learning Rate Schedule'
    
    plt.plot(epochs, lr_values, 'g-', linewidth=2, label=label, marker='o', markersize=3)
    
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.title(title, fontsize=16, fontweight='bold')
    
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    plt.yscale('log')  # 使用对数尺度显示学习率
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return save_path

def create_epoch_time_plot(epochs, time_values, save_path, use_chinese=True):
    """创建每轮训练时间曲线图"""
    plt.figure(figsize=(12, 6))
    
    if use_chinese:
        label = '每轮训练时间'
        xlabel = '训练轮次 (Epoch)'
        ylabel = '训练时间 (秒)'
        title = '每轮训练时间变化'
        avg_label = f'平均时间: {np.mean(time_values):.1f}秒'
    else:
        label = 'Epoch Time'
        xlabel = 'Epoch'
        ylabel = 'Time (seconds)'
        title = 'Training Time per Epoch'
        avg_label = f'Average: {np.mean(time_values):.1f}s'
    
    plt.plot(epochs, time_values, 'm-', linewidth=2, label=label, marker='o', markersize=3)
    
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.title(title, fontsize=16, fontweight='bold')
    
    plt.grid(True, alpha=0.3)
    
    # 添加平均时间线
    avg_time = np.mean(time_values)
    plt.axhline(y=avg_time, color='orange', linestyle='--', alpha=0.7, label=avg_label)
    plt.legend(fontsize=12)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return save_path

def create_comprehensive_training_plot(metrics, dataset_name, model_type_str, save_path, use_chinese=True):
    """创建综合的训练过程四子图可视化"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    epochs = metrics['epochs']
    
    # SDE模型名称映射
    sde_names = {
        'langevin': 'Langevin SDE',
        'linear_noise': 'Linear Noise SDE', 
        'geometric': 'Geometric SDE'
    }
    sde_display_name = sde_names.get(model_type_str, model_type_str.upper())
    
    # 设置总标题
    if use_chinese:
        main_title = f'{dataset_name} Complete Training Process (Epoch 1-{len(epochs)}) - {sde_display_name} - Original F1 & Recall'
    else:
        main_title = f'{dataset_name} Complete Training Process (Epoch 1-{len(epochs)}) - {sde_display_name} - Original F1 & Recall'
    
    fig.suptitle(main_title, fontsize=14, fontweight='bold')
    
    # 1. Loss图 (左上)
    ax1 = axes[0, 0]
    ax1.plot(epochs, metrics['train_loss'], 'b-', linewidth=2, label='Training Loss')
    ax1.plot(epochs, metrics['val_loss'], color='purple', linewidth=2, label='Validation Loss')
    ax1.set_xlabel('Training Epochs')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training & Validation Loss')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 添加最小值标注
    min_train_idx = np.argmin(metrics['train_loss'])
    min_val_idx = np.argmin(metrics['val_loss'])
    ax1.annotate(f'Min Train Loss\nEpoch {epochs[min_train_idx]}\n{metrics["train_loss"][min_train_idx]:.4f}',
                xy=(epochs[min_train_idx], metrics['train_loss'][min_train_idx]),
                xytext=(0.3, 0.7), textcoords='axes fraction',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.7),
                arrowprops=dict(arrowstyle='->', color='blue'),
                fontsize=8)
    ax1.annotate(f'Min Val Loss\nEpoch {epochs[min_val_idx]}\n{metrics["val_loss"][min_val_idx]:.4f}',
                xy=(epochs[min_val_idx], metrics['val_loss'][min_val_idx]),
                xytext=(0.6, 0.3), textcoords='axes fraction',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightpink', alpha=0.7),
                arrowprops=dict(arrowstyle='->', color='purple'),
                fontsize=8)
    
    # 2. Accuracy图 (右上)
    ax2 = axes[0, 1]
    ax2.plot(epochs, metrics['train_acc'], 'b-', linewidth=2, label='Training Accuracy')
    ax2.plot(epochs, metrics['val_acc'], color='purple', linewidth=2, label='Validation Accuracy')
    ax2.set_xlabel('Training Epochs')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training & Validation Accuracy')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # 添加最大值标注
    max_train_idx = np.argmax(metrics['train_acc'])
    max_val_idx = np.argmax(metrics['val_acc'])
    ax2.annotate(f'Max Train Acc\nEpoch {epochs[max_train_idx]}\n{metrics["train_acc"][max_train_idx]:.2f}%',
                xy=(epochs[max_train_idx], metrics['train_acc'][max_train_idx]),
                xytext=(0.05, 0.7), textcoords='axes fraction',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.7),
                arrowprops=dict(arrowstyle='->', color='blue'),
                fontsize=8)
    ax2.annotate(f'Max Val Acc\nEpoch {epochs[max_val_idx]}\n{metrics["val_acc"][max_val_idx]:.2f}%',
                xy=(epochs[max_val_idx], metrics['val_acc'][max_val_idx]),
                xytext=(0.7, 0.3), textcoords='axes fraction',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightpink', alpha=0.7),
                arrowprops=dict(arrowstyle='->', color='purple'),
                fontsize=8)
    
    # 3. F1 Score图 (左下)
    ax3 = axes[1, 0]
    ax3.plot(epochs, metrics['train_f1'], 'b-', linewidth=2, label='Training F1 Score (Original)')
    ax3.plot(epochs, metrics['val_f1'], color='purple', linewidth=2, label='Validation F1 Score (Original)')
    ax3.set_xlabel('Training Epochs')
    ax3.set_ylabel('F1 Score')
    ax3.set_title('Training & Validation F1 Score (Original from Logs)')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # 添加最大值标注
    max_train_f1_idx = np.argmax(metrics['train_f1'])
    max_val_f1_idx = np.argmax(metrics['val_f1'])
    ax3.annotate(f'Max Train F1\nEpoch {epochs[max_train_f1_idx]}\n{metrics["train_f1"][max_train_f1_idx]:.2f}',
                xy=(epochs[max_train_f1_idx], metrics['train_f1'][max_train_f1_idx]),
                xytext=(0.3, 0.3), textcoords='axes fraction',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.7),
                arrowprops=dict(arrowstyle='->', color='blue'),
                fontsize=8)
    ax3.annotate(f'Max Val F1\nEpoch {epochs[max_val_f1_idx]}\n{metrics["val_f1"][max_val_f1_idx]:.2f}',
                xy=(epochs[max_val_f1_idx], metrics['val_f1'][max_val_f1_idx]),
                xytext=(0.6, 0.7), textcoords='axes fraction',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightpink', alpha=0.7),
                arrowprops=dict(arrowstyle='->', color='purple'),
                fontsize=8)
    
    # 4. Recall图 (右下)
    ax4 = axes[1, 1]
    ax4.plot(epochs, metrics['train_recall'], 'b-', linewidth=2, label='Training Recall (Original)')
    ax4.plot(epochs, metrics['val_recall'], color='purple', linewidth=2, label='Validation Recall (Original)')
    ax4.set_xlabel('Training Epochs')
    ax4.set_ylabel('Recall')
    ax4.set_title('Training & Validation Recall (Original from Logs)')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    # 添加最大值标注
    max_train_recall_idx = np.argmax(metrics['train_recall'])
    max_val_recall_idx = np.argmax(metrics['val_recall'])
    ax4.annotate(f'Max Train Recall\nEpoch {epochs[max_train_recall_idx]}\n{metrics["train_recall"][max_train_recall_idx]:.2f}',
                xy=(epochs[max_train_recall_idx], metrics['train_recall'][max_train_recall_idx]),
                xytext=(0.3, 0.3), textcoords='axes fraction',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.7),
                arrowprops=dict(arrowstyle='->', color='blue'),
                fontsize=8)
    ax4.annotate(f'Max Val Recall\nEpoch {epochs[max_val_recall_idx]}\n{metrics["val_recall"][max_val_recall_idx]:.2f}',
                xy=(epochs[max_val_recall_idx], metrics['val_recall'][max_val_recall_idx]),
                xytext=(0.6, 0.7), textcoords='axes fraction',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightpink', alpha=0.7),
                arrowprops=dict(arrowstyle='->', color='purple'),
                fontsize=8)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return save_path

def generate_training_visualizations(log_file, dataset_name, model_type_str, timestamp_dir=None):
    """
    生成训练可视化图表的主函数
    
    Args:
        log_file (str): 训练日志文件路径
        dataset_name (str): 数据集名称 (e.g., 'LINEAR', 'ASAS', 'MACHO')
        model_type_str (str): 模型类型字符串 (e.g., 'linear_noise', 'langevin', 'geometric')
        timestamp_dir (str): 时间戳目录路径，使用新的标准化路径结构
    
    Returns:
        list: 生成的图片文件路径列表
    """
    if not os.path.exists(log_file):
        print(f"日志文件不存在: {log_file}")
        return []
    
    # 字体已在导入时配置，检查当前配置
    use_chinese = 'WenQuanYi' in str(matplotlib.rcParams['font.sans-serif'])
    
    # 使用标准化路径结构 - 训练图表放在 timestamp_dir/plots
    if timestamp_dir is not None:
        output_dir = os.path.join(timestamp_dir, "plots")
    else:
        # 后备方案：使用基础results目录
        output_dir = f'/root/autodl-fs/lnsde-contiformer/results/plots'
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 提取指标数据
    print(f"正在从 {log_file} 提取训练指标...")
    metrics = extract_metrics_from_log(log_file)
    
    if metrics is None:
        print("提取指标失败")
        return []
    
    epochs = metrics['epochs']
    generated_files = []
    
    if not epochs:
        print("没有找到有效的epoch数据")
        return []
    
    # 创建综合的训练过程四子图 (主要图表)
    # 清理model_type_str，去除不必要的字符，使文件名更简洁
    clean_model_type = model_type_str.replace('_', '').replace('+', '').replace('unknown', 'sde')
    comprehensive_filename = f'{dataset_name.lower()}_{clean_model_type}_training_{len(epochs)}epochs.png'
    comprehensive_save_path = os.path.join(output_dir, comprehensive_filename)
    
    try:
        created_path = create_comprehensive_training_plot(
            metrics, dataset_name, model_type_str, comprehensive_save_path, use_chinese
        )
        generated_files.append(created_path)
        print(f"✅ 已生成综合训练过程图表: {created_path}")
    except Exception as e:
        print(f"❌ 生成综合训练图表失败: {e}")
    
    # 可选：创建各个指标的单独曲线图 (保持原有功能)
    metric_configs = [
        ('train_loss', 'val_loss', 'Loss', '损失值' if use_chinese else 'Loss Value'),
        ('train_acc', 'val_acc', 'Accuracy', '准确率 (%)' if use_chinese else 'Accuracy (%)'),
        ('train_f1', 'val_f1', 'F1-Score', 'F1分数' if use_chinese else 'F1 Score'),
        ('train_recall', 'val_recall', 'Recall', '召回率' if use_chinese else 'Recall')
    ]
    
    for train_key, val_key, metric_name, ylabel in metric_configs:
        if train_key in metrics and val_key in metrics:
            save_path = os.path.join(output_dir, f'{train_key}_curve.png')
            try:
                created_path = create_metric_plot(
                    epochs, metrics[train_key], metrics[val_key], 
                    metric_name, ylabel, save_path, use_chinese
                )
                generated_files.append(created_path)
                print(f"已生成 {metric_name} 曲线图: {created_path}")
            except Exception as e:
                print(f"生成 {metric_name} 曲线图失败: {e}")
    
    # 创建学习率曲线图
    if 'learning_rate' in metrics and metrics['learning_rate']:
        lr_save_path = os.path.join(output_dir, 'learning_rate_curve.png')
        try:
            created_path = create_learning_rate_plot(epochs, metrics['learning_rate'], lr_save_path, use_chinese)
            generated_files.append(created_path)
            print(f"已生成学习率曲线图: {created_path}")
        except Exception as e:
            print(f"生成学习率曲线图失败: {e}")
    
    # 创建训练时间曲线图
    if 'epoch_time' in metrics and metrics['epoch_time']:
        time_save_path = os.path.join(output_dir, 'epoch_time_curve.png')
        try:
            created_path = create_epoch_time_plot(epochs, metrics['epoch_time'], time_save_path, use_chinese)
            generated_files.append(created_path)
            print(f"已生成训练时间曲线图: {created_path}")
        except Exception as e:
            print(f"生成训练时间曲线图失败: {e}")
    
    print(f"\n所有可视化图表已保存到: {output_dir}")
    print(f"共生成 {len(generated_files)} 个图表文件")
    
    return generated_files