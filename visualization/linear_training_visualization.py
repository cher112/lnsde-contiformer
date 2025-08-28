#!/usr/bin/env python3
"""
LINEAR训练日志可视化脚本
从训练日志文件中提取指标数据并为每个指标生成单独的曲线图
"""

import json
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import os

def configure_chinese_font():
    """配置中文字体显示"""
    # 添加字体到matplotlib管理器
    fm.fontManager.addfont('/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc')
    fm.fontManager.addfont('/usr/share/fonts/truetype/wqy/wqy-microhei.ttc')
    
    # 设置中文字体优先级列表
    plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'WenQuanYi Micro Hei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    return True

def extract_metrics_from_log(log_file):
    """从日志文件中提取训练指标"""
    with open(log_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    epochs = data['epochs']
    
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
    
    for epoch_data in epochs:
        metrics['epochs'].append(epoch_data['epoch'])
        metrics['train_loss'].append(epoch_data['train_loss'])
        metrics['val_loss'].append(epoch_data['val_loss'])
        metrics['train_acc'].append(epoch_data['train_acc'])
        metrics['val_acc'].append(epoch_data['val_acc'])
        metrics['train_f1'].append(epoch_data['train_f1'])
        metrics['val_f1'].append(epoch_data['val_f1'])
        metrics['train_recall'].append(epoch_data['train_recall'])
        metrics['val_recall'].append(epoch_data['val_recall'])
        metrics['learning_rate'].append(epoch_data['learning_rate'])
        metrics['epoch_time'].append(epoch_data['epoch_time'])
    
    return metrics

def create_metric_plot(epochs, train_values, val_values, metric_name, ylabel, save_path):
    """创建单个指标的训练曲线图"""
    plt.figure(figsize=(12, 8))
    
    plt.plot(epochs, train_values, 'b-', linewidth=2, label=f'训练 {metric_name}', marker='o', markersize=4)
    plt.plot(epochs, val_values, 'r-', linewidth=2, label=f'验证 {metric_name}', marker='s', markersize=4)
    
    plt.xlabel('训练轮次 (Epoch)', fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.title(f'LINEAR模型 - {metric_name} 训练曲线', fontsize=16, fontweight='bold')
    
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
    plt.annotate(f'最佳训练: {best_train_val:.4f}', 
                xy=(epochs[best_train_idx], best_train_val),
                xytext=(10, 10), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.7),
                arrowprops=dict(arrowstyle='->', color='blue'),
                fontsize=10)
    
    plt.annotate(f'最佳验证: {best_val_val:.4f}', 
                xy=(epochs[best_val_idx], best_val_val),
                xytext=(10, -20), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightcoral', alpha=0.7),
                arrowprops=dict(arrowstyle='->', color='red'),
                fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"已保存 {metric_name} 曲线图: {save_path}")

def create_learning_rate_plot(epochs, lr_values, save_path):
    """创建学习率变化曲线图"""
    plt.figure(figsize=(12, 6))
    
    plt.plot(epochs, lr_values, 'g-', linewidth=2, label='学习率', marker='o', markersize=3)
    
    plt.xlabel('训练轮次 (Epoch)', fontsize=14)
    plt.ylabel('学习率', fontsize=14)
    plt.title('LINEAR模型 - 学习率变化曲线', fontsize=16, fontweight='bold')
    
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    plt.yscale('log')  # 使用对数尺度显示学习率
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"已保存学习率曲线图: {save_path}")

def create_epoch_time_plot(epochs, time_values, save_path):
    """创建每轮训练时间曲线图"""
    plt.figure(figsize=(12, 6))
    
    plt.plot(epochs, time_values, 'm-', linewidth=2, label='每轮训练时间', marker='o', markersize=3)
    
    plt.xlabel('训练轮次 (Epoch)', fontsize=14)
    plt.ylabel('训练时间 (秒)', fontsize=14)
    plt.title('LINEAR模型 - 每轮训练时间变化', fontsize=16, fontweight='bold')
    
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    
    # 添加平均时间线
    avg_time = np.mean(time_values)
    plt.axhline(y=avg_time, color='orange', linestyle='--', alpha=0.7, 
                label=f'平均时间: {avg_time:.1f}秒')
    plt.legend(fontsize=12)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"已保存训练时间曲线图: {save_path}")

def main():
    # 配置中文字体
    configure_chinese_font()
    
    # 输入和输出路径
    log_file = '/root/autodl-tmp/lnsde-contiformer/results/logs/20250826/LINEAR/LINEAR_linear_noise_config1_20250826_111256.log'
    output_dir = '/root/autodl-tmp/lnsde-contiformer/results/pics/LINEAR'
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 提取指标数据
    print("正在提取训练指标...")
    metrics = extract_metrics_from_log(log_file)
    epochs = metrics['epochs']
    
    # 创建各个指标的曲线图
    metric_configs = [
        ('train_loss', 'val_loss', 'Loss', '损失值'),
        ('train_acc', 'val_acc', 'Accuracy', '准确率 (%)'),
        ('train_f1', 'val_f1', 'F1-Score', 'F1分数'),
        ('train_recall', 'val_recall', 'Recall', '召回率')
    ]
    
    for train_key, val_key, metric_name, ylabel in metric_configs:
        save_path = os.path.join(output_dir, f'{train_key}_curve.png')
        create_metric_plot(epochs, metrics[train_key], metrics[val_key], 
                          metric_name, ylabel, save_path)
    
    # 创建学习率曲线图
    lr_save_path = os.path.join(output_dir, 'learning_rate_curve.png')
    create_learning_rate_plot(epochs, metrics['learning_rate'], lr_save_path)
    
    # 创建训练时间曲线图
    time_save_path = os.path.join(output_dir, 'epoch_time_curve.png')
    create_epoch_time_plot(epochs, metrics['epoch_time'], time_save_path)
    
    print(f"\n所有可视化图表已保存到: {output_dir}")
    print("生成的图表包括:")
    print("- train_loss_curve.png (损失函数曲线)")
    print("- train_acc_curve.png (准确率曲线)")
    print("- train_f1_curve.png (F1分数曲线)")
    print("- train_recall_curve.png (召回率曲线)")
    print("- learning_rate_curve.png (学习率变化曲线)")
    print("- epoch_time_curve.png (每轮训练时间曲线)")

if __name__ == "__main__":
    main()