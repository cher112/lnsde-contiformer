#!/usr/bin/env python3

import json
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import os
from pathlib import Path
import seaborn as sns

# 设置seaborn风格
sns.set_style("whitegrid")

def configure_chinese_fonts():
    """配置中文字体，使用matplotlib的字体后备机制"""
    
    # 添加中文字体到matplotlib字体管理器
    font_dirs = ['/usr/share/fonts/truetype/wqy/', str(Path.home() / '.matplotlib')]
    
    for font_dir in font_dirs:
        if os.path.exists(font_dir):
            for font_file in os.listdir(font_dir):
                if font_file.endswith(('.ttf', '.ttc', '.otf')):
                    font_path = os.path.join(font_dir, font_file)
                    try:
                        fm.fontManager.addfont(font_path)
                        print(f"Added font: {font_path}")
                    except Exception as e:
                        print(f"Failed to add font {font_path}: {e}")
    
    # 使用matplotlib的字体后备机制
    plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'WenQuanYi Micro Hei', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 验证字体是否可用
    available_fonts = fm.get_font_names()
    chinese_fonts = ['WenQuanYi Zen Hei', 'WenQuanYi Micro Hei']
    
    for font_name in chinese_fonts:
        if font_name in available_fonts:
            print(f"✓ {font_name} 字体可用")
            return font_name
        else:
            print(f"✗ {font_name} 字体不可用")
    
    return None

def load_training_data():
    """加载三个数据集的训练数据"""
    
    # 文件路径
    asas_file = "/root/autodl-tmp/lnsde+contiformer/results/logs/ASAS_linear_noise_config1_20250824_000718.json"
    linear_file = "/root/autodl-tmp/lnsde+contiformer/results/logs/LINEAR_linear_noise_config1_20250824_140843.log"
    macho_file = "/root/autodl-tmp/lnsde+contiformer/results/logs/MACHO_linear_noise_config1_20250824_135530.log"
    
    datasets = {}
    
    # 加载ASAS数据
    try:
        with open(asas_file, 'r') as f:
            asas_data = json.load(f)
            datasets['ASAS'] = {
                'epochs': asas_data['training_history']['epochs'],
                'train_loss': asas_data['training_history']['train_loss'],
                'train_accuracy': asas_data['training_history']['train_accuracy'],
                'val_loss': asas_data['training_history']['val_loss'],
                'val_accuracy': asas_data['training_history']['val_accuracy']
            }
        print("✓ ASAS数据加载成功")
    except Exception as e:
        print(f"✗ 加载ASAS数据失败: {e}")
    
    # 加载LINEAR数据
    try:
        with open(linear_file, 'r') as f:
            linear_data = json.load(f)
            datasets['LINEAR'] = {
                'epochs': linear_data['training_history']['epochs'],
                'train_loss': linear_data['training_history']['train_loss'],
                'train_accuracy': linear_data['training_history']['train_accuracy'],
                'val_loss': linear_data['training_history']['val_loss'],
                'val_accuracy': linear_data['training_history']['val_accuracy']
            }
        print("✓ LINEAR数据加载成功")
    except Exception as e:
        print(f"✗ 加载LINEAR数据失败: {e}")
    
    # 加载MACHO数据
    try:
        with open(macho_file, 'r') as f:
            macho_data = json.load(f)
            datasets['MACHO'] = {
                'epochs': macho_data['training_history']['epochs'],
                'train_loss': macho_data['training_history']['train_loss'],
                'train_accuracy': macho_data['training_history']['train_accuracy'],
                'val_loss': macho_data['training_history']['val_loss'],
                'val_accuracy': macho_data['training_history']['val_accuracy']
            }
        print("✓ MACHO数据加载成功")
    except Exception as e:
        print(f"✗ 加载MACHO数据失败: {e}")
    
    return datasets

def create_training_curves(datasets, save_path):
    """创建训练曲线图"""
    
    # 科学配色方案
    colors = ['#2E86AB', '#A23B72', '#F18F01']  # 深蓝、紫红、橙色
    dataset_names = list(datasets.keys())
    
    # 创建2x2子图布局
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 图1: 训练损失
    ax1.set_title('训练损失曲线', fontweight='bold', fontsize=16, pad=20)
    ax1.set_xlabel('训练轮次 (Epoch)', fontweight='bold', fontsize=12)
    ax1.set_ylabel('训练损失', fontweight='bold', fontsize=12)
    
    for i, (dataset_name, data) in enumerate(datasets.items()):
        ax1.plot(data['epochs'], data['train_loss'], 
                label=f'{dataset_name}', color=colors[i], 
                linewidth=2.5, marker='o', markersize=4, alpha=0.8)
    
    ax1.legend(fontsize=11, loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, None)
    
    # 图2: 验证损失
    ax2.set_title('验证损失曲线', fontweight='bold', fontsize=16, pad=20)
    ax2.set_xlabel('训练轮次 (Epoch)', fontweight='bold', fontsize=12)
    ax2.set_ylabel('验证损失', fontweight='bold', fontsize=12)
    
    for i, (dataset_name, data) in enumerate(datasets.items()):
        ax2.plot(data['epochs'], data['val_loss'], 
                label=f'{dataset_name}', color=colors[i], 
                linewidth=2.5, marker='s', markersize=4, alpha=0.8)
    
    ax2.legend(fontsize=11, loc='best')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, None)
    
    # 图3: 训练准确率
    ax3.set_title('训练准确率曲线', fontweight='bold', fontsize=16, pad=20)
    ax3.set_xlabel('训练轮次 (Epoch)', fontweight='bold', fontsize=12)
    ax3.set_ylabel('训练准确率 (%)', fontweight='bold', fontsize=12)
    
    for i, (dataset_name, data) in enumerate(datasets.items()):
        ax3.plot(data['epochs'], data['train_accuracy'], 
                label=f'{dataset_name}', color=colors[i], 
                linewidth=2.5, marker='^', markersize=4, alpha=0.8)
    
    ax3.legend(fontsize=11, loc='best')
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 100)
    
    # 图4: 验证准确率
    ax4.set_title('验证准确率曲线', fontweight='bold', fontsize=16, pad=20)
    ax4.set_xlabel('训练轮次 (Epoch)', fontweight='bold', fontsize=12)
    ax4.set_ylabel('验证准确率 (%)', fontweight='bold', fontsize=12)
    
    for i, (dataset_name, data) in enumerate(datasets.items()):
        ax4.plot(data['epochs'], data['val_accuracy'], 
                label=f'{dataset_name}', color=colors[i], 
                linewidth=2.5, marker='D', markersize=4, alpha=0.8)
    
    ax4.legend(fontsize=11, loc='best')
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(0, 100)
    
    # 调整布局
    plt.tight_layout(pad=3.0)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"训练曲线图已保存至: {save_path}")

def print_training_summary(datasets):
    """打印训练摘要"""
    print("\n" + "="*80)
    print("训练过程摘要")
    print("="*80)
    print(f"{'数据集':<10} {'总轮次':<8} {'最终训练准确率(%)':<16} {'最终验证准确率(%)':<16} {'最低验证损失':<12}")
    print("-"*80)
    
    for dataset_name, data in datasets.items():
        final_train_acc = data['train_accuracy'][-1] if data['train_accuracy'] else 0
        final_val_acc = data['val_accuracy'][-1] if data['val_accuracy'] else 0
        min_val_loss = min(data['val_loss']) if data['val_loss'] else 0
        total_epochs = len(data['epochs'])
        
        print(f"{dataset_name:<10} {total_epochs:<8} {final_train_acc:<16.2f} {final_val_acc:<16.2f} {min_val_loss:<12.4f}")
    
    print("-"*80)
    print("说明:")
    print("• 训练轮次显示每个数据集的训练过程长度")
    print("• 最终准确率显示训练结束时的模型性能")
    print("• 最低验证损失显示训练过程中的最佳损失值")
    print("="*80)

def main():
    print("配置中文字体...")
    working_font = configure_chinese_fonts()
    
    if working_font:
        print(f"✅ 成功配置中文字体: {working_font}")
    else:
        print("⚠️ 中文字体配置可能有问题，但会尝试继续...")
    
    print("\n加载训练数据...")
    datasets = load_training_data()
    
    if not datasets:
        print("❌ 没有找到可用的训练数据")
        return
    
    print(f"✅ 成功加载 {len(datasets)} 个数据集的训练数据")
    
    # 创建输出目录
    output_dir = '/root/autodl-tmp/lnsde+contiformer/results/pics'
    os.makedirs(output_dir, exist_ok=True)
    
    # 生成训练曲线图
    plot_path = os.path.join(output_dir, 'training_curves_analysis.png')
    create_training_curves(datasets, plot_path)
    
    # 打印训练摘要
    print_training_summary(datasets)

if __name__ == "__main__":
    main()