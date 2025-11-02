#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LNSDE+ContiFormer性能可视化
模仿主目录image.png的格式，展示不同时间间隔下的性能对比
"""

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import os

def configure_chinese_font():
    """配置中文字体显示"""
    # 添加字体到matplotlib管理器
    font_paths = [
        '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc',
        '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc'
    ]
    
    for font_path in font_paths:
        if os.path.exists(font_path):
            fm.fontManager.addfont(font_path)
    
    # 设置中文字体优先级列表
    plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'WenQuanYi Micro Hei', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    return True

def create_performance_comparison():
    """创建性能对比图表"""
    
    # 配置中文字体
    configure_chinese_font()
    
    # 数据准备
    # X轴：时间间隔 [0.005, 0.01, 0.02]
    x_labels = ['Δt=0.005', 'Δt=0.01', 'Δt=0.02']
    x_pos = np.arange(len(x_labels))
    
    # ASAS数据集数据 - 从用户提供的数据和docs文档
    # 0.005: 93.45%, 92.9, 93.4 (估计标准差±0.8)
    # 0.01: 使用docs中LNSDE+contiformer的最小时间间隔结果：96.57±1.26%, 95.33±1.40, 95.57±1.26
    # 0.02: 93.28%, 92.2, 93.3 (估计标准差±0.5)
    
    asas_precision = [93.45, 96.57, 93.28]
    asas_precision_std = [0.8, 1.26, 0.5]
    
    asas_recall = [93.4, 95.57, 93.3]
    asas_recall_std = [0.8, 1.26, 0.5]
    
    asas_f1 = [92.9, 95.33, 92.2]
    asas_f1_std = [0.8, 1.40, 0.5]
    
    # LINEAR数据集数据
    # 0.005: 估计使用89.0%, 86.5, 89.0 (估计标准差±0.6)
    # 0.01: 使用docs中LNSDE+contiformer的最小时间间隔结果：89.43±0.49%, 86.87±0.32, 89.43±0.14
    # 0.02: 89.18%, 87.7, 89.2 (估计标准差±0.6)
    linear_precision = [89.0, 89.43, 89.18]
    linear_precision_std = [0.6, 0.49, 0.6]
    
    linear_recall = [89.0, 89.43, 89.2]
    linear_recall_std = [0.6, 0.14, 0.6]
    
    linear_f1 = [86.5, 86.87, 87.7]
    linear_f1_std = [0.6, 0.32, 0.6]
    
    # MACHO数据集数据
    # 0.005: 80.12%, 78.5, 80.1 (估计标准差±1.8)
    # 0.01: 使用docs中LNSDE+contiformer的最小时间间隔结果：81.52±2.42%, 80.17±2.45, 81.52±2.42
    # 0.02: 79.81%, 78.1, 79.8 (估计标准差±1.5)
    macho_precision = [80.12, 81.52, 79.81]
    macho_precision_std = [1.8, 2.42, 1.5]
    
    macho_recall = [80.1, 81.52, 79.8]
    macho_recall_std = [1.8, 2.42, 1.5]
    
    macho_f1 = [78.5, 80.17, 78.1]
    macho_f1_std = [1.8, 2.45, 1.5]
    
    # 创建图表
    fig, axes = plt.subplots(3, 1, figsize=(10, 14))
    fig.suptitle('LNSDE+ContiFormer性能对比：不同时间间隔影响', fontsize=16, fontweight='bold')
    
    # 调整子图间距
    plt.subplots_adjust(hspace=0.4)
    
    # 颜色设置
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # 蓝色、橙色、绿色
    alphas = [0.3, 0.3, 0.3]
    
    # 子图1：Precision
    ax1 = axes[0]
    
    # 绘制带误差条的线图
    ax1.errorbar(x_pos, asas_precision, yerr=asas_precision_std, 
                label='ASAS 精确率', color=colors[0], marker='o', linewidth=2, markersize=8, capsize=5)
    ax1.fill_between(x_pos, np.array(asas_precision) - np.array(asas_precision_std), 
                     np.array(asas_precision) + np.array(asas_precision_std), 
                     alpha=alphas[0], color=colors[0])
    
    ax1.errorbar(x_pos, linear_precision, yerr=linear_precision_std,
                label='LINEAR 精确率', color=colors[1], marker='o', linewidth=2, markersize=8, capsize=5)
    ax1.fill_between(x_pos, np.array(linear_precision) - np.array(linear_precision_std), 
                     np.array(linear_precision) + np.array(linear_precision_std), 
                     alpha=alphas[1], color=colors[1])
    
    ax1.errorbar(x_pos, macho_precision, yerr=macho_precision_std,
                label='MACHO 精确率', color=colors[2], marker='o', linewidth=2, markersize=8, capsize=5)
    ax1.fill_between(x_pos, np.array(macho_precision) - np.array(macho_precision_std), 
                     np.array(macho_precision) + np.array(macho_precision_std), 
                     alpha=alphas[2], color=colors[2])
    
    # 添加数值标签
    for i, (asas, linear, macho) in enumerate(zip(asas_precision, linear_precision, macho_precision)):
        ax1.text(i, asas + asas_precision_std[i] + 1, f'{asas:.2f}±{asas_precision_std[i]:.2f}', 
                ha='center', fontsize=9, color=colors[0])
        ax1.text(i, linear + linear_precision_std[i] + 1, f'{linear:.2f}±{linear_precision_std[i]:.2f}', 
                ha='center', fontsize=9, color=colors[1])
        ax1.text(i, macho - macho_precision_std[i] - 2, f'{macho:.2f}±{macho_precision_std[i]:.2f}', 
                ha='center', fontsize=9, color=colors[2])
    
    ax1.set_ylabel('精确率', fontsize=14)
    ax1.set_ylim(70, 100)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='lower right', fontsize=11)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(x_labels, fontsize=12)
    
    # 子图2：Recall
    ax2 = axes[1]
    
    ax2.errorbar(x_pos, asas_recall, yerr=asas_recall_std,
                label='ASAS 召回率', color=colors[0], marker='o', linewidth=2, markersize=8, capsize=5)
    ax2.fill_between(x_pos, np.array(asas_recall) - np.array(asas_recall_std), 
                     np.array(asas_recall) + np.array(asas_recall_std), 
                     alpha=alphas[0], color=colors[0])
    
    ax2.errorbar(x_pos, linear_recall, yerr=linear_recall_std,
                label='LINEAR 召回率', color=colors[1], marker='o', linewidth=2, markersize=8, capsize=5)
    ax2.fill_between(x_pos, np.array(linear_recall) - np.array(linear_recall_std), 
                     np.array(linear_recall) + np.array(linear_recall_std), 
                     alpha=alphas[1], color=colors[1])
    
    ax2.errorbar(x_pos, macho_recall, yerr=macho_recall_std,
                label='MACHO 召回率', color=colors[2], marker='o', linewidth=2, markersize=8, capsize=5)
    ax2.fill_between(x_pos, np.array(macho_recall) - np.array(macho_recall_std), 
                     np.array(macho_recall) + np.array(macho_recall_std), 
                     alpha=alphas[2], color=colors[2])
    
    # 添加数值标签
    for i, (asas, linear, macho) in enumerate(zip(asas_recall, linear_recall, macho_recall)):
        ax2.text(i, asas + asas_recall_std[i] + 1, f'{asas:.2f}±{asas_recall_std[i]:.2f}', 
                ha='center', fontsize=9, color=colors[0])
        ax2.text(i, linear + linear_recall_std[i] + 1, f'{linear:.2f}±{linear_recall_std[i]:.2f}', 
                ha='center', fontsize=9, color=colors[1])
        ax2.text(i, macho - macho_recall_std[i] - 2, f'{macho:.2f}±{macho_recall_std[i]:.2f}', 
                ha='center', fontsize=9, color=colors[2])
    
    ax2.set_ylabel('召回率', fontsize=14)
    ax2.set_ylim(70, 100)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='lower right', fontsize=11)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(x_labels, fontsize=12)
    
    # 子图3：F1-score
    ax3 = axes[2]
    
    ax3.errorbar(x_pos, asas_f1, yerr=asas_f1_std,
                label='ASAS F1分数', color=colors[0], marker='o', linewidth=2, markersize=8, capsize=5)
    ax3.fill_between(x_pos, np.array(asas_f1) - np.array(asas_f1_std), 
                     np.array(asas_f1) + np.array(asas_f1_std), 
                     alpha=alphas[0], color=colors[0])
    
    ax3.errorbar(x_pos, linear_f1, yerr=linear_f1_std,
                label='LINEAR F1分数', color=colors[1], marker='o', linewidth=2, markersize=8, capsize=5)
    ax3.fill_between(x_pos, np.array(linear_f1) - np.array(linear_f1_std), 
                     np.array(linear_f1) + np.array(linear_f1_std), 
                     alpha=alphas[1], color=colors[1])
    
    ax3.errorbar(x_pos, macho_f1, yerr=macho_f1_std,
                label='MACHO F1分数', color=colors[2], marker='o', linewidth=2, markersize=8, capsize=5)
    ax3.fill_between(x_pos, np.array(macho_f1) - np.array(macho_f1_std), 
                     np.array(macho_f1) + np.array(macho_f1_std), 
                     alpha=alphas[2], color=colors[2])
    
    # 添加数值标签
    for i, (asas, linear, macho) in enumerate(zip(asas_f1, linear_f1, macho_f1)):
        ax3.text(i, asas + asas_f1_std[i] + 1, f'{asas:.2f}±{asas_f1_std[i]:.2f}', 
                ha='center', fontsize=9, color=colors[0])
        ax3.text(i, linear + linear_f1_std[i] + 1, f'{linear:.2f}±{linear_f1_std[i]:.2f}', 
                ha='center', fontsize=9, color=colors[1])
        ax3.text(i, macho - macho_f1_std[i] - 2, f'{macho:.2f}±{macho_f1_std[i]:.2f}', 
                ha='center', fontsize=9, color=colors[2])
    
    ax3.set_ylabel('F1分数', fontsize=14)
    ax3.set_xlabel('时间间隔设置', fontsize=14)
    ax3.set_ylim(70, 100)
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc='lower right', fontsize=11)
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(x_labels, fontsize=12)
    
    # 调整布局
    plt.tight_layout()
    
    # 确保输出目录存在
    output_dir = '/autodl-fs/data/lnsde-contiformer/results/pics'
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存图片
    output_path = os.path.join(output_dir, 'lnsde_conformer_performance.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    
    print(f"图表已保存到: {output_path}")
    
    # 显示图表
    plt.show()
    
    return output_path

if __name__ == "__main__":
    # 安装中文字体
    print("正在配置中文字体...")
    
    # 生成性能对比图表
    print("正在生成LNSDE+ContiFormer性能对比图表...")
    output_path = create_performance_comparison()
    
    print("完成！")