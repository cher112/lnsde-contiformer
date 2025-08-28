#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据可视化脚本：显示每个数据集中每个类别的time-mag光变曲线样本
只显示真实数据点，不包含padding部分
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from collections import defaultdict
import os

# 正确配置中文字体 - 基于context7 MCP的建议
def configure_chinese_font():
    """配置中文字体显示"""
    # 重新扫描字体目录
    fm.fontManager.addfont('/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc')
    fm.fontManager.addfont('/usr/share/fonts/truetype/wqy/wqy-microhei.ttc')
    
    # 设置中文字体优先级列表
    plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'WenQuanYi Micro Hei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 验证字体是否可用
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    
    for font in ['WenQuanYi Zen Hei', 'WenQuanYi Micro Hei']:
        if font in available_fonts:
            print(f"✓ 中文字体 {font} 已加载")
            return True
    
    print("⚠ 中文字体加载失败，使用默认字体")
    return False

# 配置中文字体
configure_chinese_font()

def load_and_visualize_datasets():
    """加载并可视化数据集"""
    datasets = {
        'ASAS_folded_512.pkl': 'ASAS数据集',
        'LINEAR_folded_512.pkl': 'LINEAR数据集', 
        'MACHO_folded_512.pkl': 'MACHO数据集'
    }
    
    # 基础输出目录
    base_output_dir = '/root/autodl-tmp/lnsde-contiformer/results/pics'
    
    for filename, title in datasets.items():
        print(f"处理 {title}...")
        
        # 创建对应数据集的输出目录
        dataset_name = filename.replace('_folded_512.pkl', '')
        output_dir = os.path.join(base_output_dir, dataset_name)
        os.makedirs(output_dir, exist_ok=True)
        
        # 加载数据
        with open(f'data/{filename}', 'rb') as f:
            data = pickle.load(f)
        
        # 按类别分组样本
        class_samples = defaultdict(list)
        class_names = {}
        
        for sample in data:
            label = sample['label']
            class_samples[label].append(sample)
            class_names[label] = sample['class_name']
        
        # 为每个类别选择一个代表性样本（选择valid_points最多的）
        selected_samples = {}
        for label, samples in class_samples.items():
            best_sample = max(samples, key=lambda x: x['valid_points'])
            selected_samples[label] = best_sample
        
        # 创建子图
        n_classes = len(selected_samples)
        cols = min(3, n_classes)
        rows = (n_classes + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
        fig.suptitle(f'{title} - 各类别光变曲线样本', fontsize=16, fontweight='bold')
        
        if n_classes == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes if n_classes > 1 else [axes]
        else:
            axes = axes.flatten()
        
        # 绘制每个类别的样本
        for i, (label, sample) in enumerate(sorted(selected_samples.items())):
            ax = axes[i]
            
            # 获取有效数据点（mask为1的点）
            mask = sample['mask'].astype(bool)
            time = sample['time'][mask]
            mag = sample['mag'][mask]
            errmag = sample['errmag'][mask]
            
            # 绘制光变曲线
            ax.errorbar(time, mag, yerr=errmag, fmt='o-', markersize=3, linewidth=1, 
                       capsize=2, alpha=0.8, label=f'类别 {label}')
            
            # 设置标题和标签
            class_name = class_names[label]
            ax.set_title(f'类别 {label}: {class_name}\n'
                        f'周期: {sample["period"]:.4f}天, '
                        f'有效点数: {sample["valid_points"]}', 
                        fontsize=10)
            ax.set_xlabel('相位时间')
            ax.set_ylabel('星等')
            ax.grid(True, alpha=0.3)
            ax.invert_yaxis()  # 星等轴倒置
            
            # 添加统计信息
            ax.text(0.02, 0.98, f'样本数: {len(class_samples[label])}', 
                   transform=ax.transAxes, fontsize=8, 
                   verticalalignment='top', bbox=dict(boxstyle='round', 
                   facecolor='wheat', alpha=0.5))
        
        # 隐藏多余的子图
        for i in range(n_classes, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        # 保存图片
        output_file = os.path.join(output_dir, 'lc.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f'保存图片: {output_file}')
        
        plt.show()
        print(f'{title} 完成！\n')

if __name__ == '__main__':
    print("开始数据可视化...")
    load_and_visualize_datasets()
    print("所有可视化完成！")