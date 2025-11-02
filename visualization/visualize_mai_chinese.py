#!/usr/bin/env python3
"""
使用Seaborn创建中文版准确率和MAI的双Y轴柱状图
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
import pandas as pd
import pickle
import os
import shutil


def clean_font_cache():
    """清理matplotlib字体缓存"""
    cache_dir = os.path.expanduser('~/.cache/matplotlib')
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
        print("✅ 字体缓存已清理")


def configure_chinese_font():
    """配置中文字体"""
    # 清理缓存
    clean_font_cache()
    
    # 添加中文字体
    try:
        fm.fontManager.addfont('/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc')
        fm.fontManager.addfont('/usr/share/fonts/truetype/wqy/wqy-microhei.ttc')
    except:
        pass
    
    # 设置seaborn风格
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.5)
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'WenQuanYi Micro Hei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['savefig.dpi'] = 300


def load_data():
    """加载数据并计算指标"""
    
    results = []
    
    # ASAS
    with open('/root/autodl-fs/lnsde-contiformer/data/ASAS_fixed.pkl', 'rb') as f:
        data = pickle.load(f)
    labels = [s['label'] for s in data]
    unique_labels, counts = np.unique(labels, return_counts=True)
    
    K = len(unique_labels)
    n = counts.sum()
    ni_n = counts / n
    ID = 0
    for p in ni_n:
        ID += p * min(1, K * p)
    ID = 1 - ((K-1)/K) * ID
    DBC = 1.0
    MAI = ID * DBC
    
    results.append({
        '数据集': 'ASAS',
        '准确率': 96.57,
        'MAI': MAI,
        '类别数': K,
        'IR': counts.max() / counts.min()
    })
    
    # LINEAR
    with open('/root/autodl-fs/lnsde-contiformer/data/LINEAR_fixed.pkl', 'rb') as f:
        data = pickle.load(f)
    labels = [s['label'] for s in data]
    unique_labels, counts = np.unique(labels, return_counts=True)
    
    K = len(unique_labels)
    n = counts.sum()
    ni_n = counts / n
    ID = 0
    for p in ni_n:
        ID += p * min(1, K * p)
    ID = 1 - ((K-1)/K) * ID
    DBC = 1.0
    MAI = ID * DBC
    
    results.append({
        '数据集': 'LINEAR',
        '准确率': 89.43,
        'MAI': MAI,
        '类别数': K,
        'IR': counts.max() / counts.min()
    })
    
    # MACHO
    with open('/root/autodl-fs/lnsde-contiformer/data/MACHO_fixed.pkl', 'rb') as f:
        data = pickle.load(f)
    labels = [s['label'] for s in data]
    unique_labels, counts = np.unique(labels, return_counts=True)
    
    K = len(unique_labels)
    n = counts.sum()
    ni_n = counts / n
    ID = 0
    for p in ni_n:
        ID += p * min(1, K * p)
    ID = 1 - ((K-1)/K) * ID
    DBC = (7 * 6) / (5 * 4)
    MAI = ID * DBC
    
    results.append({
        '数据集': 'MACHO',
        '准确率': 81.52,
        'MAI': MAI,
        '类别数': K,
        'IR': counts.max() / counts.min()
    })
    
    return pd.DataFrame(results)


def create_visualization():
    """创建双Y轴柱状图（中文版）"""
    
    # 配置中文字体
    configure_chinese_font()
    
    # 加载数据
    df = load_data()
    
    # 创建图形
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # 设置颜色
    color_acc = '#2E86AB'  # 蓝色
    color_mai = '#A23B72'  # 紫红色
    
    # X轴位置
    x = np.arange(len(df))
    width = 0.35
    
    # 左Y轴 - 准确率
    bar1 = ax1.bar(x - width/2, df['准确率'], width, 
                   color=color_acc, alpha=0.8, 
                   edgecolor='white', linewidth=2)
    
    ax1.set_xlabel('数据集', fontsize=14, fontweight='bold')
    ax1.set_ylabel('准确率 (%)', fontsize=14, color=color_acc, fontweight='bold')
    ax1.set_ylim([75, 100])
    ax1.tick_params(axis='y', labelcolor=color_acc, labelsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(df['数据集'], fontsize=13)
    
    # 添加准确率数值标签
    for i, (rect, val) in enumerate(zip(bar1, df['准确率'])):
        height = rect.get_height()
        ax1.text(rect.get_x() + rect.get_width()/2., height + 0.5,
                f'{val:.1f}%', ha='center', va='bottom',
                fontsize=11, fontweight='bold', color=color_acc)
    
    # 右Y轴 - MAI
    ax2 = ax1.twinx()
    bar2 = ax2.bar(x + width/2, df['MAI'], width,
                   color=color_mai, alpha=0.8,
                   edgecolor='white', linewidth=2)
    
    ax2.set_ylabel('MAI (多类调整不均衡度)', fontsize=14, 
                   color=color_mai, fontweight='bold')
    ax2.set_ylim([0, 0.6])
    ax2.tick_params(axis='y', labelcolor=color_mai, labelsize=12)
    
    # 添加MAI数值标签和类别数
    for i, (rect, val, cls) in enumerate(zip(bar2, df['MAI'], df['类别数'])):
        height = rect.get_height()
        ax2.text(rect.get_x() + rect.get_width()/2., height + 0.01,
                f'{val:.3f}\n({cls}类)', ha='center', va='bottom',
                fontsize=10, color=color_mai, fontweight='bold')
    
    # 设置标题
    ax1.set_title('准确率 vs MAI (多类调整不均衡度)\nLNSDE+ContiFormer 性能分析', 
                  fontsize=16, fontweight='bold', pad=20)
    
    # 设置网格
    ax1.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)
    ax1.set_axisbelow(True)
    
    # 不添加任何图例或说明文本
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图片
    output_dir = "/root/autodl-tmp/lnsde-contiformer/results/pics/analysis"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "accuracy_mai_chinese.png")
    
    plt.savefig(output_path, bbox_inches='tight', facecolor='white')
    print(f"✅ 图片已保存: {output_path}")
    
    # 打印数据摘要
    print("\n" + "="*70)
    print("📊 数据摘要")
    print("="*70)
    print(df.to_string(index=False))
    
    print("\n" + "="*70)
    print("📈 关键发现")
    print("="*70)
    
    # 计算相关系数
    corr = df['MAI'].corr(df['准确率'])
    print(f"• 相关系数: {corr:.3f}")
    print(f"• 最高MAI: {df.loc[df['MAI'].idxmax(), '数据集']} ({df['MAI'].max():.3f})")
    print(f"• 最低准确率: {df.loc[df['准确率'].idxmin(), '数据集']} ({df['准确率'].min():.1f}%)")
    print(f"• MACHO有{df[df['数据集']=='MACHO']['类别数'].values[0]}个类别，"
          f"导致MAI达到{df[df['数据集']=='MACHO']['MAI'].values[0]:.3f}")
    print(f"• 负相关性表明：类别不均衡度越高，模型性能越低")
    
    plt.show()


if __name__ == "__main__":
    create_visualization()