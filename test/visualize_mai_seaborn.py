#!/usr/bin/env python3
"""
使用Seaborn创建准确率和MAI的双Y轴柱状图
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pickle
import os


def configure_plotting_style():
    """配置绘图风格"""
    # 设置seaborn风格
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.5)
    
    # 设置matplotlib参数
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False


def load_data():
    """加载数据并计算指标"""
    
    results = []
    
    # ASAS
    with open('/root/autodl-tmp/lnsde-contiformer/data/ASAS_folded_512_fixed.pkl', 'rb') as f:
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
        'Dataset': 'ASAS',
        'Accuracy': 96.57,
        'MAI': MAI,
        'Classes': K,
        'IR': counts.max() / counts.min()
    })
    
    # LINEAR
    with open('/root/autodl-tmp/lnsde-contiformer/data/LINEAR_folded_512_fixed.pkl', 'rb') as f:
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
        'Dataset': 'LINEAR',
        'Accuracy': 89.43,
        'MAI': MAI,
        'Classes': K,
        'IR': counts.max() / counts.min()
    })
    
    # MACHO
    with open('/root/autodl-tmp/lnsde-contiformer/data/MACHO_folded_512_fixed.pkl', 'rb') as f:
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
        'Dataset': 'MACHO',
        'Accuracy': 81.52,
        'MAI': MAI,
        'Classes': K,
        'IR': counts.max() / counts.min()
    })
    
    return pd.DataFrame(results)


def create_double_bar_plot():
    """创建双Y轴柱状图"""
    
    # 配置风格
    configure_plotting_style()
    
    # 加载数据
    df = load_data()
    
    # 创建图形
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # 设置颜色
    color_acc = '#2E86AB'  # 蓝色系
    color_mai = '#A23B72'  # 紫红色系
    
    # X轴位置
    x = np.arange(len(df))
    width = 0.35
    
    # 左Y轴 - 准确率
    bar1 = ax1.bar(x - width/2, df['Accuracy'], width, 
                   color=color_acc, alpha=0.8, 
                   edgecolor='white', linewidth=2,
                   label='Accuracy (%)')
    
    ax1.set_xlabel('Dataset', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Accuracy (%)', fontsize=14, color=color_acc, fontweight='bold')
    ax1.set_ylim([75, 100])
    ax1.tick_params(axis='y', labelcolor=color_acc, labelsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(df['Dataset'], fontsize=13)
    
    # 添加准确率数值标签
    for i, (rect, val) in enumerate(zip(bar1, df['Accuracy'])):
        height = rect.get_height()
        ax1.text(rect.get_x() + rect.get_width()/2., height + 0.5,
                f'{val:.1f}%', ha='center', va='bottom',
                fontsize=11, fontweight='bold', color=color_acc)
    
    # 右Y轴 - MAI
    ax2 = ax1.twinx()
    bar2 = ax2.bar(x + width/2, df['MAI'], width,
                   color=color_mai, alpha=0.8,
                   edgecolor='white', linewidth=2,
                   label='MAI')
    
    ax2.set_ylabel('MAI (Multi-class Adjusted Imbalance)', fontsize=14, 
                   color=color_mai, fontweight='bold')
    ax2.set_ylim([0, 0.6])
    ax2.tick_params(axis='y', labelcolor=color_mai, labelsize=12)
    
    # 添加MAI数值标签和类别数
    for i, (rect, val, cls) in enumerate(zip(bar2, df['MAI'], df['Classes'])):
        height = rect.get_height()
        ax2.text(rect.get_x() + rect.get_width()/2., height + 0.01,
                f'{val:.3f}\n({cls} classes)', ha='center', va='bottom',
                fontsize=10, color=color_mai, fontweight='bold')
    
    # 设置标题
    ax1.set_title('Accuracy vs MAI Analysis\nLNSDE+ContiFormer Performance', 
                  fontsize=16, fontweight='bold', pad=20)
    
    # 设置网格
    ax1.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)
    ax1.set_axisbelow(True)
    
    # 添加图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, 
               loc='upper right', frameon=True, 
               fancybox=True, shadow=True, fontsize=11)
    
    # 添加说明文本
    info_text = (
        "MAI = ID × DBC\n"
        "DBC (5-class) = 1.0\n"
        "DBC (7-class) = 2.1"
    )
    
    ax1.text(0.02, 0.15, info_text, transform=ax1.transAxes,
             fontsize=10, verticalalignment='bottom',
             bbox=dict(boxstyle='round,pad=0.5', 
                      facecolor='white', 
                      edgecolor='gray',
                      alpha=0.9))
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图片
    output_dir = "/root/autodl-tmp/lnsde-contiformer/results/pics/analysis"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "accuracy_mai_seaborn.png")
    
    plt.savefig(output_path, bbox_inches='tight', facecolor='white')
    print(f"✅ Figure saved: {output_path}")
    
    plt.show()
    
    return df


def create_correlation_plot(df):
    """创建相关性散点图"""
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # 散点图
    scatter = ax.scatter(df['MAI'], df['Accuracy'], 
                        s=200, alpha=0.7,
                        c=df['Classes'], cmap='coolwarm',
                        edgecolors='black', linewidth=2)
    
    # 添加数据集标签
    for idx, row in df.iterrows():
        ax.annotate(f"{row['Dataset']}\n({row['Classes']} classes)",
                   (row['MAI'], row['Accuracy']),
                   xytext=(10, -5), textcoords='offset points',
                   fontsize=10, fontweight='bold')
    
    # 添加回归线
    z = np.polyfit(df['MAI'], df['Accuracy'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(df['MAI'].min() - 0.05, df['MAI'].max() + 0.05, 100)
    ax.plot(x_line, p(x_line), 'r--', alpha=0.5, linewidth=2,
            label=f'y = {z[0]:.1f}x + {z[1]:.1f}')
    
    # 设置标签和标题
    ax.set_xlabel('MAI (Multi-class Adjusted Imbalance)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Correlation between MAI and Accuracy', fontsize=14, fontweight='bold')
    
    # 添加颜色条
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Number of Classes', fontsize=11)
    
    # 添加相关系数
    corr = df['MAI'].corr(df['Accuracy'])
    ax.text(0.05, 0.95, f'Correlation: {corr:.3f}',
            transform=ax.transAxes, fontsize=11,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.legend(loc='lower left', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存
    output_path = "/root/autodl-tmp/lnsde-contiformer/results/pics/analysis/mai_correlation.png"
    plt.savefig(output_path, bbox_inches='tight', facecolor='white')
    print(f"✅ Correlation plot saved: {output_path}")
    
    plt.show()


def print_summary(df):
    """打印数据摘要"""
    
    print("\n" + "="*70)
    print("📊 Data Summary")
    print("="*70)
    print(df.to_string(index=False))
    
    print("\n" + "="*70)
    print("📈 Key Findings")
    print("="*70)
    
    # 计算相关系数
    corr = df['MAI'].corr(df['Accuracy'])
    print(f"• Correlation coefficient: {corr:.3f}")
    print(f"• Highest MAI: {df.loc[df['MAI'].idxmax(), 'Dataset']} ({df['MAI'].max():.3f})")
    print(f"• Lowest Accuracy: {df.loc[df['Accuracy'].idxmin(), 'Dataset']} ({df['Accuracy'].min():.1f}%)")
    print(f"• MACHO has {df[df['Dataset']=='MACHO']['Classes'].values[0]} classes, "
          f"resulting in {df[df['Dataset']=='MACHO']['MAI'].values[0]:.3f} MAI")


if __name__ == "__main__":
    # 创建主要的双柱状图
    df = create_double_bar_plot()
    
    # 创建相关性散点图
    create_correlation_plot(df)
    
    # 打印摘要
    print_summary(df)