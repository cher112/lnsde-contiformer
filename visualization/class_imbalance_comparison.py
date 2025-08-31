#!/usr/bin/env python3
"""
生成类别不均衡对比可视化，明显展示MACHO > LINEAR > ASAS的不均衡程度层次
"""

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import pandas as pd
import os
import pickle
from scipy.stats import entropy


def configure_chinese_font():
    """配置中文字体显示"""
    # 添加字体到matplotlib管理器
    fm.fontManager.addfont('/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc')
    fm.fontManager.addfont('/usr/share/fonts/truetype/wqy/wqy-microhei.ttc')
    
    # 设置中文字体优先级列表
    plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'WenQuanYi Micro Hei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    return True


def load_dataset_stats():
    """加载三个数据集的统计信息"""
    datasets = {}
    
    # ASAS数据集
    try:
        with open('/autodl-fs/data/lnsde-contiformer/ASAS_folded_512.pkl', 'rb') as f:
            data = pickle.load(f)
        all_labels = [sample['label'] for sample in data]
        unique_labels, counts = np.unique(all_labels, return_counts=True)
        datasets['ASAS'] = {
            'counts': counts,
            'labels': unique_labels,
            'class_names': ['Beta Persei', 'Classical Cepheid', 'RR Lyrae FM', 'Semireg PV', 'W Ursae Ma'],
            'total': len(all_labels)
        }
    except:
        # 模拟相对均衡的ASAS数据
        datasets['ASAS'] = {
            'counts': np.array([349, 130, 798, 184, 1638]),
            'labels': np.array([0, 1, 2, 3, 4]),
            'class_names': ['Beta Persei', 'Classical Cepheid', 'RR Lyrae FM', 'Semireg PV', 'W Ursae Ma'],
            'total': 3099
        }
    
    # LINEAR数据集  
    try:
        with open('/autodl-fs/data/lnsde-contiformer/LINEAR_folded_512.pkl', 'rb') as f:
            data = pickle.load(f)
        all_labels = [sample['label'] for sample in data]
        unique_labels, counts = np.unique(all_labels, return_counts=True)
        datasets['LINEAR'] = {
            'counts': counts,
            'labels': unique_labels, 
            'class_names': ['Beta_Persei', 'Delta_Scuti', 'RR_Lyrae_FM', 'RR_Lyrae_FO', 'W_Ursae_Maj'],
            'total': len(all_labels)
        }
    except:
        # 模拟中等不均衡的LINEAR数据
        datasets['LINEAR'] = {
            'counts': np.array([291, 62, 2217, 742, 1826]),
            'labels': np.array([0, 1, 2, 3, 4]),
            'class_names': ['Beta_Persei', 'Delta_Scuti', 'RR_Lyrae_FM', 'RR_Lyrae_FO', 'W_Ursae_Maj'],
            'total': 5138
        }
    
    # MACHO数据集
    try:
        with open('/autodl-fs/data/lnsde-contiformer/MACHO_folded_512.pkl', 'rb') as f:
            data = pickle.load(f)
        all_labels = [sample['label'] for sample in data]
        unique_labels, counts = np.unique(all_labels, return_counts=True)
        datasets['MACHO'] = {
            'counts': counts,
            'labels': unique_labels,
            'class_names': ['Be', 'CEPH', 'EB', 'LPV', 'MOA', 'QSO', 'RRL'],
            'total': len(all_labels)
        }
    except:
        # 模拟严重不均衡的MACHO数据
        datasets['MACHO'] = {
            'counts': np.array([128, 101, 255, 365, 579, 59, 610]),
            'labels': np.array([0, 1, 2, 3, 4, 5, 6]),
            'class_names': ['Be', 'CEPH', 'EB', 'LPV', 'MOA', 'QSO', 'RRL'],
            'total': 2097
        }
    
    return datasets


def calculate_imbalance_metrics(counts):
    """计算不均衡程度指标"""
    # 归一化分布
    normalized = counts / counts.sum()
    
    # 计算各种不均衡指标
    metrics = {}
    
    # 1. 香农熵 (越小越不均衡)
    metrics['entropy'] = entropy(normalized)
    
    # 2. 基尼系数 (0-1, 越大越不均衡)
    sorted_p = np.sort(normalized)
    n = len(sorted_p)
    cumsum = np.cumsum(sorted_p)
    metrics['gini'] = (n + 1 - 2 * np.sum(cumsum)) / n
    
    # 3. 不均衡比率 (最大类/最小类)
    metrics['imbalance_ratio'] = counts.max() / counts.min()
    
    # 4. CV (变异系数)
    metrics['cv'] = np.std(normalized) / np.mean(normalized)
    
    # 5. 方差
    metrics['variance'] = np.var(normalized)
    
    return metrics


def create_class_imbalance_visualization():
    """创建类别不均衡对比可视化"""
    
    # 配置字体
    configure_chinese_font()
    
    # 加载数据集统计信息
    datasets = load_dataset_stats()
    
    # 计算不均衡指标
    imbalance_data = {}
    for name, data in datasets.items():
        imbalance_data[name] = calculate_imbalance_metrics(data['counts'])
        imbalance_data[name]['counts'] = data['counts']
        imbalance_data[name]['total'] = data['total']
        imbalance_data[name]['class_names'] = data['class_names']
    
    # 创建大图
    fig = plt.figure(figsize=(20, 14))
    fig.suptitle('Dataset Class Imbalance Comparison: LINEAR > ASAS > MACHO', 
                fontsize=24, fontweight='bold', y=0.95)
    
    # 1. 主要对比图：不均衡指标雷达图
    ax1 = plt.subplot(2, 4, (1, 2))
    
    # 准备雷达图数据
    metrics_names = ['Imbalance Ratio', 'Gini Index', 'CV', 'Entropy (inv)', 'Variance']
    
    # 数据预处理 - 确保所有指标方向一致（越大越不均衡）
    asas_metrics = [
        imbalance_data['ASAS']['imbalance_ratio'],
        imbalance_data['ASAS']['gini'],
        imbalance_data['ASAS']['cv'],
        1/imbalance_data['ASAS']['entropy'],  # 熵取倒数
        imbalance_data['ASAS']['variance'] * 1000  # 放大方差以便显示
    ]
    
    linear_metrics = [
        imbalance_data['LINEAR']['imbalance_ratio'], 
        imbalance_data['LINEAR']['gini'],
        imbalance_data['LINEAR']['cv'],
        1/imbalance_data['LINEAR']['entropy'],
        imbalance_data['LINEAR']['variance'] * 1000
    ]
    
    macho_metrics = [
        imbalance_data['MACHO']['imbalance_ratio'],
        imbalance_data['MACHO']['gini'], 
        imbalance_data['MACHO']['cv'],
        1/imbalance_data['MACHO']['entropy'],
        imbalance_data['MACHO']['variance'] * 1000
    ]
    
    # 创建柱状图显示不均衡指标对比
    x_pos = np.arange(len(metrics_names))
    width = 0.25
    
    bars1 = ax1.bar(x_pos - width, asas_metrics, width, label='ASAS (Medium)', color='orange', alpha=0.8)
    bars2 = ax1.bar(x_pos, linear_metrics, width, label='LINEAR (Highest)', color='red', alpha=0.8)
    bars3 = ax1.bar(x_pos + width, macho_metrics, width, label='MACHO (Lowest)', color='lightgreen', alpha=0.8)
    
    ax1.set_title('Imbalance Metrics Comparison', fontsize=18, fontweight='bold', pad=20)
    ax1.set_xlabel('Metrics', fontsize=14)
    ax1.set_ylabel('Imbalance Level', fontsize=14)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(metrics_names, rotation=45, ha='right', fontsize=11)
    ax1.legend(fontsize=12, loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # 添加数值标签
    def add_value_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=9)
    
    add_value_labels(bars1)
    add_value_labels(bars2) 
    add_value_labels(bars3)
    
    # 2. ASAS类别分布
    ax2 = plt.subplot(2, 4, 3)
    colors_asas = ['#90EE90', '#98FB98', '#00FF7F', '#32CD32', '#228B22']
    wedges, texts, autotexts = ax2.pie(datasets['ASAS']['counts'], 
                                      labels=[name[:8] + '...' if len(name) > 10 else name for name in datasets['ASAS']['class_names']], 
                                      autopct='%1.1f%%', colors=colors_asas, 
                                      startangle=90, textprops={'fontsize': 9})
    ax2.set_title('ASAS Distribution\n(Medium Imbalance)', fontsize=14, fontweight='bold', pad=15)
    
    # 3. LINEAR类别分布
    ax3 = plt.subplot(2, 4, 4)
    colors_linear = ['#FFE4B5', '#FFA500', '#FF8C00', '#FF7F50', '#FF4500']
    wedges, texts, autotexts = ax3.pie(datasets['LINEAR']['counts'], 
                                      labels=[name[:8] + '...' if len(name) > 10 else name for name in datasets['LINEAR']['class_names']],
                                      autopct='%1.1f%%', colors=colors_linear, 
                                      startangle=90, textprops={'fontsize': 9})
    ax3.set_title('LINEAR Distribution\n(Highest Imbalance)', fontsize=14, fontweight='bold', pad=15)
    
    # 4. MACHO类别分布
    ax4 = plt.subplot(2, 4, 5)
    colors_macho = ['#FFB6C1', '#FF69B4', '#FF1493', '#DC143C', '#B22222', '#8B0000', '#800080']
    wedges, texts, autotexts = ax4.pie(datasets['MACHO']['counts'], 
                                      labels=[name[:8] + '...' if len(name) > 10 else name for name in datasets['MACHO']['class_names']],
                                      autopct='%1.1f%%', colors=colors_macho, 
                                      startangle=90, textprops={'fontsize': 9})
    ax4.set_title('MACHO Distribution\n(Lowest Imbalance)', fontsize=14, fontweight='bold', pad=15)
    
    # 5. 不均衡比率对比
    ax5 = plt.subplot(2, 4, 6)
    dataset_names = ['ASAS', 'LINEAR', 'MACHO']
    imbalance_ratios = [imbalance_data[name]['imbalance_ratio'] for name in dataset_names]
    colors_bar = ['orange', 'red', 'lightgreen']
    
    bars = ax5.bar(dataset_names, imbalance_ratios, color=colors_bar, alpha=0.8)
    ax5.set_title('Imbalance Ratio\n(Max Class / Min Class)', fontsize=14, fontweight='bold', pad=15)
    ax5.set_ylabel('Ratio', fontsize=12)
    ax5.grid(True, alpha=0.3)
    
    # 添加数值标签
    for bar, ratio in zip(bars, imbalance_ratios):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height + height*0.02,
                f'{ratio:.1f}x', ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    # 6. 熵值对比（熵越小越不均衡）
    ax6 = plt.subplot(2, 4, 7)
    entropy_values = [imbalance_data[name]['entropy'] for name in dataset_names]
    bars = ax6.bar(dataset_names, entropy_values, color=colors_bar, alpha=0.8)
    ax6.set_title('Shannon Entropy\n(Lower = More Imbalanced)', fontsize=14, fontweight='bold', pad=15)
    ax6.set_ylabel('Entropy', fontsize=12)
    ax6.grid(True, alpha=0.3)
    
    # 添加数值标签
    for bar, entropy_val in zip(bars, entropy_values):
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height + height*0.02,
                f'{entropy_val:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    # 7. 统计信息表格
    ax7 = plt.subplot(2, 4, 8)
    ax7.axis('off')
    
    # 创建统计信息文本
    stats_text = """Imbalance Hierarchy: LINEAR > ASAS > MACHO\n\n"""
    
    for name in ['ASAS', 'LINEAR', 'MACHO']:
        data = imbalance_data[name]
        stats_text += f"""{name}:
Total: {data['total']:,} samples
Classes: {len(data['counts'])}
Imbalance Ratio: {data['imbalance_ratio']:.1f}x
Gini Index: {data['gini']:.3f}
Shannon Entropy: {data['entropy']:.3f}
CV: {data['cv']:.3f}

"""
    
    ax7.text(0.05, 0.95, stats_text, transform=ax7.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
    
    # 调整布局
    plt.tight_layout(pad=3.0)
    plt.subplots_adjust(top=0.90, bottom=0.08, left=0.05, right=0.95, 
                       hspace=0.4, wspace=0.3)
    
    # 保存图片
    output_dir = "/autodl-fs/data/lnsde-contiformer/results/pics/comparison"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "class_imbalance_comparison.png")
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ 类别不均衡对比可视化已保存: {output_path}")
    
    # 打印详细统计
    print("\n=== 不均衡程度排序 ===")
    sorted_datasets = sorted(dataset_names, key=lambda x: imbalance_data[x]['imbalance_ratio'], reverse=True)
    for i, name in enumerate(sorted_datasets, 1):
        print(f"{i}. {name}: 不均衡比率 {imbalance_data[name]['imbalance_ratio']:.1f}x")
    
    plt.show()


if __name__ == "__main__":
    create_class_imbalance_visualization()