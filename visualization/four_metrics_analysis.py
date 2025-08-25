#!/usr/bin/env python3

import json
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import os
from pathlib import Path

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

def get_real_dataset_info():
    """获取真实数据集信息"""
    datasets_info = {
        'ASAS': {
            'total_samples': 3100,
            'num_classes': 5,
            'class_distribution': {
                'Beta Persei': 349,
                'Classical Cepheid': 130, 
                'RR Lyrae FM': 798,
                'Semireg PV': 184,
                'W Ursae Ma': 1639
            },
            'class_mapping': {'0': 'Beta Persei', '1': 'Classical Cepheid', '2': 'RR Lyrae FM', '3': 'Semireg PV', '4': 'W Ursae Ma'},
            'best_class_accuracy': {'0': 100.0, '1': 100.0, '2': 100.0, '3': 0.0, '4': 96.67},
            'best_val_accuracy': 98.15
        },
        'LINEAR': {
            'total_samples': 5204,
            'num_classes': 5,
            'class_distribution': {
                'Beta_Persei': 291,
                'Delta_Scuti': 70,
                'RR_Lyrae_FM': 2234,
                'RR_Lyrae_FO': 749,
                'W_Ursae_Maj': 1860
            },
            'class_mapping': {'0': 'Beta_Persei', '1': 'Delta_Scuti', '2': 'RR_Lyrae_FM', '3': 'RR_Lyrae_FO', '4': 'W_Ursae_Maj'},
            'best_class_accuracy': {'0': 0.0, '1': 100.0, '2': 96.67, '3': 97.30, '4': 93.0},
            'best_val_accuracy': 89.82
        },
        'MACHO': {
            'total_samples': 2100,
            'num_classes': 7,
            'class_distribution': {
                'Be': 128,
                'CEPH': 101,
                'EB': 255,
                'LPV': 365,
                'MOA': 582,
                'QSO': 59,
                'RRL': 610
            },
            'class_mapping': {'0': 'Be', '1': 'CEPH', '2': 'EB', '3': 'LPV', '4': 'MOA', '5': 'QSO', '6': 'RRL'},
            'best_class_accuracy': {'0': 61.54, '1': 81.82, '2': 45.71, '3': 89.47, '4': 68.97, '5': 0.0, '6': 96.0},
            'best_val_accuracy': 73.81
        }
    }
    return datasets_info

def calculate_auroc_approximation(val_accuracy, class_accuracy_data):
    """基于总体准确率和类别平衡近似计算AUROC"""
    valid_class_accs = [acc for acc in class_accuracy_data.values() if acc > 0]
    
    if not valid_class_accs:
        return 0.5
    
    # 通过类别平衡调整总体准确率
    class_balance_penalty = np.std(valid_class_accs) / 100.0
    auroc_approx = (val_accuracy / 100.0) * (1 - class_balance_penalty * 0.3)
    auroc_approx = max(0.5, min(1.0, auroc_approx))
    
    return auroc_approx

def calculate_majority_minority_ratio(class_distribution):
    """计算多数类与少数类的比例"""
    counts = np.array(list(class_distribution.values()))
    majority_count = np.max(counts)
    minority_count = np.min(counts[counts > 0])  # 排除0值
    
    return majority_count / minority_count if minority_count > 0 else 0

def calculate_normalized_entropy(class_distribution):
    """计算归一化熵"""
    counts = np.array(list(class_distribution.values()))
    total = np.sum(counts)
    
    if total == 0:
        return 0
    
    # 计算概率分布
    probabilities = counts / total
    probabilities = probabilities[probabilities > 0]  # 排除0概率
    
    # 计算熵
    entropy = -np.sum(probabilities * np.log2(probabilities))
    
    # 归一化 (除以最大可能熵)
    max_entropy = np.log2(len(probabilities))
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
    
    return normalized_entropy

def create_four_metrics_plot(datasets_info, save_path):
    """创建2x2布局的四个指标对比图，图例在右侧"""
    
    datasets = list(datasets_info.keys())
    n_datasets = len(datasets)
    
    # 科研配色方案
    colors_primary = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']     # 深蓝、橙、绿、红
    colors_secondary = ['#aec7e8', '#ffbb78', '#98df8a', '#ff9999']   # 对应浅色
    
    # 计算指标
    val_accuracies = []
    aurocs = []
    majority_minority_ratios = []
    normalized_entropies = []
    
    for dataset_name in datasets:
        info = datasets_info[dataset_name]
        val_acc = info['best_val_accuracy']
        class_dist = info['class_distribution']
        class_accs = info['best_class_accuracy']
        
        auroc_approx = calculate_auroc_approximation(val_acc, class_accs)
        mm_ratio = calculate_majority_minority_ratio(class_dist)
        norm_entropy = calculate_normalized_entropy(class_dist)
        
        val_accuracies.append(val_acc)
        aurocs.append(auroc_approx * 100)
        majority_minority_ratios.append(mm_ratio)
        normalized_entropies.append(norm_entropy * 100)  # 转为百分比显示
    
    # 创建2x2子图布局
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    x_pos = np.arange(n_datasets)
    width = 0.35
    
    # 计算合适的y轴范围
    acc_min, acc_max = min(val_accuracies), max(val_accuracies)
    acc_range = acc_max - acc_min
    acc_ylim_min = max(0, acc_min - acc_range * 0.1)
    acc_ylim_max = min(100, acc_max + acc_range * 0.15)
    
    # 图1: 验证准确率 vs AUROC
    ax1_left = ax1
    bars1_left = ax1_left.bar(x_pos - width/2, val_accuracies, width, 
                              label='验证准确率', alpha=0.8, color=colors_primary[0])
    ax1_left.set_ylabel('验证准确率 (%)', color=colors_primary[0], fontweight='bold', fontsize=12)
    ax1_left.tick_params(axis='y', labelcolor=colors_primary[0])
    ax1_left.set_title('验证准确率 vs AUROC', fontweight='bold', fontsize=14)
    ax1_left.set_ylim(acc_ylim_min, acc_ylim_max)
    
    ax1_right = ax1_left.twinx()
    bars1_right = ax1_right.bar(x_pos + width/2, aurocs, width, 
                               label='AUROC', alpha=0.8, color=colors_secondary[0])
    ax1_right.set_ylabel('AUROC (%)', color=colors_secondary[0], fontweight='bold', fontsize=12)
    ax1_right.tick_params(axis='y', labelcolor=colors_secondary[0])
    
    auroc_min, auroc_max = min(aurocs), max(aurocs)
    auroc_range = auroc_max - auroc_min
    auroc_ylim_min = max(50, auroc_min - auroc_range * 0.1)
    auroc_ylim_max = min(100, auroc_max + auroc_range * 0.15)
    ax1_right.set_ylim(auroc_ylim_min, auroc_ylim_max)
    
    # 添加数值标签
    for i, (acc, auroc) in enumerate(zip(val_accuracies, aurocs)):
        ax1_left.text(i - width/2, acc + acc_range*0.02, f'{acc:.1f}%', ha='center', va='bottom', 
                     fontsize=10, color=colors_primary[0], fontweight='bold')
        ax1_right.text(i + width/2, auroc + auroc_range*0.02, f'{auroc:.1f}%', ha='center', va='bottom', 
                      fontsize=10, color=colors_secondary[0], fontweight='bold')
    
    ax1_left.set_xticks(x_pos)
    ax1_left.set_xticklabels(datasets, fontsize=11)
    ax1_left.grid(True, alpha=0.3)
    
    # 图2: 验证准确率 vs 多数-少数类比例
    ax2_left = ax2
    bars2_left = ax2_left.bar(x_pos - width/2, val_accuracies, width, 
                              label='验证准确率', alpha=0.8, color=colors_primary[1])
    ax2_left.set_ylabel('验证准确率 (%)', color=colors_primary[1], fontweight='bold', fontsize=12)
    ax2_left.tick_params(axis='y', labelcolor=colors_primary[1])
    ax2_left.set_title('验证准确率 vs 多数-少数类比例', fontweight='bold', fontsize=14)
    ax2_left.set_ylim(acc_ylim_min, acc_ylim_max)
    
    ax2_right = ax2_left.twinx()
    bars2_right = ax2_right.bar(x_pos + width/2, majority_minority_ratios, width, 
                               label='多数-少数类比例', alpha=0.8, color=colors_secondary[1])
    ax2_right.set_ylabel('多数-少数类比例', color=colors_secondary[1], fontweight='bold', fontsize=12)
    ax2_right.tick_params(axis='y', labelcolor=colors_secondary[1])
    
    mm_min, mm_max = min(majority_minority_ratios), max(majority_minority_ratios)
    mm_range = mm_max - mm_min
    mm_ylim_min = max(0, mm_min - mm_range * 0.1)
    mm_ylim_max = mm_max + mm_range * 0.15
    ax2_right.set_ylim(mm_ylim_min, mm_ylim_max)
    
    # 添加数值标签
    for i, (acc, mm) in enumerate(zip(val_accuracies, majority_minority_ratios)):
        ax2_left.text(i - width/2, acc + acc_range*0.02, f'{acc:.1f}%', ha='center', va='bottom', 
                     fontsize=10, color=colors_primary[1], fontweight='bold')
        ax2_right.text(i + width/2, mm + mm_range*0.02, f'{mm:.1f}', ha='center', va='bottom', 
                      fontsize=10, color=colors_secondary[1], fontweight='bold')
    
    ax2_left.set_xticks(x_pos)
    ax2_left.set_xticklabels(datasets, fontsize=11)
    ax2_left.grid(True, alpha=0.3)
    
    # 图3: 验证准确率 vs 归一化熵
    ax3_left = ax3
    bars3_left = ax3_left.bar(x_pos - width/2, val_accuracies, width, 
                              label='验证准确率', alpha=0.8, color=colors_primary[2])
    ax3_left.set_ylabel('验证准确率 (%)', color=colors_primary[2], fontweight='bold', fontsize=12)
    ax3_left.tick_params(axis='y', labelcolor=colors_primary[2])
    ax3_left.set_title('验证准确率 vs 归一化熵', fontweight='bold', fontsize=14)
    ax3_left.set_ylim(acc_ylim_min, acc_ylim_max)
    
    ax3_right = ax3_left.twinx()
    bars3_right = ax3_right.bar(x_pos + width/2, normalized_entropies, width, 
                               label='归一化熵', alpha=0.8, color=colors_secondary[2])
    ax3_right.set_ylabel('归一化熵 (%)', color=colors_secondary[2], fontweight='bold', fontsize=12)
    ax3_right.tick_params(axis='y', labelcolor=colors_secondary[2])
    
    ne_min, ne_max = min(normalized_entropies), max(normalized_entropies)
    ne_range = ne_max - ne_min
    ne_ylim_min = max(0, ne_min - ne_range * 0.1)
    ne_ylim_max = min(100, ne_max + ne_range * 0.15)
    ax3_right.set_ylim(ne_ylim_min, ne_ylim_max)
    
    # 添加数值标签
    for i, (acc, ne) in enumerate(zip(val_accuracies, normalized_entropies)):
        ax3_left.text(i - width/2, acc + acc_range*0.02, f'{acc:.1f}%', ha='center', va='bottom', 
                     fontsize=10, color=colors_primary[2], fontweight='bold')
        ax3_right.text(i + width/2, ne + ne_range*0.02, f'{ne:.1f}%', ha='center', va='bottom', 
                      fontsize=10, color=colors_secondary[2], fontweight='bold')
    
    ax3_left.set_xticks(x_pos)
    ax3_left.set_xticklabels(datasets, fontsize=11)
    ax3_left.grid(True, alpha=0.3)
    
    # 图4: 综合对比图 (显示所有三个指标)
    ax4_left = ax4
    bars4_left = ax4_left.bar(x_pos - width/2, val_accuracies, width, 
                              label='验证准确率', alpha=0.8, color=colors_primary[3])
    ax4_left.set_ylabel('验证准确率 (%)', color=colors_primary[3], fontweight='bold', fontsize=12)
    ax4_left.tick_params(axis='y', labelcolor=colors_primary[3])
    ax4_left.set_title('验证准确率 vs 综合不平衡指标', fontweight='bold', fontsize=14)
    ax4_left.set_ylim(acc_ylim_min, acc_ylim_max)
    
    ax4_right = ax4_left.twinx()
    # 将三个指标归一化到0-100范围进行显示
    normalized_aurocs = [(a-min(aurocs))/(max(aurocs)-min(aurocs))*100 for a in aurocs]
    normalized_mm = [(m-min(majority_minority_ratios))/(max(majority_minority_ratios)-min(majority_minority_ratios))*100 for m in majority_minority_ratios]
    
    width_small = width/3
    ax4_right.bar(x_pos + width/2 - width_small, normalized_aurocs, width_small, 
                 label='AUROC(标准化)', alpha=0.7, color=colors_secondary[0])
    ax4_right.bar(x_pos + width/2, normalized_mm, width_small, 
                 label='多数-少数类比例(标准化)', alpha=0.7, color=colors_secondary[1])
    ax4_right.bar(x_pos + width/2 + width_small, normalized_entropies, width_small, 
                 label='归一化熵', alpha=0.7, color=colors_secondary[2])
    
    ax4_right.set_ylabel('标准化指标 (%)', color='gray', fontweight='bold', fontsize=12)
    ax4_right.tick_params(axis='y', labelcolor='gray')
    ax4_right.set_ylim(0, 100)
    
    # 添加数值标签
    for i, acc in enumerate(val_accuracies):
        ax4_left.text(i - width/2, acc + acc_range*0.02, f'{acc:.1f}%', ha='center', va='bottom', 
                     fontsize=10, color=colors_primary[3], fontweight='bold')
    
    ax4_left.set_xticks(x_pos)
    ax4_left.set_xticklabels(datasets, fontsize=11)
    ax4_left.grid(True, alpha=0.3)
    
    # 为每个子图添加图例，位置在右侧
    axes_pairs = [
        (ax1_left, ax1_right),
        (ax2_left, ax2_right),
        (ax3_left, ax3_right),
        (ax4_left, ax4_right)
    ]
    
    for ax_left, ax_right in axes_pairs:
        lines1, labels1 = ax_left.get_legend_handles_labels()
        lines2, labels2 = ax_right.get_legend_handles_labels()
        ax_right.legend(lines1 + lines2, labels1 + labels2, 
                       bbox_to_anchor=(1.15, 1), loc='upper left', fontsize=10)
    
    # 调整布局
    plt.tight_layout(pad=3.0)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"四指标综合分析已保存至: {save_path}")

def print_four_metrics_summary(datasets_info):
    """打印四指标分析摘要"""
    print("\n" + "="*100)
    print("四指标综合分析 - 准确率与不平衡度量关系")
    print("="*100)
    print(f"{'数据集':<8} {'验证准确率':<10} {'AUROC':<8} {'多数-少数类比例':<15} {'归一化熵':<10}")
    print("-"*100)
    
    for dataset_name, info in datasets_info.items():
        class_dist = info['class_distribution']
        class_accs = info['best_class_accuracy']
        
        auroc_approx = calculate_auroc_approximation(info['best_val_accuracy'], class_accs)
        mm_ratio = calculate_majority_minority_ratio(class_dist)
        norm_entropy = calculate_normalized_entropy(class_dist)
        
        print(f"{dataset_name:<8} {info['best_val_accuracy']:<10.1f} {auroc_approx*100:<8.1f} "
              f"{mm_ratio:<15.1f} {norm_entropy*100:<10.1f}")
    
    print("-"*100)
    print("指标说明:")
    print("• AUROC: 基于准确率和类别平衡的近似受试者工作特征曲线下面积")
    print("• 多数-少数类比例: 最大类别样本数与最小类别样本数的比值")
    print("• 归一化熵: 类别分布的信息熵，归一化到0-1范围 (1=完全平衡，0=完全不平衡)")
    print("="*100)

def main():
    print("配置中文字体...")
    working_font = configure_chinese_fonts()
    
    if working_font:
        print(f"✅ 成功配置中文字体: {working_font}")
    else:
        print("⚠️ 中文字体配置可能有问题，但会尝试继续...")
    
    # 获取数据集信息
    datasets_info = get_real_dataset_info()
    
    # 创建输出目录
    output_dir = '/root/autodl-tmp/lnsde+contiformer/results/pics'
    os.makedirs(output_dir, exist_ok=True)
    
    # 生成四指标对比图
    plot_path = os.path.join(output_dir, 'four_metrics_analysis.png')
    create_four_metrics_plot(datasets_info, plot_path)
    
    # 打印分析摘要
    print_four_metrics_summary(datasets_info)

if __name__ == "__main__":
    main()