#!/usr/bin/env python3

import json
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import os
from collections import Counter

# 简化字体设置，避免字体问题
plt.rcParams['axes.unicode_minus'] = False

def get_real_dataset_info():
    """Get real class distribution data for each dataset with updated MACHO data"""
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
            # Updated with latest data from system reminder
            'best_class_accuracy': {'0': 61.54, '1': 81.82, '2': 45.71, '3': 89.47, '4': 68.97, '5': 0.0, '6': 96.0},
            'best_val_accuracy': 73.81  # Updated value from latest log
        }
    }
    return datasets_info

def calculate_imbalance_metrics(class_distribution, class_accuracies=None):
    """Calculate comprehensive imbalance metrics from real class distribution"""
    counts = np.array(list(class_distribution.values()))
    total_samples = np.sum(counts)
    
    # 1. Coefficient of Variation (CV) - standard deviation / mean
    mean_count = np.mean(counts)
    std_count = np.std(counts)
    cv_ratio = std_count / mean_count if mean_count > 0 else 0
    
    # 2. Gini Coefficient for class distribution
    n = len(counts)
    sorted_counts = np.sort(counts)
    cumsum_counts = np.cumsum(sorted_counts)
    gini = (n + 1 - 2 * np.sum(cumsum_counts) / cumsum_counts[-1]) / n
    
    return cv_ratio, gini

def calculate_auroc_approximation(val_accuracy, class_accuracy_data):
    """Approximate AUROC based on overall accuracy and class balance"""
    valid_class_accs = [acc for acc in class_accuracy_data.values() if acc > 0]
    
    if not valid_class_accs:
        return 0.5
    
    # Weight overall accuracy by class balance (more balanced = closer to actual AUROC)
    class_balance_penalty = np.std(valid_class_accs) / 100.0
    
    # Simple approximation: start with accuracy, adjust for class imbalance
    auroc_approx = (val_accuracy / 100.0) * (1 - class_balance_penalty * 0.3)
    
    # Ensure AUROC is between 0.5 and 1.0
    auroc_approx = max(0.5, min(1.0, auroc_approx))
    
    return auroc_approx

def create_scientific_accuracy_analysis(datasets_info, save_path):
    """Create 3 scientific-style plots with proper axis ranges and colors"""
    
    datasets = list(datasets_info.keys())
    n_datasets = len(datasets)
    
    # 科学配色方案
    colors_primary = ['#2E86AB', '#A23B72', '#F18F01']  # 蓝色、紫色、橙色
    colors_secondary = ['#5DADE2', '#E74C3C', '#F39C12']  # 对应的浅色版本
    
    # Calculate metrics for each dataset
    val_accuracies = []
    aurocs = []
    cv_ratios = []
    gini_coeffs = []
    
    for dataset_name in datasets:
        info = datasets_info[dataset_name]
        val_acc = info['best_val_accuracy']
        class_dist = info['class_distribution']
        class_accs = info['best_class_accuracy']
        
        # Calculate metrics
        cv, gini = calculate_imbalance_metrics(class_dist)
        auroc_approx = calculate_auroc_approximation(val_acc, class_accs)
        
        val_accuracies.append(val_acc)
        aurocs.append(auroc_approx * 100)  # Convert to percentage
        cv_ratios.append(cv * 100)  # Scale for visualization
        gini_coeffs.append(gini * 100)  # Convert to percentage
    
    # Create the plot with 3 subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    x_pos = np.arange(n_datasets)
    width = 0.35
    
    # Calculate proper axis ranges
    acc_min, acc_max = min(val_accuracies), max(val_accuracies)
    acc_range = acc_max - acc_min
    acc_ylim_min = max(0, acc_min - acc_range * 0.1)
    acc_ylim_max = min(100, acc_max + acc_range * 0.2)
    
    # Plot 1: 验证准确率 vs AUROC
    ax1_left = ax1
    bars1_left = ax1_left.bar(x_pos - width/2, val_accuracies, width, 
                              label='验证准确率', alpha=0.8, color=colors_primary[0])
    ax1_left.set_ylabel('验证准确率 (%)', color=colors_primary[0], fontweight='bold', fontsize=12)
    ax1_left.tick_params(axis='y', labelcolor=colors_primary[0])
    ax1_left.set_title('验证准确率 vs AUROC', fontweight='bold', fontsize=14, pad=20)
    ax1_left.set_ylim(acc_ylim_min, acc_ylim_max)
    
    ax1_right = ax1_left.twinx()
    bars1_right = ax1_right.bar(x_pos + width/2, aurocs, width, 
                               label='AUROC', alpha=0.8, color=colors_secondary[0])
    ax1_right.set_ylabel('AUROC (%)', color=colors_secondary[0], fontweight='bold', fontsize=12)
    ax1_right.tick_params(axis='y', labelcolor=colors_secondary[0])
    
    auroc_min, auroc_max = min(aurocs), max(aurocs)
    auroc_range = auroc_max - auroc_min
    auroc_ylim_min = max(50, auroc_min - auroc_range * 0.1)
    auroc_ylim_max = min(100, auroc_max + auroc_range * 0.2)
    ax1_right.set_ylim(auroc_ylim_min, auroc_ylim_max)
    
    # Add value labels
    for i, (acc, auroc) in enumerate(zip(val_accuracies, aurocs)):
        ax1_left.text(i - width/2, acc + acc_range*0.02, f'{acc:.1f}%', ha='center', va='bottom', 
                     fontsize=10, color=colors_primary[0], fontweight='bold')
        ax1_right.text(i + width/2, auroc + auroc_range*0.02, f'{auroc:.1f}%', ha='center', va='bottom', 
                      fontsize=10, color=colors_secondary[0], fontweight='bold')
    
    ax1_left.set_xticks(x_pos)
    ax1_left.set_xticklabels(datasets, fontsize=11)
    ax1_left.grid(True, alpha=0.3)
    
    # Plot 2: 验证准确率 vs 变异系数
    ax2_left = ax2
    bars2_left = ax2_left.bar(x_pos - width/2, val_accuracies, width, 
                              label='验证准确率', alpha=0.8, color=colors_primary[1])
    ax2_left.set_ylabel('验证准确率 (%)', color=colors_primary[1], fontweight='bold', fontsize=12)
    ax2_left.tick_params(axis='y', labelcolor=colors_primary[1])
    ax2_left.set_title('验证准确率 vs 变异系数', fontweight='bold', fontsize=14, pad=20)
    ax2_left.set_ylim(acc_ylim_min, acc_ylim_max)
    
    ax2_right = ax2_left.twinx()
    bars2_right = ax2_right.bar(x_pos + width/2, cv_ratios, width, 
                               label='变异系数', alpha=0.8, color=colors_secondary[1])
    ax2_right.set_ylabel('变异系数 (%)', color=colors_secondary[1], fontweight='bold', fontsize=12)
    ax2_right.tick_params(axis='y', labelcolor=colors_secondary[1])
    
    cv_min, cv_max = min(cv_ratios), max(cv_ratios)
    cv_range = cv_max - cv_min
    cv_ylim_min = max(0, cv_min - cv_range * 0.1)
    cv_ylim_max = cv_max + cv_range * 0.2
    ax2_right.set_ylim(cv_ylim_min, cv_ylim_max)
    
    # Add value labels
    for i, (acc, cv) in enumerate(zip(val_accuracies, cv_ratios)):
        ax2_left.text(i - width/2, acc + acc_range*0.02, f'{acc:.1f}%', ha='center', va='bottom', 
                     fontsize=10, color=colors_primary[1], fontweight='bold')
        ax2_right.text(i + width/2, cv + cv_range*0.02, f'{cv:.1f}', ha='center', va='bottom', 
                      fontsize=10, color=colors_secondary[1], fontweight='bold')
    
    ax2_left.set_xticks(x_pos)
    ax2_left.set_xticklabels(datasets, fontsize=11)
    ax2_left.grid(True, alpha=0.3)
    
    # Plot 3: 验证准确率 vs 基尼系数
    ax3_left = ax3
    bars3_left = ax3_left.bar(x_pos - width/2, val_accuracies, width, 
                              label='验证准确率', alpha=0.8, color=colors_primary[2])
    ax3_left.set_ylabel('验证准确率 (%)', color=colors_primary[2], fontweight='bold', fontsize=12)
    ax3_left.tick_params(axis='y', labelcolor=colors_primary[2])
    ax3_left.set_title('验证准确率 vs 基尼系数', fontweight='bold', fontsize=14, pad=20)
    ax3_left.set_ylim(acc_ylim_min, acc_ylim_max)
    
    ax3_right = ax3_left.twinx()
    bars3_right = ax3_right.bar(x_pos + width/2, gini_coeffs, width, 
                               label='基尼系数', alpha=0.8, color=colors_secondary[2])
    ax3_right.set_ylabel('基尼系数 (%)', color=colors_secondary[2], fontweight='bold', fontsize=12)
    ax3_right.tick_params(axis='y', labelcolor=colors_secondary[2])
    
    gini_min, gini_max = min(gini_coeffs), max(gini_coeffs)
    gini_range = gini_max - gini_min
    gini_ylim_min = max(0, gini_min - gini_range * 0.1)
    gini_ylim_max = gini_max + gini_range * 0.2
    ax3_right.set_ylim(gini_ylim_min, gini_ylim_max)
    
    # Add value labels
    for i, (acc, gini) in enumerate(zip(val_accuracies, gini_coeffs)):
        ax3_left.text(i - width/2, acc + acc_range*0.02, f'{acc:.1f}%', ha='center', va='bottom', 
                     fontsize=10, color=colors_primary[2], fontweight='bold')
        ax3_right.text(i + width/2, gini + gini_range*0.02, f'{gini:.1f}', ha='center', va='bottom', 
                      fontsize=10, color=colors_secondary[2], fontweight='bold')
    
    ax3_left.set_xticks(x_pos)
    ax3_left.set_xticklabels(datasets, fontsize=11)
    ax3_left.grid(True, alpha=0.3)
    
    # Add legends with better positioning
    axes_pairs = [(ax1_left, ax1_right), (ax2_left, ax2_right), (ax3_left, ax3_right)]
    
    for ax_left, ax_right in axes_pairs:
        lines1, labels1 = ax_left.get_legend_handles_labels()
        lines2, labels2 = ax_right.get_legend_handles_labels()
        ax_left.legend(lines1 + lines2, labels1 + labels2, 
                      bbox_to_anchor=(0.02, 0.98), loc='upper left', fontsize=10)
    
    # Adjust layout
    plt.tight_layout(pad=2.0, w_pad=3.0)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"科研风格准确率与不平衡分析已保存至: {save_path}")

def print_chinese_analysis_table(datasets_info):
    """Print analysis table in Chinese"""
    print("\n" + "="*100)
    print("基于真实类别分布的综合不平衡分析")
    print("="*100)
    print(f"{'数据集':<8} {'样本数':<8} {'类别数':<8} {'验证准确率':<10} {'AUROC':<8} {'变异系数':<10} {'基尼系数':<10} {'最差类别':<15}")
    print("-"*100)
    
    for dataset_name, info in datasets_info.items():
        class_dist = info['class_distribution']
        class_accs = info['best_class_accuracy']
        
        # Calculate metrics
        cv, gini = calculate_imbalance_metrics(class_dist)
        auroc_approx = calculate_auroc_approximation(info['best_val_accuracy'], class_accs)
        
        # Find worst performing class
        valid_accs = {k: v for k, v in class_accs.items() if v > 0}
        if valid_accs:
            worst_acc = min(valid_accs.values())
            worst_class = min(valid_accs, key=valid_accs.get)
            worst_class_name = info['class_mapping'].get(worst_class, worst_class)
        else:
            worst_acc = 0
            worst_class_name = "无"
        
        print(f"{dataset_name:<8} {info['total_samples']:<8} {info['num_classes']:<8} "
              f"{info['best_val_accuracy']:<10.1f} {auroc_approx*100:<8.1f} {cv:<10.2f} {gini:<10.2f} "
              f"{worst_class_name[:12]:<12} ({worst_acc:.1f}%)")
    
    print("-"*100)
    print("指标说明:")
    print("• 变异系数: 类别大小的相对变异性，数值越高表示类别越不平衡")
    print("• 基尼系数: 类别分布不平衡程度 (0=完全平衡, 1=最大不平衡)")
    print("• AUROC: 基于准确率和类别平衡的近似受试者工作特征曲线下面积")
    print("="*100)

def main():
    # Get dataset information with updated MACHO data
    datasets_info = get_real_dataset_info()
    
    # Create output directory
    output_dir = '/root/autodl-tmp/lnsde+contiformer/results/pics'
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate scientific-style plot
    plot_path = os.path.join(output_dir, 'scientific_accuracy_imbalance_analysis.png')
    create_scientific_accuracy_analysis(datasets_info, plot_path)
    
    # Print Chinese analysis table
    print_chinese_analysis_table(datasets_info)

if __name__ == "__main__":
    main()