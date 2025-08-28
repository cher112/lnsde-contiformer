#!/usr/bin/env python3

import json
import matplotlib.pyplot as plt
import numpy as np
import os
from collections import Counter

def get_real_dataset_info():
    """Get real class distribution data for each dataset"""
    datasets_info = {
        'ASAS': {
            'total_samples': 3100,  # 349+130+798+184+1639
            'num_classes': 5,
            'class_distribution': {
                'Beta Persei': 349,
                'Classical Cepheid': 130, 
                'RR Lyrae FM': 798,
                'Semireg PV': 184,
                'W Ursae Ma': 1639
            },
            'class_mapping': {'0': 'Beta Persei', '1': 'Classical Cepheid', '2': 'RR Lyrae FM', '3': 'Semireg PV', '4': 'W Ursae Ma'},
            'best_class_accuracy': {'0': 100.0, '1': 100.0, '2': 100.0, '4': 96.67},
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
            'best_class_accuracy': {'0': 0.0, '1': 100.0, '2': 96.67, '3': 97.30, '4': 92.96},
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
            'best_class_accuracy': {'0': 76.92, '1': 90.91, '2': 51.43, '3': 89.47, '4': 63.79, '5': 0.0, '6': 94.0},
            'best_val_accuracy': 74.29
        }
    }
    return datasets_info

def calculate_imbalance_metrics(class_distribution, class_accuracies=None):
    """
    Calculate comprehensive imbalance metrics from real class distribution
    
    Args:
        class_distribution: Dict with class names as keys and sample counts as values
        class_accuracies: Optional dict with class accuracies
    
    Returns:
        imbalance_ratio: Coefficient of variation of class sizes
        gini_coefficient: Gini coefficient for class distribution (0=perfect balance, 1=maximum imbalance)
        entropy_ratio: Normalized entropy (1=perfect balance, 0=maximum imbalance)
        majority_minority_ratio: Ratio of largest to smallest class
    """
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
    
    # 3. Normalized Entropy (higher = more balanced)
    proportions = counts / total_samples
    entropy = -np.sum(proportions * np.log2(proportions + 1e-10))  # Add small constant to avoid log(0)
    max_entropy = np.log2(n)  # Maximum entropy for n classes
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
    
    # 4. Majority-Minority Ratio
    maj_min_ratio = np.max(counts) / np.min(counts) if np.min(counts) > 0 else np.inf
    
    return cv_ratio, gini, normalized_entropy, maj_min_ratio

def calculate_performance_imbalance_score(val_accuracy, class_accuracies, class_distribution):
    """Calculate a composite score showing how well the model handles imbalance"""
    # Filter out classes with 0% accuracy (might not have test samples)
    valid_accs = {k: v for k, v in class_accuracies.items() if v > 0}
    
    if not valid_accs:
        return 0.0
    
    # Performance on minority classes (weighted by inverse class frequency)
    total_samples = sum(class_distribution.values())
    weighted_score = 0
    total_weight = 0
    
    for class_id, acc in valid_accs.items():
        if class_id in class_distribution:
            class_count = list(class_distribution.values())[int(class_id)]
            # Weight inversely proportional to class frequency (minority classes get higher weight)
            weight = total_samples / (class_count * len(class_distribution))
            weighted_score += acc * weight
            total_weight += weight
    
    return weighted_score / total_weight if total_weight > 0 else 0.0

def create_stacked_accuracy_imbalance_plot(datasets_info, save_path):
    """Create a stacked bar plot showing accuracy and interpretable imbalance metrics"""
    
    datasets = list(datasets_info.keys())
    n_datasets = len(datasets)
    
    # Calculate all metrics
    val_accuracies = []
    cv_ratios = []
    gini_coeffs = []
    entropy_ratios = []
    maj_min_ratios = []
    perf_imbal_scores = []
    
    for dataset_name in datasets:
        info = datasets_info[dataset_name]
        val_acc = info['best_val_accuracy']
        class_dist = info['class_distribution']
        class_accs = info['best_class_accuracy']
        
        # Calculate imbalance metrics
        cv, gini, entropy, maj_min = calculate_imbalance_metrics(class_dist)
        perf_score = calculate_performance_imbalance_score(val_acc, class_accs, class_dist)
        
        val_accuracies.append(val_acc)
        cv_ratios.append(cv * 100)  # Scale for visualization
        gini_coeffs.append(gini * 100)  # Convert to percentage
        entropy_ratios.append(entropy * 100)  # Convert to percentage
        maj_min_ratios.append(min(maj_min, 50))  # Cap at 50 for visualization
        perf_imbal_scores.append(perf_score)
    
    # Create the plot with more space
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 14))
    
    x_pos = np.arange(n_datasets)
    width = 0.3  # Make bars narrower to reduce overlap
    
    # Plot 1: Validation Accuracy vs CV Ratio (Coefficient of Variation)
    ax1_left = ax1
    bars1_left = ax1_left.bar(x_pos - width/2, val_accuracies, width, label='Val Accuracy', alpha=0.8, color='skyblue')
    ax1_left.set_ylabel('Validation Accuracy (%)', color='blue', fontweight='bold', fontsize=12)
    ax1_left.tick_params(axis='y', labelcolor='blue')
    ax1_left.set_title('Accuracy vs Coefficient of Variation', fontweight='bold', fontsize=13, pad=20)
    # Set y-axis limit so max bar is at 3/4 height
    ax1_left.set_ylim(0, max(val_accuracies) * 1.33)
    
    ax1_right = ax1_left.twinx()
    bars1_right = ax1_right.bar(x_pos + width/2, cv_ratios, width, label='CV Ratio', alpha=0.8, color='lightcoral')
    ax1_right.set_ylabel('CV Ratio (%)', color='red', fontweight='bold', fontsize=12)
    ax1_right.tick_params(axis='y', labelcolor='red')
    # Set y-axis limit so max bar is at 3/4 height
    ax1_right.set_ylim(0, max(cv_ratios) * 1.33)
    
    # Add value labels with better positioning
    for i, (acc, cv) in enumerate(zip(val_accuracies, cv_ratios)):
        ax1_left.text(i - width/2, acc + max(val_accuracies)*0.02, f'{acc:.1f}%', ha='center', va='bottom', fontsize=10, color='blue', fontweight='bold')
        ax1_right.text(i + width/2, cv + max(cv_ratios)*0.02, f'{cv:.1f}', ha='center', va='bottom', fontsize=10, color='red', fontweight='bold')
    
    ax1_left.set_xticks(x_pos)
    ax1_left.set_xticklabels(datasets, fontsize=11)
    ax1_left.grid(True, alpha=0.3)
    
    # Plot 2: Validation Accuracy vs Gini Coefficient
    ax2_left = ax2
    bars2_left = ax2_left.bar(x_pos - width/2, val_accuracies, width, label='Val Accuracy', alpha=0.8, color='skyblue')
    ax2_left.set_ylabel('Validation Accuracy (%)', color='blue', fontweight='bold', fontsize=12)
    ax2_left.tick_params(axis='y', labelcolor='blue')
    ax2_left.set_title('Accuracy vs Gini Coefficient', fontweight='bold', fontsize=13, pad=20)
    # Set y-axis limit so max bar is at 3/4 height
    ax2_left.set_ylim(0, max(val_accuracies) * 1.33)
    
    ax2_right = ax2_left.twinx()
    bars2_right = ax2_right.bar(x_pos + width/2, gini_coeffs, width, label='Gini Coeff', alpha=0.8, color='lightgreen')
    ax2_right.set_ylabel('Gini Coefficient (%)', color='green', fontweight='bold', fontsize=12)
    ax2_right.tick_params(axis='y', labelcolor='green')
    # Set y-axis limit so max bar is at 3/4 height
    ax2_right.set_ylim(0, max(gini_coeffs) * 1.33)
    
    # Add value labels with better positioning
    for i, (acc, gini) in enumerate(zip(val_accuracies, gini_coeffs)):
        ax2_left.text(i - width/2, acc + max(val_accuracies)*0.02, f'{acc:.1f}%', ha='center', va='bottom', fontsize=10, color='blue', fontweight='bold')
        ax2_right.text(i + width/2, gini + max(gini_coeffs)*0.02, f'{gini:.1f}', ha='center', va='bottom', fontsize=10, color='green', fontweight='bold')
    
    ax2_left.set_xticks(x_pos)
    ax2_left.set_xticklabels(datasets, fontsize=11)
    ax2_left.grid(True, alpha=0.3)
    
    # Plot 3: Validation Accuracy vs Entropy Ratio (Balance Measure)
    ax3_left = ax3
    bars3_left = ax3_left.bar(x_pos - width/2, val_accuracies, width, label='Val Accuracy', alpha=0.8, color='skyblue')
    ax3_left.set_ylabel('Validation Accuracy (%)', color='blue', fontweight='bold', fontsize=12)
    ax3_left.tick_params(axis='y', labelcolor='blue')
    ax3_left.set_title('Accuracy vs Normalized Entropy (Balance)', fontweight='bold', fontsize=13, pad=20)
    # Set y-axis limit so max bar is at 3/4 height
    ax3_left.set_ylim(0, max(val_accuracies) * 1.33)
    
    ax3_right = ax3_left.twinx()
    bars3_right = ax3_right.bar(x_pos + width/2, entropy_ratios, width, label='Entropy Ratio', alpha=0.8, color='orange')
    ax3_right.set_ylabel('Normalized Entropy (%)', color='darkorange', fontweight='bold', fontsize=12)
    ax3_right.tick_params(axis='y', labelcolor='darkorange')
    # Set y-axis limit so max bar is at 3/4 height
    ax3_right.set_ylim(0, max(entropy_ratios) * 1.33)
    
    # Add value labels with better positioning
    for i, (acc, entropy) in enumerate(zip(val_accuracies, entropy_ratios)):
        ax3_left.text(i - width/2, acc + max(val_accuracies)*0.02, f'{acc:.1f}%', ha='center', va='bottom', fontsize=10, color='blue', fontweight='bold')
        ax3_right.text(i + width/2, entropy + max(entropy_ratios)*0.02, f'{entropy:.1f}', ha='center', va='bottom', fontsize=10, color='darkorange', fontweight='bold')
    
    ax3_left.set_xticks(x_pos)
    ax3_left.set_xticklabels(datasets, fontsize=11)
    ax3_left.grid(True, alpha=0.3)
    
    # Plot 4: Validation Accuracy vs Majority-Minority Ratio
    ax4_left = ax4
    bars4_left = ax4_left.bar(x_pos - width/2, val_accuracies, width, label='Val Accuracy', alpha=0.8, color='skyblue')
    ax4_left.set_ylabel('Validation Accuracy (%)', color='blue', fontweight='bold', fontsize=12)
    ax4_left.tick_params(axis='y', labelcolor='blue')
    ax4_left.set_title('Accuracy vs Majority-Minority Ratio', fontweight='bold', fontsize=13, pad=20)
    # Set y-axis limit so max bar is at 3/4 height
    ax4_left.set_ylim(0, max(val_accuracies) * 1.33)
    
    ax4_right = ax4_left.twinx()
    bars4_right = ax4_right.bar(x_pos + width/2, maj_min_ratios, width, label='Maj-Min Ratio', alpha=0.7, color='purple')
    ax4_right.set_ylabel('Majority-Minority Ratio', color='purple', fontweight='bold', fontsize=12)
    ax4_right.tick_params(axis='y', labelcolor='purple')
    # Set y-axis limit so max bar is at 3/4 height
    ax4_right.set_ylim(0, max(maj_min_ratios) * 1.33)
    
    # Add value labels with better positioning
    for i, (acc, ratio) in enumerate(zip(val_accuracies, maj_min_ratios)):
        ax4_left.text(i - width/2, acc + max(val_accuracies)*0.02, f'{acc:.1f}%', ha='center', va='bottom', fontsize=10, color='blue', fontweight='bold')
        ax4_right.text(i + width/2, ratio + max(maj_min_ratios)*0.02, f'{ratio:.1f}', ha='center', va='bottom', fontsize=10, color='purple', fontweight='bold')
    
    ax4_left.set_xticks(x_pos)
    ax4_left.set_xticklabels(datasets, fontsize=11)
    ax4_left.grid(True, alpha=0.3)
    
    # Add legends with better positioning to avoid overlap
    legend_positions = ['upper right', 'upper right', 'upper right', 'upper right']
    axes_pairs = [(ax1_left, ax1_right), (ax2_left, ax2_right), (ax3_left, ax3_right), (ax4_left, ax4_right)]
    
    for (ax_left, ax_right), pos in zip(axes_pairs, legend_positions):
        lines1, labels1 = ax_left.get_legend_handles_labels()
        lines2, labels2 = ax_right.get_legend_handles_labels()
        
        # Position legends outside the plot area to avoid overlap
        if pos == 'upper right':
            ax_left.legend(lines1 + lines2, labels1 + labels2, 
                          bbox_to_anchor=(0.98, 0.98), loc='upper right', fontsize=10)
    
    # Adjust layout with more padding
    plt.tight_layout(pad=3.0, h_pad=3.0, w_pad=2.0)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Corrected accuracy vs imbalance analysis saved to: {save_path}")

def print_detailed_analysis_table(datasets_info):
    """Print comprehensive analysis table with real data"""
    print("\n" + "="*130)
    print("COMPREHENSIVE IMBALANCE ANALYSIS WITH REAL CLASS DISTRIBUTION DATA")
    print("="*130)
    print(f"{'Dataset':<8} {'Samples':<8} {'Classes':<8} {'Val Acc':<8} {'CV Ratio':<9} {'Gini':<6} {'Entropy':<8} {'Maj/Min':<8} {'Worst Class':<12}")
    print("-"*130)
    
    for dataset_name, info in datasets_info.items():
        class_dist = info['class_distribution']
        class_accs = info['best_class_accuracy']
        
        # Calculate metrics
        cv, gini, entropy, maj_min = calculate_imbalance_metrics(class_dist)
        
        # Find worst performing class
        valid_accs = {k: v for k, v in class_accs.items() if v > 0}
        worst_acc = min(valid_accs.values()) if valid_accs else 0
        worst_class = min(valid_accs, key=valid_accs.get) if valid_accs else "N/A"
        worst_class_name = info['class_mapping'].get(worst_class, worst_class)
        
        print(f"{dataset_name:<8} {info['total_samples']:<8} {info['num_classes']:<8} "
              f"{info['best_val_accuracy']:<8.1f} {cv:<9.2f} {gini:<6.2f} {entropy:<8.2f} "
              f"{maj_min:<8.1f} {worst_class_name[:10]:<10} ({worst_acc:.1f}%)")
    
    print("-"*130)
    print("INTERPRETATIONS:")
    print("• CV Ratio: Coefficient of Variation - higher means more imbalanced")
    print("• Gini: Gini coefficient (0=perfect balance, 1=maximum imbalance)")  
    print("• Entropy: Normalized entropy (1=perfect balance, 0=maximum imbalance)")
    print("• Maj/Min: Ratio of largest to smallest class")
    print("• Worst Class: Class with lowest accuracy")
    print("="*130)
    
    # Print class distributions
    print("\nCLASS DISTRIBUTIONS:")
    print("="*80)
    for dataset_name, info in datasets_info.items():
        print(f"\n{dataset_name} Dataset:")
        class_dist = info['class_distribution']
        class_accs = info['best_class_accuracy']
        class_mapping = info['class_mapping']
        
        for class_id, class_name in class_mapping.items():
            count = class_dist.get(class_name, 0)
            acc = class_accs.get(class_id, 0)
            percentage = (count / info['total_samples']) * 100
            print(f"  {class_name:<15}: {count:>4} samples ({percentage:5.1f}%) - Accuracy: {acc:5.1f}%")

def main():
    # Get real dataset information
    datasets_info = get_real_dataset_info()
    
    # Create output directory if it doesn't exist
    output_dir = '/root/autodl-tmp/lnsde+contiformer/results/pics'
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate the corrected plot
    plot_path = os.path.join(output_dir, 'corrected_accuracy_vs_imbalance.png')
    create_stacked_accuracy_imbalance_plot(datasets_info, plot_path)
    
    # Print detailed analysis
    print_detailed_analysis_table(datasets_info)

if __name__ == "__main__":
    main()