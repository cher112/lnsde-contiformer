#!/usr/bin/env python3

import json
import matplotlib.pyplot as plt
import numpy as np
import os
from collections import Counter

def load_log_data(log_path):
    """Load and parse JSON log data"""
    with open(log_path, 'r') as f:
        return json.load(f)

def calculate_class_imbalance_metrics(class_accuracy_data):
    """
    Calculate imbalance metrics from class accuracy data
    
    Args:
        class_accuracy_data: Dict with class IDs as keys and accuracy as values
        
    Returns:
        class_variance: Variance of class accuracies
        imbalance_ratio: Standard deviation of class accuracies (higher = more imbalanced)
        avg_f1_approx: Approximated average F1 score based on class accuracies
        min_class_acc: Minimum class accuracy (indicator of worst performing class)
    """
    # Remove classes with 0% accuracy as they might not have samples
    valid_class_accs = [acc for acc in class_accuracy_data.values() if acc > 0]
    
    if not valid_class_accs:
        return 100.0, 100.0, 0.0, 0.0  # Completely imbalanced
    
    # Class variance: variance of class accuracies
    class_variance = np.var(valid_class_accs)
    
    # Imbalance ratio: standard deviation of class accuracies
    imbalance_ratio = np.std(valid_class_accs)
    
    # Approximate F1 score (assuming accuracy approximates F1 for balanced precision/recall)
    avg_f1_approx = np.mean(valid_class_accs) / 100.0  # Convert to 0-1 range
    
    # Minimum class accuracy (worst performing class)
    min_class_acc = min(valid_class_accs) / 100.0  # Convert to 0-1 range
    
    return class_variance, imbalance_ratio, avg_f1_approx, min_class_acc

def calculate_auroc_approximation(val_accuracy, class_accuracy_data):
    """
    Approximate AUROC based on overall accuracy and class balance
    This is a rough approximation since we don't have actual ROC data
    """
    # For multiclass: approximate AUROC using overall accuracy and class balance
    valid_class_accs = [acc for acc in class_accuracy_data.values() if acc > 0]
    
    if not valid_class_accs:
        return 0.5  # Random guessing
    
    # Weight overall accuracy by class balance (more balanced = closer to actual AUROC)
    class_balance_penalty = np.std(valid_class_accs) / 100.0  # Normalize to 0-1
    
    # Simple approximation: start with accuracy, adjust for class imbalance
    auroc_approx = (val_accuracy / 100.0) * (1 - class_balance_penalty * 0.3)
    
    # Ensure AUROC is between 0.5 and 1.0
    auroc_approx = max(0.5, min(1.0, auroc_approx))
    
    return auroc_approx

def create_accuracy_imbalance_plot(datasets_data, save_path):
    """Create three separate dual-axis plots showing accuracy vs different imbalance metrics"""
    
    datasets = list(datasets_data.keys())
    n_datasets = len(datasets)
    
    # Calculate metrics for each dataset
    metrics_data = {}
    
    for dataset_name, data in datasets_data.items():
        # Get latest values
        val_accuracy = data['training_history']['val_accuracy'][-1]
        class_acc_history = data['training_history']['class_accuracy_history']
        latest_class_acc = class_acc_history[-1] if class_acc_history else {}
        
        # Calculate metrics
        class_variance, imbalance_ratio, f1_approx, min_class_acc = calculate_class_imbalance_metrics(latest_class_acc)
        auroc_approx = calculate_auroc_approximation(val_accuracy, latest_class_acc)
        
        metrics_data[dataset_name] = {
            'val_accuracy': val_accuracy,
            'auroc': auroc_approx * 100,  # Convert to percentage
            'class_variance': class_variance,
            'imbalance_ratio': imbalance_ratio
        }
    
    # Create the figure with 3 subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    x_pos = np.arange(n_datasets)
    width = 0.35
    
    val_accs = [metrics_data[d]['val_accuracy'] for d in datasets]
    aurocs = [metrics_data[d]['auroc'] for d in datasets]
    class_variances = [metrics_data[d]['class_variance'] for d in datasets]
    imbalance_ratios = [metrics_data[d]['imbalance_ratio'] for d in datasets]
    
    # Plot 1: Validation Accuracy vs AUROC
    ax1_left = ax1
    bars1_left = ax1_left.bar(x_pos - width/2, val_accs, width, label='Val Accuracy', alpha=0.8, color='skyblue')
    ax1_left.set_xlabel('Dataset', fontweight='bold', fontsize=12)
    ax1_left.set_ylabel('Validation Accuracy (%)', color='blue', fontweight='bold', fontsize=12)
    ax1_left.tick_params(axis='y', labelcolor='blue')
    ax1_left.set_xticks(x_pos)
    ax1_left.set_xticklabels(datasets)
    ax1_left.set_title('Validation Accuracy vs AUROC', fontweight='bold', fontsize=14)
    
    ax1_right = ax1_left.twinx()
    bars1_right = ax1_right.bar(x_pos + width/2, aurocs, width, label='AUROC', alpha=0.8, color='lightcoral')
    ax1_right.set_ylabel('AUROC (%)', color='red', fontweight='bold', fontsize=12)
    ax1_right.tick_params(axis='y', labelcolor='red')
    
    # Add value labels for plot 1
    for i, (acc, auroc) in enumerate(zip(val_accs, aurocs)):
        ax1_left.text(i - width/2, acc + 1, f'{acc:.1f}%', ha='center', va='bottom', fontsize=10, color='blue', fontweight='bold')
        ax1_right.text(i + width/2, auroc + 1, f'{auroc:.1f}%', ha='center', va='bottom', fontsize=10, color='red', fontweight='bold')
    
    ax1_left.grid(True, alpha=0.3)
    
    # Plot 2: Validation Accuracy vs Class Variance
    ax2_left = ax2
    bars2_left = ax2_left.bar(x_pos - width/2, val_accs, width, label='Val Accuracy', alpha=0.8, color='skyblue')
    ax2_left.set_xlabel('Dataset', fontweight='bold', fontsize=12)
    ax2_left.set_ylabel('Validation Accuracy (%)', color='blue', fontweight='bold', fontsize=12)
    ax2_left.tick_params(axis='y', labelcolor='blue')
    ax2_left.set_xticks(x_pos)
    ax2_left.set_xticklabels(datasets)
    ax2_left.set_title('Validation Accuracy vs Class Variance', fontweight='bold', fontsize=14)
    
    ax2_right = ax2_left.twinx()
    bars2_right = ax2_right.bar(x_pos + width/2, class_variances, width, label='Class Variance', alpha=0.8, color='lightgreen')
    ax2_right.set_ylabel('Class Accuracy Variance', color='green', fontweight='bold', fontsize=12)
    ax2_right.tick_params(axis='y', labelcolor='green')
    
    # Add value labels for plot 2
    for i, (acc, var) in enumerate(zip(val_accs, class_variances)):
        ax2_left.text(i - width/2, acc + 1, f'{acc:.1f}%', ha='center', va='bottom', fontsize=10, color='blue', fontweight='bold')
        ax2_right.text(i + width/2, var + max(class_variances)*0.02, f'{var:.1f}', ha='center', va='bottom', fontsize=10, color='green', fontweight='bold')
    
    ax2_left.grid(True, alpha=0.3)
    
    # Plot 3: Validation Accuracy vs Imbalance Ratio
    ax3_left = ax3
    bars3_left = ax3_left.bar(x_pos - width/2, val_accs, width, label='Val Accuracy', alpha=0.8, color='skyblue')
    ax3_left.set_xlabel('Dataset', fontweight='bold', fontsize=12)
    ax3_left.set_ylabel('Validation Accuracy (%)', color='blue', fontweight='bold', fontsize=12)
    ax3_left.tick_params(axis='y', labelcolor='blue')
    ax3_left.set_xticks(x_pos)
    ax3_left.set_xticklabels(datasets)
    ax3_left.set_title('Validation Accuracy vs Imbalance Ratio', fontweight='bold', fontsize=14)
    
    ax3_right = ax3_left.twinx()
    bars3_right = ax3_right.bar(x_pos + width/2, imbalance_ratios, width, label='Imbalance Ratio', alpha=0.8, color='orange')
    ax3_right.set_ylabel('Imbalance Ratio (Std Dev)', color='darkorange', fontweight='bold', fontsize=12)
    ax3_right.tick_params(axis='y', labelcolor='darkorange')
    
    # Add value labels for plot 3
    for i, (acc, ratio) in enumerate(zip(val_accs, imbalance_ratios)):
        ax3_left.text(i - width/2, acc + 1, f'{acc:.1f}%', ha='center', va='bottom', fontsize=10, color='blue', fontweight='bold')
        ax3_right.text(i + width/2, ratio + max(imbalance_ratios)*0.02, f'{ratio:.1f}', ha='center', va='bottom', fontsize=10, color='darkorange', fontweight='bold')
    
    ax3_left.grid(True, alpha=0.3)
    
    # Add legends for each plot
    lines1, labels1 = ax1_left.get_legend_handles_labels()
    lines2, labels2 = ax1_right.get_legend_handles_labels()
    ax1_left.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    lines3, labels3 = ax2_left.get_legend_handles_labels()
    lines4, labels4 = ax2_right.get_legend_handles_labels()
    ax2_left.legend(lines3 + lines4, labels3 + labels4, loc='upper left')
    
    lines5, labels5 = ax3_left.get_legend_handles_labels()
    lines6, labels6 = ax3_right.get_legend_handles_labels()
    ax3_left.legend(lines5 + lines6, labels5 + labels6, loc='upper left')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Accuracy vs Imbalance analysis plot saved to: {save_path}")

def print_detailed_metrics_table(datasets_data):
    """Print detailed metrics table"""
    print("\n" + "="*110)
    print("DETAILED PERFORMANCE AND IMBALANCE METRICS")
    print("="*110)
    print(f"{'Dataset':<8} {'Val Acc':<8} {'AUROC*':<8} {'F1*':<8} {'Class Var':<10} {'Imbal Ratio':<12} {'Min Class':<10} {'Num Classes':<12}")
    print("-"*110)
    
    for dataset_name, data in datasets_data.items():
        val_accuracy = data['training_history']['val_accuracy'][-1]
        class_acc_history = data['training_history']['class_accuracy_history']
        latest_class_acc = class_acc_history[-1] if class_acc_history else {}
        
        # Calculate metrics
        class_variance, imbalance_ratio, f1_approx, min_class_acc = calculate_class_imbalance_metrics(latest_class_acc)
        auroc_approx = calculate_auroc_approximation(val_accuracy, latest_class_acc)
        
        num_classes = len([acc for acc in latest_class_acc.values() if acc > 0])
        
        print(f"{dataset_name:<8} {val_accuracy:<8.1f} {auroc_approx*100:<8.1f} {f1_approx*100:<8.1f} "
              f"{class_variance:<10.1f} {imbalance_ratio:<12.1f} {min_class_acc*100:<10.1f} {num_classes:<12}")
    
    print("-"*110)
    print("* AUROC and F1 scores are approximated from class accuracy data")
    print("* Class Var: Variance of class accuracies (higher = more imbalanced)")
    print("* Imbalance Ratio: Standard deviation of class accuracies (higher = more imbalanced)")
    print("* Min Class: Accuracy of worst performing class")
    print("="*110)

def main():
    # Define log file paths - use JSON files for ASAS (more recent data)
    log_files = {
        'LINEAR': '/root/autodl-tmp/lnsde+contiformer/results/logs/LINEAR/LINEAR_linear_noise_config1_20250824_140843.log',
        'ASAS': '/root/autodl-tmp/lnsde+contiformer/results/logs/ASAS/ASAS_linear_noise_config1_20250824_000718.json',
        'MACHO': '/root/autodl-tmp/lnsde+contiformer/results/logs/MACHO/MACHO_linear_noise_config1_20250824_135530.log'
    }
    
    # Load data from all datasets
    datasets_data = {}
    for dataset_name, log_path in log_files.items():
        if os.path.exists(log_path):
            datasets_data[dataset_name] = load_log_data(log_path)
            print(f"Loaded data for {dataset_name}")
        else:
            print(f"Warning: Log file not found for {dataset_name}: {log_path}")
    
    if not datasets_data:
        print("No data loaded. Exiting.")
        return
    
    # Create output directory if it doesn't exist
    output_dir = '/root/autodl-tmp/lnsde+contiformer/results/pics'  # 通用输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate the accuracy vs imbalance plot
    plot_path = os.path.join(output_dir, 'accuracy_vs_imbalance_analysis.png')
    create_accuracy_imbalance_plot(datasets_data, plot_path)
    
    # Print detailed metrics
    print_detailed_metrics_table(datasets_data)

if __name__ == "__main__":
    main()