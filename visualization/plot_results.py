#!/usr/bin/env python3

import json
import matplotlib.pyplot as plt
import numpy as np
import os

def load_log_data(log_path):
    """Load and parse JSON log data"""
    with open(log_path, 'r') as f:
        return json.load(f)

def plot_accuracy_comparison(datasets_data, save_path):
    """Plot training and validation accuracy comparison for all datasets"""
    plt.figure(figsize=(15, 10))
    
    # Plot training accuracy
    plt.subplot(2, 2, 1)
    for dataset_name, data in datasets_data.items():
        epochs = data['training_history']['epochs']
        train_acc = data['training_history']['train_accuracy']
        plt.plot(epochs, train_acc, label=f'{dataset_name}', linewidth=2)
    
    plt.title('Training Accuracy Comparison', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot validation accuracy
    plt.subplot(2, 2, 2)
    for dataset_name, data in datasets_data.items():
        epochs = data['training_history']['epochs']
        val_acc = data['training_history']['val_accuracy']
        plt.plot(epochs, val_acc, label=f'{dataset_name}', linewidth=2)
    
    plt.title('Validation Accuracy Comparison', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot training loss
    plt.subplot(2, 2, 3)
    for dataset_name, data in datasets_data.items():
        epochs = data['training_history']['epochs']
        train_loss = data['training_history']['train_loss']
        plt.plot(epochs, train_loss, label=f'{dataset_name}', linewidth=2)
    
    plt.title('Training Loss Comparison', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot validation loss
    plt.subplot(2, 2, 4)
    for dataset_name, data in datasets_data.items():
        epochs = data['training_history']['epochs']
        val_loss = data['training_history']['val_loss']
        plt.plot(epochs, val_loss, label=f'{dataset_name}', linewidth=2)
    
    plt.title('Validation Loss Comparison', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Comparison plot saved to: {save_path}")

def plot_latest_metrics_bar(datasets_data, save_path):
    """Plot bar chart of latest accuracy and loss for each dataset"""
    datasets = list(datasets_data.keys())
    latest_train_acc = []
    latest_val_acc = []
    latest_train_loss = []
    latest_val_loss = []
    
    for dataset_name, data in datasets_data.items():
        train_acc = data['training_history']['train_accuracy']
        val_acc = data['training_history']['val_accuracy']
        train_loss = data['training_history']['train_loss']
        val_loss = data['training_history']['val_loss']
        
        latest_train_acc.append(train_acc[-1])
        latest_val_acc.append(val_acc[-1])
        latest_train_loss.append(train_loss[-1])
        latest_val_loss.append(val_loss[-1])
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    x = np.arange(len(datasets))
    width = 0.35
    
    # Latest Training vs Validation Accuracy
    ax1.bar(x - width/2, latest_train_acc, width, label='Training Acc', alpha=0.8)
    ax1.bar(x + width/2, latest_val_acc, width, label='Validation Acc', alpha=0.8)
    ax1.set_title('Latest Training vs Validation Accuracy', fontweight='bold')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(datasets)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, (train, val) in enumerate(zip(latest_train_acc, latest_val_acc)):
        ax1.text(i - width/2, train + 0.5, f'{train:.1f}', ha='center', va='bottom', fontsize=9)
        ax1.text(i + width/2, val + 0.5, f'{val:.1f}', ha='center', va='bottom', fontsize=9)
    
    # Latest Training vs Validation Loss
    ax2.bar(x - width/2, latest_train_loss, width, label='Training Loss', alpha=0.8)
    ax2.bar(x + width/2, latest_val_loss, width, label='Validation Loss', alpha=0.8)
    ax2.set_title('Latest Training vs Validation Loss', fontweight='bold')
    ax2.set_ylabel('Loss')
    ax2.set_xticks(x)
    ax2.set_xticklabels(datasets)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, (train, val) in enumerate(zip(latest_train_loss, latest_val_loss)):
        ax2.text(i - width/2, train + 0.002, f'{train:.3f}', ha='center', va='bottom', fontsize=9)
        ax2.text(i + width/2, val + 0.002, f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    
    # Latest Validation Accuracy only
    bars3 = ax3.bar(datasets, latest_val_acc, color=['skyblue', 'lightcoral', 'lightgreen'], alpha=0.8)
    ax3.set_title('Latest Validation Accuracy by Dataset', fontweight='bold')
    ax3.set_ylabel('Accuracy (%)')
    ax3.grid(True, alpha=0.3)
    
    for bar, acc in zip(bars3, latest_val_acc):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                f'{acc:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Latest Validation Loss only
    bars4 = ax4.bar(datasets, latest_val_loss, color=['skyblue', 'lightcoral', 'lightgreen'], alpha=0.8)
    ax4.set_title('Latest Validation Loss by Dataset', fontweight='bold')
    ax4.set_ylabel('Loss')
    ax4.grid(True, alpha=0.3)
    
    for bar, loss in zip(bars4, latest_val_loss):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002, 
                f'{loss:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Latest metrics bar chart saved to: {save_path}")

def print_summary_table(datasets_data):
    """Print a summary table of latest metrics"""
    print("\n" + "="*80)
    print("LATEST METRICS SUMMARY")
    print("="*80)
    print(f"{'Dataset':<10} {'Train Acc':<10} {'Val Acc':<10} {'Train Loss':<12} {'Val Loss':<12} {'Best Val Acc':<12}")
    print("-"*80)
    
    for dataset_name, data in datasets_data.items():
        train_acc = data['training_history']['train_accuracy'][-1]
        val_acc = data['training_history']['val_accuracy'][-1]
        train_loss = data['training_history']['train_loss'][-1]
        val_loss = data['training_history']['val_loss'][-1]
        best_val_acc = data.get('best_metrics', {}).get('best_val_accuracy', data.get('best_val_accuracy', 'N/A'))
        
        print(f"{dataset_name:<10} {train_acc:<10.2f} {val_acc:<10.2f} {train_loss:<12.4f} {val_loss:<12.4f} {best_val_acc:<12}")
    
    print("="*80)

def main():
    # Define log file paths - use JSON files for ASAS (more recent data)
    log_files = {
        'LINEAR': '/root/autodl-tmp/lnsde+contiformer/results/logs/LINEAR_linear_noise_config1_20250824_140843.log',
        'ASAS': '/root/autodl-tmp/lnsde+contiformer/results/logs/ASAS_linear_noise_config1_20250824_000718.json',  # JSON has 98% accuracy
        'MACHO': '/root/autodl-tmp/lnsde+contiformer/results/logs/MACHO_linear_noise_config1_20250824_135530.log'
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
    output_dir = '/root/autodl-tmp/lnsde+contiformer/results/pics'
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate plots
    comparison_path = os.path.join(output_dir, 'training_comparison_all_datasets.png')
    plot_accuracy_comparison(datasets_data, comparison_path)
    
    latest_metrics_path = os.path.join(output_dir, 'latest_metrics_comparison.png')
    plot_latest_metrics_bar(datasets_data, latest_metrics_path)
    
    # Print summary table
    print_summary_table(datasets_data)
    
    print(f"\nAll plots saved to: {output_dir}")

if __name__ == "__main__":
    main()