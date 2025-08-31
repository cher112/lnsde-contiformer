#!/usr/bin/env python3
import json
import matplotlib.pyplot as plt
import numpy as np
import os

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

def load_merged_log():
    """加载合并后的MACHO log文件"""
    log_path = "/autodl-fs/data/lnsde-contiformer/results/logs/MACHO/MACHO_linear_noise_config1_merged.log"
    with open(log_path, 'r') as f:
        return json.load(f)

def create_training_curves(data):
    """创建训练曲线图"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    epochs = data['training_history']['epochs']
    
    # 1. 损失曲线
    ax1.plot(epochs, data['training_history']['train_loss'], 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs, data['training_history']['val_loss'], 'r-', label='Val Loss', linewidth=2)
    ax1.set_title('MACHO Training & Validation Loss', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 准确率曲线
    ax2.plot(epochs, data['training_history']['train_accuracy'], 'b-', label='Train Accuracy', linewidth=2)
    ax2.plot(epochs, data['training_history']['val_accuracy'], 'r-', label='Val Accuracy', linewidth=2)
    ax2.set_title('MACHO Training & Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 添加最佳验证准确率标记
    best_epoch = data['best_metrics']['best_epoch']
    best_acc = data['best_metrics']['best_val_accuracy']
    ax2.axvline(x=best_epoch, color='g', linestyle='--', alpha=0.7)
    ax2.annotate(f'Best: {best_acc:.1f}% @ Epoch {best_epoch}', 
                xy=(best_epoch, best_acc), xytext=(best_epoch+2, best_acc+2),
                arrowprops=dict(arrowstyle='->', color='green'), fontsize=10)
    
    # 3. 学习率曲线
    ax3.plot(epochs[:len(data['training_history']['learning_rates'])], 
             data['training_history']['learning_rates'], 'g-', linewidth=2)
    ax3.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Learning Rate')
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale('log')
    
    # 4. 各类别准确率变化（选择最后几个epoch）
    class_names = ['Class 0', 'Class 1', 'Class 2', 'Class 3', 'Class 4', 'Class 5', 'Class 6']
    colors = plt.cm.Set3(np.linspace(0, 1, 7))
    
    # 取最后10个epoch的数据
    last_10_epochs = epochs[-10:] if len(epochs) >= 10 else epochs
    last_10_class_acc = data['training_history']['class_accuracy_history'][-10:] if len(data['training_history']['class_accuracy_history']) >= 10 else data['training_history']['class_accuracy_history']
    
    for i, class_name in enumerate(class_names):
        class_accs = [epoch_data.get(str(i), 0) for epoch_data in last_10_class_acc]
        ax4.plot(last_10_epochs, class_accs, color=colors[i], label=class_name, linewidth=2, marker='o', markersize=4)
    
    ax4.set_title('Class Accuracy (Last 10 Epochs)', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Accuracy (%)')
    ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图片
    pics_dir = "/autodl-fs/data/lnsde-contiformer/results/pics/MACHO"
    os.makedirs(pics_dir, exist_ok=True)
    plt.savefig(os.path.join(pics_dir, "macho_merged_training_curves.png"), dpi=300, bbox_inches='tight')
    print(f"训练曲线图已保存到: {pics_dir}/macho_merged_training_curves.png")
    
    return fig

def create_class_performance_analysis(data):
    """创建类别性能分析图"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 最佳性能时的类别准确率
    best_class_acc = data['best_metrics']['best_class_accuracy']
    classes = list(best_class_acc.keys())
    accuracies = list(best_class_acc.values())
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(classes)))
    bars = ax1.bar(classes, accuracies, color=colors, alpha=0.8, edgecolor='black')
    ax1.set_title('Best Class Accuracy (MACHO Dataset)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Class')
    ax1.set_ylabel('Accuracy (%)')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 在柱子上显示数值
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # 各类别准确率的变化趋势（热力图）
    class_history = data['training_history']['class_accuracy_history']
    epochs = data['training_history']['epochs']
    
    # 创建热力图数据
    heatmap_data = []
    for class_id in range(7):
        class_accs = [epoch_data.get(str(class_id), 0) for epoch_data in class_history]
        heatmap_data.append(class_accs)
    
    heatmap_data = np.array(heatmap_data)
    
    im = ax2.imshow(heatmap_data, cmap='RdYlBu_r', aspect='auto', interpolation='nearest')
    ax2.set_title('Class Accuracy Evolution', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Class')
    ax2.set_yticks(range(7))
    ax2.set_yticklabels([f'Class {i}' for i in range(7)])
    
    # 设置x轴标签（每隔5个epoch显示一次）
    step = max(1, len(epochs) // 10)
    ax2.set_xticks(range(0, len(epochs), step))
    ax2.set_xticklabels([str(epochs[i]) for i in range(0, len(epochs), step)])
    
    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax2)
    cbar.set_label('Accuracy (%)')
    
    plt.tight_layout()
    
    # 保存图片
    pics_dir = "/autodl-fs/data/lnsde-contiformer/results/pics/MACHO"
    plt.savefig(os.path.join(pics_dir, "macho_class_performance_analysis.png"), dpi=300, bbox_inches='tight')
    print(f"类别性能分析图已保存到: {pics_dir}/macho_class_performance_analysis.png")
    
    return fig

def create_summary_stats(data):
    """创建训练总结统计图"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    epochs = data['training_history']['epochs']
    train_acc = data['training_history']['train_accuracy']
    val_acc = data['training_history']['val_accuracy']
    
    # 1. 训练进程概览
    ax1.fill_between(epochs, train_acc, alpha=0.3, color='blue', label='Train Accuracy')
    ax1.fill_between(epochs, val_acc, alpha=0.3, color='red', label='Val Accuracy')
    ax1.plot(epochs, train_acc, 'b-', linewidth=2)
    ax1.plot(epochs, val_acc, 'r-', linewidth=2)
    ax1.set_title('MACHO Training Progress Overview', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy (%)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 损失分布
    train_loss = data['training_history']['train_loss']
    val_loss = data['training_history']['val_loss']
    
    ax2.hist(train_loss, bins=20, alpha=0.6, color='blue', label='Train Loss', density=True)
    ax2.hist(val_loss, bins=20, alpha=0.6, color='red', label='Val Loss', density=True)
    ax2.set_title('Loss Distribution', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Loss Value')
    ax2.set_ylabel('Density')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 准确率改进
    train_improvement = np.diff(train_acc)
    val_improvement = np.diff(val_acc)
    
    ax3.bar(epochs[1:], train_improvement, alpha=0.6, color='blue', label='Train Acc Change')
    ax3.bar(epochs[1:], val_improvement, alpha=0.6, color='red', label='Val Acc Change')
    ax3.set_title('Accuracy Improvement per Epoch', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Accuracy Change (%)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # 4. 关键统计信息
    ax4.axis('off')
    
    # 计算统计信息
    final_train_acc = train_acc[-1]
    final_val_acc = val_acc[-1]
    best_val_acc = max(val_acc)
    best_epoch = epochs[val_acc.index(best_val_acc)]
    total_epochs = len(epochs)
    
    stats_text = f"""
MACHO Dataset Training Summary

Total Epochs: {total_epochs}
Final Train Accuracy: {final_train_acc:.2f}%
Final Val Accuracy: {final_val_acc:.2f}%
Best Val Accuracy: {best_val_acc:.2f}%
Best Epoch: {best_epoch}

Model: Linear Noise SDE + ContiFormer
SDE Config: 1
"""
    
    ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes, fontsize=12, 
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    
    # 保存图片
    pics_dir = "/autodl-fs/data/lnsde-contiformer/results/pics/MACHO"
    plt.savefig(os.path.join(pics_dir, "macho_training_summary.png"), dpi=300, bbox_inches='tight')
    print(f"训练总结图已保存到: {pics_dir}/macho_training_summary.png")
    
    return fig

def main():
    """主函数"""
    print("开始创建MACHO合并数据的可视化图表...")
    
    # 加载数据
    data = load_merged_log()
    print(f"加载数据完成，共{len(data['training_history']['epochs'])}个epoch")
    
    # 创建图表
    create_training_curves(data)
    create_class_performance_analysis(data)
    create_summary_stats(data)
    
    print("所有可视化图表创建完成！")

if __name__ == "__main__":
    main()