#!/usr/bin/env python3
import json
import matplotlib.pyplot as plt
import numpy as np
import os

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

def load_linear_history():
    """加载LINEAR历史日志文件"""
    log_path = "/root/autodl-tmp/lnsde+contiformer/results/logs/LINEAR/LINEAR_linear_noise_history.json"
    with open(log_path, 'r') as f:
        return json.load(f)

def create_training_curves(data):
    """创建训练曲线图"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    epochs = list(range(1, len(data['train_losses']) + 1))
    
    # 1. 损失曲线
    ax1.plot(epochs, data['train_losses'], 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs, data['val_losses'], 'r-', label='Val Loss', linewidth=2)
    ax1.set_title('LINEAR Training & Validation Loss', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 准确率曲线
    ax2.plot(epochs, data['train_accs'], 'b-', label='Train Accuracy', linewidth=2)
    ax2.plot(epochs, data['val_accs'], 'r-', label='Val Accuracy', linewidth=2)
    ax2.set_title('LINEAR Training & Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 添加最佳验证准确率标记
    best_val_acc = data['best_val_acc']
    best_epoch = data['val_accs'].index(best_val_acc) + 1
    ax2.axvline(x=best_epoch, color='g', linestyle='--', alpha=0.7)
    ax2.annotate(f'Best: {best_val_acc:.2f}% @ Epoch {best_epoch}', 
                xy=(best_epoch, best_val_acc), xytext=(best_epoch+5, best_val_acc+1),
                arrowprops=dict(arrowstyle='->', color='green'), fontsize=10)
    
    # 3. 损失平滑曲线（移动平均）
    window_size = 10
    if len(data['train_losses']) >= window_size:
        train_loss_smooth = np.convolve(data['train_losses'], 
                                      np.ones(window_size)/window_size, mode='valid')
        val_loss_smooth = np.convolve(data['val_losses'], 
                                    np.ones(window_size)/window_size, mode='valid')
        epochs_smooth = epochs[window_size-1:]
        
        ax3.plot(epochs_smooth, train_loss_smooth, 'b-', label='Train Loss (Smoothed)', linewidth=2)
        ax3.plot(epochs_smooth, val_loss_smooth, 'r-', label='Val Loss (Smoothed)', linewidth=2)
    else:
        ax3.plot(epochs, data['train_losses'], 'b-', label='Train Loss', linewidth=2)
        ax3.plot(epochs, data['val_losses'], 'r-', label='Val Loss', linewidth=2)
    
    ax3.set_title('LINEAR Loss Curves (Smoothed)', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Loss')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 准确率改进趋势
    train_acc_diff = np.diff(data['train_accs'])
    val_acc_diff = np.diff(data['val_accs'])
    
    ax4.bar(epochs[1:], train_acc_diff, alpha=0.6, color='blue', label='Train Acc Change', width=0.8)
    ax4.bar(epochs[1:], val_acc_diff, alpha=0.6, color='red', label='Val Acc Change', width=0.8)
    ax4.set_title('LINEAR Accuracy Change per Epoch', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Accuracy Change (%)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图片
    pics_dir = "/root/autodl-tmp/lnsde+contiformer/results/pics/LINEAR"
    os.makedirs(pics_dir, exist_ok=True)
    plt.savefig(os.path.join(pics_dir, "linear_history_training_curves.png"), dpi=300, bbox_inches='tight')
    print(f"LINEAR训练曲线图已保存到: {pics_dir}/linear_history_training_curves.png")
    
    return fig

def create_performance_analysis(data):
    """创建性能分析图"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    epochs = list(range(1, len(data['train_losses']) + 1))
    
    # 1. 学习效率分析 - 每个epoch的准确率提升
    train_acc_diff = np.diff([0] + data['train_accs'])  # 添加初始0来计算第一个epoch的提升
    val_acc_diff = np.diff([0] + data['val_accs'])
    
    # 使用颜色编码显示正负变化
    colors_train = ['green' if x > 0 else 'red' if x < 0 else 'gray' for x in train_acc_diff]
    colors_val = ['darkgreen' if x > 0 else 'darkred' if x < 0 else 'darkgray' for x in val_acc_diff]
    
    ax1.scatter(epochs, train_acc_diff, c=colors_train, alpha=0.6, s=30, label='Train Acc Change')
    ax1.scatter(epochs, val_acc_diff, c=colors_val, alpha=0.8, s=30, marker='^', label='Val Acc Change')
    ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax1.set_title('LINEAR Learning Efficiency per Epoch', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy Change (%)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 添加平滑趋势线
    from scipy import signal
    if len(val_acc_diff) > 10:
        smoothed_val = signal.savgol_filter(val_acc_diff, 11, 3)
        ax1.plot(epochs, smoothed_val, 'purple', linewidth=2, alpha=0.7, label='Val Trend')
        ax1.legend()
    
    # 2. 性能指标热力图
    # 创建一个综合性能矩阵
    performance_metrics = []
    window_size = 5
    
    for i in range(len(epochs)):
        # 计算局部窗口的平均性能
        start_idx = max(0, i - window_size // 2)
        end_idx = min(len(epochs), i + window_size // 2 + 1)
        
        local_train_acc = np.mean(data['train_accs'][start_idx:end_idx])
        local_val_acc = np.mean(data['val_accs'][start_idx:end_idx])
        local_train_loss = np.mean(data['train_losses'][start_idx:end_idx])
        local_val_loss = np.mean(data['val_losses'][start_idx:end_idx])
        
        performance_metrics.append([local_train_acc, local_val_acc, 1/local_train_loss*10, 1/local_val_loss*10])
    
    performance_matrix = np.array(performance_metrics).T
    
    im = ax2.imshow(performance_matrix, cmap='RdYlBu_r', aspect='auto', interpolation='bilinear')
    ax2.set_title('LINEAR Performance Heatmap', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Metrics')
    ax2.set_yticks(range(4))
    ax2.set_yticklabels(['Train Acc', 'Val Acc', 'Train Loss⁻¹', 'Val Loss⁻¹'])
    
    # 设置x轴标签
    step = max(1, len(epochs) // 10)
    ax2.set_xticks(range(0, len(epochs), step))
    ax2.set_xticklabels([str(epochs[i]) for i in range(0, len(epochs), step)])
    
    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax2, shrink=0.8)
    cbar.set_label('Performance Score')
    
    # 3. 损失-准确率相关性分析
    ax3.scatter(data['train_losses'], data['train_accs'], alpha=0.6, color='blue', 
                s=20, label='Train: Loss vs Acc')
    ax3.scatter(data['val_losses'], data['val_accs'], alpha=0.6, color='red', 
                s=20, label='Val: Loss vs Acc')
    
    # 添加趋势线
    from scipy.stats import linregress
    train_slope, train_intercept, train_r, _, _ = linregress(data['train_losses'], data['train_accs'])
    val_slope, val_intercept, val_r, _, _ = linregress(data['val_losses'], data['val_accs'])
    
    train_line = np.array(data['train_losses']) * train_slope + train_intercept
    val_line = np.array(data['val_losses']) * val_slope + val_intercept
    
    ax3.plot(data['train_losses'], train_line, 'b--', alpha=0.8, 
             label=f'Train Trend (R²={train_r**2:.3f})')
    ax3.plot(data['val_losses'], val_line, 'r--', alpha=0.8, 
             label=f'Val Trend (R²={val_r**2:.3f})')
    
    ax3.set_title('LINEAR Loss-Accuracy Correlation', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Loss')
    ax3.set_ylabel('Accuracy (%)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 训练里程碑和关键统计
    ax4.axis('off')
    
    # 计算关键统计信息
    best_val_acc = data['best_val_acc']
    best_epoch = data['val_accs'].index(best_val_acc) + 1
    final_train_acc = data['train_accs'][-1]
    final_val_acc = data['val_accs'][-1]
    
    # 计算性能提升里程碑
    val_milestones = []
    for threshold in [80, 85, 87, 89]:
        for i, acc in enumerate(data['val_accs']):
            if acc >= threshold:
                val_milestones.append((threshold, i + 1))
                break
    
    # 计算收敛性指标
    late_epochs = data['val_accs'][-20:] if len(data['val_accs']) >= 20 else data['val_accs'][-10:]
    stability = np.std(late_epochs)
    
    # 计算过拟合程度
    overfitting_score = np.mean([t - v for t, v in zip(data['train_accs'][-10:], data['val_accs'][-10:])])
    
    stats_text = f"""LINEAR Dataset Performance Analysis

[Best Performance]
   • Best Val Accuracy: {best_val_acc:.2f}% @ Epoch {best_epoch}
   • Final Val Accuracy: {final_val_acc:.2f}%
   • Final Train Accuracy: {final_train_acc:.2f}%

[Learning Milestones]"""
    
    for threshold, epoch in val_milestones:
        stats_text += f"\n   • {threshold}% accuracy reached @ Epoch {epoch}"
    
    stats_text += f"""

[Model Analysis]
   • Late-stage Stability: {stability:.2f}% (std of last epochs)
   • Overfitting Score: {overfitting_score:.2f}% (train-val gap)
   • Loss-Acc Correlation: R²={val_r**2:.3f}

[Training Efficiency]
   • Total Epochs: {len(epochs)}
   • Avg Val Improvement: {np.mean([x for x in val_acc_diff if x > 0]):.3f}%/epoch
   • Best Improvement: {max(val_acc_diff):.2f}% @ Epoch {np.argmax(val_acc_diff) + 1}
"""
    
    ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes, fontsize=10, 
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=1', facecolor='lightcyan', alpha=0.8))
    
    plt.tight_layout()
    
    # 保存图片
    pics_dir = "/root/autodl-tmp/lnsde+contiformer/results/pics/LINEAR"
    plt.savefig(os.path.join(pics_dir, "linear_history_performance_analysis.png"), dpi=300, bbox_inches='tight')
    print(f"LINEAR性能分析图已保存到: {pics_dir}/linear_history_performance_analysis.png")
    
    return fig

def create_convergence_analysis(data):
    """创建收敛分析图"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    epochs = list(range(1, len(data['train_losses']) + 1))
    
    # 1. 学习曲线对数尺度
    ax1.semilogy(epochs, data['train_losses'], 'b-', label='Train Loss', linewidth=2)
    ax1.semilogy(epochs, data['val_losses'], 'r-', label='Val Loss', linewidth=2)
    ax1.set_title('LINEAR Loss Curves (Log Scale)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss (Log Scale)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 梯度分析（损失差分作为梯度代理）
    train_loss_grad = np.abs(np.diff(data['train_losses']))
    val_loss_grad = np.abs(np.diff(data['val_losses']))
    
    ax2.plot(epochs[1:], train_loss_grad, 'b-', label='Train Loss Gradient', linewidth=2)
    ax2.plot(epochs[1:], val_loss_grad, 'r-', label='Val Loss Gradient', linewidth=2)
    ax2.set_title('LINEAR Loss Gradient Magnitude', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('|ΔLoss|')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    
    # 3. 准确率收敛
    # 计算距离最佳准确率的差距
    best_val_acc = data['best_val_acc']
    val_acc_gap = [best_val_acc - acc for acc in data['val_accs']]
    
    ax3.plot(epochs, val_acc_gap, 'r-', linewidth=2, label='Gap to Best Val Acc')
    ax3.set_title('LINEAR Validation Accuracy Convergence', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Gap to Best Accuracy (%)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale('log')
    
    # 4. 过拟合分析
    overfitting_gap = [train_acc - val_acc for train_acc, val_acc in 
                      zip(data['train_accs'], data['val_accs'])]
    
    ax4.plot(epochs, overfitting_gap, 'purple', linewidth=2, label='Train - Val Acc Gap')
    ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax4.set_title('LINEAR Overfitting Analysis', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Train - Val Accuracy (%)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 标注过拟合程度
    final_gap = overfitting_gap[-1]
    max_gap = max(overfitting_gap)
    mean_gap = np.mean(overfitting_gap)
    
    ax4.annotate(f'Final Gap: {final_gap:.2f}%\nMax Gap: {max_gap:.2f}%\nMean Gap: {mean_gap:.2f}%', 
                xy=(0.7, 0.7), xycoords='axes fraction',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
    
    plt.tight_layout()
    
    # 保存图片
    pics_dir = "/root/autodl-tmp/lnsde+contiformer/results/pics/LINEAR"
    plt.savefig(os.path.join(pics_dir, "linear_history_convergence_analysis.png"), dpi=300, bbox_inches='tight')
    print(f"LINEAR收敛分析图已保存到: {pics_dir}/linear_history_convergence_analysis.png")
    
    return fig

def main():
    """主函数"""
    print("开始创建LINEAR历史数据的可视化图表...")
    
    # 加载数据
    data = load_linear_history()
    print(f"加载数据完成，共{len(data['train_losses'])}个epoch")
    
    # 创建图表
    print("创建训练曲线图...")
    create_training_curves(data)
    
    print("创建性能分析图...")
    create_performance_analysis(data)
    
    print("创建收敛分析图...")
    create_convergence_analysis(data)
    
    print("所有LINEAR可视化图表创建完成！")
    
    # 显示数据摘要
    print("\n=== LINEAR训练数据摘要 ===")
    print(f"最佳验证准确率: {data['best_val_acc']:.2f}%")
    print(f"最终训练准确率: {data['train_accs'][-1]:.2f}%")
    print(f"最终验证准确率: {data['val_accs'][-1]:.2f}%")
    print(f"最终训练损失: {data['train_losses'][-1]:.4f}")
    print(f"最终验证损失: {data['val_losses'][-1]:.4f}")

if __name__ == "__main__":
    main()