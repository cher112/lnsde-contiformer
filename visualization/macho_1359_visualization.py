#!/usr/bin/env python3
"""
MACHO 1359数据集训练可视化脚本
基于合并的日志文件生成训练曲线（epochs 21-50）
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
import os

def configure_chinese_font():
    """配置中文字体显示"""
    # 添加字体到matplotlib管理器
    fm.fontManager.addfont('/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc')
    fm.fontManager.addfont('/usr/share/fonts/truetype/wqy/wqy-microhei.ttc')
    
    # 设置中文字体优先级列表
    plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'WenQuanYi Micro Hei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    return True

def setup_seaborn_style():
    """设置Seaborn科研绘图风格"""
    configure_chinese_font()
    
    sns.set_style("whitegrid", {
        "axes.edgecolor": "0.15",
        "axes.linewidth": 1.25,
        "axes.spines.left": True,
        "axes.spines.bottom": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.linewidth": 0.8,
        "grid.color": "0.9"
    })
    
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#7209B7']
    sns.set_palette(colors)
    
    plt.rcParams.update({
        'font.size': 11,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 11,
        'figure.titlesize': 16,
        'lines.linewidth': 2.5,
        'lines.markersize': 8
    })

def calculate_metrics_from_confusion_matrix(cm):
    """从混淆矩阵计算精确率、召回率和F1分数"""
    cm = np.array(cm)
    n_classes = cm.shape[0]
    
    # 计算每个类别的TP, FP, FN
    tp = np.diag(cm)
    fp = np.sum(cm, axis=0) - tp
    fn = np.sum(cm, axis=1) - tp
    
    # 避免除零
    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp, dtype=float), where=(tp + fp) != 0)
    recall = np.divide(tp, tp + fn, out=np.zeros_like(tp, dtype=float), where=(tp + fn) != 0)
    
    # 计算F1分数
    f1 = np.divide(2 * precision * recall, precision + recall, 
                   out=np.zeros_like(precision, dtype=float), 
                   where=(precision + recall) != 0)
    
    # 计算总体指标（微平均）
    total_tp = np.sum(tp)
    total_fp = np.sum(fp)
    total_fn = np.sum(fn)
    
    # 微平均精确率、召回率和F1
    micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0
    
    return {
        'precision_per_class': precision,
        'recall_per_class': recall,
        'f1_per_class': f1,
        'micro_precision': micro_precision * 100,
        'micro_recall': micro_recall * 100,
        'micro_f1': micro_f1 * 100,
        'macro_precision': np.mean(precision) * 100,
        'macro_recall': np.mean(recall) * 100,
        'macro_f1': np.mean(f1) * 100
    }

def load_macho_1359_data():
    """加载MACHO 1359合并的训练数据"""
    log_file = "/root/autodl-tmp/lnsde-contiformer/results/20250829/MACHO/1359/logs/MACHO_1359_geometric_config1_merged.log"
    
    with open(log_file, 'r') as f:
        data = json.load(f)
    
    training_history = data.get("training_history", {})
    
    epochs = training_history.get("epochs", [])
    train_losses = training_history.get("train_loss", [])
    train_accs = training_history.get("train_accuracy", [])
    val_losses = training_history.get("val_loss", [])
    val_accs = training_history.get("val_accuracy", [])
    train_f1s = training_history.get("train_f1", [])
    train_recalls = training_history.get("train_recall", [])
    val_f1s = training_history.get("val_f1", [])
    val_recalls = training_history.get("val_recall", [])
    learning_rates = training_history.get("learning_rates", [])
    epoch_times = training_history.get("epoch_times", [])
    confusion_matrices = training_history.get("confusion_matrices", [])
    
    return {
        'epochs': epochs,
        'train_loss': train_losses,
        'train_acc': train_accs,
        'val_loss': val_losses,
        'val_acc': val_accs,
        'train_f1': train_f1s,
        'train_recall': train_recalls,
        'val_f1': val_f1s,
        'val_recall': val_recalls,
        'learning_rates': learning_rates,
        'epoch_times': epoch_times,
        'confusion_matrices': confusion_matrices
    }

def create_training_visualization(data, output_path):
    """创建8子图训练可视化"""
    setup_seaborn_style()
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 12))
    fig.suptitle('MACHO 1359数据集训练过程可视化 (Epochs 21-50)', fontsize=18, y=0.98)
    
    epochs = data['epochs']
    
    # 1. 训练和验证损失
    ax = axes[0, 0]
    ax.plot(epochs, data['train_loss'], 'o-', label='训练损失', color='#2E86AB', linewidth=2, markersize=6)
    ax.plot(epochs, data['val_loss'], 's-', label='验证损失', color='#A23B72', linewidth=2, markersize=6)
    ax.set_title('损失函数')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 标注最优值
    min_train_loss_idx = np.argmin(data['train_loss'])
    min_val_loss_idx = np.argmin(data['val_loss'])
    ax.annotate(f'最小训练损失: {data["train_loss"][min_train_loss_idx]:.3f}', 
                xy=(epochs[min_train_loss_idx], data['train_loss'][min_train_loss_idx]), 
                xytext=(10, 10), textcoords='offset points', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    
    # 2. 训练和验证准确率
    ax = axes[0, 1]
    ax.plot(epochs, data['train_acc'], 'o-', label='训练准确率', color='#2E86AB', linewidth=2, markersize=6)
    ax.plot(epochs, data['val_acc'], 's-', label='验证准确率', color='#A23B72', linewidth=2, markersize=6)
    ax.set_title('准确率')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy (%)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 标注最优值
    max_train_acc_idx = np.argmax(data['train_acc'])
    max_val_acc_idx = np.argmax(data['val_acc'])
    ax.annotate(f'最高验证准确率: {data["val_acc"][max_val_acc_idx]:.2f}%', 
                xy=(epochs[max_val_acc_idx], data['val_acc'][max_val_acc_idx]), 
                xytext=(10, -20), textcoords='offset points', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.7))
    
    # 3. F1分数
    ax = axes[0, 2]
    ax.plot(epochs, data['train_f1'], 'o-', label='训练F1', color='#F18F01', linewidth=2, markersize=6)
    ax.plot(epochs, data['val_f1'], 's-', label='验证F1', color='#C73E1D', linewidth=2, markersize=6)
    ax.set_title('F1分数')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('F1 Score')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. 召回率
    ax = axes[0, 3]
    ax.plot(epochs, data['train_recall'], 'o-', label='训练召回率', color='#7209B7', linewidth=2, markersize=6)
    ax.plot(epochs, data['val_recall'], 's-', label='验证召回率', color='#2E86AB', linewidth=2, markersize=6)
    ax.set_title('召回率')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Recall')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 5. 学习率变化
    ax = axes[1, 0]
    ax.plot(epochs, data['learning_rates'], 'o-', color='#A23B72', linewidth=2, markersize=6)
    ax.set_title('学习率变化')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning Rate')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    # 6. Epoch训练时间
    ax = axes[1, 1]
    epoch_times_min = [t/60 for t in data['epoch_times']]  # 转换为分钟
    ax.plot(epochs, epoch_times_min, 'o-', color='#F18F01', linewidth=2, markersize=6)
    ax.set_title('每个Epoch训练时间')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('时间 (分钟)')
    ax.grid(True, alpha=0.3)
    
    # 7. 训练进度总览
    ax = axes[1, 2]
    # 归一化各指标到0-100范围进行比较
    norm_train_acc = np.array(data['train_acc'])
    norm_val_acc = np.array(data['val_acc'])
    norm_train_f1 = np.array(data['train_f1'])
    norm_val_f1 = np.array(data['val_f1'])
    
    ax.plot(epochs, norm_train_acc, '--', label='训练准确率', alpha=0.7)
    ax.plot(epochs, norm_val_acc, '--', label='验证准确率', alpha=0.7)
    ax.plot(epochs, norm_train_f1, '-', label='训练F1')
    ax.plot(epochs, norm_val_f1, '-', label='验证F1')
    ax.set_title('综合指标对比')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('指标值')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # 8. 最终混淆矩阵热图
    ax = axes[1, 3]
    if data['confusion_matrices']:
        final_cm = np.array(data['confusion_matrices'][-1])
        im = ax.imshow(final_cm, interpolation='nearest', cmap='Blues')
        ax.set_title(f'最终混淆矩阵 (Epoch {epochs[-1]})')
        
        # 添加数值标注
        for i in range(final_cm.shape[0]):
            for j in range(final_cm.shape[1]):
                ax.text(j, i, str(final_cm[i, j]), 
                       ha="center", va="center", color="black" if final_cm[i, j] < final_cm.max()/2 else "white")
        
        ax.set_xlabel('预测类别')
        ax.set_ylabel('真实类别')
        
        # 添加colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('样本数量', rotation=270, labelpad=20)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    
    # 保存图片
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"训练可视化已保存到: {output_path}")
    
    # 计算并返回最优值
    best_values = {
        'train_acc': {'value': max(data['train_acc']), 'epoch': epochs[np.argmax(data['train_acc'])]},
        'val_acc': {'value': max(data['val_acc']), 'epoch': epochs[np.argmax(data['val_acc'])]},
        'train_loss': {'value': min(data['train_loss']), 'epoch': epochs[np.argmin(data['train_loss'])]},
        'val_loss': {'value': min(data['val_loss']), 'epoch': epochs[np.argmin(data['val_loss'])]},
        'train_f1': {'value': max(data['train_f1']), 'epoch': epochs[np.argmax(data['train_f1'])]},
        'val_f1': {'value': max(data['val_f1']), 'epoch': epochs[np.argmax(data['val_f1'])]},
        'train_recall': {'value': max(data['train_recall']), 'epoch': epochs[np.argmax(data['train_recall'])]},
        'val_recall': {'value': max(data['val_recall']), 'epoch': epochs[np.argmax(data['val_recall'])]}
    }
    
    return best_values

def main():
    """主函数"""
    print("开始生成MACHO 1359数据集训练可视化...")
    
    # 加载数据
    data = load_macho_1359_data()
    print(f"成功加载数据，包含epochs: {min(data['epochs'])} - {max(data['epochs'])}")
    
    # 创建输出目录
    output_dir = "/root/autodl-tmp/lnsde-contiformer/results/pics/MACHO/"
    os.makedirs(output_dir, exist_ok=True)
    
    # 生成可视化
    output_path = os.path.join(output_dir, "macho_1359_training_epochs21-50.png")
    best_values = create_training_visualization(data, output_path)
    
    # 输出最优值总结
    print(f"\n=== MACHO 1359 训练最优值总结 (Epochs 21-50) ===")
    print(f"🏆 最高训练准确率: {best_values['train_acc']['value']:.2f}% (Epoch {best_values['train_acc']['epoch']})")
    print(f"🏆 最高验证准确率: {best_values['val_acc']['value']:.2f}% (Epoch {best_values['val_acc']['epoch']})")
    print(f"📉 最低训练损失: {best_values['train_loss']['value']:.4f} (Epoch {best_values['train_loss']['epoch']})")
    print(f"📉 最低验证损失: {best_values['val_loss']['value']:.4f} (Epoch {best_values['val_loss']['epoch']})")
    print(f"🎯 最高训练F1: {best_values['train_f1']['value']:.2f} (Epoch {best_values['train_f1']['epoch']})")
    print(f"🎯 最高验证F1: {best_values['val_f1']['value']:.2f} (Epoch {best_values['val_f1']['epoch']})")
    print(f"📊 最高训练召回率: {best_values['train_recall']['value']:.2f} (Epoch {best_values['train_recall']['epoch']})")
    print(f"📊 最高验证召回率: {best_values['val_recall']['value']:.2f} (Epoch {best_values['val_recall']['epoch']})")
    
    print(f"\n✅ MACHO 1359 训练可视化完成，图片保存至: {output_path}")

if __name__ == "__main__":
    main()