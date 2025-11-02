#!/usr/bin/env python3
"""
ASAS实验0137可视化脚本
可视化训练曲线和混淆矩阵
"""

import json
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import seaborn as sns
from pathlib import Path

def configure_chinese_font():
    """配置中文字体显示"""
    # 添加字体到matplotlib管理器
    fm.fontManager.addfont('/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc')
    fm.fontManager.addfont('/usr/share/fonts/truetype/wqy/wqy-microhei.ttc')
    
    # 设置中文字体优先级列表
    plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'WenQuanYi Micro Hei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    return True

def load_training_log(log_path):
    """加载训练日志"""
    with open(log_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def plot_training_curves(log_data, output_dir):
    """绘制训练曲线"""
    epochs = []
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    train_f1s = []
    val_f1s = []
    train_recalls = []
    val_recalls = []
    
    # 提取数据
    for epoch_data in log_data['epochs']:
        epochs.append(epoch_data['epoch'])
        train_losses.append(epoch_data['train_loss'])
        val_losses.append(epoch_data['val_loss'])
        train_accs.append(epoch_data['train_acc'])
        val_accs.append(epoch_data['val_acc'])
        
        # F1和Recall可能不存在于早期epoch
        train_f1s.append(epoch_data.get('train_f1', 0))
        val_f1s.append(epoch_data.get('val_f1', 0))
        train_recalls.append(epoch_data.get('train_recall', 0))
        val_recalls.append(epoch_data.get('val_recall', 0))
    
    # 创建子图
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 损失曲线
    ax1.plot(epochs, train_losses, 'b-', label='训练损失', linewidth=2)
    ax1.plot(epochs, val_losses, 'r-', label='验证损失', linewidth=2)
    ax1.set_xlabel('轮次')
    ax1.set_ylabel('损失')
    ax1.set_title('损失变化曲线')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 准确率曲线
    ax2.plot(epochs, train_accs, 'b-', label='训练准确率', linewidth=2)
    ax2.plot(epochs, val_accs, 'r-', label='验证准确率', linewidth=2)
    ax2.set_xlabel('轮次')
    ax2.set_ylabel('准确率 (%)')
    ax2.set_title('准确率变化曲线')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # F1分数曲线
    ax3.plot(epochs, train_f1s, 'b-', label='训练F1分数', linewidth=2)
    ax3.plot(epochs, val_f1s, 'r-', label='验证F1分数', linewidth=2)
    ax3.set_xlabel('轮次')
    ax3.set_ylabel('F1分数 (%)')
    ax3.set_title('F1分数变化曲线')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 召回率曲线
    ax4.plot(epochs, train_recalls, 'b-', label='训练召回率', linewidth=2)
    ax4.plot(epochs, val_recalls, 'r-', label='验证召回率', linewidth=2)
    ax4.set_xlabel('轮次')
    ax4.set_ylabel('召回率 (%)')
    ax4.set_title('召回率变化曲线')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图片
    output_path = output_dir / 'asas_0137_training_curves.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"训练曲线已保存到: {output_path}")
    plt.show()

def plot_confusion_matrix(log_data, output_dir):
    """绘制最终混淆矩阵"""
    # 获取最后一个epoch的混淆矩阵
    last_epoch = log_data['epochs'][-1]
    
    if 'confusion_matrix' not in last_epoch:
        print("警告: 最后一个epoch没有混淆矩阵数据")
        return
    
    cm = np.array(last_epoch['confusion_matrix'])
    
    # ASAS数据集的类别标签
    class_names = ['RRAB', 'RRC', 'ROT', 'LPV', 'NV']
    
    plt.figure(figsize=(10, 8))
    
    # 绘制混淆矩阵
    sns.heatmap(cm, 
                annot=True, 
                fmt='d', 
                cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names,
                cbar_kws={'label': '样本数量'})
    
    plt.title(f'ASAS混淆矩阵 (第{last_epoch["epoch"]}轮)', fontsize=16, pad=20)
    plt.xlabel('预测标签', fontsize=14)
    plt.ylabel('真实标签', fontsize=14)
    
    # 计算并显示各类别准确率
    class_accuracies = []
    for i in range(len(class_names)):
        if cm[i].sum() > 0:
            acc = cm[i, i] / cm[i].sum() * 100
            class_accuracies.append(acc)
        else:
            class_accuracies.append(0)
    
    # 在图上添加准确率信息
    info_text = "各类别准确率:\n"
    for i, (name, acc) in enumerate(zip(class_names, class_accuracies)):
        info_text += f"{name}: {acc:.1f}%\n"
    
    plt.text(1.02, 0.5, info_text, transform=plt.gca().transAxes, 
             verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # 保存图片
    output_path = output_dir / 'asas_0137_confusion_matrix.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"混淆矩阵已保存到: {output_path}")
    plt.show()

def print_training_summary(log_data):
    """打印训练摘要"""
    print("=" * 60)
    print("ASAS实验0137训练摘要")
    print("=" * 60)
    
    first_epoch = log_data['epochs'][0]
    last_epoch = log_data['epochs'][-1]
    
    print(f"数据集: {log_data['dataset']}")
    print(f"模型类型: {log_data['model_type']}")
    print(f"开始时间: {log_data['start_time']}")
    print(f"总轮次: {len(log_data['epochs'])}")
    print()
    
    print("初始性能:")
    print(f"  训练损失: {first_epoch['train_loss']:.4f}")
    print(f"  验证损失: {first_epoch['val_loss']:.4f}")
    print(f"  训练准确率: {first_epoch['train_acc']:.2f}%")
    print(f"  验证准确率: {first_epoch['val_acc']:.2f}%")
    print()
    
    print("最终性能:")
    print(f"  训练损失: {last_epoch['train_loss']:.4f}")
    print(f"  验证损失: {last_epoch['val_loss']:.4f}")
    print(f"  训练准确率: {last_epoch['train_acc']:.2f}%")
    print(f"  验证准确率: {last_epoch['val_acc']:.2f}%")
    
    if 'val_f1' in last_epoch:
        print(f"  验证F1分数: {last_epoch['val_f1']:.2f}%")
    if 'val_recall' in last_epoch:
        print(f"  验证召回率: {last_epoch['val_recall']:.2f}%")
    
    print()
    
    # 找到最佳验证准确率
    best_val_acc = max(epoch['val_acc'] for epoch in log_data['epochs'])
    best_epoch = next(epoch for epoch in log_data['epochs'] if epoch['val_acc'] == best_val_acc)
    
    print(f"最佳验证准确率: {best_val_acc:.2f}% (第{best_epoch['epoch']}轮)")
    print("=" * 60)

def main():
    # 配置中文字体
    configure_chinese_font()
    
    # 文件路径
    log_path = "/root/autodl-tmp/lnsde-contiformer/results/20250902/ASAS/0137/logs/ASAS_modellinear_noise_sde_cf_config1_lr1e-04_bs50_hc128_cd256.log"
    output_dir = Path("/root/autodl-tmp/lnsde-contiformer/results/pics/ASAS")
    
    # 确保输出目录存在
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载训练日志
    try:
        log_data = load_training_log(log_path)
        print(f"成功加载训练日志: {log_path}")
    except Exception as e:
        print(f"加载训练日志失败: {e}")
        return
    
    # 打印训练摘要
    print_training_summary(log_data)
    
    # 绘制训练曲线
    print("\n正在生成训练曲线...")
    plot_training_curves(log_data, output_dir)
    
    # 绘制混淆矩阵
    print("\n正在生成混淆矩阵...")
    plot_confusion_matrix(log_data, output_dir)
    
    print(f"\n所有图片已保存到: {output_dir}")

if __name__ == "__main__":
    main()