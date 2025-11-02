#!/usr/bin/env python3
import json
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
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

def load_merged_log():
    """加载MACHO 0-200 epoch合并后的日志文件"""
    log_path = "/root/autodl-tmp/lnsde-contiformer/results/20250901/MACHO/MACHO_0_200_epochs_merged.log"
    with open(log_path, 'r') as f:
        return json.load(f)

def create_training_curves(data):
    """创建训练曲线图"""
    print("创建训练曲线图...")
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 提取数据
    epochs = [ep["epoch"] for ep in data["epochs"]]
    train_loss = [ep["train_loss"] for ep in data["epochs"]]
    val_loss = [ep["val_loss"] for ep in data["epochs"]]
    train_acc = [ep["train_acc"] for ep in data["epochs"]]
    val_acc = [ep["val_acc"] for ep in data["epochs"]]
    
    # 1. 损失曲线
    ax1.plot(epochs, train_loss, 'b-', label='训练损失', linewidth=2, alpha=0.8)
    ax1.plot(epochs, val_loss, 'r-', label='验证损失', linewidth=2, alpha=0.8)
    ax1.set_title('MACHO训练损失曲线 (0-200 Epochs)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('损失值')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 200)
    
    # 2. 准确率曲线
    ax2.plot(epochs, train_acc, 'b-', label='训练准确率', linewidth=2, alpha=0.8)
    ax2.plot(epochs, val_acc, 'r-', label='验证准确率', linewidth=2, alpha=0.8)
    ax2.set_title('MACHO准确率曲线 (0-200 Epochs)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('准确率 (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 200)
    
    # 标记最佳验证准确率
    best_val_acc = max(val_acc)
    best_epoch = epochs[val_acc.index(best_val_acc)]
    ax2.axvline(x=best_epoch, color='g', linestyle='--', alpha=0.7)
    ax2.annotate(f'最佳: {best_val_acc:.2f}% @ Epoch {best_epoch}', 
                xy=(best_epoch, best_val_acc), xytext=(best_epoch+10, best_val_acc+2),
                arrowprops=dict(arrowstyle='->', color='green'), fontsize=10)
    
    # 3. F1分数曲线
    train_f1 = [ep["train_f1"] for ep in data["epochs"]]
    val_f1 = [ep["val_f1"] for ep in data["epochs"]]
    
    ax3.plot(epochs, train_f1, 'b-', label='训练F1', linewidth=2, alpha=0.8)
    ax3.plot(epochs, val_f1, 'r-', label='验证F1', linewidth=2, alpha=0.8)
    ax3.set_title('MACHO F1分数曲线 (0-200 Epochs)', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('F1分数')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, 200)
    
    # 4. 召回率曲线
    train_recall = [ep["train_recall"] for ep in data["epochs"]]
    val_recall = [ep["val_recall"] for ep in data["epochs"]]
    
    ax4.plot(epochs, train_recall, 'b-', label='训练召回率', linewidth=2, alpha=0.8)
    ax4.plot(epochs, val_recall, 'r-', label='验证召回率', linewidth=2, alpha=0.8)
    ax4.set_title('MACHO召回率曲线 (0-200 Epochs)', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('召回率 (%)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(0, 200)
    
    plt.tight_layout()
    
    # 保存图片到规范目录
    pics_dir = "/root/autodl-tmp/lnsde-contiformer/results/pics/MACHO"
    os.makedirs(pics_dir, exist_ok=True)
    plt.savefig(os.path.join(pics_dir, "macho_0_200_training_curves.png"), dpi=300, bbox_inches='tight')
    print(f"训练曲线图已保存到: {pics_dir}/macho_0_200_training_curves.png")
    
    return fig

def create_confusion_matrix_evolution(data):
    """创建混淆矩阵演化图 - MACHO 7类分类"""
    print("创建混淆矩阵演化图...")
    
    # MACHO数据集的7个类别
    class_names = ['RR Lyrae (ab)', 'RR Lyrae (c)', 'Cepheid', 'EB', 'Long Period Variable', 'Non-Variable', 'Quasar']
    
    # 选择关键epoch的混淆矩阵进行展示
    key_epochs = [50, 100, 150, 200]  # 选择4个关键节点
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.ravel()
    
    for idx, epoch_num in enumerate(key_epochs):
        if epoch_num <= len(data["epochs"]):
            # 获取对应epoch的混淆矩阵
            cm = np.array(data["epochs"][epoch_num-1]["confusion_matrix"])
            
            # 计算百分比
            cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
            cm_percent = np.nan_to_num(cm_percent)  # 处理除零情况
            
            # 绘制混淆矩阵热力图
            sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap='Blues', 
                       xticklabels=[f'C{i}' for i in range(7)],
                       yticklabels=[f'C{i}' for i in range(7)],
                       ax=axes[idx], cbar_kws={'label': '预测百分比 (%)'})
            
            axes[idx].set_title(f'Epoch {epoch_num} 混淆矩阵\\n准确率: {data["epochs"][epoch_num-1]["val_acc"]:.2f}%', 
                               fontsize=12, fontweight='bold')
            axes[idx].set_xlabel('预测类别')
            axes[idx].set_ylabel('真实类别')
    
    # 添加类别说明
    class_info = "\\n".join([f"C{i}: {name}" for i, name in enumerate(class_names)])
    fig.text(0.02, 0.02, f"类别说明:\\n{class_info}", fontsize=8, 
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    
    # 保存图片
    pics_dir = "/root/autodl-tmp/lnsde-contiformer/results/pics/MACHO"
    plt.savefig(os.path.join(pics_dir, "macho_confusion_matrix_evolution.png"), dpi=300, bbox_inches='tight')
    print(f"混淆矩阵演化图已保存到: {pics_dir}/macho_confusion_matrix_evolution.png")
    
    return fig

def create_final_confusion_matrix(data):
    """创建最终混淆矩阵详细分析图"""
    print("创建最终混淆矩阵详细分析图...")
    
    # MACHO数据集的7个类别（完整名称）
    class_names = ['RR Lyrae (ab)', 'RR Lyrae (c)', 'Cepheid', 'EB', 'Long Period Variable', 'Non-Variable', 'Quasar']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    
    # 获取最后一个epoch的混淆矩阵
    final_cm = np.array(data["epochs"][-1]["confusion_matrix"])
    final_epoch = data["epochs"][-1]["epoch"]
    final_acc = data["epochs"][-1]["val_acc"]
    
    # 左图：原始数量
    sns.heatmap(final_cm, annot=True, fmt='d', cmap='Blues', 
               xticklabels=[f'C{i}' for i in range(7)],
               yticklabels=[f'C{i}' for i in range(7)],
               ax=ax1, cbar_kws={'label': '样本数量'})
    
    ax1.set_title(f'MACHO最终混淆矩阵 (Epoch {final_epoch})\\n验证准确率: {final_acc:.2f}%', 
                  fontsize=14, fontweight='bold')
    ax1.set_xlabel('预测类别')
    ax1.set_ylabel('真实类别')
    
    # 右图：归一化百分比
    cm_percent = final_cm.astype('float') / final_cm.sum(axis=1)[:, np.newaxis] * 100
    cm_percent = np.nan_to_num(cm_percent)
    
    sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap='RdYlBu_r', 
               xticklabels=[f'C{i}' for i in range(7)],
               yticklabels=[f'C{i}' for i in range(7)],
               ax=ax2, cbar_kws={'label': '预测百分比 (%)'})
    
    ax2.set_title(f'归一化混淆矩阵 (百分比)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('预测类别')
    ax2.set_ylabel('真实类别')
    
    # 添加类别说明
    class_info = "\\n".join([f"C{i}: {name}" for i, name in enumerate(class_names)])
    fig.text(0.02, 0.02, f"MACHO类别说明:\\n{class_info}", fontsize=9, 
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    # 计算每类的精确率、召回率、F1分数
    precision = np.diag(final_cm) / np.sum(final_cm, axis=0)
    recall = np.diag(final_cm) / np.sum(final_cm, axis=1)
    f1 = 2 * (precision * recall) / (precision + recall)
    
    # 处理除零情况
    precision = np.nan_to_num(precision)
    recall = np.nan_to_num(recall)
    f1 = np.nan_to_num(f1)
    
    metrics_text = "每类性能指标:\\n"
    for i in range(7):
        metrics_text += f"C{i}: P={precision[i]:.3f} R={recall[i]:.3f} F1={f1[i]:.3f}\\n"
    
    fig.text(0.02, 0.5, metrics_text, fontsize=8, fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.tight_layout()
    
    # 保存图片
    pics_dir = "/root/autodl-tmp/lnsde-contiformer/results/pics/MACHO"
    plt.savefig(os.path.join(pics_dir, "macho_final_confusion_matrix.png"), dpi=300, bbox_inches='tight')
    print(f"最终混淆矩阵图已保存到: {pics_dir}/macho_final_confusion_matrix.png")
    
    return fig

def create_class_performance_trends(data):
    """创建各类别性能趋势图"""
    print("创建各类别性能趋势图...")
    
    class_names = ['RR Lyrae (ab)', 'RR Lyrae (c)', 'Cepheid', 'EB', 'Long Period Variable', 'Non-Variable', 'Quasar']
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.ravel()
    
    epochs = [ep["epoch"] for ep in data["epochs"]]
    
    # 为每个类别计算性能趋势
    for class_id in range(7):
        class_precisions = []
        class_recalls = []
        class_f1s = []
        
        for epoch_data in data["epochs"]:
            cm = np.array(epoch_data["confusion_matrix"])
            
            # 计算精确率
            if np.sum(cm[:, class_id]) > 0:
                precision = cm[class_id, class_id] / np.sum(cm[:, class_id])
            else:
                precision = 0
                
            # 计算召回率  
            if np.sum(cm[class_id, :]) > 0:
                recall = cm[class_id, class_id] / np.sum(cm[class_id, :])
            else:
                recall = 0
                
            # 计算F1分数
            if precision + recall > 0:
                f1 = 2 * precision * recall / (precision + recall)
            else:
                f1 = 0
            
            class_precisions.append(precision * 100)
            class_recalls.append(recall * 100)
            class_f1s.append(f1 * 100)
        
        # 绘制该类别的性能曲线
        axes[class_id].plot(epochs, class_precisions, 'b-', label='精确率', linewidth=2, alpha=0.8)
        axes[class_id].plot(epochs, class_recalls, 'r-', label='召回率', linewidth=2, alpha=0.8)
        axes[class_id].plot(epochs, class_f1s, 'g-', label='F1分数', linewidth=2, alpha=0.8)
        
        axes[class_id].set_title(f'类别 {class_id}: {class_names[class_id]}', fontsize=11, fontweight='bold')
        axes[class_id].set_xlabel('Epoch')
        axes[class_id].set_ylabel('性能指标 (%)')
        axes[class_id].legend(fontsize=8)
        axes[class_id].grid(True, alpha=0.3)
        axes[class_id].set_xlim(0, 200)
        axes[class_id].set_ylim(0, 100)
    
    # 隐藏最后一个空的子图
    axes[7].axis('off')
    
    plt.tight_layout()
    
    # 保存图片
    pics_dir = "/root/autodl-tmp/lnsde-contiformer/results/pics/MACHO"
    plt.savefig(os.path.join(pics_dir, "macho_class_performance_trends.png"), dpi=300, bbox_inches='tight')
    print(f"各类别性能趋势图已保存到: {pics_dir}/macho_class_performance_trends.png")
    
    return fig

def create_training_summary(data):
    """创建训练总结图"""
    print("创建训练总结图...")
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.axis('off')
    
    # 计算关键统计信息
    epochs = [ep["epoch"] for ep in data["epochs"]]
    val_accs = [ep["val_acc"] for ep in data["epochs"]]
    val_f1s = [ep["val_f1"] for ep in data["epochs"]]
    
    final_epoch = epochs[-1]
    final_val_acc = val_accs[-1]
    final_val_f1 = val_f1s[-1]
    best_val_acc = max(val_accs)
    best_epoch = epochs[val_accs.index(best_val_acc)]
    
    # 获取最终混淆矩阵
    final_cm = np.array(data["epochs"][-1]["confusion_matrix"])
    
    # 计算每类的性能
    class_names = ['RR Lyrae (ab)', 'RR Lyrae (c)', 'Cepheid', 'EB', 'Long Period Variable', 'Non-Variable', 'Quasar']
    class_performance = []
    
    for i in range(7):
        if np.sum(final_cm[:, i]) > 0:
            precision = final_cm[i, i] / np.sum(final_cm[:, i])
        else:
            precision = 0
            
        if np.sum(final_cm[i, :]) > 0:
            recall = final_cm[i, i] / np.sum(final_cm[i, :])
        else:
            recall = 0
            
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0
            
        class_performance.append((i, class_names[i], precision*100, recall*100, f1*100))
    
    # 创建总结文本
    summary_text = f"""
MACHO数据集训练完整报告 (0-200 Epochs)

═══════════════════════════════════════════════════════
训练配置信息
═══════════════════════════════════════════════════════
数据集: {data["dataset"]}
模型类型: {data["model_type"]} 
SDE配置: {data["sde_config"]}
开始时间: {data["start_time"]}
总训练轮次: {final_epoch}

═══════════════════════════════════════════════════════
整体性能指标
═══════════════════════════════════════════════════════
最终验证准确率: {final_val_acc:.2f}%
最终验证F1分数: {final_val_f1:.2f}%
最佳验证准确率: {best_val_acc:.2f}% (Epoch {best_epoch})

═══════════════════════════════════════════════════════
各类别详细性能 (最终epoch)
═══════════════════════════════════════════════════════"""

    for class_id, class_name, precision, recall, f1 in class_performance:
        summary_text += f"\\n类别{class_id} ({class_name[:15]}...):\\n"
        summary_text += f"    精确率: {precision:6.2f}%  召回率: {recall:6.2f}%  F1分数: {f1:6.2f}%"

    summary_text += f"""\\n
═══════════════════════════════════════════════════════
训练收敛情况分析
═══════════════════════════════════════════════════════
初始验证准确率: {val_accs[0]:.2f}%
最终验证准确率: {final_val_acc:.2f}%
性能提升: {final_val_acc - val_accs[0]:+.2f}%

模型在第{best_epoch}轮达到最佳性能，最终性能稳定在{final_val_acc:.2f}%
"""
    
    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=10, 
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    
    # 保存图片
    pics_dir = "/root/autodl-tmp/lnsde-contiformer/results/pics/MACHO"
    plt.savefig(os.path.join(pics_dir, "macho_training_summary.png"), dpi=300, bbox_inches='tight')
    print(f"训练总结图已保存到: {pics_dir}/macho_training_summary.png")
    
    return fig

def main():
    """主函数"""
    print("配置中文字体...")
    configure_chinese_font()
    
    print("开始创建MACHO 0-200 epochs可视化图表...")
    
    # 加载数据
    print("加载合并后的训练数据...")
    data = load_merged_log()
    print(f"数据加载完成，共{len(data['epochs'])}个epoch")
    
    # 创建所有可视化图表
    create_training_curves(data)
    create_confusion_matrix_evolution(data)
    create_final_confusion_matrix(data)
    create_class_performance_trends(data)
    create_training_summary(data)
    
    print("\\n🎉 所有MACHO可视化图表创建完成！")
    print("图表保存位置: /root/autodl-tmp/lnsde-contiformer/results/pics/MACHO/")

if __name__ == "__main__":
    main()