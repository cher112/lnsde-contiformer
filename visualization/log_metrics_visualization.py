#!/usr/bin/env python3

import json
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import os

def configure_chinese_font():
    """配置中文字体显示"""
    try:
        # 添加字体到matplotlib管理器
        fm.fontManager.addfont('/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc')
        fm.fontManager.addfont('/usr/share/fonts/truetype/wqy/wqy-microhei.ttc')
        
        # 设置中文字体优先级列表
        plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'WenQuanYi Micro Hei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 抑制字体警告
        import warnings
        warnings.filterwarnings("ignore", category=UserWarning, message=".*Glyph.*missing from font.*")
        warnings.filterwarnings("ignore", category=UserWarning, message=".*Font.*does not have a glyph.*")
        
        return True
    except Exception as e:
        print(f"字体配置失败: {e}")
        # 使用英文标签作为后备
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        return False

def load_log_data(log_path):
    """加载日志数据"""
    with open(log_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def calculate_basic_metrics_from_confusion_matrix(cm):
    """从混淆矩阵计算基础（非宏观）准确度、召回率、F1分数"""
    n_classes = cm.shape[0]
    total_samples = np.sum(cm)
    
    # 准确度 = 正确预测的样本数 / 总样本数
    accuracy = np.trace(cm) / total_samples
    
    # 计算每个类别的精确率、召回率、F1分数
    per_class_precision = []
    per_class_recall = []
    per_class_f1 = []
    
    for i in range(n_classes):
        # True Positives: 正确预测为类别i的样本数
        tp = cm[i, i]
        
        # False Negatives: 实际是类别i但预测错误的样本数
        fn = np.sum(cm[i, :]) - tp
        
        # False Positives: 实际不是类别i但预测为类别i的样本数  
        fp = np.sum(cm[:, i]) - tp
        
        # 召回率 = TP / (TP + FN)
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        per_class_recall.append(recall)
        
        # 精确率 = TP / (TP + FP) 
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        per_class_precision.append(precision)
        
        # F1分数 = 2 * (precision * recall) / (precision + recall)
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        per_class_f1.append(f1)
    
    # 计算加权平均（按每个类别的真实样本数加权）
    class_sample_counts = np.sum(cm, axis=1)  # 每个类别的真实样本数
    
    weighted_recall = np.sum([per_class_recall[i] * class_sample_counts[i] for i in range(n_classes)]) / total_samples
    weighted_f1 = np.sum([per_class_f1[i] * class_sample_counts[i] for i in range(n_classes)]) / total_samples
    
    return accuracy * 100, weighted_recall * 100, weighted_f1 * 100

def extract_training_metrics(data):
    """提取训练过程中的指标"""
    epochs = []
    accuracies = []
    recalls = []
    f1_scores = []
    
    for epoch_data in data['epochs']:
        epochs.append(epoch_data['epoch'])
        
        # 从混淆矩阵重新计算基础指标
        cm = np.array(epoch_data['confusion_matrix'])
        acc, recall, f1 = calculate_basic_metrics_from_confusion_matrix(cm)
        
        accuracies.append(acc)
        recalls.append(recall)
        f1_scores.append(f1)
    
    return epochs, accuracies, recalls, f1_scores

def create_metric_plot(epochs, values, metric_name, ylabel, save_path, use_chinese=True):
    """创建单个指标的训练曲线图 - 模仿main函数风格"""
    plt.figure(figsize=(12, 8))
    
    # 根据是否支持中文选择标签
    if use_chinese:
        xlabel = '训练轮次 (Epoch)'
        title_prefix = '训练曲线'
        best_text = '最佳'
    else:
        xlabel = 'Epoch'
        title_prefix = 'Training Curve'
        best_text = 'Best'
    
    # 绘制曲线 - 使用验证集风格（红色）
    plt.plot(epochs, values, 'r-', linewidth=2, label=f'验证 {metric_name}', marker='s', markersize=4)
    
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.title(f'{metric_name} {title_prefix}', fontsize=16, fontweight='bold')
    
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    
    # 添加最佳值标注
    best_idx = np.argmax(values)
    best_val = values[best_idx]
    
    # 标注最佳点
    plt.annotate(f'{best_text}: {best_val:.4f}', 
                xy=(epochs[best_idx], best_val),
                xytext=(10, -20), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightcoral', alpha=0.7),
                arrowprops=dict(arrowstyle='->', color='red'),
                fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return save_path

def main():
    # 配置中文字体
    use_chinese = configure_chinese_font()
    
    # 日志文件路径
    log_paths = {
        'LINEAR': '/root/autodl-tmp/lnsde-contiformer/results/20250828/LINEAR/2116/logs/LINEAR_linear_noise_config1_20250828_194129.log',
        'MACHO': '/root/autodl-tmp/lnsde-contiformer/results/20250828/MACHO/2116/logs/MACHO_linear_noise_config1_20250828_194135.log'
    }
    
    # 创建输出目录
    output_base = '/root/autodl-tmp/lnsde-contiformer/results/pics'
    
    # 为每个数据集生成图表
    for dataset_name, log_path in log_paths.items():
        if not os.path.exists(log_path):
            print(f"警告: 日志文件不存在: {log_path}")
            continue
            
        print(f"正在处理 {dataset_name} 数据集...")
        
        # 创建数据集目录
        dataset_dir = os.path.join(output_base, dataset_name)
        os.makedirs(dataset_dir, exist_ok=True)
        
        # 加载数据
        data = load_log_data(log_path)
        epochs, accuracies, recalls, f1_scores = extract_training_metrics(data)
        
        # 生成各个指标的曲线图 - 按照main函数风格
        metric_configs = [
            (accuracies, 'Accuracy', '准确率 (%)' if use_chinese else 'Accuracy (%)', 'accuracy'),
            (recalls, 'Recall', '召回率 (%)' if use_chinese else 'Recall (%)', 'recall'),
            (f1_scores, 'F1-Score', 'F1分数 (%)' if use_chinese else 'F1 Score (%)', 'f1')
        ]
        
        generated_files = []
        for values, metric_name, ylabel, file_suffix in metric_configs:
            save_path = os.path.join(dataset_dir, f'{file_suffix}_curve.png')
            created_path = create_metric_plot(epochs, values, metric_name, ylabel, save_path, use_chinese)
            generated_files.append(created_path)
            print(f"已生成 {dataset_name} {metric_name} 曲线图: {created_path}")
        
        print(f"{dataset_name} 数据集处理完成，生成了 {len(generated_files)} 个图表")
    
    print("\n所有可视化完成！")

if __name__ == "__main__":
    main()