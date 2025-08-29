#!/usr/bin/env python3
"""
从训练日志中提取每个数据集的最佳原始F1分数、召回率和准确率
基于混淆矩阵计算，而非macro平均
"""

import os
import json
import glob
from typing import Dict, List, Optional
import numpy as np

def parse_confusion_matrix(matrix_text: str) -> Optional[np.ndarray]:
    """解析混淆矩阵文本"""
    try:
        lines = matrix_text.strip().split('\n')
        matrix_lines = []
        
        for line in lines:
            line = line.strip()
            if line and not line.startswith('[') and not line.startswith('tensor'):
                # 提取数字
                numbers = re.findall(r'\d+(?:\.\d+)?', line)
                if numbers:
                    matrix_lines.append([float(n) for n in numbers])
        
        if matrix_lines:
            return np.array(matrix_lines)
    except Exception as e:
        print(f"解析混淆矩阵出错: {e}")
    return None

def calculate_metrics_from_cm(cm: np.ndarray) -> Dict[str, float]:
    """从混淆矩阵计算原始指标（非macro）"""
    if cm.shape[0] != 2 or cm.shape[1] != 2:
        return {}
    
    tn, fp, fn, tp = cm.ravel()
    
    # 计算准确率
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    
    # 计算召回率 (真正率)
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    # 计算精确率
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    
    # 计算F1分数
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'accuracy': accuracy,
        'precision': precision, 
        'recall': recall,
        'f1': f1,
        'tp': tp,
        'tn': tn,
        'fp': fp,
        'fn': fn
    }

def extract_metrics_from_log(log_file: str) -> List[Dict]:
    """从日志文件中提取所有epoch的指标"""
    metrics_list = []
    
    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 查找所有测试结果块
        test_blocks = re.findall(r'Testing\s+results.*?(?=\n(?:Epoch|Testing|$))', content, re.DOTALL | re.MULTILINE)
        
        for block in test_blocks:
            # 提取epoch信息
            epoch_match = re.search(r'Epoch\s+(\d+)', block)
            epoch = int(epoch_match.group(1)) if epoch_match else None
            
            # 查找混淆矩阵
            cm_match = re.search(r'Confusion Matrix.*?\n(.*?)(?=\n\w|\n$)', block, re.DOTALL)
            if cm_match:
                cm_text = cm_match.group(1)
                cm = parse_confusion_matrix(cm_text)
                
                if cm is not None:
                    metrics = calculate_metrics_from_cm(cm)
                    if metrics:
                        metrics['epoch'] = epoch
                        metrics['log_file'] = os.path.basename(log_file)
                        metrics_list.append(metrics)
    
    except Exception as e:
        print(f"处理日志文件 {log_file} 时出错: {e}")
    
    return metrics_list

def find_best_metrics_for_dataset(dataset: str) -> Dict:
    """查找指定数据集的最佳指标"""
    log_pattern = f"/root/autodl-tmp/lnsde-contiformer/results/**/{dataset}/**/logs/*.log"
    log_files = glob.glob(log_pattern, recursive=True)
    
    all_metrics = []
    for log_file in log_files:
        metrics = extract_metrics_from_json_log(log_file)
        all_metrics.extend(metrics)
    
    if not all_metrics:
        return {'dataset': dataset, 'status': 'no_data'}
    
    # 按F1分数排序找最佳
    best_by_f1 = max(all_metrics, key=lambda x: x['f1'])
    best_by_accuracy = max(all_metrics, key=lambda x: x['accuracy'])
    best_by_recall = max(all_metrics, key=lambda x: x['recall'])
    
    return {
        'dataset': dataset,
        'total_experiments': len(all_metrics),
        'best_f1': {
            'f1': best_by_f1['f1'],
            'recall': best_by_f1['recall'],
            'accuracy': best_by_f1['accuracy'],
            'precision': best_by_f1['precision'],
            'epoch': best_by_f1['epoch'],
            'log_file': best_by_f1['log_file'],
            'confusion_matrix': {
                'tp': int(best_by_f1['tp']),
                'tn': int(best_by_f1['tn']), 
                'fp': int(best_by_f1['fp']),
                'fn': int(best_by_f1['fn'])
            }
        },
        'best_accuracy': {
            'accuracy': best_by_accuracy['accuracy'],
            'f1': best_by_accuracy['f1'],
            'recall': best_by_accuracy['recall'],
            'precision': best_by_accuracy['precision'],
            'epoch': best_by_accuracy['epoch'],
            'log_file': best_by_accuracy['log_file']
        },
        'best_recall': {
            'recall': best_by_recall['recall'],
            'f1': best_by_recall['f1'],
            'accuracy': best_by_recall['accuracy'],
            'precision': best_by_recall['precision'],
            'epoch': best_by_recall['epoch'],
            'log_file': best_by_recall['log_file']
        }
    }

def main():
    datasets = ['ASAS', 'LINEAR', 'MACHO']
    
    print("="*80)
    print("数据集最佳指标汇总报告")
    print("="*80)
    print()
    
    for dataset in datasets:
        print(f"📊 {dataset} 数据集分析")
        print("-" * 40)
        
        results = find_best_metrics_for_dataset(dataset)
        
        if results['status'] == 'no_data':
            print(f"❌ 未找到 {dataset} 数据集的日志文件")
            print()
            continue
        
        print(f"📈 总实验次数: {results['total_experiments']}")
        print()
        
        # 最佳F1分数结果
        best_f1 = results['best_f1']
        print("🏆 最佳F1分数结果:")
        print(f"   F1 Score: {best_f1['f1']:.4f}")
        print(f"   召回率 (Recall): {best_f1['recall']:.4f}")
        print(f"   准确率 (Accuracy): {best_f1['accuracy']:.4f}")
        print(f"   精确率 (Precision): {best_f1['precision']:.4f}")
        print(f"   Epoch: {best_f1['epoch']}")
        print(f"   日志文件: {best_f1['log_file']}")
        print(f"   混淆矩阵: TP={best_f1['confusion_matrix']['tp']}, TN={best_f1['confusion_matrix']['tn']}, FP={best_f1['confusion_matrix']['fp']}, FN={best_f1['confusion_matrix']['fn']}")
        print()
        
        # 如果最佳准确率和最佳F1不同，也显示
        if results['best_accuracy']['accuracy'] != best_f1['accuracy']:
            best_acc = results['best_accuracy']
            print("🎯 最佳准确率结果:")
            print(f"   准确率 (Accuracy): {best_acc['accuracy']:.4f}")
            print(f"   F1 Score: {best_acc['f1']:.4f}")
            print(f"   召回率 (Recall): {best_acc['recall']:.4f}")
            print(f"   Epoch: {best_acc['epoch']}")
            print(f"   日志文件: {best_acc['log_file']}")
            print()
        
        # 如果最佳召回率和最佳F1不同，也显示
        if results['best_recall']['recall'] != best_f1['recall']:
            best_rec = results['best_recall']
            print("🔍 最佳召回率结果:")
            print(f"   召回率 (Recall): {best_rec['recall']:.4f}")
            print(f"   F1 Score: {best_rec['f1']:.4f}")
            print(f"   准确率 (Accuracy): {best_rec['accuracy']:.4f}")
            print(f"   Epoch: {best_rec['epoch']}")
            print(f"   日志文件: {best_rec['log_file']}")
            print()
        
        print()
    
    # 汇总表格
    print("="*80)
    print("三个数据集最佳指标汇总表")
    print("="*80)
    print(f"{'数据集':<10} {'最佳F1':<10} {'最佳召回率':<12} {'最佳准确率':<12}")
    print("-" * 50)
    
    for dataset in datasets:
        results = find_best_metrics_for_dataset(dataset)
        if results.get('status') != 'no_data':
            best_f1 = results['best_f1']['f1']
            best_recall = max(results['best_f1']['recall'], results['best_recall']['recall'])
            best_acc = max(results['best_f1']['accuracy'], results['best_accuracy']['accuracy'])
            print(f"{dataset:<10} {best_f1:<10.4f} {best_recall:<12.4f} {best_acc:<12.4f}")

if __name__ == '__main__':
    main()